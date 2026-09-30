# %% [markdown]
# # Переход обычной железной дороги на сервис МЦД и результаты фирм
#
# Независимый кейс. Сначала берём широкий предварительный реестр железнодорожных
# станций, затем строим одинаковые catchment-гексы у станций
# D1–D4 и у направлений, оставшихся обычной железной дорогой. На выходе две
# панели Parquet для R и слои GeoPackage для QGIS. Крупные расчёты кэшируются.

# %% [markdown]
# ## 0. Среда и параметры
#
# Линия МЦД, а не отдельная платформа, получает treatment при запуске сервиса.
# В годовой отчётности первый полный год — 2020 для D1/D2 и 2024 для D3/D4.
# Выбираем Python 3.11 и запускаем notebook из корня этого репозитория.

# %%
from pathlib import Path
import math
import os
import tempfile
import warnings

os.environ.setdefault("USE_PYGEOS", "0")
import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import requests
from pyproj import Transformer
from shapely.geometry import Polygon

ROOT = Path.cwd()
assert (ROOT / "input" / "mcd_line_launches.csv").exists(), "Откройте корень репозитория в VS Code"
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

YEARS = range(2014, 2026)
CRS_METRIC = 32637
BBOX = (54.20, 35.10, 57.05, 40.30)  # south, west, north, east: Москва + Московская область с запасом
RADIUS_M = 800
HEX_AREA_M2 = 62_500
MAP_YEAR = 2023
USE_HOUSES = False
HOUSES_CSV = ROOT / "input" / "houses.csv"
FORCE_REBUILD = False
GEOCODING_QUALITY = ("house",)
RAW_RROOT = ROOT / "input" / "RFSD_hf"

def sql_path(path):
    return str(path).replace("\\", "/").replace("'", "''")

print("Проект:", ROOT)
print("RFSD:", RAW_RROOT)
print("DuckDB:", duckdb.__version__)

# %% [markdown]
# ## 1. Скачивание RFSD и аудит схемы
#
# Существующие partitions из input/RFSD_hf переиспользуются. Иначе Hugging Face
# скачивает 2014–2025 гг. Загрузка может занять долгое время.

# %%
raw_files = [RAW_RROOT / "RFSD" / f"year={year}" / "part-0.parquet" for year in YEARS]
if not all(path.exists() for path in raw_files):
    from huggingface_hub import snapshot_download
    RAW_RROOT.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="irlspbru/RFSD", repo_type="dataset", revision="main",
        allow_patterns=[f"RFSD/year={year}/*.parquet" for year in YEARS],
        local_dir=RAW_RROOT, max_workers=4,
    )
    raw_files = [RAW_RROOT / "RFSD" / f"year={year}" / "part-0.parquet" for year in YEARS]
assert all(path.exists() for path in raw_files), "Не все годовые partitions RFSD доступны"

con = duckdb.connect(str(OUT / "work.duckdb"))
con.execute("PRAGMA threads=4")
raw_glob = sql_path(RAW_RROOT / "RFSD" / "year=*" / "*.parquet")
reader = f"read_parquet('{raw_glob}', hive_partitioning=true, union_by_name=true)"
display(con.sql(f"DESCRIBE SELECT * FROM {reader}").df())

# %% [markdown]
# ## 2. Железнодорожные станции и даты запуска МЦД
#
# Реестр охватывает все найденные в Wikidata радиальные направления МЖД,
# Ленинградское направление ОЖД (для D3) и Большое кольцо. P1192 задаёт сервис
# МЦД, P81 — физическую линию, P1619 — открытие станции. Это предварительные
# данные: пустая дата МЦД у станций на его коридорах требует ручной сверки.

# %%
launches = pd.read_csv(ROOT / "input" / "mcd_line_launches.csv", parse_dates=["launch_date"])
assert set(launches.mcd_line) == {"D1", "D2", "D3", "D4"}

import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.build_station_registry import build as build_station_registry

REGISTRY = ROOT / "input" / "rail_station_treatment_registry.csv"
if not REGISTRY.exists():
    build_station_registry()
all_stations = pd.read_csv(REGISTRY, sep=";", encoding="utf-8-sig", dtype={"station_id": str})
display(all_stations.groupby(["treatment_group", "review_status"]).size().rename("stations"))
all_stations_gdf = gpd.GeoDataFrame(
    all_stations, geometry=gpd.points_from_xy(all_stations.lon, all_stations.lat), crs=4326
).to_crs(CRS_METRIC)
all_stations_gdf.to_file(OUT / "qgis_rail_stations_all.gpkg", layer="rail_stations_all", driver="GPKG")
stations = all_stations.loc[all_stations.did_eligible_candidate.eq(1)].copy()
stations["mcd_line"] = stations.mcd_line.fillna("ORDINARY")
print(f"Реестр: {len(all_stations)} станций; строгая предварительная выборка: {len(stations)}")

required = {"station_id", "station_name", "lon", "lat", "mcd_line",
            "corridor_id", "baseline_operating", "source_url"}
assert required <= set(stations), f"Не хватает полей: {required - set(stations)}"
stations["baseline_operating"] = pd.to_numeric(
    stations.baseline_operating, errors="coerce"
).fillna(0)
for column in ["lon", "lat"]:
    stations[column] = pd.to_numeric(stations[column], errors="coerce")
for column in ["mcd_line", "corridor_id"]:
    stations[column] = stations[column].fillna("").astype(str).str.strip().str.upper()
stations = stations[stations.baseline_operating.eq(1)].copy()
assert len(stations), "Нет станций, работавших до 2014 г.: проверьте реестр"
assert stations.station_id.is_unique
assert stations.mcd_line.isin(["D1", "D2", "D3", "D4", "ORDINARY"]).all()
assert stations.corridor_id.notna().all() and stations.corridor_id.ne("").all()
assert stations.loc[stations.mcd_line.ne("ORDINARY"), "corridor_id"].eq(
    stations.loc[stations.mcd_line.ne("ORDINARY"), "mcd_line"]
).all()
assert stations.loc[stations.mcd_line.eq("ORDINARY"), "corridor_id"].ne("ORDINARY").all()
assert stations.lon.between(BBOX[1], BBOX[3]).all()
assert stations.lat.between(BBOX[0], BBOX[2]).all()
assert stations.source_url.fillna("").str.len().gt(0).all()
stations = stations.merge(launches[["mcd_line", "launch_date"]], on="mcd_line", how="left", validate="many_to_one")
stations["g"] = stations.launch_date.dt.year.add(1).fillna(0).astype(int)
stations_gdf = gpd.GeoDataFrame(
    stations, geometry=gpd.points_from_xy(stations.lon, stations.lat), crs=4326
).to_crs(CRS_METRIC)
display(stations.groupby(["mcd_line", "g"]).size().rename("stations"))
assert stations.mcd_line.eq("ORDINARY").any(), "Нет станций обычной железной дороги для контроля"
assert stations.g.gt(0).any(), "Нет станций МЦД"

# %% [markdown]
# ## 3. Гексы и exposure
#
# Гексы строятся вокруг всех проверенных железнодорожных остановок. При
# наложении catchment разных направлений ячейка остаётся в экспорте, но
# исключается из причинной выборки. Когорты определяются линией МЦД.

# %%
side = math.sqrt(2 * HEX_AREA_M2 / (3 * math.sqrt(3)))
height = math.sqrt(3) * side
records = {}
for point in stations_gdf.geometry:
    col_min = math.floor((point.x - RADIUS_M - side) / (1.5 * side))
    col_max = math.ceil((point.x + RADIUS_M + side) / (1.5 * side))
    for col in range(col_min, col_max + 1):
        cx = 1.5 * side * col
        offset = height / 2 if col % 2 else 0
        row_min = math.floor((point.y - RADIUS_M - height - offset) / height)
        row_max = math.ceil((point.y + RADIUS_M + height - offset) / height)
        for row in range(row_min, row_max + 1):
            cy = height * row + offset
            if math.hypot(cx - point.x, cy - point.y) <= RADIUS_M:
                records[(col, row)] = (cx, cy)

grid_rows = []
for (col, row), (cx, cy) in records.items():
    angles = np.deg2rad(np.arange(0, 360, 60))
    vertices = [(cx + side * math.cos(a), cy + side * math.sin(a)) for a in angles]
    grid_rows.append((f"h_{col}_{row}", Polygon(vertices)))
grid = gpd.GeoDataFrame(grid_rows, columns=["cell_id", "geometry"], crs=CRS_METRIC)
assert grid.cell_id.is_unique and grid.geometry.is_valid.all()

centers = grid[["cell_id", "geometry"]].copy()
centers.geometry = centers.centroid
buffers = stations_gdf[["station_id", "corridor_id", "geometry"]].copy()
buffers.geometry = buffers.buffer(RADIUS_M)
pairs = gpd.sjoin(centers, buffers, how="inner", predicate="within")
points_by_id = stations_gdf.set_index("station_id").geometry.to_dict()
pairs["distance_m"] = [
    geom.distance(points_by_id[sid]) for geom, sid in zip(pairs.geometry, pairs.station_id)
]
cross_corridor = pairs.groupby("cell_id").corridor_id.nunique().rename("n_corridors")
nearest = (pairs.sort_values(["cell_id", "distance_m", "station_id"])
           .drop_duplicates("cell_id")[["cell_id", "station_id", "distance_m"]])
exposure = nearest.merge(cross_corridor, on="cell_id").merge(
    stations.drop(columns=["lon", "lat", "railway_tag", "line_hint"], errors="ignore"),
    on="station_id", validate="many_to_one"
)
exposure["ambiguous"] = exposure.n_corridors.gt(1).astype("int8")
exposure["did_sample"] = exposure.ambiguous.eq(0).astype("int8")
exposure["event_line"] = exposure.mcd_line
exposure["station_cluster_id"] = exposure.corridor_id
cells = grid.merge(exposure, on="cell_id", validate="one_to_one")
assert len(cells) == len(exposure)
display(exposure.groupby(["mcd_line", "g", "did_sample"]).size().rename("hexes"))

# %% [markdown]
# ## 4. Отбор RFSD в DuckDB
#
# Читаем годовые Parquet без загрузки всего RFSD в память. Оставляем московский
# bbox, год, юридический адрес с точностью до дома, ОКВЭД и строки отчётности.
# Кэш этого проекта отделён от кэша метро.

# %%
FIRMS_CACHE = OUT / "firms_moscow_oblast_filtered.parquet"
section_case = """CASE
 WHEN code2 BETWEEN 1 AND 3 THEN 'A' WHEN code2 BETWEEN 5 AND 9 THEN 'B'
 WHEN code2 BETWEEN 10 AND 33 THEN 'C' WHEN code2=35 THEN 'D'
 WHEN code2 BETWEEN 36 AND 39 THEN 'E' WHEN code2 BETWEEN 41 AND 43 THEN 'F'
 WHEN code2 BETWEEN 45 AND 47 THEN 'G' WHEN code2 BETWEEN 49 AND 53 THEN 'H'
 WHEN code2 BETWEEN 55 AND 56 THEN 'I' WHEN code2 BETWEEN 58 AND 63 THEN 'J'
 WHEN code2 BETWEEN 64 AND 66 THEN 'K' WHEN code2=68 THEN 'L'
 WHEN code2 BETWEEN 69 AND 75 THEN 'M' WHEN code2 BETWEEN 77 AND 82 THEN 'N'
 WHEN code2=84 THEN 'O' WHEN code2=85 THEN 'P'
 WHEN code2 BETWEEN 86 AND 88 THEN 'Q' WHEN code2 BETWEEN 90 AND 93 THEN 'R'
 WHEN code2 BETWEEN 94 AND 96 THEN 'S' WHEN code2 BETWEEN 97 AND 98 THEN 'T'
 WHEN code2=99 THEN 'U' END"""
quality_filter = ", ".join(f"'{x}'" for x in GEOCODING_QUALITY)
if FORCE_REBUILD or not FIRMS_CACHE.exists():
    query = f"""
    WITH parsed AS (
      SELECT cast(inn AS VARCHAR) AS firm_id, try_cast(year AS INTEGER) AS year,
        try_cast(replace(cast(lon AS VARCHAR), ',', '.') AS DOUBLE) AS lon,
        try_cast(replace(cast(lat AS VARCHAR), ',', '.') AS DOUBLE) AS lat,
        replace(cast(okved AS VARCHAR), ',', '.') AS okved,
        try_cast(substr(replace(cast(okved AS VARCHAR), ',', '.'), 1, 2) AS INTEGER) AS code2,
        cast(geocoding_quality AS VARCHAR) AS geocoding_quality,
        try_cast(filed AS DOUBLE) AS filed,
        try_cast(eligible AS DOUBLE) AS eligible,
        try_cast(outlier AS DOUBLE) AS outlier,
        try_cast(replace(cast(line_2110 AS VARCHAR), ',', '.') AS DOUBLE) AS revenue,
        try_cast(replace(cast(line_1600 AS VARCHAR), ',', '.') AS DOUBLE) AS assets,
        try_cast(replace(cast(line_2400 AS VARCHAR), ',', '.') AS DOUBLE) AS profit,
        try_cast(replace(cast(line_2120 AS VARCHAR), ',', '.') AS DOUBLE) AS costs
      FROM {reader}
    ), eligible AS (
      SELECT *, {section_case} AS okved_section FROM parsed
      WHERE year BETWEEN {min(YEARS)} AND {max(YEARS)}
        AND lon BETWEEN {BBOX[1]} AND {BBOX[3]}
        AND lat BETWEEN {BBOX[0]} AND {BBOX[2]}
        AND geocoding_quality IN ({quality_filter})
        AND coalesce(eligible, 1)=1 AND coalesce(outlier, 0)=0
        AND firm_id IS NOT NULL
    )
    SELECT firm_id, year, lon, lat, okved, okved_section, filed,
           revenue, assets, profit, costs
    FROM eligible
    QUALIFY row_number() OVER (
      PARTITION BY firm_id, year ORDER BY filed DESC NULLS LAST,
      revenue DESC NULLS LAST
    )=1
    ORDER BY year, firm_id
    """
    con.execute(f"COPY ({query}) TO '{sql_path(FIRMS_CACHE)}' (FORMAT PARQUET, COMPRESSION ZSTD)")
display(con.sql(f"SELECT year, count(*) firms FROM read_parquet('{sql_path(FIRMS_CACHE)}') GROUP BY year ORDER BY year").df())

# %% [markdown]
# ## 5. Пространственная привязка firm-year
#
# Обрабатываем Arrow batch-ами. Сохраняем и фирмы вне catchment: это позволяет
# проверить переезды после первой географической привязки.

# %%
SPATIAL_CACHE = OUT / "firm_year_spatial_moscow_oblast.parquet"
if (FORCE_REBUILD or not SPATIAL_CACHE.exists()
        or REGISTRY.stat().st_mtime > SPATIAL_CACHE.stat().st_mtime):
    dataset = pads.dataset(FIRMS_CACHE, format="parquet")
    transformer = Transformer.from_crs(4326, CRS_METRIC, always_xy=True)
    lookup = grid[["cell_id", "geometry"]]
    writer = None
    temporary = tempfile.NamedTemporaryFile(suffix=".parquet", dir=OUT, delete=False)
    try:
        for batch in dataset.to_batches(batch_size=200_000):
            x = batch.to_pandas()
            x["firm_id"] = x.firm_id.astype("string")
            x["okved_section"] = x.okved_section.astype("string")
            x["okved"] = x.okved.astype("string")
            x["year"] = x.year.astype("int32")
            x["filed"] = pd.to_numeric(x.filed, errors="coerce").astype("float64")
            for col in ["revenue", "assets", "profit", "costs"]:
                x[col] = pd.to_numeric(x[col], errors="coerce").astype("float64")
            mx, my = transformer.transform(x.lon.to_numpy(), x.lat.to_numpy())
            points = gpd.GeoDataFrame(x, geometry=gpd.points_from_xy(mx, my), crs=CRS_METRIC)
            joined = gpd.sjoin(points, lookup, how="left", predicate="within")
            joined = pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))
            joined["cell_id"] = joined.cell_id.astype("string")
            table = pa.Table.from_pandas(joined, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary.name, table.schema, compression="zstd")
            else:
                table = table.cast(writer.schema, safe=False)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
        temporary.close()
    Path(temporary.name).replace(SPATIAL_CACHE)
display(con.sql(f"SELECT count(*) AS n_rows, count(cell_id) AS near_rail FROM read_parquet('{sql_path(SPATIAL_CACHE)}')").df())

# %% [markdown]
# ## 6. Дома АИС ФРТ: только предопределённые характеристики
#
# Необязательная тяжёлая ячейка. Текущую оценку числа жителей не используем как
# население 2014 года. Площадь и число квартир домов с годом постройки до 2014
# служат baseline переменными.

# %%
HOUSES_CACHE = OUT / "houses_baseline_by_hex.parquet"
if USE_HOUSES and (FORCE_REBUILD or not HOUSES_CACHE.exists()
                   or REGISTRY.stat().st_mtime > HOUSES_CACHE.stat().st_mtime):
    assert HOUSES_CSV.exists(), f"Нет {HOUSES_CSV}"
    columns = pd.read_csv(HOUSES_CSV, nrows=0).columns
    needed = [c for c in ["id", "built_year", "area_residential", "quarters_count",
                          "longitude", "latitude"] if c in columns]
    assert {"built_year", "longitude", "latitude"} <= set(needed)
    transformer = Transformer.from_crs(4326, CRS_METRIC, always_xy=True)
    chunks = []
    for chunk in pd.read_csv(HOUSES_CSV, usecols=needed, chunksize=150_000, low_memory=False):
        chunk["built_year"] = pd.to_numeric(chunk.built_year, errors="coerce")
        chunk = chunk[chunk.built_year.le(2013)].copy()
        chunk["longitude"] = pd.to_numeric(chunk.longitude, errors="coerce")
        chunk["latitude"] = pd.to_numeric(chunk.latitude, errors="coerce")
        chunk = chunk.dropna(subset=["longitude", "latitude"])
        if chunk.empty:
            continue
        mx, my = transformer.transform(chunk.longitude.to_numpy(), chunk.latitude.to_numpy())
        points = gpd.GeoDataFrame(chunk, geometry=gpd.points_from_xy(mx, my), crs=CRS_METRIC)
        joined = gpd.sjoin(points, grid[["cell_id", "geometry"]], how="inner", predicate="within")
        for col in ["area_residential", "quarters_count"]:
            if col not in joined:
                joined[col] = 0
            joined[col] = pd.to_numeric(joined[col], errors="coerce").fillna(0)
        chunks.append(joined.groupby("cell_id", as_index=False).agg(
            pre2014_res_area=("area_residential", "sum"),
            pre2014_units=("quarters_count", "sum"),
            pre2014_buildings=("built_year", "size"),
        ))
    if chunks:
        houses = pd.concat(chunks, ignore_index=True).groupby("cell_id", as_index=False).sum()
    else:
        houses = pd.DataFrame(columns=[
            "cell_id", "pre2014_res_area", "pre2014_units", "pre2014_buildings"
        ])
    houses.to_parquet(HOUSES_CACHE, index=False)
elif USE_HOUSES:
    houses = pd.read_parquet(HOUSES_CACHE)
else:
    houses = pd.DataFrame({"cell_id": grid.cell_id})
    for col in ["pre2014_res_area", "pre2014_units", "pre2014_buildings"]:
        houses[col] = 0
display(houses.head())

# %% [markdown]
# ## 7. Две панели: фирма × год и гекс × год
#
# Cohort фирмы фиксируется по её первой наблюдаемой ячейке. Гексы включают нули
# в годы без зарегистрированных фирм. Counts строятся отдельно по секциям
# ОКВЭД A–U, чтобы не смешивать отрасли с разными механизмами.

# %%
EXPOSURE_CACHE = OUT / "cell_exposure.parquet"
exposure.drop(columns="geometry", errors="ignore").to_parquet(EXPOSURE_CACHE, index=False)
HOUSE_EXPORT = OUT / "houses_baseline_for_join.parquet"
houses.to_parquet(HOUSE_EXPORT, index=False)
FIRM_PANEL = OUT / "r_firm_year.parquet"
HEX_PANEL = OUT / "r_hex_year.parquet"

if (FORCE_REBUILD or not FIRM_PANEL.exists()
        or SPATIAL_CACHE.stat().st_mtime > FIRM_PANEL.stat().st_mtime):
    con.execute(f"""COPY (
      WITH obs AS (
        SELECT * FROM read_parquet('{sql_path(SPATIAL_CACHE)}')
      ), ranked AS (
        SELECT *, row_number() OVER (PARTITION BY firm_id ORDER BY year) AS rn,
               count(DISTINCT cell_id) OVER (PARTITION BY firm_id) AS n_cells,
               sum(CASE WHEN cell_id IS NULL THEN 1 ELSE 0 END)
                 OVER (PARTITION BY firm_id) AS years_outside_grid,
               min(year) OVER (PARTITION BY firm_id) AS first_year
        FROM obs
      ), base AS (
        SELECT r.firm_id, r.cell_id AS baseline_cell_id,
               e.station_id AS baseline_station_id, e.station_name AS baseline_station_name,
               e.mcd_line, e.corridor_id, e.station_cluster_id,
               e.launch_date, e.g, e.did_sample, e.ambiguous,
               e.distance_m AS baseline_distance_m
        FROM ranked r JOIN read_parquet('{sql_path(EXPOSURE_CACHE)}') e
          ON r.cell_id=e.cell_id
        WHERE r.rn=1
      )
      SELECT r.firm_id, r.year, r.lon, r.lat, r.okved, r.okved_section,
             r.cell_id AS current_cell_id,
             b.* EXCLUDE (firm_id),
             r.filed, r.revenue, r.assets, r.profit, r.costs,
             cast(r.n_cells=1 AND r.years_outside_grid=0 AS INTEGER) AS stable_location_full,
             cast(CASE WHEN b.g=0 THEN r.first_year=2014
                       ELSE r.first_year<=b.g-3 END AS INTEGER) AS did_preperiod_ok,
             CASE WHEN b.g>0 THEN r.year-b.g END AS event_time,
             asinh(r.revenue/1000000.0) AS asinh_revenue_mln,
             asinh(r.assets/1000000.0) AS asinh_assets_mln,
             asinh(r.profit/1000000.0) AS asinh_profit_mln,
             asinh(r.costs/1000000.0) AS asinh_costs_mln,
             coalesce(h.pre2014_res_area, 0) AS baseline_pre2014_res_area,
             coalesce(h.pre2014_units, 0) AS baseline_pre2014_units
      FROM ranked r JOIN base b ON r.firm_id=b.firm_id
      LEFT JOIN read_parquet('{sql_path(HOUSE_EXPORT)}') h
        ON b.baseline_cell_id=h.cell_id
      ORDER BY r.firm_id, r.year
    ) TO '{sql_path(FIRM_PANEL)}' (FORMAT PARQUET, COMPRESSION ZSTD)""")

sections = "ABCDEFGHIJKLMNOPQRSTU"
counts_sql = ", ".join(
    f"count(*) FILTER (WHERE okved_section='{section}') AS firms_{section}"
    for section in sections
)
agg = con.sql(f"""
  SELECT cell_id, year, count(*) AS active_firms,
         sum(coalesce(filed,0)) AS reporting_firms,
         sum(revenue) AS total_revenue, sum(assets) AS total_assets,
         {counts_sql}
  FROM read_parquet('{sql_path(SPATIAL_CACHE)}')
  WHERE cell_id IS NOT NULL GROUP BY cell_id, year
""").df()

hex_panel = exposure.merge(pd.DataFrame({"year": list(YEARS)}), how="cross")
hex_panel = hex_panel.merge(agg, on=["cell_id", "year"], how="left", validate="one_to_one")
hex_panel = hex_panel.merge(houses, on="cell_id", how="left", validate="many_to_one")
count_cols = ["active_firms", "reporting_firms"] + [f"firms_{s}" for s in sections]
hex_panel[count_cols] = hex_panel[count_cols].fillna(0)
for column in ["pre2014_res_area", "pre2014_units", "pre2014_buildings"]:
    hex_panel[column] = hex_panel[column].fillna(0)
hex_panel["event_time"] = np.where(hex_panel.g.gt(0), hex_panel.year - hex_panel.g, np.nan)
for column in ["active_firms"] + [f"firms_{s}" for s in sections]:
    hex_panel[f"log1p_{column}"] = np.log1p(hex_panel[column])
hex_panel["asinh_total_revenue_mln"] = np.arcsinh(hex_panel.total_revenue.fillna(0) / 1e6)
hex_panel["asinh_total_assets_mln"] = np.arcsinh(hex_panel.total_assets.fillna(0) / 1e6)
hex_panel.to_parquet(HEX_PANEL, index=False)
assert not hex_panel.duplicated(["cell_id", "year"]).any()
assert hex_panel.groupby("cell_id").year.nunique().eq(len(YEARS)).all()
display(hex_panel.groupby(["mcd_line", "g"]).cell_id.nunique().rename("hexes"))
print("R:", FIRM_PANEL, HEX_PANEL)

# %% [markdown]
# ## 8. Карта и экспорт в QGIS
#
# Для исторической карты окрашиваем только линии, запущенные к `MAP_YEAR`.
# Ненаступившие D3/D4 в 2022 г. остаются обычной железной дорогой на карте.

# %%
map_cells = cells.copy()
map_outcomes = ["cell_id", "year", "active_firms", "total_revenue"] + [
    f"firms_{section}" for section in sections
]
map_cells = map_cells.merge(
    hex_panel.loc[hex_panel.year.eq(MAP_YEAR), map_outcomes],
    on="cell_id", how="left", validate="one_to_one"
)
map_cells["opened_by_map_year"] = (
    map_cells.launch_date.notna() &
    pd.to_datetime(map_cells.launch_date).dt.year.le(MAP_YEAR)
).astype("int8")
map_cells["map_line"] = np.where(map_cells.opened_by_map_year.eq(1), map_cells.mcd_line, "ordinary/future")
colors = {"D1": "#377eb8", "D2": "#e41a1c", "D3": "#4daf4a",
          "D4": "#984ea3", "ordinary/future": "#dddddd"}
fig, ax = plt.subplots(figsize=(10, 9))
for label, group in map_cells.groupby("map_line"):
    group.plot(ax=ax, color=colors[label], linewidth=0.1, edgecolor="white", label=label)
ax.legend(title=f"Сервис к {MAP_YEAR} г.")
ax.set_axis_off()
ax.set_title("Железнодорожные catchment-гексы и запуск МЦД")
plt.show()

map_cells.to_file(OUT / "qgis_cells.gpkg", layer="cells", driver="GPKG")
stations_gdf.to_file(OUT / "qgis_rail_stations.gpkg", layer="rail_stations", driver="GPKG")
print("QGIS:", OUT / "qgis_cells.gpkg", OUT / "qgis_rail_stations.gpkg")

# %% [markdown]
# ## 9. Все фирмы на карте: четыре взаимоисключающие группы
#
# Берём срез за MAP_YEAR из пространственного кэша, включая фирмы вне зон
# станций. Группы основаны на зоне гекса (расстояние от центра гекса до
# станции), как и панели для R. overlap_excluded включает все пересечения
# разных направлений; near_both выделяет среди них именно МЦД + контроль.
# Это карта расположения фирм за один год, не финальная DiD-выборка.

# %%
firm_map_sql = f"""
  SELECT firm_id, year, lon, lat, okved_section, revenue, assets, cell_id
  FROM read_parquet('{sql_path(SPATIAL_CACHE)}')
  WHERE year = {MAP_YEAR} AND lon IS NOT NULL AND lat IS NOT NULL
"""
firm_map = con.sql(firm_map_sql).df()
firm_map = gpd.GeoDataFrame(
    firm_map,
    geometry=gpd.points_from_xy(firm_map.lon, firm_map.lat),
    crs=4326,
).to_crs(CRS_METRIC)
region = gpd.read_file(
    ROOT / "input" / "moscow_moscow_oblast_boundary.geojson"
)[["geometry"]].to_crs(CRS_METRIC)
firm_map = gpd.sjoin(firm_map, region, how="inner", predicate="within")
firm_map = firm_map.drop(columns="index_right").drop_duplicates("firm_id")

pair_groups = (
    pairs[["cell_id", "station_id"]].drop_duplicates()
    .merge(stations[["station_id", "g"]], on="station_id", validate="many_to_one")
)
pair_groups["near_treatment"] = pair_groups.g.gt(0).astype("int8")
pair_groups["near_control"] = pair_groups.g.eq(0).astype("int8")
cell_flags = pair_groups.groupby("cell_id", as_index=False)[
    ["near_treatment", "near_control"]
].max()
firm_map = firm_map.merge(
    exposure[["cell_id", "station_name", "mcd_line", "g", "launch_date",
              "distance_m", "n_corridors", "ambiguous", "did_sample"]],
    on="cell_id", how="left", validate="many_to_one",
)
firm_map = firm_map.merge(cell_flags, on="cell_id", how="left", validate="many_to_one")
for column in ["near_treatment", "near_control"]:
    firm_map[column] = firm_map[column].fillna(0).astype("int8")
firm_map["near_both"] = (
    firm_map.near_treatment.eq(1) & firm_map.near_control.eq(1)
).astype("int8")
firm_map["firm_group"] = np.select(
    [firm_map.ambiguous.eq(1), firm_map.g.gt(0), firm_map.g.eq(0)],
    ["overlap_excluded", "treatment", "control"],
    default="neither",
)
firm_map["overlap_type"] = np.select(
    [firm_map.near_both.eq(1),
     firm_map.firm_group.eq("overlap_excluded") & firm_map.near_treatment.eq(1),
     firm_map.firm_group.eq("overlap_excluded") & firm_map.near_control.eq(1)],
    ["treatment_and_control", "multiple_treated_corridors",
     "multiple_control_corridors"],
    default="",
)
firm_map["opened_by_map_year"] = (
    firm_map.launch_date.notna()
    & pd.to_datetime(firm_map.launch_date).dt.year.le(MAP_YEAR)
).astype("int8")
assert firm_map.firm_id.is_unique
assert firm_map.loc[firm_map.firm_group.eq("overlap_excluded"), "ambiguous"].eq(1).all()
assert firm_map.loc[firm_map.near_both.eq(1), "firm_group"].eq("overlap_excluded").all()

firm_map = firm_map.drop(columns=["lon", "lat", "launch_date"])
firm_map_path = OUT / f"qgis_firms_four_groups_{MAP_YEAR}.gpkg"
firm_map.to_file(firm_map_path, layer="firms_four_groups", driver="GPKG")
group_counts = firm_map.groupby(["firm_group", "overlap_type"], dropna=False).size().rename("firms")
display(group_counts)
print("QGIS firms:", firm_map_path)

