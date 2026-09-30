"""Build a broad Moscow-region suburban-rail station registry.

The registry is a reproducible *candidate frame*, not a hand-verified historical
list. Wikidata P81 identifies physical corridors; P1192 identifies MCD service.
The existing OSM CSV is used to improve coordinates and is never overwritten.
"""

from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer
from shapely.geometry import Point, shape
from shapely.ops import unary_union
from shapely.prepared import prep


ROOT = Path(__file__).resolve().parents[1]
BBOX = (54.20, 35.10, 57.05, 40.30)
ENDPOINT = "https://query.wikidata.org/sparql"
BOUNDARY_PATH = ROOT / "input" / "moscow_moscow_oblast_boundary.geojson"
BOUNDARY_URL = (
    "https://github.com/wmgeolab/geoBoundaries/raw/9469f09/releaseData/gbOpen/"
    "RUS/ADM1/geoBoundaries-RUS-ADM1_simplified.geojson"
)

# Radial corridors on Moscow Railway, plus Leningrad (October Railway, D3).
# The Greater Ring is retained in the registry but not used as a DiD control.
CORRIDORS = {
    "Q4146162": ("GORKOVSKOE", "Горьковское", "МЖД", True),
    "Q4207854": ("KAZANSKOE", "Казанское", "МЖД", True),
    "Q4220266": ("KIEVSKOE", "Киевское", "МЖД", True),
    "Q4248874": ("KURSKOE", "Курское", "МЖД", True),
    "Q4341281": ("PAVELETSKOE", "Павелецкое", "МЖД", True),
    "Q4394582": ("RIZHSKOE", "Рижское", "МЖД", True),
    "Q4402675": ("RYAZANSKOE", "Рязанское", "МЖД", True),
    "Q4404352": ("SAVELOVSKOE", "Савёловское", "МЖД", True),
    "Q4424999": ("BELORUSSKOE", "Белорусское", "МЖД", True),
    "Q4538927": ("YAROSLAVSKOE", "Ярославское", "МЖД", True),
    "Q1548805": ("BIG_RING", "Большое кольцо МЖД", "МЖД", False),
    "Q4178843": ("LENINGRADSKOE", "Ленинградское", "ОЖД", True),
    "Q114425691": ("LENINGRADSKOE", "Ленинградское", "ОЖД", True),
    "Q114425725": ("LENINGRADSKOE", "Ленинградское", "ОЖД", True),
}
MCD = {
    "Q62091003": "D1", "Q62091175": "D2",
    "Q65127804": "D3", "Q63928373": "D4",
}
# Дополнительные контрольные направления для демонстрационной выборки.
# Ветки без подтверждённого пассажирского сервиса (например, МЦК/грузовые)
# остаются исключёнными. Граница D3 сверена с конечной Ипподром/Раменское.
EXTRA_CONTROL_BRANCHES = {
    "Q19909429",  # Мытищи — Фрязево
    "Q109746531", # Подлипки — Фрязино
    "Q109746679", # Софрино — Красноармейск
    "Q16651550",  # Голицыно — Звенигород
    "Q123080970", # Реутов — Балашиха
}
EXTRA_CONTROL_STATUSES = {
    "analysis_control_kazanskoe", "analysis_control_beyond_d3", "analysis_control_branch"
}


def fetch(query: str) -> list[dict]:
    response = requests.get(
        ENDPOINT, params={"query": query, "format": "json"},
        headers={"User-Agent": "mcd-firm-effect-demo/2.0 (academic research)"},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["results"]["bindings"]


def qid(uri: str) -> str:
    return uri.rsplit("/", 1)[-1]


def station_query(property_id: str, values: dict[str, object]) -> str:
    ids = " ".join(f"wd:{key}" for key in values)
    return f"""
    SELECT ?station ?stationLabel ?line ?lineLabel ?coord ?opening ?precision WHERE {{
      VALUES ?line {{ {ids} }}
      ?station wdt:{property_id} ?line; wdt:P625 ?coord.
      OPTIONAL {{
        ?station p:P1619/psv:P1619 ?date_node.
        ?date_node wikibase:timeValue ?opening;
                   wikibase:timePrecision ?precision.
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "ru,en". }}
    }}
    """


OWNED_QUERY = """
SELECT ?station ?stationLabel ?line ?lineLabel ?coord ?opening ?precision WHERE {
  ?station wdt:P127 wd:Q1765011; wdt:P625 ?coord.
  OPTIONAL { ?station wdt:P81 ?line. }
  OPTIONAL {
    ?station p:P1619/psv:P1619 ?date_node.
    ?date_node wikibase:timeValue ?opening;
               wikibase:timePrecision ?precision.
  }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "ru,en". }
}
"""


def decode(bindings: list[dict], kind: str) -> pd.DataFrame:
    rows = []
    for item in bindings:
        point = re.fullmatch(r"Point\(([-+\d.eE]+) ([-+\d.eE]+)\)", item["coord"]["value"])
        if point is None:
            continue
        lon, lat = map(float, point.groups())
        if not (BBOX[1] <= lon <= BBOX[3] and BBOX[0] <= lat <= BBOX[2]):
            continue
        opening = item.get("opening", {}).get("value", "")
        rows.append({
            "qid": qid(item["station"]["value"]),
            "name": item["stationLabel"]["value"],
            "lon": lon, "lat": lat,
            "line_qid": qid(item["line"]["value"]) if "line" in item else "",
            "line_name": item.get("lineLabel", {}).get("value", ""),
            "kind": kind,
            "opening": opening[:10] if opening else "",
            "opening_precision": int(item.get("precision", {}).get("value", 0)),
        })
    return pd.DataFrame(rows)


def normalize_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е")
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"\b(станция|платформа|вокзал|остановочный|пункт|жд)\b", " ", value)
    return " ".join(re.findall(r"\w+", value))


def match_osm(registry: pd.DataFrame, osm_path: Path) -> pd.DataFrame:
    registry = registry.copy()
    registry["osm_station_id"] = ""
    registry["osm_match_m"] = np.nan
    registry["osm_name_score"] = np.nan
    if not osm_path.exists():
        return registry
    osm = pd.read_csv(osm_path, sep=None, engine="python", encoding="utf-8-sig")
    osm = osm.dropna(subset=["station_name", "lon", "lat"])
    if not {"station_id", "station_name", "lon", "lat"} <= set(osm):
        return registry
    transformer = Transformer.from_crs(4326, 32637, always_xy=True)
    ox, oy = transformer.transform(osm.lon.to_numpy(), osm.lat.to_numpy())
    names = osm.station_name.map(normalize_name).tolist()
    for idx, row in registry.iterrows():
        sx, sy = transformer.transform(row.lon, row.lat)
        distances = np.hypot(ox - sx, oy - sy)
        nearby = np.flatnonzero(distances <= 1000)
        if not len(nearby):
            continue
        target = normalize_name(row.station_name)
        best = max(nearby, key=lambda j: SequenceMatcher(None, target, names[j]).ratio() - distances[j] / 3000)
        score = SequenceMatcher(None, target, names[best]).ratio()
        if score < (0.7 if distances[best] <= 500 else 0.85):
            continue
        registry.at[idx, "osm_station_id"] = str(osm.iloc[best].station_id)
        registry.at[idx, "osm_match_m"] = round(float(distances[best]), 1)
        registry.at[idx, "osm_name_score"] = round(score, 2)
        registry.at[idx, "lon"] = float(osm.iloc[best].lon)
        registry.at[idx, "lat"] = float(osm.iloc[best].lat)
    return registry


def append_osm_unmatched(registry: pd.DataFrame, osm_path: Path) -> pd.DataFrame:
    """Keep OSM-only stops visible without fabricating line or treatment data."""
    if not osm_path.exists():
        return registry
    osm = pd.read_csv(osm_path, sep=None, engine="python", encoding="utf-8-sig")
    osm = osm.dropna(subset=["station_id", "station_name", "lon", "lat"])
    osm = osm.loc[~osm.station_id.astype(str).isin(set(registry.osm_station_id))]
    extras = []
    for row in osm.itertuples(index=False):
        extras.append({
            "station_id": f"osm_{row.station_id}", "station_name": row.station_name,
            "lon": float(row.lon), "lat": float(row.lat),
            "corridor_id": "", "rail_corridor_id": "", "corridor_candidates": "",
            "corridor_name": "", "railway": "", "corridor_qids": "",
            "mcd_line": "", "mcd_lines": "", "treatment_date": "",
            "line_launch_date": "", "treatment_date_quality": "not_classified",
            "treatment_group": "unclassified", "opening_date": "",
            "opening_date_precision": 0, "opening_year": "", "baseline_operating": "",
            "did_eligible_candidate": 0, "review_status": "osm_only_line_unknown",
            "source_url": row.source_url,
            "osm_station_id": str(row.station_id), "osm_match_m": 0,
            "osm_name_score": 1,
        })
    return pd.concat([registry, pd.DataFrame(extras)], ignore_index=True)


def study_region():
    if not BOUNDARY_PATH.exists():
        response = requests.get(BOUNDARY_URL, timeout=90)
        response.raise_for_status()
        features = [
            feature for feature in response.json()["features"]
            if feature["properties"].get("shapeISO") in {"RU-MOW", "RU-MOS"}
        ]
        if len(features) != 2:
            raise RuntimeError("Moscow and Moscow Oblast boundaries not found")
        BOUNDARY_PATH.write_text(
            json.dumps({"type": "FeatureCollection", "features": features}, ensure_ascii=False),
            encoding="utf-8",
        )
    features = json.loads(BOUNDARY_PATH.read_text(encoding="utf-8"))["features"]
    if {f["properties"]["shapeISO"] for f in features} != {"RU-MOW", "RU-MOS"}:
        raise ValueError("Unexpected study-area boundary")
    return prep(unary_union([shape(feature["geometry"]) for feature in features]))


def build() -> pd.DataFrame:
    physical = decode(fetch(station_query("P81", CORRIDORS)), "physical")
    services = decode(fetch(station_query("P1192", MCD)), "service")
    owned = decode(fetch(OWNED_QUERY), "owned")
    all_rows = pd.concat([physical, services, owned], ignore_index=True)
    if all_rows.empty:
        raise RuntimeError("Wikidata returned no stations; no registry was written")
    launches = pd.read_csv(ROOT / "input" / "mcd_line_launches.csv")
    launch_by_line = dict(zip(launches.mcd_line, launches.launch_date))
    result = []
    for station_qid, group in all_rows.groupby("qid", sort=True):
        first = group.iloc[0]
        physical_qids = sorted(set(group.loc[
            group.kind.isin(["physical", "owned"]) & group.line_qid.ne(""), "line_qid"
        ]))
        mcd_lines = sorted({MCD[q] for q in group.loc[group.kind.eq("service"), "line_qid"]})
        corridors = sorted({CORRIDORS[q][0] if q in CORRIDORS else q for q in physical_qids})
        line_names = dict(zip(group.line_qid, group.line_name))
        candidate_dates = sorted(launch_by_line[line] for line in mcd_lines)
        launch_date = candidate_dates[0] if candidate_dates else ""
        dates = group.loc[group.opening.ne(""), ["opening", "opening_precision"]].drop_duplicates()
        if len(dates):
            opening = dates.sort_values("opening").iloc[0]
            opening_date = opening.opening
            precision = int(opening.opening_precision)
            opening_year = int(opening_date[:4])
        else:
            opening_date, precision, opening_year = "", 0, ""

        # A station built after service launch was not treated on launch day.
        postlaunch = bool(launch_date and opening_year != "" and (
            opening_year > int(launch_date[:4]) or
            (opening_year == int(launch_date[:4]) and precision >= 11 and opening_date > launch_date)
        ))
        same_year_imprecise = bool(launch_date and opening_year == int(launch_date[:4]) and precision < 11)
        if postlaunch:
            treatment_date = opening_date if precision >= 11 else ""
            date_quality = "station_opening_day" if treatment_date else "postlaunch_opening_date_unknown"
        elif same_year_imprecise:
            treatment_date, date_quality = "", "same_year_opening_ambiguous"
        elif launch_date:
            treatment_date = launch_date
            date_quality = "line_launch" if opening_year != "" else "line_launch_opening_unverified"
        else:
            treatment_date, date_quality = "", "never_mcd_in_source"

        if len(mcd_lines) > 1:
            status = "multiple_mcd_lines"
        elif not mcd_lines and not physical_qids:
            status = "line_unknown"
        elif opening_year == "":
            status = "opening_unknown"
        elif opening_year > 2013:
            status = "opened_after_baseline"
        elif len(corridors) != 1 and not mcd_lines:
            status = "multiple_corridors"
        elif not mcd_lines and any(q not in CORRIDORS for q in physical_qids):
            status = "branch_line_review"
        elif physical_qids and not any(CORRIDORS[q][3] for q in physical_qids if q in CORRIDORS):
            status = "nonradial_corridor"
        else:
            status = "candidate"

        # Never treat missing P1192 as proof of no treatment on a D1-D4 corridor.
        treated_corridors = {
            "BELORUSSKOE", "SAVELOVSKOE", "KURSKOE", "RIZHSKOE",
            "LENINGRADSKOE", "KAZANSKOE", "RYAZANSKOE", "KIEVSKOE", "GORKOVSKOE",
        }
        if not mcd_lines and set(corridors) & treated_corridors:
            status = "mcd_corridor_service_uncertain"
        # Явное расширение контроля: Казанское направление (включая Шатуру),
        # Рязанское восточнее D3 и отдельные пригородные ветки. Не переносим
        # это правило на все станции коридоров D1–D4 без проверки.
        if not mcd_lines and opening_year != "" and opening_year <= 2013:
            if status == "mcd_corridor_service_uncertain" and corridors == ["KAZANSKOE"]:
                status = "analysis_control_kazanskoe"
            elif (status == "mcd_corridor_service_uncertain"
                  and corridors == ["RYAZANSKOE"] and float(first.lon) > 38.3):
                status = "analysis_control_beyond_d3"
            elif status == "branch_line_review" and corridors[0] in EXTRA_CONTROL_BRANCHES:
                status = "analysis_control_branch"
        control = not mcd_lines and (status == "candidate" or status in EXTRA_CONTROL_STATUSES)
        eligible = (bool(mcd_lines) and status == "candidate") or control
        result.append({
            "station_id": f"wd_{station_qid}", "station_name": first["name"],
            "lon": first.lon, "lat": first.lat,
            "corridor_id": mcd_lines[0] if len(mcd_lines) == 1 else corridors[0] if len(corridors) == 1 else "",
            "rail_corridor_id": corridors[0] if len(corridors) == 1 else "",
            "corridor_candidates": "|".join(corridors),
            "corridor_name": (
                CORRIDORS[physical_qids[0]][1] if physical_qids[0] in CORRIDORS
                else line_names.get(physical_qids[0], "")
            ) if len(corridors) == 1 else "",
            "railway": "МЖД" if group.kind.eq("owned").any() else (
                CORRIDORS[physical_qids[0]][2] if len(corridors) == 1 else ""
            ),
            "corridor_qids": "|".join(physical_qids),
            "mcd_line": mcd_lines[0] if len(mcd_lines) == 1 else "",
            "mcd_lines": "|".join(mcd_lines),
            "treatment_date": treatment_date,
            "line_launch_date": launch_date,
            "treatment_date_quality": date_quality,
            "treatment_group": "treated" if mcd_lines else "control_candidate" if physical_qids else "unclassified",
            "opening_date": opening_date,
            "opening_date_precision": precision,
            "opening_year": opening_year,
            "baseline_operating": int(opening_year <= 2013) if opening_year != "" else "",
            "did_eligible_candidate": int(eligible),
            "review_status": status,
            "source_url": f"https://www.wikidata.org/wiki/{station_qid}",
        })
    registry = pd.DataFrame(result)
    osm_path = ROOT / "input" / "rail_stations_verified.csv"
    registry = match_osm(registry, osm_path)
    shared_osm = registry.osm_station_id.ne("") & registry.osm_station_id.duplicated(keep=False)
    registry.loc[shared_osm, "did_eligible_candidate"] = 0
    registry.loc[shared_osm, "review_status"] = "shared_osm_stop"
    registry = append_osm_unmatched(registry, osm_path)
    region = study_region()
    registry = registry.loc[[
        region.covers(Point(row.lon, row.lat)) for row in registry.itertuples(index=False)
    ]].copy()
    registry = registry.sort_values(["corridor_id", "station_name", "station_id"]).reset_index(drop=True)
    if registry.station_id.duplicated().any():
        raise ValueError("Duplicate station IDs")
    output = ROOT / "input" / "rail_station_treatment_registry.csv"
    registry.to_csv(output, sep=";", index=False, encoding="utf-8-sig")
    return registry


if __name__ == "__main__":
    stations = build()
    print(f"Wrote {len(stations)} stations to {ROOT / 'input' / 'rail_station_treatment_registry.csv'}")
    print(stations.groupby(["treatment_group", "review_status"]).size().to_string())
