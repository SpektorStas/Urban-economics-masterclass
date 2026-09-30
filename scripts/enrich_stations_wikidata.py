"""Prepare a small, reviewable MCD/ordinary-rail station table from Wikidata.

The user's OSM candidate CSV is read for coordinate/name cross-checking only.
It is never modified. Wikidata dates describe station opening, not MCD launch.
"""

from __future__ import annotations

import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from pyproj import Transformer


MCD = {
    "Q62091003": ("D1", "D1"),
    "Q62091175": ("D2", "D2"),
    "Q65127804": ("D3", "D3"),
    "Q63928373": ("D4", "D4"),
}
CONTROLS = {
    "Q4538927": ("ORDINARY", "YAROSLAVSKOE"),
    "Q4341281": ("ORDINARY", "PAVELETSKOE"),
}
LINE_LOOKUP = {**MCD, **CONTROLS}
SPARQL = """
SELECT ?station ?stationLabel ?line ?coord ?opening WHERE {
  ?station wdt:P625 ?coord.
  {
    VALUES ?line { wd:Q62091003 wd:Q62091175 wd:Q65127804 wd:Q63928373 }
    ?station wdt:P1192 ?line.
  } UNION {
    VALUES ?line { wd:Q4538927 wd:Q4341281 }
    ?station wdt:P81 ?line.
  }
  OPTIONAL { ?station wdt:P1619 ?opening. }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "ru,en". }
}
"""


def normalized_name(value: str) -> str:
    value = unicodedata.normalize("NFKC", str(value)).casefold().replace("ё", "е")
    value = re.sub(r"\([^)]*\)", " ", value)
    value = re.sub(r"\b(станция|платформа|вокзал|остановочный|пункт|жд)\b", " ", value)
    return " ".join(re.findall(r"[\w]+", value, flags=re.UNICODE))


def wikidata_rows(bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    response = requests.get(
        "https://query.wikidata.org/sparql",
        params={"query": SPARQL, "format": "json"},
        headers={"User-Agent": "mcd-firm-effect-demo/1.0 (research notebook)"},
        timeout=90,
    )
    response.raise_for_status()
    rows = []
    for binding in response.json()["results"]["bindings"]:
        point = binding["coord"]["value"]
        match = re.fullmatch(r"Point\(([-+\d.eE]+) ([-+\d.eE]+)\)", point)
        if not match:
            continue
        lon, lat = map(float, match.groups())
        if not (bbox[1] <= lon <= bbox[3] and bbox[0] <= lat <= bbox[2]):
            continue
        qid = binding["station"]["value"].rsplit("/", 1)[-1]
        line_qid = binding["line"]["value"].rsplit("/", 1)[-1]
        opening = binding.get("opening", {}).get("value", "")
        year_match = re.match(r"^[+-]?(\d{4})", opening)
        rows.append({
            "wikidata_id": qid,
            "wd_station_name": binding["stationLabel"]["value"],
            "wd_lon": lon, "wd_lat": lat,
            "line_qid": line_qid,
            "opening_year": int(year_match.group(1)) if year_match else np.nan,
        })
    if not rows:
        raise RuntimeError("Wikidata не вернула станций в выбранном bbox")
    return pd.DataFrame(rows).drop_duplicates()


def enrich_stations(
    candidate_csv: Path | None,
    output_csv: Path,
    bbox: tuple[float, float, float, float],
) -> pd.DataFrame:
    wd = wikidata_rows(bbox)
    candidate = None
    if candidate_csv is not None and candidate_csv.exists():
        candidate = pd.read_csv(candidate_csv, sep=None, engine="python", encoding="utf-8-sig")
        needed = {"station_id", "station_name", "lon", "lat", "source_url"}
        if not needed <= set(candidate):
            raise ValueError(f"В OSM CSV не хватает полей: {needed - set(candidate)}")
        candidate["lon"] = pd.to_numeric(candidate.lon, errors="coerce")
        candidate["lat"] = pd.to_numeric(candidate.lat, errors="coerce")
        candidate = candidate.dropna(subset=["lon", "lat"]).reset_index(drop=True)

    transformer = Transformer.from_crs(4326, 32637, always_xy=True)
    if candidate is not None and len(candidate):
        osm_x, osm_y = transformer.transform(candidate.lon.to_numpy(), candidate.lat.to_numpy())
        osm_names = candidate.station_name.fillna("").map(normalized_name).tolist()

    rows = []
    for qid, group in wd.groupby("wikidata_id", sort=True):
        row = group.iloc[0]
        lines = sorted({LINE_LOOKUP[q][0] for q in group.line_qid})
        corridors = sorted({LINE_LOOKUP[q][1] for q in group.line_qid})
        single_line = len(corridors) == 1
        years = group.opening_year.dropna().astype(int)
        opening_year = int(years.min()) if len(years) else ""
        baseline = int(opening_year <= 2013) if opening_year != "" else ""
        lon, lat = float(row.wd_lon), float(row.wd_lat)
        osm_id, osm_url, distance_m, name_score = "", "", "", ""
        match_status = "no_osm_candidate"
        if candidate is not None and len(candidate):
            wx, wy = transformer.transform(lon, lat)
            distances = np.hypot(osm_x - wx, osm_y - wy)
            nearby = np.flatnonzero(distances <= 600)
            if len(nearby):
                wd_name = normalized_name(row.wd_station_name)
                scores = [
                    SequenceMatcher(None, wd_name, osm_names[i]).ratio() for i in nearby
                ]
                best = min(
                    range(len(nearby)),
                    key=lambda j: (-(scores[j] - distances[nearby[j]] / 3000), distances[nearby[j]]),
                )
                idx = int(nearby[best])
                distance_m = round(float(distances[idx]), 1)
                name_score = round(float(scores[best]), 2)
                if name_score >= 0.70 and distance_m <= 500:
                    selected = candidate.iloc[idx]
                    lon, lat = float(selected.lon), float(selected.lat)
                    osm_id = str(selected.station_id)
                    osm_url = str(selected.source_url)
                    match_status = "name_and_location"
                else:
                    match_status = "nearby_name_differs"

        row_out = {
            "station_id": f"wd_{qid}",
            "station_name": row.wd_station_name,
            "lon": lon, "lat": lat,
            "mcd_line": lines[0] if single_line else "",
            "corridor_id": corridors[0] if single_line else "",
            "baseline_operating": baseline,
            "verified": 0,
            "source_url": f"https://www.wikidata.org/wiki/{qid}",
            "opening_year": opening_year,
            "line_candidates": "|".join(lines),
            "corridor_candidates": "|".join(corridors),
            "osm_station_id": osm_id,
            "osm_source_url": osm_url,
            "osm_match_m": distance_m,
            "osm_name_score": name_score,
            "match_status": match_status,
            "review_note": (
                "multiple_lines" if not single_line else
                "opening_unknown" if opening_year == "" else
                "opened_after_2013" if baseline == 0 else
                "check_history_and_location"
            ),
        }
        rows.append(row_out)
    result = pd.DataFrame(rows).sort_values(["corridor_id", "station_name"])
    shared_osm_stop = (
        result.osm_station_id.ne("") & result.osm_station_id.duplicated(keep=False)
    )
    result.loc[shared_osm_stop, ["mcd_line", "corridor_id"]] = ""
    result.loc[shared_osm_stop, "review_note"] = "shared_osm_stop"
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, sep=";", index=False, encoding="utf-8-sig")
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    candidate_path = root / "input" / "rail_stations_verified.csv"
    if not candidate_path.exists():
        candidate_path = root / "output" / "rail_stations_to_verify.csv"
    output_path = root / "output" / "rail_stations_wikidata_review.csv"
    result = enrich_stations(
        candidate_path if candidate_path.exists() else None,
        output_path, (54.20, 35.10, 57.05, 40.30),
    )
    print(f"Wrote {len(result)} station candidates: {output_path}")
    print(result.groupby(["mcd_line", "review_note"], dropna=False).size())
