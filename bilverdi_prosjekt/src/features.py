"""Felles feature engineering – brukes både i trening og i web-appen."""
import re
import numpy as np
import pandas as pd

CURRENT_YEAR = 2025  # datasettet er fra 2024/2025
CAT = ["brand", "model_base", "fuel_type", "trans_type", "ext_col", "int_col"]
NUM = ["age", "milage", "hp", "liters", "cylinders", "gears", "accident", "clean_title", "miles_per_year"]
FEATURES = CAT + NUM


def _first(pattern, s):
    m = re.search(pattern, s)
    return float(m.group(1)) if m else np.nan


def parse_engine(e):
    e = str(e)
    cyl = _first(r"(\d+)\s*Cylinder", e)
    if np.isnan(cyl):
        cyl = _first(r"\b[VvIiHh]-?(\d{1,2})\b", e)
    return pd.Series({
        "hp": _first(r"(\d+(?:\.\d+)?)\s*HP", e),
        "liters": _first(r"(\d+(?:\.\d+)?)\s*(?:L\b|Liter)", e),
        "cylinders": cyl,
    })


def trans_type(t):
    t = str(t).lower()
    if "dual" in t or "dct" in t:
        return "dual_clutch"
    if "cvt" in t or "variable" in t:
        return "cvt"
    if "m/t" in t or "manual" in t:
        return "manual"
    if "a/t" in t or "auto" in t:
        return "automatic"
    return "other"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tar rå rader (kolonner som i CSV, milage som tall) og returnerer feature-matrise."""
    d = df.copy()
    out = pd.DataFrame(index=d.index)
    out["brand"] = d["brand"].astype(str).str.strip()
    out["model_base"] = d["model"].astype(str).str.split().str[0].str.upper()
    fuel = d["fuel_type"].astype(str).str.strip()
    out["fuel_type"] = fuel.where(~fuel.isin(["–", "-", "nan", "not supported"]), "Unknown").fillna("Unknown")
    out["trans_type"] = d["transmission"].map(trans_type)
    out["ext_col"] = d["ext_col"].astype(str).str.strip().str.lower()
    out["int_col"] = d["int_col"].astype(str).str.strip().str.lower()
    out["age"] = CURRENT_YEAR - d["model_year"].astype(float)
    out["milage"] = d["milage"].astype(float)
    eng = d["engine"].apply(parse_engine)
    out["hp"], out["liters"], out["cylinders"] = eng["hp"], eng["liters"], eng["cylinders"]
    out["gears"] = d["transmission"].astype(str).map(lambda s: _first(r"(\d+)-Speed", s))
    out["accident"] = d["accident"].astype(str).str.contains("At least 1", na=False).astype(int)
    out["clean_title"] = d["clean_title"].astype(str).str.contains("Yes", na=False).astype(int)
    out["miles_per_year"] = out["milage"] / out["age"].clip(lower=1)
    return out[FEATURES]
