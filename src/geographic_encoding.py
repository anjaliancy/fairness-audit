"""
Geographic region encoding for fairness preprocessing.

Replaces raw geographic identifiers (ZIP, State, MSA) with US Census Division
labels, reducing redlining risk while preserving legitimate area-level signal.

Census Bureau 9-Division schema:
  1  New England           CT ME MA NH RI VT
  2  Middle Atlantic       NJ NY PA
  3  East North Central    IL IN MI OH WI
  4  West North Central    IA KS MN MO NE ND SD
  5  South Atlantic        DC DE FL GA MD NC SC VA WV
  6  East South Central    AL KY MS TN
  7  West South Central    AR LA OK TX
  8  Mountain              AZ CO ID MT NV NM UT WY
  9  Pacific               AK CA HI OR WA
  10 Territories           PR GU VI (non-standard, present in Freddie Mac data)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# State → Census Division lookup
# ---------------------------------------------------------------------------

STATE_TO_CENSUS_DIVISION: dict[str, str] = {
    # New England
    "CT": "New England",
    "ME": "New England",
    "MA": "New England",
    "NH": "New England",
    "RI": "New England",
    "VT": "New England",
    # Middle Atlantic
    "NJ": "Middle Atlantic",
    "NY": "Middle Atlantic",
    "PA": "Middle Atlantic",
    # East North Central
    "IL": "East North Central",
    "IN": "East North Central",
    "MI": "East North Central",
    "OH": "East North Central",
    "WI": "East North Central",
    # West North Central
    "IA": "West North Central",
    "KS": "West North Central",
    "MN": "West North Central",
    "MO": "West North Central",
    "NE": "West North Central",
    "ND": "West North Central",
    "SD": "West North Central",
    # South Atlantic
    "DC": "South Atlantic",
    "DE": "South Atlantic",
    "FL": "South Atlantic",
    "GA": "South Atlantic",
    "MD": "South Atlantic",
    "NC": "South Atlantic",
    "SC": "South Atlantic",
    "VA": "South Atlantic",
    "WV": "South Atlantic",
    # East South Central
    "AL": "East South Central",
    "KY": "East South Central",
    "MS": "East South Central",
    "TN": "East South Central",
    # West South Central
    "AR": "West South Central",
    "LA": "West South Central",
    "OK": "West South Central",
    "TX": "West South Central",
    # Mountain
    "AZ": "Mountain",
    "CO": "Mountain",
    "ID": "Mountain",
    "MT": "Mountain",
    "NV": "Mountain",
    "NM": "Mountain",
    "UT": "Mountain",
    "WY": "Mountain",
    # Pacific
    "AK": "Pacific",
    "CA": "Pacific",
    "HI": "Pacific",
    "OR": "Pacific",
    "WA": "Pacific",
    # Territories (Freddie Mac data includes PR, GU, VI)
    "PR": "Territories",
    "GU": "Territories",
    "VI": "Territories",
}

# Numeric codes for ordered / label-encoded use downstream
DIVISION_TO_CODE: dict[str, int] = {
    "New England": 1,
    "Middle Atlantic": 2,
    "East North Central": 3,
    "West North Central": 4,
    "South Atlantic": 5,
    "East South Central": 6,
    "West South Central": 7,
    "Mountain": 8,
    "Pacific": 9,
    "Territories": 10,
    "Unknown": 0,
}

# Binary grouping for DIR / BinaryLabelDatasetMetric.
# Southern divisions (South Atlantic + East South Central + West South Central)
# historically show higher disparate impact in credit lending.
SOUTHERN_DIVISIONS = {"South Atlantic", "East South Central", "West South Central"}


def encode_state_to_division(state_series: pd.Series) -> pd.Series:
    """Map a State column (2-letter USPS code) to Census Division labels."""
    return state_series.map(STATE_TO_CENSUS_DIVISION).fillna("Unknown")


def build_zip3_to_division_map(
    zip_series: pd.Series,
    state_series: pd.Series,
) -> dict[int, str]:
    """
    Derive a ZIP3-to-division lookup from the dataset itself.

    The dataset stores ZIP codes as 3-digit ZIP3 prefix × 100 (e.g. 92000 = ZIP3 920).
    This function maps each ZIP3 to the Census Division of its most common associated state.
    Using the data avoids hardcoding USPS prefix ranges, which have historical exceptions.
    """
    zip3 = (zip_series // 100).astype(int)
    division = encode_state_to_division(state_series)
    mapping_df = pd.DataFrame({"zip3": zip3, "division": division})
    # For each ZIP3 pick the modal division (handles rare cross-state prefixes)
    lookup = (
        mapping_df.groupby("zip3")["division"]
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )
    return lookup


def encode_zip3_to_division(
    zip_series: pd.Series,
    zip3_division_map: dict[int, str],
) -> pd.Series:
    """Apply a pre-built ZIP3→division map. ZIP values are stored as ZIP3 × 100."""
    zip3 = (zip_series // 100).astype(int)
    return zip3.map(zip3_division_map).fillna("Unknown")


def build_msa_to_division_map(
    msa_series: pd.Series,
    state_series: pd.Series,
) -> dict[float, str]:
    """
    Derive an MSA (CBSA code) → Census Division map from the data.

    Each MSA is assigned the division of its most frequently co-occurring state.
    Null MSAs (rural / non-metro areas) are left as NaN and handled separately.
    """
    division = encode_state_to_division(state_series)
    mapping_df = pd.DataFrame({"msa": msa_series, "division": division}).dropna(
        subset=["msa"]
    )
    lookup = (
        mapping_df.groupby("msa")["division"]
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )
    return lookup


def encode_msa_to_division(
    msa_series: pd.Series,
    msa_division_map: dict[float, str],
) -> pd.Series:
    """Apply a pre-built MSA→division map. Nulls map to 'Non-Metro'."""
    return msa_series.map(msa_division_map).fillna("Non-Metro")


def add_geographic_region_features(
    df: pd.DataFrame,
    zip_col: str = "ZIP",
    state_col: str = "State",
    msa_col: str = "MSA",
    drop_originals: bool = True,
    fit_maps: Optional[tuple[dict, dict]] = None,
) -> tuple[pd.DataFrame, tuple[dict, dict]]:
    """
    Replace raw geographic identifiers with Census Division encodings.

    Added columns
    -------------
    census_division        : Census Division label (string) derived from State
    census_division_code   : Integer code 0-10 (0 = Unknown)
    is_southern_region     : Binary flag — 1 if South Atlantic / East/West South Central
    zip3_division          : Census Division label derived from ZIP3 prefix
    msa_division           : Census Division label derived from MSA CBSA code
                             (or 'Non-Metro' if MSA is null)

    Parameters
    ----------
    df             : Input dataframe (may be train or test).
    drop_originals : If True, drop ZIP, State, MSA raw columns after encoding.
    fit_maps       : Pass (zip3_map, msa_map) from a previously fitted train set
                     to apply the same lookup on a test set without refitting.

    Returns
    -------
    (encoded_df, (zip3_map, msa_map))
        encoded_df : DataFrame with geographic region columns added.
        maps tuple : ZIP3 and MSA lookup dicts (save from train; reuse on test).
    """
    df = df.copy()

    # --- State → Census Division ---
    df["census_division"] = encode_state_to_division(df[state_col])
    df["census_division_code"] = df["census_division"].map(DIVISION_TO_CODE).fillna(0).astype(int)
    df["is_southern_region"] = df["census_division"].isin(SOUTHERN_DIVISIONS).astype(int)

    # --- ZIP3 → Census Division ---
    if fit_maps is not None:
        zip3_map, msa_map = fit_maps
    else:
        zip3_map = build_zip3_to_division_map(df[zip_col], df[state_col])
        msa_map = build_msa_to_division_map(df[msa_col], df[state_col])

    df["zip3_division"] = encode_zip3_to_division(df[zip_col], zip3_map)
    df["msa_division"] = encode_msa_to_division(df[msa_col], msa_map)

    if drop_originals:
        cols_to_drop = [c for c in [zip_col, state_col, msa_col] if c in df.columns]
        df = df.drop(columns=cols_to_drop)

    return df, (zip3_map, msa_map)
