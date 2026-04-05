"""Load and clean SCATS traffic data from Excel files."""

import os
import datetime
import pandas as pd
import numpy as np
import yaml


def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_scats_data(filepath=None):
    """
    Load SCATS traffic volume data from the Excel file.

    The Excel file has:
    - Row 0: actual column names (SCATS Number, Location, ..., Date, V00..V95)
    - Columns 10-105 have datetime.time headers (00:00, 00:15, etc.)
    - These correspond to V00..V95

    Returns a DataFrame with columns:
        scats_number, location, latitude, longitude, date, V00..V95
    """
    if filepath is None:
        config = load_config()
        base_dir = os.path.dirname(os.path.dirname(__file__))
        filepath = os.path.join(base_dir, config["data"]["scats_path"])

    # Read with first row as header
    df = pd.read_excel(filepath, sheet_name="Data", header=0)

    # The first row of data contains the actual column names
    # Row 0 has: "SCATS Number", "Location", ..., "Date", "V00", "V01", ...
    # The actual Excel headers are: "Unnamed:0", ..., time objects
    # So row 0 IS actually a header row read as data

    # Get the actual header names from row 0
    actual_headers = df.iloc[0].values
    # Drop the header row
    df = df.iloc[1:].reset_index(drop=True)

    # Build column mapping
    col_map = {}
    for i, (old_col, new_name) in enumerate(zip(df.columns, actual_headers)):
        if isinstance(new_name, str):
            col_map[old_col] = new_name
        else:
            # datetime.time objects -> V00, V01, etc.
            col_map[old_col] = f"V{i - 10:02d}" if i >= 10 else str(new_name)

    df.rename(columns=col_map, inplace=True)

    # Standardize column names
    rename_map = {
        "SCATS Number": "scats_number",
        "Location": "location",
        "NB_LATITUDE": "latitude",
        "NB_LONGITUDE": "longitude",
        "Date": "date",
    }
    df.rename(columns=rename_map, inplace=True)

    # Convert types
    df["scats_number"] = pd.to_numeric(df["scats_number"], errors="coerce").astype(int)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce").astype(float)
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce").astype(float)
    df["date"] = pd.to_datetime(df["date"])

    # Volume columns
    volume_cols = [f"V{i:02d}" for i in range(96)]
    for col in volume_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Keep only needed columns
    meta_cols = ["scats_number", "location", "latitude", "longitude", "date"]
    keep_cols = meta_cols + [c for c in volume_cols if c in df.columns]
    df = df[keep_cols].copy()

    # Fix missing coordinates
    df = _fix_missing_coords(df)

    df = df.sort_values(["scats_number", "date"]).reset_index(drop=True)

    return df


def _fix_missing_coords(df):
    """Fix sites with lat=0 or lon=0 by interpolating from nearby sites."""
    bad_mask = (df["latitude"] == 0) | (df["longitude"] == 0) | \
               df["latitude"].isna() | df["longitude"].isna()
    bad_sites = df.loc[bad_mask, "scats_number"].unique()

    if len(bad_sites) == 0:
        return df

    manual_coords = {
        4266: (-37.8220, 145.0440),
    }

    for site in bad_sites:
        site_mask = df["scats_number"] == site
        good_rows = df.loc[site_mask & ~bad_mask]
        if len(good_rows) > 0:
            lat = good_rows["latitude"].iloc[0]
            lon = good_rows["longitude"].iloc[0]
        elif site in manual_coords:
            lat, lon = manual_coords[site]
        else:
            continue
        df.loc[site_mask, "latitude"] = lat
        df.loc[site_mask, "longitude"] = lon

    return df


def get_site_locations(df):
    """
    Extract unique SCATS site locations.
    Returns dict: {scats_number: (latitude, longitude)}
    """
    sites = {}
    for _, row in df.drop_duplicates("scats_number").iterrows():
        sites[int(row["scats_number"])] = (float(row["latitude"]), float(row["longitude"]))
    return sites


def get_site_list(df):
    """Get sorted list of unique SCATS site numbers."""
    return sorted(df["scats_number"].unique().tolist())


def get_site_descriptions(df):
    """
    Get location descriptions for each site.
    Returns dict: {scats_number: "ROAD_A/ROAD_B description"}
    """
    desc = {}
    for _, row in df.drop_duplicates("scats_number").iterrows():
        desc[int(row["scats_number"])] = str(row["location"])
    return desc
