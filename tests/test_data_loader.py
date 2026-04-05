"""Tests for data loading and cleaning."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import pandas as pd
from src.data_loader import load_scats_data, get_site_locations, get_site_list, get_site_descriptions


@pytest.fixture(scope="module")
def scats_df():
    return load_scats_data()


def test_data_loading_shape(scats_df):
    """Test 1: SCATS data loads with expected shape."""
    assert scats_df.shape[0] == 4192, f"Expected 4192 rows, got {scats_df.shape[0]}"
    # 5 meta columns + 96 volume columns = 101
    assert scats_df.shape[1] == 101, f"Expected 101 columns, got {scats_df.shape[1]}"


def test_40_sites(scats_df):
    """Test 2: Dataset contains exactly 40 SCATS sites."""
    sites = get_site_list(scats_df)
    assert len(sites) == 40, f"Expected 40 sites, got {len(sites)}"


def test_volume_columns_present(scats_df):
    """Test 3: All 96 volume columns V00-V95 are present."""
    for i in range(96):
        col = f"V{i:02d}"
        assert col in scats_df.columns, f"Missing column {col}"


def test_missing_coords_fixed(scats_df):
    """Test 4: Site 4266 coordinates are not zero after fixing."""
    locations = get_site_locations(scats_df)
    lat, lon = locations[4266]
    assert lat != 0, "Site 4266 latitude should not be 0"
    assert lon != 0, "Site 4266 longitude should not be 0"
    assert -38.0 < lat < -37.5, f"Site 4266 latitude {lat} out of Boroondara range"
    assert 144.9 < lon < 145.2, f"Site 4266 longitude {lon} out of Boroondara range"


def test_date_column_is_datetime(scats_df):
    """Test 5: Date column is properly converted to datetime."""
    assert pd.api.types.is_datetime64_any_dtype(scats_df["date"]), \
        "Date column should be datetime type"
    # Check October 2006 range
    assert scats_df["date"].min().year == 2006
    assert scats_df["date"].min().month == 10


def test_volume_values_non_negative(scats_df):
    """Test 6: All volume values are non-negative."""
    volume_cols = [f"V{i:02d}" for i in range(96)]
    for col in volume_cols:
        assert (scats_df[col] >= 0).all(), f"Negative values found in {col}"


def test_site_descriptions_exist(scats_df):
    """Test 7: All sites have location descriptions."""
    descriptions = get_site_descriptions(scats_df)
    assert len(descriptions) == 40
    for site_id, desc in descriptions.items():
        assert isinstance(desc, str) and len(desc) > 0, \
            f"Site {site_id} has empty description"
