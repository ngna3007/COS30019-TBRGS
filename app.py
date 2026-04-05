"""
Streamlit GUI for the Traffic-Based Route Guidance System (TBRGS).
Usage: streamlit run app.py
"""

import os
import sys
import json
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_scats_data, get_site_locations, get_site_descriptions, load_config
from src.preprocessing import melt_to_timeseries, aggregate_by_site, add_time_features, prepare_site_data, inverse_scale_volume
from src.graph_builder import load_adjacency, build_traffic_graph
from src.route_finder import find_routes
from src.traffic_conversion import flow_to_speed, flow_15min_to_hourly
from src.visualization import create_route_map, create_network_map
try:
    from src.models.lstm_model import LSTMTrafficModel
    from src.models.gru_model import GRUTrafficModel
    HAS_TF = True
except ImportError:
    HAS_TF = False

from src.models.rf_model import RandomForestTrafficModel


@st.cache_data
def load_data():
    """Load and preprocess SCATS data (cached)."""
    df = load_scats_data()
    locations = get_site_locations(df)
    descriptions = get_site_descriptions(df)
    ts = melt_to_timeseries(df)
    agg = aggregate_by_site(ts)
    return df, locations, descriptions, agg


@st.cache_resource
def load_trained_models(model_type, site_ids):
    """Load trained models from disk."""
    base_dir = os.path.dirname(__file__)
    models_dir = os.path.join(base_dir, "trained_models")
    models = {}

    for site_id in site_ids:
        if model_type == "RandomForest":
            path = os.path.join(models_dir, f"RandomForest_{site_id}.pkl")
            if os.path.exists(path):
                m = RandomForestTrafficModel()
                m.load(path)
                models[site_id] = m
        elif HAS_TF:
            ext = ".keras"
            path = os.path.join(models_dir, f"{model_type}_{site_id}{ext}")
            if os.path.exists(path):
                if model_type == "LSTM":
                    m = LSTMTrafficModel()
                else:
                    m = GRUTrafficModel()
                m.load(path)
                models[site_id] = m
    return models


def get_predictions_for_time(agg, site_locations, model_type, hour, minute, day_type):
    """
    Get traffic predictions for a specific time.
    Falls back to historical average if models not available.
    """
    predictions = {}

    # Use historical average as baseline/fallback
    agg_with_time = agg.copy()
    agg_with_time["hour"] = agg_with_time["datetime"].dt.hour
    agg_with_time["minute"] = agg_with_time["datetime"].dt.minute
    agg_with_time["is_weekend"] = agg_with_time["datetime"].dt.dayofweek >= 5

    weekend = (day_type == "Weekend")
    time_mask = (agg_with_time["hour"] == hour) & \
                (agg_with_time["minute"] == minute) & \
                (agg_with_time["is_weekend"] == weekend)

    for site_id in site_locations:
        site_mask = agg_with_time["scats_number"] == site_id
        matching = agg_with_time[site_mask & time_mask]
        if len(matching) > 0:
            predictions[site_id] = matching["volume"].mean()
        else:
            # Broader fallback: just same hour
            hour_mask = (agg_with_time["hour"] == hour) & \
                        (agg_with_time["is_weekend"] == weekend)
            matching = agg_with_time[site_mask & hour_mask]
            if len(matching) > 0:
                predictions[site_id] = matching["volume"].mean()
            else:
                predictions[site_id] = 50  # default

    return predictions


def main():
    config = load_config()

    st.set_page_config(
        page_title=config["gui"]["title"],
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("Traffic-Based Route Guidance System")
    st.markdown("**Boroondara Area, Melbourne** - COS30019 Assignment 2B")

    # Load data
    df, locations, descriptions, agg = load_data()
    adjacency = load_adjacency()

    site_ids = sorted(locations.keys())
    site_options = {s: f"{s} - {descriptions.get(s, '')}" for s in site_ids}

    # ---- Sidebar ----
    st.sidebar.header("Route Settings")

    origin = st.sidebar.selectbox(
        "Origin SCATS Site",
        options=site_ids,
        format_func=lambda s: site_options[s],
        index=site_ids.index(2000) if 2000 in site_ids else 0,
    )

    destination = st.sidebar.selectbox(
        "Destination SCATS Site",
        options=site_ids,
        format_func=lambda s: site_options[s],
        index=site_ids.index(3002) if 3002 in site_ids else 1,
    )

    time_val = st.sidebar.slider(
        "Time of Day",
        min_value=0, max_value=95,
        value=32,  # 08:00
        format="",
        help="Drag to select time (0=00:00, 32=08:00, 72=18:00)"
    )
    hour = (time_val * 15) // 60
    minute = (time_val * 15) % 60
    st.sidebar.write(f"Selected time: **{hour:02d}:{minute:02d}**")

    day_type = st.sidebar.radio("Day Type", ["Weekday", "Weekend"])

    model_options = ["LSTM", "GRU", "RandomForest"] if HAS_TF else ["RandomForest"]
    model_type = st.sidebar.selectbox("ML Model", model_options)

    num_routes = st.sidebar.slider("Number of Routes", 1, 5, 5)

    find_button = st.sidebar.button("Find Routes", type="primary")

    # ---- Main content ----
    tab1, tab2, tab3, tab4 = st.tabs([
        "Routes", "Traffic Predictions", "Model Evaluation", "Network"
    ])

    with tab1:
        if find_button:
            if origin == destination:
                st.warning("Origin and destination are the same.")
            else:
                with st.spinner("Finding routes..."):
                    predictions = get_predictions_for_time(
                        agg, locations, model_type, hour, minute, day_type
                    )

                    routes = find_routes(
                        origin, destination, None,
                        predictions, adjacency, locations, k=num_routes
                    )

                if not routes or "error" in routes[0]:
                    st.error("No routes found between the selected sites.")
                else:
                    st.success(f"Found {len(routes)} route(s)")

                    # Results table
                    table_data = []
                    for i, route in enumerate(routes):
                        path_str = " -> ".join(str(s) for s in route["path"])
                        table_data.append({
                            "Route": i + 1,
                            "Path": path_str,
                            "Travel Time (min)": route["travel_time_minutes"],
                            "Distance (km)": route["distance_km"],
                            "Intersections": route["num_intersections"],
                        })
                    st.dataframe(pd.DataFrame(table_data), use_container_width=True)

                    # Map
                    try:
                        route_map = create_route_map(
                            routes, locations, adjacency, descriptions, predictions
                        )
                        st.components.v1.html(route_map._repr_html_(), height=500)
                    except Exception as e:
                        st.warning(f"Map visualization unavailable: {e}")
        else:
            st.info("Configure route settings in the sidebar and click 'Find Routes'.")

    with tab2:
        st.subheader("Traffic Volume Predictions")
        selected_site = st.selectbox(
            "Select Site",
            options=site_ids,
            format_func=lambda s: site_options[s],
            key="pred_site",
        )

        site_data = agg[agg["scats_number"] == selected_site].copy()
        if len(site_data) > 0:
            site_data = add_time_features(site_data)
            daily = site_data.groupby(site_data["datetime"].dt.hour)["volume"].mean()

            st.line_chart(daily, use_container_width=True)
            st.caption("Average traffic volume by hour of day (all October 2006)")

    with tab3:
        st.subheader("Model Evaluation Results")
        results_path = os.path.join(os.path.dirname(__file__), "results", "results.json")
        if os.path.exists(results_path):
            with open(results_path) as f:
                results = json.load(f)

            if "aggregate" in results:
                st.markdown("### Aggregate Metrics")
                comp_df = pd.DataFrame(results["aggregate"]).T
                comp_df.index.name = "Model"
                st.dataframe(comp_df.round(4), use_container_width=True)

            # Show saved charts
            chart_files = [
                ("metrics_comparison.png", "Metrics Comparison"),
                ("per_site_rmse_boxplot.png", "Per-Site RMSE Distribution"),
                ("per_site_rmse_heatmap.png", "Per-Site RMSE Heatmap"),
            ]
            for fname, title in chart_files:
                fpath = os.path.join(os.path.dirname(__file__), "results", fname)
                if os.path.exists(fpath):
                    st.markdown(f"### {title}")
                    st.image(fpath)

            # Training curves
            for model_name in ["LSTM", "GRU"]:
                fpath = os.path.join(os.path.dirname(__file__), "results",
                                     f"{model_name}_training_curves.png")
                if os.path.exists(fpath):
                    st.markdown(f"### {model_name} Training Curves")
                    st.image(fpath)
        else:
            st.info("No evaluation results yet. Run `python train.py` first.")

    with tab4:
        st.subheader("SCATS Network - Boroondara")
        try:
            # Show network with current time predictions
            predictions = get_predictions_for_time(
                agg, locations, model_type, hour, minute, day_type
            )
            network_map = create_network_map(
                locations, adjacency, descriptions, predictions
            )
            st.components.v1.html(network_map._repr_html_(), height=600)
        except Exception as e:
            st.warning(f"Network visualization unavailable: {e}")


if __name__ == "__main__":
    main()
