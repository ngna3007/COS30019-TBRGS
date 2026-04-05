"""
CLI entry point for the Traffic-Based Route Guidance System (TBRGS).

Usage:
    python main.py <origin> <destination> [--time HH:MM] [--model LSTM|GRU|RandomForest] [--k 5]

Example:
    python main.py 2000 3002 --time 08:00 --model LSTM --k 5
"""

import os
import sys
import argparse
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from src.data_loader import load_scats_data, get_site_locations, get_site_descriptions, load_config
from src.preprocessing import melt_to_timeseries, aggregate_by_site
from src.graph_builder import load_adjacency
from src.route_finder import find_routes, format_route_display


def get_historical_predictions(agg, locations, hour, minute):
    """Get predictions from historical averages."""
    agg_t = agg.copy()
    agg_t["hour"] = agg_t["datetime"].dt.hour
    agg_t["minute"] = agg_t["datetime"].dt.minute

    predictions = {}
    for site in locations:
        site_data = agg_t[agg_t["scats_number"] == site]
        time_data = site_data[(site_data["hour"] == hour) & (site_data["minute"] == minute)]
        if len(time_data) > 0:
            predictions[site] = time_data["volume"].mean()
        else:
            hour_data = site_data[site_data["hour"] == hour]
            predictions[site] = hour_data["volume"].mean() if len(hour_data) > 0 else 50
    return predictions


def main():
    parser = argparse.ArgumentParser(description="TBRGS - Find routes in Boroondara")
    parser.add_argument("origin", type=int, help="Origin SCATS site number")
    parser.add_argument("destination", type=int, help="Destination SCATS site number")
    parser.add_argument("--time", type=str, default="08:00", help="Time of day (HH:MM)")
    parser.add_argument("--model", type=str, default="LSTM",
                        choices=["LSTM", "GRU", "RandomForest"], help="ML model to use")
    parser.add_argument("--k", type=int, default=5, help="Number of routes to find")
    args = parser.parse_args()

    # Parse time
    hour, minute = map(int, args.time.split(":"))

    print(f"TBRGS - Traffic-Based Route Guidance System")
    print(f"=" * 50)

    # Load data
    print("Loading data...")
    df = load_scats_data()
    locations = get_site_locations(df)
    descriptions = get_site_descriptions(df)
    adjacency = load_adjacency()

    # Validate inputs
    if args.origin not in locations:
        print(f"Error: Origin site {args.origin} not found.")
        print(f"Available sites: {sorted(locations.keys())}")
        sys.exit(1)
    if args.destination not in locations:
        print(f"Error: Destination site {args.destination} not found.")
        print(f"Available sites: {sorted(locations.keys())}")
        sys.exit(1)

    # Get predictions
    print(f"Getting traffic predictions for {hour:02d}:{minute:02d}...")
    ts = melt_to_timeseries(df)
    agg = aggregate_by_site(ts)
    predictions = get_historical_predictions(agg, locations, hour, minute)

    # Find routes
    print(f"Finding top-{args.k} routes from {args.origin} to {args.destination}...\n")
    routes = find_routes(
        args.origin, args.destination, None,
        predictions, adjacency, locations, k=args.k
    )

    if not routes or "error" in routes[0]:
        print("No routes found.")
        sys.exit(1)

    print(f"Origin:      {args.origin} ({descriptions.get(args.origin, '')})")
    print(f"Destination: {args.destination} ({descriptions.get(args.destination, '')})")
    print(f"Time:        {hour:02d}:{minute:02d}")
    print(f"Model:       {args.model}")
    print(f"Routes found: {len(routes)}\n")

    for i, route in enumerate(routes):
        print(f"--- Route {i + 1} ---")
        print(format_route_display(route, descriptions))
        print()


if __name__ == "__main__":
    main()
