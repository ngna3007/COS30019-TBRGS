"""OpenStreetMap visualization using Folium for routes and traffic."""

import folium
import numpy as np


def create_base_map(site_locations, zoom_start=13):
    """Create a base Folium map centered on Boroondara."""
    lats = [loc[0] for loc in site_locations.values()]
    lons = [loc[1] for loc in site_locations.values()]
    center = [np.mean(lats), np.mean(lons)]

    m = folium.Map(location=center, zoom_start=zoom_start,
                   tiles="OpenStreetMap")
    return m


def add_site_markers(m, site_locations, site_descriptions=None,
                     predictions=None):
    """Add SCATS site markers to the map."""
    for site_id, (lat, lon) in site_locations.items():
        desc = ""
        if site_descriptions:
            desc = site_descriptions.get(site_id, "")

        popup_text = f"<b>Site {site_id}</b><br>{desc}"
        if predictions and site_id in predictions:
            vol = predictions[site_id]
            popup_text += f"<br>Predicted Volume: {vol:.0f} veh/15min"

        folium.CircleMarker(
            location=[lat, lon],
            radius=6,
            popup=folium.Popup(popup_text, max_width=200),
            tooltip=f"Site {site_id}",
            color="blue",
            fill=True,
            fill_opacity=0.7,
        ).add_to(m)
    return m


def add_route(m, path, site_locations, color="blue", weight=4,
              opacity=0.8, label=None):
    """Draw a route on the map as a polyline."""
    coords = []
    for site_id in path:
        if site_id in site_locations:
            lat, lon = site_locations[site_id]
            coords.append([lat, lon])

    if len(coords) < 2:
        return m

    popup = label or "Route"
    folium.PolyLine(
        coords,
        color=color,
        weight=weight,
        opacity=opacity,
        popup=popup,
    ).add_to(m)

    # Mark origin and destination
    folium.Marker(
        coords[0],
        popup=f"Origin: {path[0]}",
        icon=folium.Icon(color="green", icon="play"),
    ).add_to(m)
    folium.Marker(
        coords[-1],
        popup=f"Destination: {path[-1]}",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(m)

    return m


def add_multiple_routes(m, routes, site_locations):
    """Add multiple routes with different colors."""
    colors = ["blue", "red", "green", "purple", "orange"]
    for i, route in enumerate(routes):
        color = colors[i % len(colors)]
        path = route["path"]
        label = (f"Route {i + 1}: {route['travel_time_minutes']:.1f} min, "
                 f"{route['distance_km']:.2f} km")
        add_route(m, path, site_locations, color=color, weight=4 - i * 0.5,
                  label=label)
    return m


def add_congestion_overlay(m, adjacency, site_locations, predictions):
    """
    Color edges by congestion level.
    Green = free flow, Yellow = moderate, Red = congested.
    """
    from .traffic_conversion import flow_15min_to_hourly

    for site_id, neighbors in adjacency.items():
        if site_id not in site_locations:
            continue
        lat1, lon1 = site_locations[site_id]

        flow_15min = predictions.get(site_id, 0)
        flow_hourly = flow_15min_to_hourly(flow_15min)

        # Color based on flow level
        if flow_hourly <= 351:
            color = "green"
        elif flow_hourly <= 900:
            color = "orange"
        else:
            color = "red"

        for neighbor_id, _ in neighbors:
            if neighbor_id not in site_locations:
                continue
            lat2, lon2 = site_locations[neighbor_id]
            folium.PolyLine(
                [[lat1, lon1], [lat2, lon2]],
                color=color,
                weight=3,
                opacity=0.6,
                popup=f"Flow: {flow_hourly:.0f} veh/hr",
            ).add_to(m)

    return m


def create_network_map(site_locations, adjacency, site_descriptions=None,
                       predictions=None):
    """Create a full network visualization map."""
    m = create_base_map(site_locations)
    if predictions:
        m = add_congestion_overlay(m, adjacency, site_locations, predictions)
    m = add_site_markers(m, site_locations, site_descriptions, predictions)
    return m


def create_route_map(routes, site_locations, adjacency,
                     site_descriptions=None, predictions=None):
    """Create a map showing routes with network overlay."""
    m = create_base_map(site_locations)
    if predictions:
        m = add_congestion_overlay(m, adjacency, site_locations, predictions)
    m = add_site_markers(m, site_locations, site_descriptions, predictions)
    m = add_multiple_routes(m, routes, site_locations)
    return m
