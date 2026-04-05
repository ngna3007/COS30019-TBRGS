# COS30019 Assignment 2B - TBRGS Implementation Plan
## Target: 100/100

---

## Marking Breakdown

| Component | Marks |
|---|---|
| Data processing | 9 |
| 3 ML algorithms (7 x 3) | 21 |
| Comprehensive ML evaluation | 15 |
| Integration (Part A + Part B) end-to-end TBRGS | 15 |
| Testing (10+ test cases) | 10 |
| Report (8-10 pages) | 15 |
| Research initiatives | 15 |
| Code quality penalty | up to -10 |
| **Total** | **100** |

---

## Project Structure

```
ASM2B/
  config.yaml                    # Central configuration
  requirements.txt               # Dependencies
  main.py                        # CLI entry point
  train.py                       # Train all models
  evaluate.py                    # Run evaluation + generate charts
  app.py                         # Streamlit GUI entry point
  src/
    __init__.py
    data_loader.py               # Load + clean SCATS data
    preprocessing.py             # Feature engineering, sequences, scaling
    models/
      __init__.py
      base_model.py              # Abstract base class
      lstm_model.py              # LSTM
      gru_model.py               # GRU
      sarima_model.py            # SARIMA (3rd model)
    evaluation.py                # Metrics, charts, comparison tables
    traffic_conversion.py        # Flow -> speed -> travel time
    graph_builder.py             # Build SCATS adjacency graph
    search.py                    # Ported + fixed search from 2A
    route_finder.py              # Yen's K-shortest paths + integration
    visualization.py             # Folium/OpenStreetMap maps
  data/
    adjacency.json               # SCATS site adjacency (manually verified)
    (existing data files stay here)
  trained_models/                # Saved model weights
  results/                       # Generated charts, tables
  tests/
    __init__.py
    test_data_loader.py
    test_preprocessing.py
    test_models.py
    test_traffic_conversion.py
    test_graph_builder.py
    test_search.py
    test_route_finder.py
    test_integration.py
```

---

## Phase 1: Data Processing (9 marks)

### 1a. `src/data_loader.py`

**Load SCATS Excel data:**
- Read "Data" sheet from `Scats Data October 2006.xls`
- 4,192 rows x 106 columns
- 40 SCATS sites, October 2006, 96 volume readings (V00-V95) per 15-min interval
- Convert Excel serial dates to datetime
- Fix site 4266 missing coordinates (lat=0, lon=0) by interpolating from nearby Burwood Rd sites
- Return clean DataFrame: `scats_number, location, latitude, longitude, date, V00...V95`

### 1b. `src/preprocessing.py`

**Feature engineering + ML-ready data:**
- Melt wide format (96 cols) to long format: (site, timestamp, volume) = ~400K rows
- Aggregate per-site total volume per timestamp (sum across measurement directions)
- Multiply 15-min count x 4 = vehicles/hour (for flow-to-speed formula)
- Create sliding window sequences: past 12 intervals (3 hours) -> predict next interval
- Add time features: hour_sin, hour_cos, day_of_week_sin, day_of_week_cos, is_weekend
- Temporal train/test split: first 25 days train, last 6 days test (no data leakage)
- MinMaxScaler fitted on training data only

---

## Phase 2: ML Models (21 marks = 7 x 3)

### 2a. `src/models/base_model.py`
Abstract interface: `train()`, `predict()`, `save()`, `load()`, `get_name()`

### 2b. `src/models/lstm_model.py` (7 marks)
- Framework: TensorFlow/Keras
- Architecture: 2 LSTM layers (64 units) + dropout (0.2) + Dense output
- Training: Adam, MSE loss, early stopping (patience=10), ReduceLROnPlateau
- Input: (window_size, num_features)
- One model per site

### 2c. `src/models/gru_model.py` (7 marks)
- Same framework, same pipeline
- 2 GRU layers (64 units) + dropout (0.2) + Dense output
- GRU: fewer parameters, faster training (note in report)

### 2d. `src/models/sarima_model.py` (7 marks)
- Framework: statsmodels SARIMAX
- Aggregate to hourly data (s=24) to avoid s=96 being too slow
- Order: (2,1,2)(1,1,1,24) or use pmdarima auto_arima
- Provides classical stats vs deep learning contrast
- Fit per-site, save via pickle

---

## Phase 3: Evaluation (15 marks)

### `src/evaluation.py`

**Metrics:** MAE, RMSE, MAPE (handle zero-flow), R-squared

**Visualizations to generate:**
1. Comparison table: model x metrics (per-site and aggregate)
2. Actual vs Predicted line plots (selected sites, peak/off-peak)
3. Training loss curves (LSTM, GRU)
4. Error distribution histograms per model
5. Box plots: per-site error comparison
6. Bar chart: side-by-side metric comparison
7. Heatmap: per-site performance matrix

All saved to `results/` with matplotlib/seaborn.

---

## Phase 4: Traffic Conversion + Graph

### 4a. `src/traffic_conversion.py`

**Formula:** `flow = -1.4648375 * speed^2 + 93.75 * speed`

**`flow_to_speed(flow_per_hour)`:**
- If flow <= 351: return 60.0 (speed limit, free flow)
- If flow > 351: solve quadratic for green curve (higher speed root, under capacity)
  - `speed = (93.75 - sqrt(8789.0625 - 5.85935 * flow)) / 2.929675`
- Cap at 60, floor at 5 km/hr
- Max flow from formula: 1500 veh/hr (discriminant = 0)

**`compute_travel_time(distance_km, flow_per_hour)`:**
- `speed = flow_to_speed(flow)`
- `travel_time = (distance_km / speed) * 3600 + 30` seconds
- 30 seconds = intersection delay

**IMPORTANT from spec:** "the travel time from SCATS site A to SCATS site B can be approximated by... the accumulated volume per hour at the SCATS site **B**"
- Use flow at the DESTINATION site (site B), not the starting site
- Wait, the conversion doc says: "Between two SCATS sites, the flow is calculated from the **starting** SCATS site"
- These contradict! The conversion doc takes precedence since it's specific to this topic
- **Use flow at the STARTING site A**

### 4b. `src/graph_builder.py`

**Determine adjacency from Boroondara map:**
- Parse location descriptions to get road names per site
- For each road, sort sites geographically, connect consecutive pairs only
- Compute distances via Haversine formula (lat/lon)
- Store in `data/adjacency.json` (manually verified against map PDF)

**Expected ~50-60 bidirectional edges. Key corridors:**
- Burke Rd (N-S): ~8 sites chain
- Warrigal Rd (N-S): ~5 sites chain
- Canterbury Rd (E-W): ~4 sites
- High St (E-W): ~6 sites
- Toorak Rd (E-W): ~3 sites

**`build_traffic_graph(adjacency, predictions, time_index)`:**
- For each edge (A, B): get predicted flow at A, compute travel_time
- Return: `{node: [(neighbor, travel_time), ...]}`

---

## Phase 5: Search Integration (15 marks)

### 5a. `src/search.py` (ported from 2A with fixes)

**Fixes to apply:**
1. **IDS bug:** Remove `explored` set in `_depth_limited_search`. Use path-based cycle detection (walk up parent chain). This is required for IDS optimality.
2. **visit_all:** Replace `itertools.permutations` with Held-Karp DP (bitmask). O(n^2 * 2^n) vs O(n!).
3. **Input validation:** Handle missing nodes, disconnected graphs gracefully.
4. **Adapt for SCATS:** Node IDs are SCATS numbers (integers). Coords are (lat, lon).

### 5b. `src/route_finder.py`

**Yen's K-Shortest Paths algorithm:**
1. Find shortest path P1 via A*
2. For k=2..K: spur from each node in P_{k-1}, remove used edges, find spur path, collect candidates
3. Return up to 5 distinct paths with travel times

**End-to-end function:**
```
route_with_predictions(origin, dest, time_of_day, model, adjacency, site_data)
  -> [(path, travel_time), ...] up to 5 routes
```

---

## Phase 6: GUI (Streamlit)

### `app.py`

**Layout:**
- **Sidebar:** Origin dropdown, Destination dropdown, Time slider (00:00-23:45), Day type, ML model selector, Num routes slider, [Find Routes] button
- **Tab 1 - Routes:** Folium map with color-coded routes + results table
- **Tab 2 - Traffic Predictions:** Line chart of predicted flow
- **Tab 3 - Model Evaluation:** Comparison table + charts
- **Tab 4 - Network:** Full SCATS network visualization

**Config file:** `config.yaml` with all defaults (paths, hyperparams, GUI settings)

---

## Phase 7: Testing (10 marks)

### 14 test cases across 8 test files:

| # | Test | File | What it tests |
|---|------|------|---------------|
| 1 | test_data_loading | test_data_loader.py | 40 sites, 4192 rows, 96 V columns |
| 2 | test_missing_coords | test_data_loader.py | Site 4266 coordinates corrected |
| 3 | test_sliding_window | test_preprocessing.py | Window size 12 on 100 items = 88 samples |
| 4 | test_train_test_split | test_preprocessing.py | Temporal split, no leakage |
| 5 | test_lstm_shape | test_models.py | LSTM output matches expected dims |
| 6 | test_gru_shape | test_models.py | GRU output matches expected dims |
| 7 | test_flow_free | test_traffic_conversion.py | flow=200 -> speed=60 |
| 8 | test_flow_congested | test_traffic_conversion.py | flow=1000 -> 0 < speed < 60 |
| 9 | test_travel_time_positive | test_traffic_conversion.py | Always > 0 |
| 10 | test_adjacency | test_graph_builder.py | Burke Rd chain verified |
| 11 | test_ids_fixed | test_search.py | IDS returns shallowest path |
| 12 | test_same_origin_dest | test_route_finder.py | Returns empty path |
| 13 | test_top_k | test_route_finder.py | Returns up to 5 distinct paths |
| 14 | test_peak_vs_offpeak | test_integration.py | 8am travel > 2am travel |

---

## Phase 8: Research (15 marks)

### R1. OpenStreetMap Visualization
- Folium map with SCATS markers, route polylines (green=fast, red=slow)
- Popups showing predicted flow/speed
- Congestion heatmap overlay

### R2. Traffic Pattern Analysis
- Daily/weekly traffic heatmaps (hour x day-of-week)
- Peak hour identification per site
- Cross-site correlation analysis

### R3. Hyperparameter Sensitivity
- Window size: 6, 12, 24
- LSTM/GRU units: 32, 64, 128
- Learning curve: how much data each model needs

### R4. Route Comparison Dashboard
- How recommended route changes by time of day
- ML-predicted vs static shortest-distance routing
- Quantify travel time savings from traffic-aware routing

---

## Implementation Order

### Week 1: Foundation
1. `config.yaml`
2. `src/data_loader.py` + tests
3. `src/preprocessing.py` + tests
4. `src/models/base_model.py`

### Week 2: Models + Evaluation
5. `src/models/lstm_model.py`
6. `src/models/gru_model.py`
7. `src/models/sarima_model.py`
8. `train.py`
9. `src/evaluation.py` + generate results

### Week 3: Integration + Routing
10. `src/traffic_conversion.py` + tests
11. `data/adjacency.json` (build + verify)
12. `src/graph_builder.py` + tests
13. `src/search.py` (port + fix IDS, visit_all)
14. `src/route_finder.py` (Yen's K-shortest) + tests

### Week 4: GUI + Research + Report
15. `src/visualization.py`
16. `app.py` (Streamlit GUI)
17. Research analyses
18. `tests/test_integration.py`
19. Report writing

---

## Key Spec Details (Don't Miss!)

1. **Flow uses STARTING site:** "Between two SCATS sites, the flow is calculated from the starting SCATS site"
2. **Speed capped at 60 km/hr** when flow <= 351 veh/hr
3. **Green curve (under capacity)** assumed for all traffic
4. **30 seconds delay** per controlled intersection
5. **Top-5 routes** from origin to destination
6. **SCATS site numbers** as input (e.g., O=2000, D=3002)
7. **Lat/lon don't map correctly** to actual intersections on Google Maps - need adjustments
8. **Report 8-10 pages** excluding cover + TOC
9. **Config file required** for default parameters

## Dependencies

```
tensorflow>=2.10
pandas
numpy
scikit-learn
statsmodels
matplotlib
seaborn
streamlit
folium
streamlit-folium
pyyaml
xlrd
pytest
pmdarima  # optional, for auto_arima
```
