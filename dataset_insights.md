# Dataset Overview and Insights

## What is SCATS?

SCATS (Sydney Coordinated Adaptive Traffic System) is the traffic management system used by VicRoads across Victoria. Detectors embedded in the road at each intersection count how many vehicles pass over them every 15 minutes.

## Dataset Structure

| Property | Value |
|---|---|
| Source | VicRoads, Boroondara municipality |
| File | Scats Data October 2006.xls |
| Period | 1-31 October 2006 (31 days) |
| SCATS sites | 40 intersections |
| Readings per day | 96 (one every 15 minutes) |
| Raw rows | 4,192 (multiple measurement directions per site per day) |
| Volume columns | V00 (00:00) through V95 (23:45) |
| Total data points | 4,192 x 96 = 402,432 individual volume readings |

### After Aggregation (per site, per timestamp)

| Property | Value |
|---|---|
| Aggregated records | 116,160 |
| Per site | ~2,904 timesteps |
| Train (70%) | ~2,071 samples per site |
| Validation (10%) | ~285 samples per site |
| Test (20%) | ~584 samples per site |
| Input features | 6 (volume, hour_sin, hour_cos, dow_sin, dow_cos, is_weekend) |
| Window size | 12 timesteps (3 hours) |

## Key Insights

### 1. Very Small Dataset

The spec explicitly calls this "a very small dataset." With only 31 days of data per site, deep learning models have limited training material. This is a known limitation:
- LSTM/GRU still perform well (R2 ~0.98) because traffic patterns are highly repetitive and periodic.
- Random Forest performs comparably because it doesn't need as much data as neural networks.
- A larger dataset (multiple months/years) would likely improve all models and especially benefit deep learning.

### 2. Multiple Measurement Directions

Some sites have multiple rows per day representing different approach directions (e.g., site 3120 has "BURKE_RD N of CANTERBURY_RD" and "BURKE_RD S of CANTERBURY_RD" as separate entries). We aggregate (sum) all directions per site per timestamp to get total intersection flow. This is appropriate for the travel time formula which uses total flow.

### 3. Contradicting Flow Specification

- The assignment spec (page 1) says: "accumulated volume per hour at the SCATS site **B**" (destination site).
- The conversion document says: "Between two SCATS sites, the flow is calculated from the **starting** SCATS site."
- We follow the conversion document since it is the dedicated reference for this calculation.
- This should be noted as an assumption in the report.

### 4. Lat/Lon Coordinates Are Inaccurate

The spec warns (page 3): "the Latitude and Longitude of the SCATS sites do not map correctly to the actual intersections on Google Maps, you'll have to make adjustments."
- Site 4266 (AUBURN_RD/BURWOOD_RD) has lat=0, lon=0 (completely missing). We interpolated from nearby sites.
- Other sites' coordinates may be slightly off from their true positions.
- For our purposes (relative distances between sites, Haversine calculation), the provided coordinates are sufficient.

### 5. Data Quality Issues (~10% Unreliable)

The VicRoads notes sheet documents:
- Some data removed due to road works, maintenance, and equipment failure.
- Detectors can over-count due to poor lane discipline (vehicles straddling lanes trigger multiple detectors).
- This means ~10% of the raw data may be unreliable.
- We handle this by using aggregation (smooths out individual detector errors) and robust ML models.

### 6. Strong Daily Periodicity

Traffic flow shows a very clear daily cycle:
- Low volume overnight (00:00-06:00): ~10-30 vehicles per 15 min
- Morning peak (07:00-09:00): ~200-350 vehicles per 15 min
- Midday moderate (10:00-15:00): ~150-250 vehicles per 15 min
- Evening peak (16:00-18:00): ~250-350 vehicles per 15 min
- Tapering off (19:00-23:00): ~100-150 vehicles per 15 min

This strong periodicity is why all models achieve high R2 - the pattern is predictable.

### 7. Weekday vs Weekend Difference

Weekday traffic shows the classic double-peak (AM/PM rush hours).
Weekend traffic is lower overall with a single broader peak around midday.
This distinction is captured by our `is_weekend` and `dow_sin/dow_cos` features.

### 8. The Spec Prioritizes Insights Over Accuracy

The report section explicitly states (bold in spec): "We are most interested in the insights you have gained, especially those **using data obtained by running your software**."

The objective is NOT to beat Google Maps. It is to demonstrate understanding of:
- ML pipeline (data processing, training, evaluation)
- Model comparison (which is better and why)
- Integration of ML predictions with graph search
- The end-to-end TBRGS system

## Data Split Strategy

### Why 70/10/20 (Train/Validation/Test)?

For small datasets, using a proper 3-way temporal split is important:

| Approach | Pros | Cons |
|---|---|---|
| **No val set (80/20)** | More training data | Data leakage: test set used for early stopping decisions |
| **3-way split (70/10/20)** | Clean evaluation, no leakage | Less training data |
| **K-fold cross validation** | Uses all data efficiently | Cannot shuffle time series data; complex to implement |
| **Time series cross validation** | Best for temporal data | Very complex; overkill for this assignment |

We chose **70/10/20 temporal split** because:
1. It prevents data leakage (test set is never seen during training or hyperparameter tuning).
2. The temporal ordering is preserved (no future data in training).
3. It is simple and transparent for the grader to understand.
4. With ~2,000 training samples per site, models still converge well.

**Temporal split detail:**
- Train: Days 1-21 (first 70% of October)
- Validation: Days 22-24 (next 10%, used for early stopping in LSTM/GRU)
- Test: Days 25-31 (last 20%, final evaluation only)

## Files Provided

| File | Purpose |
|---|---|
| Scats Data October 2006.xls | Primary dataset: traffic volumes |
| Traffic_Count_Locations_with_LONG_LAT.csv | Site metadata: coordinates, AADT, road names |
| SCATSSiteListingSpreadsheet_VicRoads.xls | Reference data, data quality notes |
| IBM data_Bor.pdf | Boroondara map showing SCATS site locations |
| Traffic Flow to Travel Time Conversion v1.0.pdf | Quadratic formula for flow-to-speed conversion |
