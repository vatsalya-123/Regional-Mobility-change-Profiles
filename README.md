# Regional Mobility Change Profiles

[![Live Dashboard](https://img.shields.io/badge/Live-Dashboard-55C6B3?style=for-the-badge&logo=streamlit&logoColor=white)](https://regional-mobility-change-profiles-bzwchyhnexko3xh4s5dfsk.streamlit.app/)

**Live Demo:** [Open the interactive dashboard](https://regional-mobility-change-profiles-bzwchyhnexko3xh4s5dfsk.streamlit.app/)

An unsupervised machine learning project that discovers regional mobility change patterns during the COVID-19 pandemic using Google Community Mobility Reports.

## Project Overview

The COVID-19 pandemic changed how people travelled, worked, shopped, and used public spaces. However, these changes were not identical across regions.

This project applies unsupervised learning to identify groups of regions with similar mobility-change patterns and examine how those patterns evolved over time.

Rather than predicting predefined disease or mobility labels, the project discovers patterns from mobility features without using class labels.

## Research Objective

To discover and analyse regional mobility change profiles across different geographic levels using preprocessing, feature engineering, clustering, and validation techniques.

### Geographic Scopes

The analysis covers four geographic scopes:

- **National:** Country-level mobility profiles.
- **India States:** State-level mobility profiles within India.
- **India Districts:** District-level mobility profiles within India.
- **Global States:** Subnational regions across countries where sufficient data is available.

## Dataset

**Source:** Google COVID-19 Community Mobility Reports.

**Study period:** February 2020 – October 2022.

The dataset contains daily percentage changes in mobility compared with a weekday-specific baseline.

Six mobility categories are used:

1. Retail and recreation
2. Grocery and pharmacy
3. Parks
4. Transit stations
5. Workplaces
6. Residential

The raw dataset is not included in the repository. After obtaining the Google Community Mobility Reports CSV, place it at:

`data/raw/Global_Mobility_Report.csv`

The project expects this exact filename and location, as specified in `config.yaml`.

## Technology Stack

- Python 3.12
- Pandas and NumPy
- SciPy
- Scikit-learn
- Matplotlib and Seaborn
- Plotly
- PyArrow
- Streamlit
- PyYAML
- UMAP-learn

## Project Structure

```text
Regional-Mobility-Change-Profiles/
│
├── app/
│   ├── streamlit_app.py
│   ├── style.py
│   └── theme.css
│
├── data/                         # Raw data and caches are local; processed results and the India boundary are tracked
│   ├── raw/
│   ├── interim/
│   ├── processed/
│   └── boundaries/
│       └── india_full.geojson
│
├── outputs/
│   ├── figures/
│   └── tables/
│
├── reports/
│   └── preprocessing_decisions.md
│
├── src/
│   ├── __init__.py
│   ├── load_data.py
│   ├── profile_data.py
│   ├── preprocess.py
│   ├── features.py
│   ├── outliers.py
│   ├── cluster.py
│   ├── evaluate.py
│   ├── hierarchical.py
│   ├── interpret.py
│   ├── temporal.py
│   └── holdout.py
│
├── tests/
│   └── test_processed_data.py
│
├── .streamlit/
├── .gitignore
├── check_env.py
├── config.yaml
├── requirements.in
├── requirements.txt
└── run_pipeline.py
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/vatsalya-123/Regional-Mobility-change-Profiles.git
cd Regional-Mobility-change-Profiles
```

### 2. Create a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
```

macOS/Linux:

```bash
python3 -m venv .venv
```

### 3. Activate the environment

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
python check_env.py
```

### 5. Add the dataset

Place the downloaded CSV file at:

```text
data/raw/Global_Mobility_Report.csv
```

## Running the Project

### Execute the complete pipeline

From the project root:

```bash
python run_pipeline.py
```

The pipeline performs the following stages:

1. Load the raw mobility dataset.
2. Generate an initial data profile.
3. Preprocess mobility observations.
4. Engineer regional mobility features.
5. Detect and document extreme outliers.
6. Perform clustering.
7. Evaluate clustering configurations.
8. Compare hierarchical clustering results.
9. Interpret discovered clusters.
10. Analyse temporal patterns and transitions.
11. Perform hold-out validation.
12. Run the test suite.

### View available pipeline steps

```bash
python run_pipeline.py --list
```

### Run a specific step

```bash
python run_pipeline.py --only preprocess
```

### Resume the pipeline from a selected step

```bash
python run_pipeline.py --from cluster
```

Use the step names displayed by `--list`.

## Launching the Dashboard

Start the interactive Streamlit dashboard from the project root:

```bash
python -m streamlit run app/streamlit_app.py
```

The dashboard loads its styling from `app/theme.css` through `app/style.py`; both files are required. The India state map reads `data/boundaries/india_full.geojson`. The full upstream boundary distribution is kept locally as `india-geojson-assets.zip`; it is not required to run the dashboard.

**Important:** The processed Parquet files and dashboard tables are included in the repository, so the dashboard can run immediately after setup. Re-run `python run_pipeline.py` if you want to regenerate them from `data/raw/Global_Mobility_Report.csv`.

The dashboard provides:

- **Overview:** Research question, headline metrics, workflow and interpretation.
- **Profiles:** Cluster overview, feature-space views, profile table, signatures and membership.
- **Geography:** National choropleth and India state map.
- **Validation:** Model-selection metrics, stability, metrics at the selected k and hold-out checks.
- **Timeline:** Behavioural states across pandemic periods and state transitions.
- **Compare:** Daily trajectories and profile values for selected regions.
- **Method:** Methodology, glossary and guidance for responsible reading.

The dashboard supports national, India state, India district, and global state scopes where corresponding processed results are available.

## Methodology

### 1. Data Loading

The raw CSV is read in chunks to handle the large dataset efficiently. Geographic levels are identified, and the observations are stored as compressed Parquet files.

### 2. Data Preprocessing

Preprocessing includes:

- Constructing daily date sequences.
- Handling internal missing-value gaps up to 14 days through interpolation.
- Filling eligible edge gaps up to 7 days.
- Excluding regions when a mobility feature exceeds the configured 30% missingness threshold.
- Standardising numerical features where required.

### 3. Feature Engineering

Regional mobility profiles are constructed using:

- First-wave mean mobility changes.
- Late-pandemic mean mobility changes.
- First-wave trough depth.
- Recovery duration.

The six mobility categories contribute to the period-based features. Additional recovery features describe the depth and duration of the initial mobility decline.

Recovery is measured using a composite of retail and recreation, transit stations, and workplace mobility, with a 7-day rolling average and a configured recovery threshold of -10 percentage points.

### 4. Outlier Detection

Extreme regional profiles are identified using a robust z-score approach based on the median and median absolute deviation.

The configured exclusion threshold is a robust z-score greater than 10.

Excluded regions are documented separately for transparency.

### 5. Unsupervised Clustering

The project investigates clustering configurations using:

- K-Means clustering.
- K-Medians clustering.
- Euclidean and Manhattan distance comparisons.
- Multiple cluster counts.
- Standard and robust scaling approaches.

The project also compares clustering results with hierarchical clustering.

### 6. Evaluation and Validation

Clustering configurations are examined using:

- Within-cluster sum of squares (WCSS).
- Silhouette score.
- Davies–Bouldin index.
- Subsample stability.
- Adjusted Rand Index (ARI).
- Hold-out prediction strength.

These measures help assess cluster compactness, separation, stability, and reproducibility.

### 7. Temporal Analysis

Temporal analysis examines how regional mobility profiles and cluster distributions change across different pandemic periods.

## Configuration

The `config.yaml` file contains the main project settings, including:

- Mobility feature definitions.
- Study periods.
- Missing-data thresholds.
- Interpolation and edge-filling limits.
- Clustering search ranges.
- Distance metrics.
- Number of initialisations.
- Bootstrap runs.
- Random seed.

The pipeline also defines selected cluster counts for its scope-specific evaluation stages.

## Tests

Run the project tests using:

```bash
python -m pytest
```

The current processed-data test suite contains 21 tests.

## Outputs

The project generates results under:

- `outputs/figures/` — clustering, evaluation, and temporal visualisations.
- `outputs/tables/` — cluster assignments, profiles, validation metrics, and regional analysis.
- `reports/` — documented preprocessing and analytical decisions.

## Limitations

- Google mobility data represents changes relative to historical baselines, not absolute movement or population counts.
- Coverage varies across geographic regions.
- Regions with insufficient data are excluded by the configured missingness threshold.
- The discovered clusters are exploratory profiles rather than definitive natural categories.
- Some validation results indicate moderate cluster reproducibility.
- The analysis describes associations and patterns; it does not establish causal relationships.
- Results are limited to the available dataset and selected study period.

## Research Interpretation

The objective is to understand how mobility patterns differed across regions during the pandemic and to identify groups with similar responses and recovery trajectories.

The resulting clusters should be interpreted as data-driven summaries that support regional comparison and further investigation, rather than as fixed classifications of regions.

---

**Project:** Regional Mobility Change Profiles
**Domain:** Unsupervised Machine Learning | Exploratory Data Analysis | Mobility Analytics
**Language:** Python
