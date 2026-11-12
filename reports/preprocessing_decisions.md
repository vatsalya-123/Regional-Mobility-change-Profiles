# Preprocessing Decision Log

| # | Decision | Choice | Reason |

| D1 | Variables | Keep the 6 `*_percent_change_from_baseline` columns. Drop `census_fips_code` and `place_id`. | The 6 columns are the behavioural signal. The dropped ones are identifiers or US-only. Names and codes are used for labelling only, never as features. |

| D2 | Geographic levels | National, state, district and metro rows are kept in separate datasets and never mixed. | Mixing a country with a district in one clustering is not meaningful. |

| D3 | Scope | A: national (131 countries). B: India states (33) and districts (533). C: global states (958), supplementary. Metro areas excluded. | National data is the cleanest. India is the deep dive. Global states carry selection bias (see D8). |

| D4 | Missing values | Full daily grid. A region is dropped if any feature is over 30% missing. Linear interpolation for gaps up to 14 days. Edge fill up to 7 days. Remaining gaps handled by a 70% coverage rule in feature engineering. | Short gaps are interpolated safely. Long gaps are not invented. Dropped regions are listed in `outputs/tables/dropped_regions_*.csv`. |

| D5 | Data quirk | Namibia's country code "NA" was parsed as missing. Fixed with `keep_default_na=False`. | Found through a region-count mismatch (134 vs 135). Tests now guard against it. |

| D6 | Encoding | No categorical fields enter the clustering matrix. | All clustering features are numeric. Region names are identifiers. |

| D7 | Standardisation | StandardScaler vs RobustScaler, compared, applied to the profile vectors. | Features have different scales and outliers. |

| D8 | Known biases | Regions exceeding the 30% missingness threshold are excluded; 49% of global-state regions are dropped, and 8 countries lose all state-level coverage. | Clusters describe regions with sufficient Google Location History coverage, not the world. |

| D9 | Profile features | Per region: first-wave mean and late-pandemic mean for each of the 6 categories, plus trough depth and recovery days (14 features). | Captures the shock, the new normal, and the speed of return. |

| D10 | Recovery | Composite of retail, transit and workplaces, 7-day smoothed. Recovered = within 10 points of baseline after the first-wave trough. Never-recovered regions flagged as censored. | Grocery is essential travel, parks are seasonal, residential moves opposite. |

| D11 | Parks seasonality | Baseline is Jan to Feb 2020, so parks conflate season and pandemic. Run all clustering with and without parks. | Sensitivity test, reported in the stability section. |

| D12 | Min coverage | A region-period needs 70% of days present or it is dropped from the profile table. | Avoids profiles built on little data. |

| D13 | Clustering grid | K-Means (Euclidean) and K-Medians (Manhattan, own implementation), k=2..10, Standard vs Robust scaling, with vs without parks. | Tests sensitivity to distance, scaling and the seasonal parks feature. |

| D14 | Primary configuration | K-Means, Euclidean, k=3, RobustScaler, parks excluded. | Retained as an interpretable exploratory configuration with strong full-data subsample stability, while acknowledging moderate hold-out reproducibility and the stronger India-district hold-out result at k=2. Robust scaling reduces sensitivity to extreme values; parks are excluded because of baseline seasonality concerns. |

| D15 | Stability protocol | 50 random 80% subsamples per configuration, ARI against the full-data clustering. | Quantifies how reproducible the grouping is. |

| D16 | Interpretation limits | Silhouette about 0.3 indicates weak separation. Values are changes against a fixed baseline. Clusters are exploratory profiles, not categories. | Responsible-use requirement. |

| D17 | Hierarchical comparison | Ward, complete and average linkage (Euclidean), plus complete and average (Manhattan), cut at k=3, compared with K-Means via ARI, cophenetic correlation and silhouette. | Tests whether the grouping depends on the algorithm. Ward and complete agree with K-Means for 83 to 85% of countries. Average linkage is rejected because it chains (clusters of size 2 and 3). |

| D18 | Feature contribution | Kruskal-Wallis H with epsilon-squared effect sizes, reported descriptively. | Clusters were built from the same features, so p-values are circular and are not interpreted as significance. |

| D19 | Borderline regions | Flagged if membership confidence < 0.8 across 50 subsamples, or silhouette < 0.1. 12 of 126 countries flagged, all on the moderate/deep boundary. | Quantifies the soft boundary. Flagged regions are not assigned confident labels in interpretation. |

| D20 | Surrogate rules | Depth-limited decision tree reproducing the clusters, with 5-fold cross-validated accuracy (0.85). | Descriptive summary only, not a causal model. |

| D21 | Behavioural states | K-Means (k=4, RobustScaler) on region-period means of 5 categories (parks excluded), ordered by mean activity. | Pre-pandemic period lands 99% in the near-baseline state, which validates the states. k=4 chosen for interpretability (silhouette prefers k=2). |

| D22 | Transition analysis | Descriptive transition probabilities between consecutive periods (national: 882 observations). State 3 has no observed exits, based on about a dozen transitions. "Above baseline" is relative to a Jan-Feb 2020 reference. | Small counts, so no claim of permanence. |


| D23 | Outlier rule | Regions with a robust z-score above 10 on any profile feature are excluded from model fitting and listed in outputs/tables/excluded_outlier_regions.csv. | National max is 7.9 and India districts 9.0 (no exclusions). India states: Jammu and Kashmir scores 35.8 against 3.3 for the next state. Global states: J&K 31.1 and two Greek island regions at 17.1 and 12.3. Likely baseline or seasonal artifacts (untested). |

| D23a | India states outlier | Jammu and Kashmir shows extreme late-period mobility values (+614% retail, +882% grocery) and dominates the India-states profile. | Excluded from model fitting under the robust-z-score threshold in D23; documented in outputs/tables/outlier_diagnostics.csv. |

| D24 | India districts | k=3 results are stable (ARI 0.92 to 0.96) with silhouette 0.16 to 0.35. Retained districts are those with enough Location History data. | Weak separation and urban coverage bias are stated as limitations. |

| D25 | District geography check | The clusters are not given location, yet Delhi's 11 districts all fall in the deep-dip cluster and 54 of 71 Uttar Pradesh districts in the strong-rebound cluster. | Independent sanity check. No causal explanation claimed. |

| D26 | J&K evidence | Jammu and Kashmir shows retail +151% and transit +252% in Feb-Mar 2020 and transit +217% in Apr-Jun 2020 (lockdown), rising to +650% by 2022. | Implausible as behaviour; consistent with an anomalously low baseline. Excluded under rule D23. |

| D27 | India states | n=32 after exclusion. Silhouette 0.25 to 0.31, stability 0.62 to 0.74 at k=3 (0.96 to 0.98 at k=2). Reported as a supporting descriptive analysis. Districts (n=524) are the primary India result. | Small sample makes cluster structure fragile. |

| D28 | Hierarchical, India districts | Ward agrees with K-Means for 73% (ARI 0.39). The 179 mild-dip districts stay together in all 179 cases. Average linkage chains (clusters of 2). | One robust group plus a continuum, as at national level. |

| D29 | Temporal, India | States and districts show a winter-2020/21 lull and a Delta-wave relapse, unlike the national pattern. | Fixed global period windows blur country-specific wave timing (limitation). |

| D30 | Borderline rule | A region is flagged if membership confidence < 0.8 or silhouette < 0.1. India districts: 94 of 524 flagged. | The count depends on the rule when overall silhouette is low, so both components are reported. |

| D31 | Reproducibility | Full rebuild from preprocessing (python run_pipeline.py --from preprocess) took 12.1 min on an 8 GB laptop and passed all 21 tests. Rerunning changed only 9 files. | Fixed seeds make results reproducible. Changes traced to a documented code change (temporal analysis restricted to regions with complete profiles) and to the metrics table (checked). |

| D32 | Temporal scope | Temporal state analysis uses only regions present in the clustering profile table (national: 126 countries x 7 periods = 882 observations). | Keeps the temporal and static analyses on the same set of regions. |

| D33 | Numeric precision | Saved metrics are rounded to 6 decimals. | Reruns differed only in the 16th digit of WCSS (floating-point summation order), so rounding keeps diffs clean without affecting any reported value. |

| D34 | Hold-out validation | National: K=3 mean prediction strength approximately 0.64 and mean ARI 0.69. India districts: 100 random 70/30 splits (367 train, 157 test), using no_parks and RobustScaler. At K=2, mean prediction strength 0.776 and mean ARI 0.746. At K=3, mean prediction strength 0.660 and mean ARI 0.635. No tested K from 2 to 8 achieved mean prediction strength >= 0.8 for India districts. | Hold-out validation measures out-of-sample reproducibility and is distinct from full-data subsample stability. K=2 performs better than K=3 on hold-out validation. K=3 is retained as an interpretable exploratory configuration, not as a strongly generalisable or naturally occurring classification. |
