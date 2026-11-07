# Preprocessing Decision Log

| # | Decision | Choice | Reason |

| D1 | Variables | Keep the 6 `*_percent_change_from_baseline` columns. Drop `census_fips_code` and `place_id`. | The 6 columns are the behavioural signal. The dropped ones are identifiers or US-only. Names and codes are used for labelling only, never as features. |

| D2 | Geographic levels | National, state, district and metro rows are kept in separate datasets and never mixed. | Mixing a country with a district in one clustering is not meaningful. |

| D3 | Scope | A: national (131 countries). B: India states (33) and districts (533). C: global states (958), supplementary. Metro areas excluded. | National data is the cleanest. India is the deep dive. Global states carry selection bias (see D8). |

| D4 | Missing values | Full daily grid. A region is dropped if any feature is over 30% missing. Linear interpolation for gaps up to 14 days. Edge fill up to 7 days. Remaining gaps handled by a 70% coverage rule in feature engineering. | Short gaps are interpolated safely. Long gaps are not invented. Dropped regions are listed in `outputs/tables/dropped_regions_*.csv`. |

| D5 | Data quirk | Namibia's country code "NA" was parsed as missing. Fixed with `keep_default_na=False`. | Found through a region-count mismatch (134 vs 135). Tests now guard against it. |

| D6 | Encoding | No categorical fields enter the clustering matrix. | All clustering features are numeric. Region names are identifiers. |

| D7 | Standardisation (planned) | StandardScaler vs RobustScaler, compared, applied to the profile vectors. | Features have different scales and outliers. |

| D8 | Known biases | Location History users only. Small or rural regions are withheld. 49% of global states dropped, and 8 countries lose all states. | Clusters describe covered regions, not the world. |

| D9  | Profile features | Per region: first-wave mean and late-pandemic mean for each of the 6 categories, plus trough depth and recovery days (14 features). | Captures the shock, the new normal, and the speed of return. |

| D10 | Recovery | Composite of retail, transit and workplaces, 7-day smoothed. Recovered = within 10 points of baseline after the first-wave trough. Never-recovered regions flagged as censored. | Grocery is essential travel, parks are seasonal, residential moves opposite. |

| D11 | Parks seasonality | Baseline is Jan to Feb 2020, so parks conflate season and pandemic. Run all clustering with and without parks. | Sensitivity test, reported in the stability section. |

| D12 | Min coverage | A region-period needs 70% of days present or it is dropped from the profile table. | Avoids profiles built on little data. |

