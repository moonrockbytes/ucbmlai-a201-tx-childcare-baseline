### Applying Machine Learning to Test Whether Texas Childcare Inspections Reflect Family Experience

**Rakesh Chandrasekaran**

---

#### Executive Summary

Texas licenses more than 13,000 childcare facilities statewide. Each facility is inspected by the Health and Human Services Commission (HHSC), producing a public record of deficiencies, corrective actions, and capacity information. Families choosing a childcare center also leave Google reviews that aggregate into a public star rating. This project asks whether those two signals agree.

Using 2,468 active licensed childcare centers across the five largest Texas cities — Austin, Dallas, Fort Worth, Houston, and San Antonio — a Ridge Regression baseline model was trained to predict Google star ratings from 15 HHSC regulatory features. The model explains only 3% of the variance in Google ratings (R² = 0.032, RMSE = 0.671 vs a naive baseline of 0.685). All 15 regulatory features show weak negative correlations with Google ratings (range: -0.22 to -0.02), and no linear structure is visible in any feature-vs-rating scatter plot.

The central finding is that Texas childcare inspection records are not reflected in family ratings: centers with high deficiency rates and frequent violations still commonly receive 4 to 5 star Google ratings. The planned Module 24 follow-up will compare this regulatory-feature model against a text-based model trained on 42,268 review texts, and will systematically identify centers where the two signals diverge most.

---

#### Rationale

Childcare quality is a critical public health and child development issue. Parents making childcare decisions must rely on informal word-of-mouth, online reviews, or state inspection data — but it is unknown whether those signals are consistent with one another. If regulatory compliance records and family experience diverge, then either state inspections are missing quality dimensions that families notice, or family reviews are driven by factors unrelated to licensing standards. Understanding this gap has direct implications for how policymakers, parents, and regulators should use and interpret public childcare data.

---

#### Research Question

**Do Texas HHSC regulatory compliance records predict the Google star ratings that families assign to licensed childcare centers in the five largest Texas cities?**

If inspection outcomes (deficiency counts, deficiency rates, corrective actions) systematically predict lower Google ratings, regulatory data is a useful quality proxy for families. If they do not, a second question follows: can the review text itself — the words families use — predict ratings more accurately than the inspection record?

---

#### Data Sources

**Texas HHSC Licensed Childcare Operations** (`hhsc_data.csv`)
- Source: Texas Health and Human Services Commission public data portal
- 13,000+ statewide licensed childcare operations (all types, all status)
- Filtered to: 5 target cities, active centers (OPERATION_STATUS = Y), childcare operation types only
- Fields used: deficiency counts by severity tier, total inspections, total assessments, total capacity, years in operation, accreditation status, subsidy acceptance, corrective action flags

**Google Places API** (fetched via `places/searchText` and `place/details`)
- Matched each HHSC center to its Google Places listing using fuzzy name matching (token_sort_ratio) and geographic proximity
- 2,468 centers matched at high or medium confidence with at least one Google review
- Fields used: Google star rating (target variable), up to 5 review texts per center (Model 2 input)
- 42,268 reviews total; median 77 words per review

---

#### Methodology

**Data Collection and Matching (Sections 1–5)**
- HHSC CSV downloaded and filtered to active childcare centers in Austin, Dallas, Fort Worth, Houston, and San Antonio
- Each center matched to its Google Places listing using a two-stage process: text search by name and city, then fuzzy name matching to classify each match as high-confidence, medium-confidence, or low-confidence
- High- and medium-confidence matches with at least one Google review are retained for modelling (2,468 centers)

**Feature Engineering (Section 6)**
- 15 regulatory features engineered from raw HHSC fields: 6 deficiency count tiers (HIGH, MEDIUM_HIGH, MEDIUM, MEDIUM_LOW, LOW, TOTAL), 2 rate features (DEFICIENCY_RATE, HIGH_DEFICIENCY_RATE), 2 scale features (TOTAL_CAPACITY, TOTAL_INSPECTIONS), and 3 binary features (IS_ACCREDITED, ACCEPTS_SUBSIDIES, HAS_CORRECTIVE_ACTION), plus TOTAL_ASSESSMENTS and YEARS_IN_OPERATION
- Two candidate features (HAS_ADVERSE_ACTION, WAS_TEMP_CLOSED) were zero-variance after filtering and excluded from modelling

**Data Cleaning (Section 7)**
- Missing values imputed using column medians (YEARS_IN_OPERATION: 24 missing values); ACCEPTS_SUBSIDIES non-Y/N values mapped to 0
- No duplicate center records found after Google Places deduplication in Section 5
- Outliers retained — all extreme values represent legitimate operational variation; Ridge Regression is robust to outliers through its L2 penalty

**Exploratory Data Analysis (Section 8)**
- Target distribution: mean 4.47, median 4.60, std 0.573; 61.5% of centers rated 4.5–5.0; strong positivity bias consistent across all five cities
- Feature correlations: all 15 features weakly negatively correlated with Google rating (strongest: TOTAL_INSPECTIONS at -0.22)
- Deficiency tier features are highly multicollinear (pairwise correlations 0.63 to 0.95)
- No linear structure visible in any of the 12 continuous feature vs rating scatter plots
- Outlier analysis: DEFICIENCY_LOW has the highest outlier count (332); YEARS_IN_OPERATION has zero IQR outliers despite a 40-year range

**Baseline Model (Section 9)**
- **Evaluation Metric:** RMSE — chosen because it is in the same units as the target (star rating points), making results directly interpretable; R² is structurally suppressed by the compressed target range and is not the primary metric
- **Naive Baseline:** DummyRegressor (predict mean) — RMSE 0.6852, establishes the performance floor
- **Model Comparison:** Linear Regression, Ridge, Lasso, and ElasticNet compared with GridSearchCV on 5-fold cross-validation; Ridge selected for marginal RMSE advantage, stability under multicollinearity, and full coefficient retention for SHAP analysis in Module 24
- **Polynomial Analysis:** Degree 2 (RMSE 0.6782) and degree 3 (RMSE 0.7423) both worse than linear (RMSE 0.6705) — linear features confirmed as sufficient
- **Ridge Baseline:** RMSE 0.6707, MAE 0.4509, R² 0.0322, best alpha = 100 (GridSearchCV)

---

#### Results

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Naive baseline (predict mean) | 0.6852 | 0.4759 | -0.0102 |
| Ridge Regression (GridSearchCV, alpha = 100) | 0.6707 | 0.4509 | 0.0322 |

Ridge Regression improves over the naive baseline by 2.1%, explaining approximately 3% of the variance in Google star ratings. This marginal gain is consistent with the EDA finding that all 15 regulatory features have weak correlations with the target.

Predictions are compressed into a narrow band (approximately 4.2 to 4.8) regardless of actual rating, reflecting the positivity bias in the training data. Centers with actual ratings of 1.0 to 3.5 are predicted as 4.2 to 4.6, resulting in large errors for the minority of poorly-rated centers.

The largest model coefficients are city dummies (CITY DALLAS: -0.09, CITY SAN ANTONIO: -0.075), suggesting city-level effects account for more variation than any individual regulatory measure. ACCEPTS_SUBSIDIES (-0.065) is the strongest regulatory coefficient, likely a socioeconomic confound.

**The central finding is that Texas HHSC inspection records are not meaningful predictors of Google star ratings.** Families are not assigning ratings based on observable inspection outcomes.

---

#### Next Steps

The following work is planned for Module 24:

1. **Model 1 — Final:** XGBoost regressor with hyperparameter tuning and SHAP value analysis on the same 15 regulatory features to identify which features drive predictions and whether direction is consistent across cities.

2. **Model 2 — Review Text:** TF-IDF vectorisation of 42,268 review texts followed by Ridge or Lasso regression on the same Google star rating target. This tests whether the words families use are more predictive than inspection records.

3. **Direct Comparison:** Both models evaluated on the same 20% holdout test set using RMSE, MAE, and R², enabling a like-for-like comparison of regulatory data versus family language as quality signals.

4. **Divergence Analysis:** Centers where the two models disagree most will be systematically identified — including centers where regulatory records suggest high quality but family ratings are poor, and centers where ratings are high despite a weak compliance record. This analysis will surface the gap between formal compliance measurement and lived family experience.

---

#### Outline of Project

- [tx_childcare_eda_baseline.ipynb](tx_childcare_eda_baseline.ipynb) — Full EDA and baseline model notebook covering data collection (Sections 1–5), feature engineering (Section 6), data cleaning (Section 7), exploratory data analysis (Section 8), and Ridge Regression baseline (Section 9)

---

##### Contact and Further Information

Rakesh Chandrasekaran — talk2rakeshc@gmail.com
