### Applying Machine Learning to Test Whether Texas Childcare Inspections Reflect Family Experience

**Rakesh Chandrasekaran**

---

#### Executive Summary

Texas licenses more than 15,000 childcare facilities statewide. Each facility is inspected by the Health and Human Services Commission (HHSC), producing a public record of deficiencies, corrective actions, and capacity information. Families choosing a childcare center also leave Google reviews that aggregate into a public star rating. This project asks whether those two signals agree.

Using 2,468 active licensed childcare centers across the five largest Texas cities — Austin, Dallas, Fort Worth, Houston, and San Antonio — a Ridge Regression baseline model was trained to predict Google star ratings from 15 HHSC regulatory features. The model explains only 3% of the variance in Google ratings (R² = 0.032, RMSE = 0.671 vs a naive baseline of 0.685). All 15 regulatory features show weak negative correlations with Google ratings (range: -0.22 to -0.02), and no linear structure is visible in any feature-vs-rating scatter plot.

The central finding is that Texas childcare inspection records are not reflected in family ratings: centers with high deficiency rates and frequent violations still commonly receive 4 to 5 star Google ratings. The planned Module 24 follow-up will compare this regulatory-feature model against a text-based model trained on 42,268 review texts, and will systematically identify centers where the two signals diverge most.

---

#### Rationale

Childcare quality is a critical public health and child development issue. Parents making childcare decisions must rely on informal word-of-mouth, online reviews, or state inspection data — but it is unknown whether those signals are consistent with one another. If regulatory compliance records and family experience diverge, then either state inspections are missing quality dimensions that families notice, or family reviews are driven by factors unrelated to licensing standards. Understanding this gap has direct implications for how policymakers, parents, and regulators should use and interpret public childcare data.

---

#### Research Question

**Do Texas HHSC regulatory compliance records predict the Google star ratings that families assign to licensed childcare centers in the five largest Texas cities?**

---

#### Data Sources

**Source 1 — Texas HHSC Licensed Childcare Operations** (`hhsc_data.csv`)
- Direct download: `https://data.texas.gov/views/bc5r-88dy/rows.csv?accessType=DOWNLOAD`
- 15,141 statewide licensed childcare operations (all types, all status)
- Filtered to active licensed centers across Austin, Houston, San Antonio, Dallas, and Fort Worth
- Features used: inspection violation counts by severity level, total licensed capacity, total inspections conducted, years in continuous operation, accreditation status, subsidy acceptance status, and flags for corrective action, adverse action, and temporary closure
- These features serve as the inputs to Model 1

**Source 2 — Google Places API**
- Each HHSC center matched to its Google Places listing using the `findplacefromtext` endpoint by name and city; details fetched via the `place/details` endpoint
- 2,468 centers matched at high or medium confidence with at least one Google review
- Google star rating (1–5 scale) serves as the labeled output for both models; review text serves as the input to Model 2
- 42,268 reviews collected; median 77 words per review

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
- Missing values imputed using column medians (YEARS_IN_OPERATION: 24 missing values); centers with unknown subsidy status treated as not accepting subsidies
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

**EDA Findings**
- All 15 regulatory features show weak negative correlations with Google rating (range: -0.22 to -0.02) — no single inspection measure is a meaningful predictor of family ratings
- No linear structure is visible in any of the 12 continuous feature vs rating scatter plots across all five cities
- Google ratings are heavily concentrated between 4.5 and 5.0 (61.5% of centers), with a standard deviation of 0.573 — this positivity bias structurally limits how much any model can explain

**Baseline Model Performance**

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

2. **Model 2 — Review Text:** TF-IDF vectorisation of 42,268 review texts followed by Ridge Regression on the same Google star rating target. This tests whether the words families use are more predictive than inspection records.

3. **Direct Comparison:** Both models evaluated on the same 20% holdout test set using RMSE, MAE, and R², enabling a like-for-like comparison of regulatory data versus family language as quality signals.

4. **Divergence Analysis:** Centers where the two models disagree most will be systematically identified — including centers where regulatory records suggest high quality but family ratings are poor, and centers where ratings are high despite a weak compliance record. This analysis will surface the gap between formal compliance measurement and lived family experience.

---

#### Outline of Project

- [tx_childcare_eda_baseline.ipynb](tx_childcare_eda_baseline.ipynb) — Full EDA and baseline model notebook covering data collection (Sections 1–5), feature engineering (Section 6), data cleaning (Section 7), exploratory data analysis (Section 8), and Ridge Regression baseline (Section 9)

##### Notebook Structure

```
0.  Setup
1.  Download HHSC Data
2.  Google Places Matching
3.  Match Quality Validation
4.  Fetch Ratings and Reviews
5.  Coverage Assessment
6.  Feature Engineering
7.  Data Overview and Cleaning
8.  Exploratory Data Analysis
    8.1  Target Variable — Google Star Rating
    8.2  Regulatory Feature Distributions
    8.3  Feature Correlations with Google Rating
    8.4  Scatter Plots — Regulatory Features vs Google Rating
    8.5  Outlier Analysis
    8.6  Review Text Overview (Model 2 Input Data)
    8.7  EDA Summary and Interpretation
9.  Baseline Model (Model 1)
    9.1  Prepare Model Data
    9.2  Naive Baseline
    9.3  Model Comparison with GridSearchCV
    9.4  Polynomial Analysis
    9.5  Ridge Regression Diagnostics
         9.5.1  Performance Summary
         9.5.2  Predicted vs Actual Plot
         9.5.3  Feature Coefficient Plot
10. Export Datasets
11. Summary and Next Steps
```

##### Project Structure

```
ucbmlai-a201-tx-childcare-baseline/
├── tx_childcare_eda_baseline.ipynb   # Main notebook
├── README.md
├── .gitignore
├── images/                           # All plots saved by the notebook
│   ├── coverage_assessment.png
│   ├── match_quality_validation.png
│   ├── eda_8_1_target_distribution.png
│   ├── eda_8_2_deficiency_counts.png
│   ├── eda_8_2_rate_scale_features.png
│   ├── eda_8_2_binary_features.png
│   ├── eda_8_3_correlation_heatmap.png
│   ├── eda_8_4_scatter_deficiency_counts.png
│   ├── eda_8_4_scatter_rate_scale.png
│   ├── eda_8_5_outlier_analysis.png
│   ├── eda_8_6_review_text_overview.png
│   ├── model_9_3_comparison_rmse.png
│   ├── model_9_4_polynomial_analysis.png
│   ├── model_9_5_2_predicted_vs_actual.png
│   └── model_9_5_3_feature_coefficients.png
├── data/
│   ├── raw/                          # Not tracked — downloaded at runtime
│   │   ├── hhsc_data.csv
│   │   ├── place_ids.json
│   │   └── place_details.json
│   └── processed/                    # Not tracked — generated by notebook
│       ├── df_matched_rated.pkl
│       ├── df_cleaned.pkl
│       ├── model1_dataset.csv
│       └── model2_reviews.csv
```

---

#### Setup Instructions

**Prerequisites:** Python 3.9+, Jupyter Notebook or JupyterLab, and a Google Places API key.

**1. Clone the repository**
```bash
git clone https://github.com/moonrockbytes/ucbmlai-a201-tx-childcare-baseline.git
cd ucbmlai-a201-tx-childcare-baseline
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn requests rapidfuzz scikit-learn python-dotenv
```

**3. Create a `.env` file in the project root**
```
GOOGLE_API_KEY=your_google_places_api_key_here
CAPSTONE_BASE_DIR=/path/to/ucbmlai-a201-tx-childcare-baseline
```

**4. Run the notebook**

- To run the full pipeline from scratch, execute all cells from Section 0 onwards. Sections 1–3 will call the Google Places API and HHSC data portal.
- To skip data extraction and resume from saved data, run Section 0 first, then run the reload checkpoint cell at the top of Section 6. This loads `df_matched_rated.pkl` from `data/processed/`.

---

##### Contact and Further Information

Rakesh Chandrasekaran — talk2rakeshc@gmail.com
