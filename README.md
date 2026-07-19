# Steel Industry Energy Consumption — EDA, Feature Engineering & Baseline Regression Modeling

End-to-end machine learning workflow on real energy consumption data from a steel
manufacturing plant — covering exploratory data analysis, feature engineering, outlier
detection, correlation analysis, and baseline regression modeling to predict energy usage.

## Dataset

- **Source:** [UCI Machine Learning Repository — Steel Industry Energy Consumption Dataset](https://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption)
- **File:** `data/Steel_industry_data.csv`
- **Rows:** 35,040 (15-minute interval readings over one year)
- **Columns:** 11, including energy usage, reactive power, CO2 output, power factor, load
  type, and timestamp

## Project structure

```
├── data/
│   ├── Steel_industry_data.csv                    # Raw dataset
│   └── Steel_industry_data_cleaned.csv            # Cleaned dataset with engineered features (output of Part 1, input to Part 2)
├── steel_industry_energy_consumption_eda.ipynb   # Part 1: EDA & Feature Engineering
├── steel_industry_modeling.ipynb                 # Part 2: Baseline Regression Modeling
├── requirements.txt                               # Python dependencies
└── README.md
```

---

## Part 1 — Exploratory Data Analysis & Feature Engineering

1. **Load & inspect** — checks shape, data types, missing values, and duplicates. Dataset is
   fully clean: 0 missing values, 0 duplicate rows.
2. **Datetime feature extraction** — parses the timestamp and derives:
   - `hour` — hour of day (0–23)
   - `day_of_week_num` — day of week (0 = Monday)
   - `month` — month number
   - `is_weekend` — 1 if Saturday/Sunday, else 0
3. **`power_factor_ratio`** — Leading Current Power Factor ÷ Lagging Current Power Factor.
4. **`High_Load`** — binary flag marking rows where `Usage_kWh` is above the 75th percentile
   (1) or not (0).
5. **Outlier detection (IQR method)** on `Usage_kWh`, visualized with a boxplot.
   **328 outliers found (0.94% of rows)** — almost all high-usage spikes, not sensor errors.
6. **Correlation heatmap** — top 3 features correlated with `Usage_kWh`:
   - `CO2(tCO2)` (≈0.99)
   - `Lagging_Current_Reactive.Power_kVarh` (≈0.90)
   - `Lagging_Current_Power_Factor` (≈0.39)
7. **Average consumption by Load Type** — grouped bar chart (Light / Medium / Maximum load).
8. **Average usage by hour of day** — line chart showing the daily consumption rhythm.
9. **EDA summary & hypothesis** — written findings on data quality, correlations, patterns,
   and what likely drives energy spikes.
10. **Exported cleaned dataset** — `Steel_industry_data_cleaned.csv`, used directly by Part 2.

### Key findings (Part 1)

- No missing/duplicate data — the main cleanup consideration was statistical outliers, which
  represent genuine peak-production periods rather than errors.
- Energy usage correlates almost perfectly with CO2 output and reactive power draw.
- Usage follows a clear daily/shift-based pattern, with `Maximum_Load` periods consuming
  **~7x more energy** on average than `Light_Load` periods.
- **Hypothesis:** energy spikes are driven by the plant running at `Maximum_Load` during
  weekday daytime shifts, compounded by lower power factor efficiency at those times.

---

## Part 2 — Baseline Regression Modeling

1. **Load** the cleaned dataset (with all engineered features) exported from Part 1.
2. **Drop leakage/redundant columns:**
   - `date` — raw timestamp, already captured by `hour`/`month`/`is_weekend`
   - `High_Load` — directly derived from `Usage_kWh`, so it leaks the target
   - `WeekStatus` — redundant duplicate of `is_weekend`
3. **Encode categorical columns** — `Load_Type` and `Day_of_week` via **one-hot encoding**
   (chosen over label encoding, since both are nominal categories with no natural order —
   label encoding would falsely imply ranking/spacing, which especially hurts linear models).
4. **Train/test split** — 80% train / 20% test, `random_state=42` for reproducibility.
5. **Trained 4 models:** Linear Regression, Ridge Regression, Decision Tree Regressor,
   Random Forest Regressor.
6. **Test-set metrics** (MAE, RMSE, R²) calculated for each model.
7. **5-fold cross-validation** — mean RMSE reported per model for stability check.
8. **Bar chart** comparing test RMSE across all 4 models.
9. **Scatter plot** of Predicted vs Actual values for the best model.
10. **Model Selection write-up** — best model, overfitting signs, and final choice.

### Results

| Model | MAE | RMSE | R² | CV RMSE (mean) |
|---|---|---|---|---|
| Linear Regression | 2.63 | 4.15 | 0.985 | 4.51 |
| Ridge Regression | 4.36 | 6.27 | 0.965 | 6.23 |
| Decision Tree | 0.53 | 1.39 | 0.998 | 1.48 |
| **Random Forest** | **0.35** | **1.04** | **0.999** | **1.01** |

### Model Selection (Part 2)

**Random Forest** was selected as the best model — lowest test RMSE and MAE, highest R², and
a cross-validation RMSE (1.01) that closely matches its test RMSE (1.04), indicating stable,
generalizable performance rather than a lucky train/test split.

The **Decision Tree** showed clear overfitting: a near-zero training RMSE (perfect fit on
training data) paired with a noticeably higher test RMSE — a classic memorization pattern.
Random Forest's averaging across many trees keeps this in check.

**Linear** and **Ridge Regression** underfit the data — their RMSE was substantially higher
on both train and test sets, since a straight-line model can't capture the step-like,
regime-based usage patterns (Light/Medium/Maximum load) found in the EDA.

**Final model carried forward: Random Forest Regressor.**

---

## Tools used

Python, pandas, NumPy, Matplotlib, Seaborn, scikit-learn (Jupyter Notebook)

Install dependencies: `pip install -r requirements.txt`
