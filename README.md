# Social Media User Behavior — ML Pipeline

A machine learning project that predicts **sleep disruption level** based on social media behavioral and demographic data using four classification models — Logistic Regression, Random Forest, Gradient Boosting, and XGBoost.

---

## Authors

| Name | Roll Number |
|------|-------------|
| *Dhruv* | *245UAI037* |
---

## Problem Statement

Given a set of behavioral and demographic attributes reported by a social media user, predict the most likely **sleep disruption level** from 4 possible categories. This is a **multi-class classification problem** with 33 input features (numeric + categorical) and 4 target classes.

---

## Dataset

| Property | Detail |
|----------|--------|
| **File** | `social_media_user_behavior.csv` |
| **Source** | Kaggle — Social Media User Behavior Dataset |
| **Rows** | 2,000 |
| **Features** | 33 (behavioral, demographic, engagement, platform) |
| **Target** | `sleep_disruption` — 4 classes |
| **Missing Values** | None |

**Target Classes:**

| Class | Label | Count |
|-------|-------|-------|
| 0 | No impact | 588 |
| 1 | Mild impact | 705 |
| 2 | Moderate impact | 487 |
| 3 | Severe impact | 220 |

---

## Exploratory Data Analysis

The following EDA plots were generated:

- **Target Distribution** — Confirmed mild-to-moderate class imbalance; Severe impact is the least frequent category
- **Daily Usage Hours by Sleep Disruption** — KDE curves showing users with severe disruption tend to use social media significantly more hours per day
- **Mental Health Score vs Daily Usage** — Scatter plot revealing inverse correlation between mental health score and usage hours, colored by disruption class
- **Correlation Heatmap** — Numeric feature correlations highlighting strong relationships between `daily_usage_hours`, `sessions_per_day`, and `avg_session_duration_min`
- **Categorical Distributions** — Count plots for primary platform, preferred device, and gender across the dataset

---

## Preprocessing

- Dropped `user_id` (non-informative identifier)
- Engineered `account_age_years` from `account_join_date` (days since 2025-01-01 / 365), then dropped the original date column
- Separated features (X) and target (y)
- Encoded target labels using `LabelEncoder` with a fixed class order
- Applied `StandardScaler` to all numeric features
- Applied `OneHotEncoder(handle_unknown="ignore")` to all categorical features
- Both transformations wrapped in a `ColumnTransformer` inside an `sklearn Pipeline` to prevent data leakage
- Used an 80/20 stratified train/test split (`random_state=42`)

---

## Models Used

| Model | Description |
|-------|-------------|
| Logistic Regression | Linear baseline with L2 regularization (`C=1.0`, `max_iter=1000`) |
| Random Forest | Ensemble of 200 decision trees (`max_depth=12`, `n_jobs=-1`) |
| Gradient Boosting | Sequential boosting with 200 estimators (`learning_rate=0.1`, `max_depth=5`) |
| XGBoost | Extreme gradient boosting with 200 estimators (`learning_rate=0.1`, `max_depth=6`) |

---

## Results

| Model | CV Accuracy | Test Accuracy | Test F1 (weighted) |
|-------|-------------|--------------|-------------------|
| Logistic Regression | 29.8% ± 2.1% | 31.3% | 0.289 |
| Random Forest | 34.8% ± 1.6% | **36.0%** | 0.278 |
| Gradient Boosting | 32.1% ± 1.5% | 30.3% | 0.289 |
| XGBoost | 33.1% ± 2.9% | 32.0% | **0.303** |

- Random Forest achieved the highest test accuracy at 36.0%
- XGBoost achieved the highest weighted F1 score at 0.303
- All models consistently outperform the random baseline of ~25% (1 in 4 classes)
- The modest absolute scores reflect the inherently subjective and multifactorial nature of self-reported sleep disruption — a well-known challenge in behavioral health prediction

---

## Project Structure

```
Mental Health Prediction Model/
|
|-- README.md                         # Project documentation
|-- .gitignore                        # Git ignore rules
|-- social_media_ml.ipynb             # Main Jupyter notebook
|-- social_media_user_behavior.csv    # Raw dataset
|-- best_model.pkl                    # Serialized best model
|-- label_encoder.pkl                 # Fitted target LabelEncoder
|-- model_metrics.json                # CV and test scores for all models
|-- categorical_distributions.png     # Categorical feature distribution plots
|-- confusion_matrix.png              # Best model confusion matrix
|-- correlation_heatmap.png           # Numeric feature correlation heatmap
|-- feature_importances.png           # Feature importance plot
|-- mental_health_scatter.png         # Mental health score vs usage plot
|-- model_comparison.png              # Model performance comparison plot
|-- target_distribution.png           # Target class distribution plot
`-- usage_by_disruption.png           # Daily usage by disruption class plot
```

---

## Libraries Used

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- joblib
- warnings

Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib
```

---

## How to Run

1. Clone or download this repository
2. Place `social_media_user_behavior.csv` in the same folder as the notebook
3. Open `social_media_ml.ipynb` in Jupyter Notebook or JupyterLab
4. Run all cells sequentially from top to bottom

---

## Key Observations

- The dataset has a mild class imbalance — Severe impact accounts for only 11% of samples
- `daily_usage_hours`, `sessions_per_day`, and `self_reported_mental_health_score` are among the most predictive features
- Users with severe sleep disruption show higher daily usage hours and lower mental health scores on average
- Random Forest is recommended for real-world deployment due to robustness against noise and unseen feature combinations
- XGBoost is recommended when weighted F1 is the primary evaluation criterion

---

## References

- Kaggle Dataset: https://www.kaggle.com/datasets/hamnamunir/social-media-user-behavior-dataset?select=social_media_user_behavior.csv
- Scikit-learn Documentation: https://scikit-learn.org
- XGBoost Documentation: https://xgboost.readthedocs.io
- Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.
