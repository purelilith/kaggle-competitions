# Playground Series S6E7 - Health Condition Classification

Solution for a multiclass classification task (predicting `health_condition`) using an ensemble of **LightGBM + CatBoost + XGBoost**.

**Public score: `0.94968`**

## Data

```
/kaggle/input/competitions/playground-series-s6e7/{train,test,sample_submission}.csv
```

Tabular data on user health and activity (sleep, heart rate, BMI, calories, steps, etc.), with `health_condition` as the target category.

## Pipeline

### 1. Feature engineering (numerical features)
- Derived metrics: `calories_per_step`, `calories_per_minute`, `steps_per_minute`, `total_activity_score`
- Missing value indicators (`*_was_missing`) for the key numerical features
- Missing values filled with the median (computed on train)
- Composite features: `healthy_score` (average of lifestyle columns), `sleep_score` (Gaussian of sleep duration x sleep quality), `calorie_score`, `steps_score`, `sport_score`
- Medical flags: `bradycardia`/`tachycardia` (from heart rate), `underweight`/`overweight` (from BMI)

### 2. Categorical features and binning
- Binning of `calorie_expenditure`, `water_intake`, `step_count` into categorical buckets
- Quantile bins (`KBinsDiscretizer`) for `sleep_duration` and `water_intake`
- Missing values filled with the `"Missing"` label

### 3. Encoding
- `OrdinalEncoder` — for LightGBM and XGBoost
- Raw string categories (no ordinal encoding) - for CatBoost, which handles them natively via `cat_features`

### 4. Training (5-fold `StratifiedKFold`)
Three models are trained on each fold, and their test-set predictions are averaged across folds:

| Model | Key parameters |
|---|---|
| **LightGBM** | `n_estimators=1500`, `lr=0.05`, `max_depth=6`, `class_weight='balanced'`, early stopping (50 rounds) |
| **CatBoost** | `iterations=1500`, `lr=0.075`, `depth=8`, `l2_leaf_reg=7.0`, `Bernoulli` bootstrap, native categorical handling |
| **XGBoost** | `n_estimators=1500`, `lr=0.05`, `max_depth=6`, `sample_weight='balanced'`, early stopping (50 rounds) |

### 5. Blending
Weighted average of class probabilities across all folds:

```python
final_probs = 0.45 * cat_probs + 0.45 * lgb_probs + 0.10 * xgb_probs
```

The final class is obtained via `argmax` over the averaged probabilities, then decoded back with `LabelEncoder`.

## Stack

`Python`, `LightGBM`, `CatBoost`, `XGBoost`, `scikit-learn`, `pandas`, `numpy`

## Result

| Metric | Value |
|---|---|
| Public score | **0.94968** |
