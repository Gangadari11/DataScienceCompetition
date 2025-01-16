# DataScienceCompetition

This repository contains our solution to the customer experience prediction challenge. Below, we share our approach, preprocessing steps, modeling process, challenges, and key learnings.

---

## Our Approach

We chose **XGBoost** as our primary model because of its ability to handle both numerical and categorical data effectively. It’s robust, efficient, and has performed well on similar datasets in the past.

---

## Data Preprocessing

We spent significant time understanding and preparing the data to ensure the model could learn effectively. Key steps included:

### Handling Missing Values  
We filled missing values with the median for numerical columns and the mode for categorical columns.

```python
# Fill numeric columns with median
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

# Fill categorical columns with mode
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])

### Feature Engineering with Dates  
We broke down date columns into multiple components and created new features like time differences and weekend indicators.

```python
data[f'{col}_year'] = data[col].dt.year.fillna(-1).astype(int)
data[f'{col}_month'] = data[col].dt.month.fillna(-1).astype(int)
data[f'{col}_is_weekend'] = (data[col].dt.dayofweek >= 5).astype(int)
```

### Derived Pricing Features  
We calculated discount amounts and percentages to capture customer behavior related to pricing.

```python
data['discount_amount'] = data['Product_value'] - data['final_payment']
data['discount_percentage'] = (data['discount_amount'] / data['Product_value'] * 100).clip(0, 100)
```

### Loyalty Program Features  
We included features like `has_redeemed_points` and `points_to_value_ratio` to reflect the influence of loyalty programs.

```python
data['has_redeemed_points'] = (data['loyalty_points_redeemed'] > 0).astype(int)
data['points_to_value_ratio'] = data['loyalty_points_redeemed'] / (data['Product_value'] + 1)
```

### Encoding Categorical Variables  
All text-based categorical variables were converted into numerical representations for the model.

---

## Modeling

### XGBoost with Tuned Parameters  
We optimized hyperparameters through trial and error to maximize performance.

```python
params = {
    'n_estimators': 1000,
    'max_depth': 9,
    'learning_rate': 0.02,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'min_child_weight': 1,
    'gamma': 0.04
}
```

### Cross-Validation  
We used 5-fold cross-validation with stratified sampling to ensure model stability.

```python
from sklearn.model_selection import StratifiedKFold

n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
```

### PCA Features  
We applied Principal Component Analysis (PCA) to numeric features to capture additional complex patterns.

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=10)
pca_features = pca.fit_transform(X_numeric_transformed)
```

### Early Stopping  
Early stopping was used during training to avoid overfitting and improve generalization.

---

## Evaluation

We chose the **weighted F1 score** as our primary metric to address class imbalance effectively.

```python
from sklearn.metrics import f1_score

weighted_f1 = f1_score(y_val, y_pred, average='weighted')
```

Additionally, we analyzed the **classification report** to evaluate performance across categories. Cross-validation results confirmed the reliability of our approach.

---

## Biggest Challenges

1. **Date Handling**  
   Breaking down dates and calculating meaningful time differences required careful attention.
2. **Categorical Data**  
   Managing numerous categorical variables and ensuring optimal encoding was challenging.
3. **Balancing Complexity**  
   Striking the right balance between model complexity and performance required extensive experimentation.
4. **Feature Importance**  
   Ensuring the engineered features added predictive value to the model.

---

## What We Learned

1. **Feature Engineering is Key**  
   Features derived from dates and loyalty programs significantly improved model performance.
2. **Pricing Insights Matter**  
   Pricing-related features like discounts and loyalty points were among the most impactful predictors.
3. **Validation is Crucial**  
   Cross-validation ensured our model was robust and generalized well to unseen data.
4. **Dimensionality Reduction Helps**  
   PCA uncovered patterns that were not immediately obvious, enhancing model accuracy.

---

