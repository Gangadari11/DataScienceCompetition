# DataScienceCompetition

Welcome to our solution for the **Customer Experience Prediction Challenge**! This includes our approach, data preprocessing techniques, modeling strategies, and key learnings.


## Our Approach

For this challenge, we selected **XGBoost** as our primary model. XGBoost's ability to handle both numerical and categorical data, coupled with its robust performance, made it a natural choice for the structured dataset we worked with.


## Data Preprocessing

We dedicated significant effort to understanding and preparing the data. Below are the key steps we followed:

### 1. Handling Missing Values
- **Numeric Columns:** Filled with the median.
- **Categorical Columns:** Filled with the mode.
```python
# Fill numeric columns with median
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())

# Fill categorical columns with mode
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])
```

### 2. Date Features
We engineered date-related features, breaking down date columns into components like year, month, day, and weekend indicators. Additionally, we calculated meaningful time differences:
```python
data[f'{col}_year'] = data[col].dt.year.fillna(-1).astype(int)
data[f'{col}_month'] = data[col].dt.month.fillna(-1).astype(int)
data[f'{col}_is_weekend'] = (data[col].dt.dayofweek >= 5).astype(int)
```

### 3. Feature Engineering
To enhance the dataset, we created new features:
- **Pricing Features:**
    - Discount amounts and percentages:
    ```python
    data['discount_amount'] = data['Product_value'] - data['final_payment']
    data['discount_percentage'] = (data['discount_amount'] / data['Product_value'] * 100).clip(0, 100)
    ```
- **Loyalty Program Features:**
    - Indicators for redeemed points and their value ratio:
    ```python
    data['has_redeemed_points'] = (data['loyalty_points_redeemed'] > 0).astype(int)
    data['points_to_value_ratio'] = data['loyalty_points_redeemed'] / (data['Product_value'] + 1)
    ```

### 4. Encoding Categorical Variables
We converted text-based categories into numeric representations using label encoding and one-hot encoding.


## Modeling

### 1. Model Selection
We utilized **XGBoost**, a gradient boosting algorithm, for its suitability to structured datasets.

### 2. Hyperparameter Tuning
We optimized the following hyperparameters through trial and error:
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

### 3. Cross-Validation
To validate the model's performance, we implemented 5-fold stratified cross-validation:
```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X_train, y_train):
    model = xgb.XGBClassifier(**params)
    model.fit(X_train[train_idx], y_train[train_idx],
              eval_set=[(X_train[val_idx], y_train[val_idx])],
              early_stopping_rounds=50, verbose=False)
```

### 4. PCA Features
We applied Principal Component Analysis (PCA) to capture latent patterns in the data:
```python
pca = PCA(n_components=10)
pca_features = pca.fit_transform(X_numeric_transformed)
for i in range(pca_features.shape[1]):
    X[f'pca_feature_{i}'] = pca_features[:, i]
```

### 5. Early Stopping
To prevent overfitting, we used early stopping based on validation loss during training.


## How We Evaluated It

Our primary evaluation metric was the **weighted F1 score** to account for imbalanced class distributions. We also:
- Reviewed the classification report to analyze predictions for each category.
- Used cross-validation to ensure stability across data splits.
```python
weighted_f1 = f1_score(y_val, y_pred, average='weighted')
```


## Biggest Challenges

1. **Date Handling:** Engineering meaningful features from date columns required careful thought.
2. **Categorical Variables:** Balancing effective encoding with model performance.
3. **Feature Selection:** Ensuring that all new features contributed positively to predictions.


## What We Learned

1. **Feature Engineering:** Time spent on creating meaningful features, especially around dates and loyalty programs, greatly improved model performance.
2. **Importance of Pricing Features:** Discount-related features were highly predictive.
3. **Cross-Validation:** Ensured that our model was generalizable.
4. **PCA Features:** Helped uncover patterns missed by manual feature creation.



```

