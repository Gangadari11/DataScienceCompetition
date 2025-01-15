# DataScienceCompetition

Hey everyone! We wanted to share how we approached this customer experience prediction challenge. Let we walk you through what we did:

### Our Approach
 We went with XGBoost because it's really good at handling both numerical and categorical data, which we had plenty of in this dataset.

### Data Preprocessing

First, We really dug into the data to understand what we're working with. Here's what I did:

*  Fixed missing values (used median for numbers and mode for categories)

```bash 
# Fill numeric columns with median
for col in numeric_cols:
    data[col] = data[col].fillna(data[col].median())
    
# Fill categorical columns with mode
for col in categorical_cols:
    data[col] = data[col].fillna(data[col].mode()[0])
```
* Worked a lot with dates - broke them down into year, month, day, etc.
* Created time differences between important dates

```bash
data[f'{col}_year'] = data[col].dt.year.fillna(-1).astype(int)
data[f'{col}_month'] = data[col].dt.month.fillna(-1).astype(int)
data[f'{col}_is_weekend'] = (data[col].dt.dayofweek >= 5).astype(int)
```

* Made new features around pricing (like discount amounts and percentages)

```bash
data['discount_amount'] = data['Product_value'] - data['final_payment']
data['discount_percentage'] = (data['discount_amount'] / data['Product_value'] * 100).clip(0, 100)
```

* Added some loyalty program features (like whether points were redeemed)

```bash
data['has_redeemed_points'] = (data['loyalty_points_redeemed'] > 0).astype(int)
data['points_to_value_ratio'] = data['loyalty_points_redeemed'] / (data['Product_value'] + 1)
```

* Converted all text categories into numbers the model could understand

### Modeling
For the actual model building:

* Used XGBoost with parameters I tuned through trial and error

```bash
params = {
    'n_estimators': 1000,
    'max_depth': 9,
    'learning_rate': 0.02,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'min_child_weight': 1,
    'gamma': 0.04
    # ... and more
}
```

* Did 5-fold cross-validation to make sure the model was stable

```bash
n_splits = 5
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
```

* Added PCA features to capture complex patterns

```bash
pca = PCA(n_components=10)
pca_features = pca.fit_transform(X_numeric_transformed)
```

* Used early stopping to prevent overfitting

### How We Evaluated It

* Mainly looked at the weighted F1 score since our classes weren't evenly distributed

```bash
weighted_f1 = f1_score(y_val, y_pred, average='weighted')

```

* Used cross-validation to make sure the model worked consistently
* Kept an eye on the classification report to see how well it predicted each category

### Biggest Challenges

* Dealing with dates was tricky - had to make sure all the time differences made sense
* Lots of categorical variables needed careful handling
* Had to balance model complexity with performance
* Making sure the features actually helped predict customer experience

### What We Learned
Looking back, here are our main takeaways:

* Feature engineering made a huge difference - especially the date-related features
* Price and loyalty program features were super important
* Cross-validation helped a lot in making sure the model was reliable
* The PCA features helped capture some patterns we might have missed otherwise

Let me know if you want me to explain any part in more detail!