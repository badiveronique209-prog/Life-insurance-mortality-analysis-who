# Life Insurance Mortality Analysis - WHO Dataset

What drives life expectancy across 193 countries?

This project analyzes the WHO Life Expectancy dataset (2000-2015) to understand which demographic, economic and health factors are associated with life expectancy differences between countries. It is an actuarial data exploration, not a pricing model.

It complements my non-life insurance project (Auto Insurance Pricing with GLM) to show both sides of an actuarial profile: Non-Life and Life.

### Dataset
WHO Global Health Observatory 2938 observations, 22 variables, 193 countries, 2000 to 2015.
Source: https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who

### What I found
- Developed countries: tight distribution between 79 and 82 years. Developing countries: much wider spread, some below 50 years.
- Correlation: schooling (r = +0.74) and income composition (r = +0.66) most positively associated. Adult mortality (r = -0.69) and HIV/AIDS (r = -0.55) most negatively associated.
- Multiple linear regression: R2 ~0.83, RMSE ~3 years. Coefficients interpreted conditionally with 95% CI and residual analysis.
- Classification (Developed vs Developing): logistic regression >90% accuracy.

### Methodology
The dataset is longitudinal: each country appears multiple times. A random split would put the same country in both train and test (e.g. France 2005 in train, France 2006 in test) and inflate performance.

I used GroupShuffleSplit grouped by Country so no country appears in both sets.

Pipeline to avoid leakage:
```python
Pipeline([
  ('imputer', SimpleImputer(strategy='median')),
  ('scaler', StandardScaler()),
  ('model', LogisticRegression(max_iter=1000))
])
