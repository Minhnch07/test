%md
# Section 2: Data Processing - Complete Study Guide

---

## Sub-topic 1: Compute Summary Statistics

### Key Methods Comparison

| Method | Returns | Statistics Included | Notes |
| --- | --- | --- | --- |
| `df.describe()` | Spark DataFrame | count, mean, stddev, min, max | String cols get count/min/max only |
| `df.summary()` | Spark DataFrame | count, mean, stddev, min, **25%**, **50%**, **75%**, max | Superset of describe(); accepts args |
| `dbutils.data.summarize(df)` | HTML (visual) | Types, nulls, distributions, stats | Interactive profile in cell output |

### Critical Distinctions

- **`df.summary()` vs `df.describe()`**: summary() adds the 25th, 50th, and 75th percentiles. You can also pass specific stats: `df.summary('count', 'mean', '50%')`
- **`dbutils.data.summarize(df)`**: Does NOT return a DataFrame. It renders an interactive HTML visualization directly in the notebook output. No arguments needed.
- **String columns in describe()**: They are NOT excluded. They show count, min, max (lexicographic order). Mean/stddev appear as null.

---

## Sub-topic 2: Remove Outliers

### IQR Method

```
IQR = Q3 - Q1
Lower Bound = Q1 - 1.5 * IQR
Upper Bound = Q3 + 1.5 * IQR
```

**PySpark quantile computation:**
```python
Q1, Q3 = df.approxQuantile('col_name', [0.25, 0.75], 0.01)
#                                        probabilities    relative error
```

> There is NO `df.quantile()` or `df.stat.percentile()` in PySpark.

### Standard Deviation Method

```
Lower Bound = mean - N * stddev
Upper Bound = mean + N * stddev
```

Common thresholds: 2 SD (~5% removed), 3 SD (~0.3% removed)

**PySpark filter:**
```python
df.filter((col('x') >= lower) & (col('x') <= upper))
# or equivalently:
df.where((col('x') >= lower) & (col('x') <= upper))
```

### IQR vs Standard Deviation - When to Use

| Property | IQR | Standard Deviation |
| --- | --- | --- |
| Robustness | More robust (percentiles ignore extremes) | Less robust (mean/std pulled by outliers) |
| Distribution assumption | None | Assumes approximate normality |
| Best for | Skewed data, unknown distributions | Roughly normal data |

---

## Sub-topic 3: Visualizations

### Choosing the Right Chart

| Data Type | Goal | Visualization |
| --- | --- | --- |
| Categorical | Show frequency/counts | **Bar chart** |
| Categorical | Show proportions (few classes) | Pie chart (only if 2-5 classes) |
| Continuous | Show distribution/skewness | **Histogram** or KDE plot |
| Continuous | Show central tendency + spread + outliers | **Box plot** |
| Continuous across groups | Compare distributions | Grouped box plot or violin plot |

### Databricks-Specific

- Use `display(df)` to get the interactive visualization panel (built-in charts)
- There is NO `df.plot.hist()`, NO `spark.plot()`, NO `dbutils.visualize()`
- For programmatic plots: use **matplotlib** or **seaborn** after `.toPandas()`

---

## Sub-topic 4: Comparing Features

### Two Categorical Features

| Method | Purpose | Output |
| --- | --- | --- |
| **Chi-squared test** | Test if association exists (significant or not) | p-value |
| **Cramer's V** | Measure association strength (normalized) | 0 to 1 |

### Two Continuous Features

| Method | Purpose | Output |
| --- | --- | --- |
| **Pearson correlation** | Measure LINEAR relationship strength | -1 to +1 |
| **Scatter plot** | Visualize relationship (detect non-linear) | Visual |
| Spearman correlation | Measure MONOTONIC relationship | -1 to +1 |

### Common Traps

- **Pearson r = 0** means NO LINEAR relationship. There could still be a strong non-linear one.
- **Chi-squared** only tells you IF association exists. **Cramer's V** tells you HOW STRONG.
- **Never use Pearson/Spearman for categorical data.** Never use Chi-squared for continuous data.

---

## Sub-topic 5: Comparing Imputation Strategies

### Decision Matrix

| Feature Type | Distribution | Best Imputation | Why |
| --- | --- | --- | --- |
| Continuous | Normal/symmetric | **Mean** | Mean = median for symmetric data |
| Continuous | Skewed/outliers | **Median** | Robust to extreme values |
| Categorical | Any | **Mode** (most_frequent) | Mean/median undefined for categories |
| Binary (0/1) | Any | **Mode** | Treats as categorical |

### Key Concepts

- **Mean imputation reduces variance** (all imputed values = same number, pulling toward center)
- **Mean imputation distorts correlations** with other variables
- **Median is ROBUST to outliers** because percentiles are not affected by extreme values
- **Mode replaces ALL nulls with the single most frequent value** (no randomization)

### Quick Rule
> Skewed? Use **median**. Symmetric? Use **mean**. Categorical? Use **mode**.

---

## Sub-topic 6: Imputation Implementation

### scikit-learn: SimpleImputer

```python
from sklearn.impute import SimpleImputer

# Valid strategies: 'mean', 'median', 'most_frequent', 'constant'
imp = SimpleImputer(strategy='most_frequent')  # This is MODE
imp.fit_transform(X)
```

> The mode strategy is called `'most_frequent'` -- NOT `'mode'`!

### PySpark ML: Imputer

```python
from pyspark.ml.feature import Imputer

imputer = Imputer(inputCols=['age'], outputCols=['age_imputed'])
imputer.setStrategy('median')  # Valid: 'mean', 'median', 'mode'
model = imputer.fit(df)        # Learns the statistic
df_imputed = model.transform(df)  # Applies it
```

> **fit() computes the statistic. transform() fills the nulls.** This is the Estimator/Transformer pattern.

### Manual PySpark (without Imputer)

```python
from pyspark.sql.functions import median
med_val = df.select(median('col')).collect()[0][0]
df_filled = df.fillna(med_val, subset=['col'])
```

### Multiple Column Types (ColumnTransformer)

```python
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer([
    ('num', SimpleImputer(strategy='mean'), numeric_cols),
    ('cat', SimpleImputer(strategy='most_frequent'), categorical_cols)
])
```

> Spark ML Imputer handles ONLY numeric columns. Use ColumnTransformer in sklearn for mixed types.

---

## Sub-topic 7: One-Hot Encoding

### PySpark ML Pipeline (2 steps required)

```python
from pyspark.ml.feature import StringIndexer, OneHotEncoder

# Step 1: String -> Numeric index
indexer = StringIndexer(inputCol='color', outputCol='color_idx')
# Step 2: Index -> Binary vector
encoder = OneHotEncoder(inputCol='color_idx', outputCol='color_ohe')
```

> OneHotEncoder CANNOT accept strings directly. StringIndexer is mandatory first.

### Default Behavior Comparison

| Library | Default Drop | Output Dimensions (N categories) | Output Format |
| --- | --- | --- | --- |
| PySpark OneHotEncoder | `dropLast=True` | **N-1** | Sparse vector |
| sklearn OneHotEncoder | `drop=None` | **N** | Sparse matrix |
| pandas get_dummies | `drop_first=False` | **N** | Dense DataFrame |

### pandas

```python
pd.get_dummies(df, columns=['category'])  # One-hot encoding
df['category'].factorize()                 # Label encoding (NOT one-hot)
```

### Why dropLast=True?

Prevents the **dummy variable trap** (multicollinearity). If you have N binary features that sum to 1, one is perfectly predictable from the others. Linear models break with perfect multicollinearity.

---

## Sub-topic 8: When OHE is (Not) Appropriate

### Model-Encoding Matrix

| Model Type | Needs OHE? | Why |
| --- | --- | --- |
| Linear/Logistic Regression | **YES** | Interprets numbers as ordered magnitudes |
| KNN (Euclidean distance) | **YES** | Label encoding creates false distances |
| SVM | **YES** | Relies on numeric feature space |
| Decision Trees / Random Forest | **NO** | Splits on thresholds; handles labels natively |
| XGBoost / LightGBM | **NO** | Sparse OHE dilutes split effectiveness |
| Neural Networks | Optional | Works, but **embeddings** preferred for high-cardinality |

### When NOT to Use OHE

1. **High cardinality** (>100 categories): Creates extremely wide sparse vectors. Use target encoding, hashing, or embeddings instead.
2. **Ordinal features** (has natural order): Use ordinal/label encoding (0,1,2,3) to preserve the order. OHE destroys ordinality.
3. **Tree-based models**: OHE dilutes feature importance and requires more splits. Integer encoding is sufficient.

### The Label Encoding Trap for Linear Models

If `department` is nominal (HR=0, Engineering=1, Sales=2, Marketing=3):
- The model thinks Engineering > HR and Marketing - Sales = Sales - Engineering
- This creates **false ordinal assumptions** in any model that uses feature magnitudes

---

## Sub-topic 9: Log Transformation

### When to Apply Log Transform

| Scenario | Apply Log? |
| --- | --- |
| Right-skewed positive values (income, prices) | **YES** |
| Wide range spanning orders of magnitude | **YES** |
| Multiplicative relationships in data | **YES** |
| Data contains zeros | Use **log1p(x) = log(1+x)** |
| Data contains negatives | **NO** (log undefined) |
| Already symmetric/normal | **NO** (unnecessary) |
| Binary or categorical features | **NO** |

### Critical: Reversing the Transform

If you log-transform the **target variable** before training:
```
prediction_original_scale = exp(model_prediction)
```

> The model does NOT automatically reverse this. You must exponentiate predictions manually.

### log(x) vs log1p(x)

- `log(0)` = undefined (negative infinity)
- `log1p(0)` = log(1) = 0 (safe!)

> Use **log1p** whenever your data contains zeros (counts, quantities).

### Which Models Benefit?

| Model | Benefit from Log Transform? | Why |
| --- | --- | --- |
| Linear Regression | **HIGH** | Assumes normal residuals; log helps symmetry |
| Logistic Regression | **HIGH** | Linear decision boundary benefits from symmetric features |
| KNN | **HIGH** | Distance calculations affected by scale |
| **Decision Trees** | **NONE** | Invariant to monotonic transforms (same splits) |
| Random Forest / GBT | **NONE** | Same reason as decision trees |

> **Key insight**: Tree-based models are invariant to monotonic transformations because they only care about relative ordering, not absolute values.

---

## Quick Reference: Common Exam Traps

1. `df.summary()` includes percentiles; `df.describe()` does not
2. `dbutils.data.summarize()` returns HTML, not a DataFrame
3. `df.approxQuantile()` is the ONLY way to compute percentiles in PySpark DataFrame API
4. IQR is MORE robust than standard deviation (not less)
5. Mean imputation REDUCES variance (does not increase it)
6. Mode in sklearn = `strategy='most_frequent'` (not 'mode')
7. PySpark OHE requires StringIndexer first
8. PySpark OHE default: N-1 dimensions (dropLast=True)
9. sklearn OHE default: N dimensions (drop=None)
10. Log transform gives NO benefit to tree-based models
11. Pearson r = 0 means no LINEAR relationship (non-linear may exist)
12. For nominal categories in linear models: MUST use OHE (label encoding implies false order)
