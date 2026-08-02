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

## Sub-topic 7: Encoding Categorical Features

### One-Hot Encoding (Nominal Data)

**PySpark ML Pipeline (3 steps required for complete pipeline):**
```python
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml import Pipeline

# Step 1: String -> Numeric index
indexer = StringIndexer(inputCol='color', outputCol='color_idx')

# Step 2: Index -> Binary vector
encoder = OneHotEncoder(inputCol='color_idx', outputCol='color_ohe')

# Step 3: Combine all features into single vector
assembler = VectorAssembler(
    inputCols=['color_ohe', 'age', 'salary'],
    outputCol='features'
)

pipeline = Pipeline(stages=[indexer, encoder, assembler])
```

> OneHotEncoder CANNOT accept strings directly. StringIndexer is mandatory first.

### Label Encoding (Ordinal Data)

**PySpark ML Pipeline (2 steps):**
```python
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline

# Step 1: String -> Integer (0, 1, 2, 3...)
indexer = StringIndexer(inputCol='education', outputCol='education_idx')

# Step 2: Assemble features
assembler = VectorAssembler(inputCols=['education_idx', 'age', 'salary'], 
                            outputCol='features')

pipeline = Pipeline(stages=[indexer, assembler])
```

**Output:** `education_idx` = 0 (High School), 1 (Bachelor), 2 (Master), 3 (PhD)

> NO OneHotEncoder step! Keep the integers directly to preserve order.

### Ordinal Encoding (Custom Order)

```python
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.sql.functions import when

# Option 1: Use StringIndexer with custom order
indexer = StringIndexer(
    inputCol='size',
    outputCol='size_idx',
    stringOrderType='alphabetAsc'  # Or 'frequencyDesc', 'frequencyAsc'
)

# Option 2: Manual mapping for precise control
df_encoded = df.withColumn('size_idx',
    when(df['size'] == 'Small', 0)
    .when(df['size'] == 'Medium', 1)
    .when(df['size'] == 'Large', 2)
    .otherwise(3)
)

assembler = VectorAssembler(inputCols=['size_idx', 'age'], outputCol='features')
```

> Use when you need to enforce a specific order (not alphabetical).

### Target Encoding (High-Cardinality Data)

```python
from pyspark.sql.functions import avg

# Step 1: Learn the target mean per category
target_means = df.groupBy('city').agg(avg('target').alias('city_target_mean'))

# Step 2: Join back to original data
df_encoded = df.join(target_means, on='city', how='left')

# Step 3: Replace category with the mean
df_encoded = df_encoded.drop('city').withColumnRenamed('city_target_mean', 'city_encoded')

# Step 4: Assemble features
from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=['city_encoded', 'age', 'salary'], 
                            outputCol='features')
```

> Typically manual (not a built-in Transformer). Use for >100 categories.

### Encoding Techniques Comparison

| Encoding | Pipeline Steps | Output Type | Dimensions | When to Use |
| --- | --- | --- | --- | --- |
| **One-Hot** | StringIndexer → OneHotEncoder → VectorAssembler | Sparse binary vector | N-1 | Nominal categorical (no order) |
| **Label** | StringIndexer → VectorAssembler | Integer 0,1,2,... | 1 | Ordinal data (has natural order) |
| **Ordinal** | StringIndexer (custom order) → VectorAssembler | Integer with meaning | 1 | Ordinal with specific order |
| **Target** | Manual join/groupby → VectorAssembler | Numeric (mean/median) | 1 | High-cardinality (>100 categories) |

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

## Sub-topic 9: Data Transformations (Log & Scaling)

### Log Transformation

#### When to Apply Log Transform

| Scenario | Apply Log? |
| --- | --- |
| Right-skewed positive values (income, prices) | **YES** |
| Wide range spanning orders of magnitude | **YES** |
| Multiplicative relationships in data | **YES** |
| Data contains zeros | Use **log1p(x) = log(1+x)** |
| Data contains negatives | **NO** (log undefined) |
| Already symmetric/normal | **NO** (unnecessary) |
| Binary or categorical features | **NO** |

#### Critical: Reversing the Transform

If you log-transform the **target variable** before training:
```
prediction_original_scale = exp(model_prediction)
```

> The model does NOT automatically reverse this. You must exponentiate predictions manually.

#### log(x) vs log1p(x)

- `log(0)` = undefined (negative infinity)
- `log1p(0)` = log(1) = 0 (safe!)

> Use **log1p** whenever your data contains zeros (counts, quantities).

#### Which Models Benefit from Log Transform?

| Model | Benefit? | Why |
| --- | --- | --- |
| Linear Regression | **HIGH** | Assumes normal residuals; log helps symmetry |
| Logistic Regression | **HIGH** | Linear decision boundary benefits from symmetric features |
| KNN | **HIGH** | Distance calculations affected by scale |
| **Decision Trees** | **NONE** | Invariant to monotonic transforms (same splits) |
| Random Forest / GBT | **NONE** | Same reason as decision trees |

> **Key insight**: Tree-based models are invariant to monotonic transformations because they only care about relative ordering, not absolute values.

---

### Feature Scaling/Standardization

#### Why Scaling Matters

Scaling is **critical** for models that use distance or gradient descent:

| Model Type | Needs Scaling? | Why |
| --- | --- | --- |
| Linear/Logistic Regression | **YES** | Gradient descent converges faster; coefficient magnitude matters |
| KNN (Euclidean distance) | **YES** | Features with larger ranges dominate distance calculations |
| SVM | **YES** | Distance-based; large-range features get too much weight |
| K-means | **YES** | Clusters based on Euclidean distance |
| Neural Networks | **YES** | Gradient descent needs normalized inputs |
| **Decision Trees** | **NO** | Splits on thresholds; scale-invariant |
| **Random Forest** | **NO** | Same reason as trees |
| **XGBoost/LightGBM** | **NO** | Tree-based |

#### Common Scaling Techniques

| Technique | Formula | Range | When to Use | Robustness |
| --- | --- | --- | --- | --- |
| **StandardScaler** | (x - mean) / stddev | ~[-3, 3] | **Most common**; assumes normal distribution | Sensitive to outliers |
| **MinMaxScaler** | (x - min) / (max - min) | [0, 1] | Bounded range needed; preserve zero | Very sensitive to outliers |
| **RobustScaler** | (x - median) / IQR | Variable | Has outliers; more robust than StandardScaler | Robust (uses percentiles) |
| **Normalizer** | x / \|\|x\|\| | [0, 1] | Text/sparse data | N/A |

#### PySpark ML Implementation

```python
from pyspark.ml.feature import StandardScaler, MinMaxScaler, RobustScaler, VectorAssembler

# First: Assemble features into a single vector
assembler = VectorAssembler(inputCols=['age', 'salary'], outputCol='features')
df_assembled = assembler.transform(df)

# StandardScaler (most common)
scaler = StandardScaler(inputCol='features', outputCol='scaled_features')
scaler_model = scaler.fit(df_assembled)
df_scaled = scaler_model.transform(df_assembled)

# MinMaxScaler
minmax = MinMaxScaler(inputCol='features', outputCol='minmax_features')
minmax_model = minmax.fit(df_assembled)
df_minmax = minmax_model.transform(df_assembled)

# RobustScaler
robust = RobustScaler(inputCol='features', outputCol='robust_features')
robust_model = robust.fit(df_assembled)
df_robust = robust_model.transform(df_assembled)
```

> **fit() learns min/max/mean/stddev. transform() applies scaling.** This is the Estimator/Transformer pattern.

#### scikit-learn Implementation

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.pipeline import Pipeline

# Standalone
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# In a pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipeline.fit(X_train, y_train)
```

#### Critical: Scaling Order in Pipeline

```
1. Fit scaler on TRAINING data only
2. Scale TRAINING data
3. Scale TEST/VALIDATION data using the SAME learned scaler

# ❌ WRONG:
scaler = StandardScaler()
X_all_scaled = scaler.fit_transform(X_train + X_test)  # Leaks test info!

# ✅ RIGHT:
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use SAME scaler
```

> **Data leakage**: If you fit the scaler on combined train+test data, test statistics influence training!

---

### Log Transform vs Feature Scaling (NOT the Same!)

| Aspect | Log Transformation | Feature Scaling |
| --- | --- | --- |
| **Purpose** | Fix distribution shape (skewed → symmetric) | Normalize numeric ranges to same scale |
| **When applied** | BEFORE scaling (during data exploration) | AFTER encoding, BEFORE model training |
| **What it affects** | Distribution shape | Magnitude/range only |
| **Example** | `log1p(price)` or `log(income)` | `(x - mean) / stddev` |
| **Output range** | Compressed but variable | Fixed: [0,1] or [-3,3] |
| **Reversible** | Yes: `exp(log_x)` | Yes: can invert scaling |
| **Helps tree-based models?** | NO (invariant) | NO (invariant) |

> Apply log transform FIRST to fix distribution, then apply scaling SECOND to normalize ranges.

---

## Complete ML Data Processing Pipeline

### Full Order with All Steps

```
1. ⭐ Summary Statistics (explore data)
   └─ df.describe(), df.summary(), dbutils.data.summarize()

2. ⭐ Remove Outliers (IQR or standard deviation methods)
   └─ df.approxQuantile(), df.filter()

3. ⭐ Imputation (fill nulls)
   └─ PySpark Imputer or sklearn SimpleImputer

4. ⭐ Log Transformation (if data is skewed/right-tailed)
   └─ log1p(x) for data with zeros
   └─ ONLY for right-skewed numeric columns
   └─ Skip for tree-based models

5. ⭐ Encoding (categorical → numeric)
   └─ StringIndexer + OneHotEncoder (OHE) for nominal
   └─ StringIndexer (2 steps) for ordinal
   └─ Manual target encoding for high-cardinality

6. ⭐ Feature Scaling/Standardization (normalize ranges)
   └─ StandardScaler (most common)
   └─ MinMaxScaler (bounded [0,1])
   └─ RobustScaler (with outliers)
   └─ SKIP for tree-based models

7. ⭐ Model Training & Prediction
   └─ Fit model on processed features
   └─ Predictions automatically un-scaled if needed
```

### PySpark Full Pipeline Example

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression
from pyspark.sql.functions import log1p, col

# Assume df has: color (string), price (numeric, skewed), age (numeric), target (label)

# Step 1: Log transform skewed features
df = df.withColumn('price_log', log1p(col('price')))

# Step 2: Encode categorical
indexer = StringIndexer(inputCol='color', outputCol='color_idx')
encoder = OneHotEncoder(inputCol='color_idx', outputCol='color_ohe')

# Step 3: Assemble features
assembler = VectorAssembler(inputCols=['color_ohe', 'price_log', 'age'], 
                            outputCol='features')

# Step 4: Scale features
scaler = StandardScaler(inputCol='features', outputCol='scaled_features')

# Step 5: Model
model = LogisticRegression(featuresCol='scaled_features', labelCol='target')

# Chain together
pipeline = Pipeline(stages=[indexer, encoder, assembler, scaler, model])

# Fit and predict
fitted_model = pipeline.fit(df_train)
predictions = fitted_model.transform(df_test)
```

---

## Quick Reference: Common Exam Traps

1. `df.summary()` includes percentiles; `df.describe()` does not
2. `dbutils.data.summarize()` returns HTML, not a DataFrame
3. `df.approxQuantile()` is the ONLY way to compute percentiles in PySpark DataFrame API
4. IQR is MORE robust than standard deviation (not less)
5. Mean imputation REDUCES variance (does not increase it)
6. Mode in sklearn = `strategy='most_frequent'` (not 'mode')
7. PySpark OHE requires StringIndexer first (2-step requirement)
8. PySpark OHE default: N-1 dimensions (dropLast=True)
9. sklearn OHE default: N dimensions (drop=None)
10. Log transform gives NO benefit to tree-based models
11. Pearson r = 0 means no LINEAR relationship (non-linear may exist)
12. For nominal categories in linear models: MUST use OHE (label encoding implies false order)
13. **Log transform and feature scaling are DIFFERENT steps** — log fixes distribution shape, scaling normalizes ranges
14. **StandardScaler is most common** for linear/distance-based models
15. **Tree-based models need NO scaling or log transform** (both scale-invariant)
16. **MUST fit scaler on training data only** — scale test data with the same learned scaler (prevents data leakage)
17. **3-step OHE pipeline**: StringIndexer → OneHotEncoder → VectorAssembler
18. **2-step label encoding**: StringIndexer → VectorAssembler (no OneHotEncoder)
19. **Apply log transform BEFORE scaling** to fix distribution first, then normalize ranges
20. **Label encoding for ordinal data preserves order** (0 < 1 < 2); OHE destroys ordinality
