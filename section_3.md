%md
# Section 3: Model Development - Complete Study Guide

---

## Sub-topic 1: Algorithm Selection (ML Foundations)

### Choosing the Right Algorithm

| Problem Type | Algorithm | When to Use |
| --- | --- | --- |
| **Binary Classification** | Logistic Regression | Linear boundaries, interpretability needed |
| **Binary Classification** | Random Forest | Non-linear, handles mixed features |
| **Binary Classification** | Gradient Boosted Trees (XGBoost/LightGBM) | High accuracy, tabular data |
| **Binary Classification** | SVM (Support Vector Machine) | High-dimensional, clear margin between classes |
| **Binary Classification** | KNN (K-Nearest Neighbors) | Small data, non-parametric quick baseline |
| **Multi-class Classification** | Multinomial Logistic Regression | Linear, many classes |
| **Multi-class Classification** | Random Forest / XGBoost | Non-linear, many classes |
| **Regression** | Linear Regression | Linear relationships, interpretable |
| **Regression** | Decision Tree Regressor | Non-linear, interpretable splits |
| **Regression** | Gradient Boosted Trees | High accuracy, complex patterns |
| **Regression** | KNN Regressor | Small data, local patterns, non-parametric |
| **Clustering** | K-Means | Spherical clusters, known k |
| **Dimensionality Reduction** | PCA | Reduce features, remove multicollinearity |

### Key Decision Factors

| Factor | Simpler Models (Linear/Logistic) | Complex Models (Ensemble/Tree-based) |
| --- | --- | --- |
| **Interpretability** | High — coefficients are meaningful | Low — black box |
| **Training speed** | Fast | Slower |
| **Data size needed** | Works with small data | Needs more data |
| **Feature engineering** | Requires more (scaling, encoding) | Less sensitive to feature form |
| **Overfitting risk** | Low (high bias) | Higher (low bias, high variance) |
| **Non-linear patterns** | Cannot capture | Captures well |

### Algorithm Selection Rules of Thumb

1. **Start simple** — Linear/Logistic Regression as baseline
2. **Tabular data with many features** → Gradient Boosted Trees (XGBoost, LightGBM)
3. **Need interpretability** → Linear models or Decision Trees
4. **High-dimensional sparse data** → Linear models with regularization
5. **Mixed feature types** → Tree-based models (handle categorical natively)
6. **Small dataset** → Simpler models to avoid overfitting

> **Exam tip**: The question will describe a scenario — match the constraints (interpretability, data size, linearity) to the algorithm.

---

### Deep Dive: How Each Model Works (So You Choose, Not Memorize)

#### Linear Regression

**Mechanism**: Finds the best-fit line (or hyperplane) by minimizing the sum of squared residuals: ŷ = w₁x₁ + w₂x₂ + ... + b

**Intuition**: Imagine stretching a rubber band across your data points. The model finds the position where the total "pull" from all points is balanced. Each weight (w) tells you: "for every 1-unit increase in this feature, the prediction changes by w units."

**When it shines**:
- Relationship between features and target is approximately linear
- You need to explain WHY ("age increases price by $2,000 per year")
- You have fewer samples than features (with regularization)

**When it fails**:
- Non-linear patterns (e.g., house price peaks at middle size then drops)
- Interaction effects not manually specified (x₁ × x₂)

**Regularization variants**:
- **Ridge (L2)**: Shrinks all weights toward zero but keeps all features → use when all features might matter
- **Lasso (L1)**: Drives some weights to exactly zero → use for automatic feature selection
- **ElasticNet**: Mix of both → use when you're unsure

#### Logistic Regression

**Mechanism**: Linear Regression wrapped in a sigmoid function: P(y=1) = 1 / (1 + e^-(wx+b)). This squashes the output to [0, 1] for probability.

**Intuition**: The model draws a straight line (decision boundary) in feature space. Everything on one side is class A, the other side is class B. The sigmoid converts "distance from the boundary" into confidence.

**When it shines**:
- Binary or multi-class classification with roughly linear separability
- You need probability outputs ("this patient has 73% risk")
- Regulatory/healthcare settings requiring explainability
- High-dimensional sparse data (e.g., text with TF-IDF)

**When it fails**:
- Complex, non-linear decision boundaries (use trees)
- Heavily correlated features (multicollinearity distorts coefficients)

**Key insight**: Despite the name, it's a CLASSIFICATION algorithm, not regression. The "regression" refers to the mathematical technique (modeling log-odds).

#### Decision Trees

**Mechanism**: Repeatedly asks binary questions ("is age > 35?") to split data into purer subsets. Each leaf holds a prediction (majority class or mean value).

**Intuition**: Think of a flowchart. The tree learns the most informative questions to ask first (measured by information gain or Gini impurity). It partitions the feature space into rectangles.

**When it shines**:
- Need a human-readable model ("if income > 50K AND age > 30 → approve")
- Mixed feature types without preprocessing
- Non-linear relationships and interaction effects (captured naturally by nested splits)

**When it fails**:
- Prone to overfitting (memorizes training data if not pruned)
- Unstable — small data changes → completely different tree
- Axis-aligned splits can't capture diagonal boundaries efficiently

**Why no OHE needed**: The tree asks "is color_encoded ≤ 1.5?" — this creates group {0,1} vs {2,3}. The numeric value is just a split threshold, never multiplied by a weight.

#### Random Forest

**Mechanism**: Trains many decision trees on random subsets of data AND features, then averages (regression) or votes (classification).

**Intuition**: Instead of one expert (who might be wrong), ask 100 experts who each saw different evidence. Their consensus is more reliable than any individual. Each tree is intentionally made imperfect (random feature subsets) so their errors are uncorrelated.

**When it shines**:
- Almost any tabular dataset as a strong baseline
- You want robustness without careful hyperparameter tuning
- Parallel training (each tree is independent → fast on clusters)

**When it fails**:
- Rarely beats gradient boosting on accuracy (good but not best)
- Cannot extrapolate beyond training data range
- Memory-heavy with many deep trees

**Key hyperparameters**:
- `n_estimators`: More trees = more stable (diminishing returns after ~100-300)
- `max_depth`: Deeper = more flexible but more overfit risk
- `max_features`: Lower = more diversity between trees

#### Gradient Boosted Trees (XGBoost, LightGBM)

**Mechanism**: Trains trees SEQUENTIALLY. Each new tree focuses specifically on the errors (residuals) of the previous ensemble. This "boosts" weak learners into a strong learner.

**Intuition**: First tree makes predictions. Second tree is trained on "what the first tree got wrong." Third tree fixes the remaining errors. Each tree is shallow (weak), but their sum is powerful.

**When it shines**:
- State-of-the-art accuracy on tabular data (most Kaggle competitions)
- Complex non-linear patterns with interactions
- Handles missing values natively (XGBoost/LightGBM)

**When it fails**:
- Overfits small datasets if not regularized (more sensitive than RF)
- Slower training than Random Forest (sequential, not parallel)
- Less interpretable than single trees

**Random Forest vs Gradient Boosting**:

| Property | Random Forest | Gradient Boosting |
| --- | --- | --- |
| Training | Parallel (independent trees) | Sequential (each fixes previous errors) |
| Typical accuracy | Good | Often better |
| Overfitting risk | Lower (averaging reduces variance) | Higher (must tune learning_rate, depth) |
| Tuning effort | Low (robust defaults) | Higher (learning_rate, n_estimators, depth) |
| Speed | Faster training | Slower training, faster inference |

#### K-Nearest Neighbors (KNN)

**What it actually does**: KNN memorizes the entire training dataset. When you ask it to predict something new, it measures the distance from your new point to EVERY stored point, finds the K closest ones, and takes a vote.

**Concrete example**: Imagine you move to a new neighborhood and want to guess your house price. KNN says: "Find the 5 houses closest to yours (in size, age, location). Average their prices. That's your prediction."

**The critical insight — KNN has NO training phase**:
- Linear Regression learns weights, then discards the training data
- Decision Trees learn splits, then discards the training data
- KNN learns NOTHING. It stores the data and does ALL the work at prediction time
- This is why it's called a "lazy learner" — it postpones everything until you ask

**Why this matters practically**:
- Training = instant (just store the data)
- Prediction = SLOW (scan entire dataset per prediction)
- Large datasets = unusable (1M training points means 1M distance calculations per prediction)

**Why feature scaling is mandatory**:
- Distance formula: √((x₁-y₁)² + (x₂-y₂)²)
- If `salary` ranges 30,000–200,000 and `age` ranges 20–70, salary DOMINATES the distance
- A difference of 1 year in age becomes invisible compared to $1 difference in salary
- Solution: StandardScaler → both features contribute equally to distance

**Why OHE is mandatory for categorical features**:
- If color is encoded as blue=0, green=1, red=2
- Distance(blue, red) = |0-2| = 2, Distance(blue, green) = |0-1| = 1
- This says "green is more similar to blue than red is" — which is meaningless for colors
- With OHE: blue=[1,0,0], green=[0,1,0], red=[0,0,1] → all pairs have equal distance (√2)

**When to choose KNN**: Small dataset + you want a quick non-parametric baseline + no time to tune hyperparameters

**When NOT to choose KNN**: Large datasets, high dimensions (curse of dimensionality makes all points equidistant), or when you need fast predictions

---

#### Support Vector Machines (SVM)

**What it actually does**: Among all possible lines (or hyperplanes) that separate two classes, SVM finds the ONE that has the widest "road" (margin) between the classes.

**Concrete example**: Imagine you have red dots and blue dots on a table. You need to place a ruler between them. Many positions work, but SVM picks the position where the ruler is as far as possible from the closest red dot AND the closest blue dot simultaneously.

**Why the "widest road" matters**:
- A narrow separation = fragile (new points near the boundary could be misclassified)
- A wide margin = confident (even slightly noisy new points land on the correct side)
- The closest points to the boundary are called "support vectors" — they DEFINE where the line goes. Move any other point and nothing changes.

**The kernel trick (for non-linear boundaries)**:
- What if the two classes form concentric circles? No straight line can separate them.
- Trick: imagine lifting the data into 3D (add a new dimension = x² + y²). Now in 3D, a flat plane CAN separate them.
- The kernel computes "what the distance would be in that higher dimension" without actually transforming all the data
- Common kernels: `linear` (straight line), `rbf` (curved boundaries), `poly` (polynomial curves)

**When to choose SVM**:
- More features than samples (text with 50,000 TF-IDF features but 1,000 documents)
- Clean data with a clear gap between classes
- Binary classification (SVM is inherently binary; multi-class is hacked via one-vs-one)

**When NOT to choose SVM**:
- Large datasets (training is O(n²) to O(n³) — 100K samples is already painful)
- You need probability outputs (SVM gives you "which side of the line" not "how confident")
- Noisy data with overlapping classes (hard to find a clean margin)

**Key comparison with Logistic Regression**:
- Both draw a linear boundary in the same feature space
- Logistic Regression uses ALL points to find it (via maximum likelihood)
- SVM uses only the CLOSEST points (support vectors) to find it
- SVM's boundary is more robust to outliers far from the boundary

---

#### K-Means Clustering (Unsupervised)

**What it actually does**: Groups unlabeled data into K clusters by iteratively assigning points to the nearest cluster center, then moving each center to the average of its assigned points.

**THE fundamental difference from everything above**: K-Means has NO target variable. There is no "correct answer" to learn from. It discovers structure that humans haven't labeled.

**Concrete example**: You have 10,000 customers with purchase history. Nobody has labeled them as "budget", "premium", or "occasional." K-Means looks at their behavior and says "I see 3 natural groups" — then YOU interpret what those groups mean.

**The algorithm step by step**:
1. Pick K random points as initial centers ("centroids")
2. Assign every data point to its nearest centroid
3. Move each centroid to the average position of its assigned points
4. Repeat steps 2-3 until centroids stop moving (convergence)

**Why you must choose K in advance**:
- K-Means cannot discover how many clusters exist — you must tell it
- **Elbow method**: run K=1,2,3,...10. Plot total within-cluster distance. Look for the "elbow" where improvement slows dramatically
- **Silhouette score**: measures how similar points are to their own cluster vs. nearest neighbor cluster (ranges -1 to +1, higher = better)

**When K-Means fails (and why)**:
- **Non-spherical clusters**: K-Means assigns by DISTANCE to center. A crescent-shaped cluster has points far from its center → gets broken into pieces. Use DBSCAN instead.
- **Different-sized clusters**: The large cluster's center pulls points from the small cluster
- **Unscaled features**: Same problem as KNN — distance is dominated by large-scale features

**Exam trap**: K-Means is NEVER the answer to a classification or regression question. If the question says "predict" or "classify" → it's NOT K-Means. K-Means is only for "segment", "group", or "discover patterns."

---

#### K-Means vs DBSCAN: When Each Wins

**The core difference in one sentence**:
- K-Means: "I'll divide ALL points into exactly K circular groups around K centers."
- DBSCAN: "I'll follow chains of densely packed points and call each chain a cluster. Isolated points are noise."

**How DBSCAN works (step by step)**:
1. Pick a point. Count how many other points are within a radius ε (epsilon)
2. If ≥ minPts neighbors exist → this is a "core point" → start a cluster
3. Visit each of those neighbors. If any of THEM also have ≥ minPts within ε → expand the cluster
4. Keep expanding until no more dense points are reachable
5. Points that aren't dense enough and aren't reachable from any core point → labeled as **noise** (-1)

**Why DBSCAN handles crescents/bananas**: It doesn't use a center. It follows the chain of dense points along the curve. As long as each point has enough neighbors within ε, the cluster continues — regardless of shape.

**Comparison Table**:

| Property | K-Means | DBSCAN |
| --- | --- | --- |
| **Must specify K (number of clusters)?** | YES — you tell it how many | NO — discovers clusters automatically |
| **Cluster shape** | Spherical/circular only | ANY shape (crescents, rings, blobs) |
| **Handles noise/outliers?** | No — forces every point into a cluster | Yes — labels isolated points as noise (-1) |
| **Parameters** | K (number of clusters) | ε (radius) and minPts (density threshold) |
| **Cluster sizes** | Tends to create equal-sized clusters | Handles very different-sized clusters |
| **Speed** | Fast (O(n·K·iterations)) | Slower (O(n²) without spatial index) |
| **Deterministic?** | No — different random starts → different results | Yes — same params always gives same clusters |

**Concrete analogy**:
- **K-Means** = dropping K pins on a map and saying "everyone go to your nearest pin." Pins can only attract circular neighborhoods.
- **DBSCAN** = starting a rumor. A person tells everyone within earshot (ε). If those people also have enough neighbors (minPts), the rumor spreads. It follows crowds of any shape. Isolated people never hear it (noise).

**When to choose which**:

| Scenario | Choose |
| --- | --- |
| You know how many groups exist | K-Means |
| Clusters are round/spherical | K-Means |
| You DON'T know how many groups exist | DBSCAN |
| Clusters are irregular shapes | DBSCAN |
| Data has outliers you want to ignore | DBSCAN |
| Clusters have very different densities | Neither works well (use HDBSCAN) |

**Exam signal**: If a question mentions "arbitrary-shaped clusters", "unknown number of clusters", "outlier detection", or "density-based" → DBSCAN. If it mentions "segment into K groups" or "spherical clusters" → K-Means.

---

#### PCA (Principal Component Analysis)

**What it actually does**: Rotates your coordinate system so that the first axis captures the most variance, the second axis captures the most remaining variance (perpendicular to the first), and so on. Then you keep only the top-k axes and throw away the rest.

**PCA is NOT a predictive model. It's a data compression tool.**

**Concrete example**: Imagine a spreadsheet with 100 columns describing customer behavior. Many columns are correlated (e.g., "minutes on site" and "pages viewed" move together). PCA discovers these correlations and says: "You actually have ~15 independent patterns. I can compress your 100 columns into 15 numbers that capture 95% of the information."

**Why "maximum variance" = "most information"**:
- If a feature has zero variance (same value for everyone), it tells you nothing
- If a feature has high variance (different for everyone), it distinguishes between samples
- PCA finds the DIRECTIONS in which your data varies most — these are the informative directions

**What PCA removes**:
- Redundancy: if feature A and feature B are 95% correlated, one principal component captures both
- Noise: the last principal components capture random noise (low variance = not informative)

**The tradeoff**:
- Keep more components = more information preserved, but less compression
- Keep fewer components = more compression, but lose subtle patterns
- Rule of thumb: keep enough components to explain 90-95% of total variance

**When to use PCA**:
- Before Linear Regression: removes multicollinearity (correlated features confuse linear models)
- Visualization: reduce 100D data to 2D/3D for plotting
- Speed: fewer features = faster training for downstream models

**When NOT to use PCA**:
- You need to interpret features (PC1 = 0.3*age + 0.5*income + 0.2*score — what does that MEAN?)
- Tree-based models: they handle many features and correlations natively; PCA just makes features uninterpretable
- Categorical data: PCA assumes linear relationships between continuous variables

---

### Putting It All Together: How These Four Differ

| Question | KNN | SVM | K-Means | PCA |
| --- | --- | --- | --- | --- |
| **Is it supervised?** | Yes (needs labels) | Yes (needs labels) | NO (no labels) | NO (no labels) |
| **Does it predict?** | Yes (class or value) | Yes (class) | No (assigns groups) | No (transforms features) |
| **Does it train a model?** | No (stores data) | Yes (learns boundary) | Yes (learns centroids) | Yes (learns axes) |
| **Uses distance/magnitude?** | Yes → needs scaling + OHE | Yes → needs scaling + OHE | Yes → needs scaling | Yes → needs scaling |
| **What it outputs** | Predicted label/value | Predicted class | Cluster assignment (0,1,...K-1) | Transformed features (fewer columns) |
| **Typical exam context** | "small data, quick baseline" | "high-dimensional, clear separation" | "segment customers, find groups" | "reduce dimensions, remove correlation" |

---

### The Selection Flowchart (Reasoning, Not Memorizing)

```
Step 1: What's the task?
  - Predict a number (price, temperature) → REGRESSION
  - Predict a category (churn/not, species) → CLASSIFICATION
  - Find groups without labels → CLUSTERING
  - Reduce feature count → DIMENSIONALITY REDUCTION

Step 2: What are the constraints?
  - Need to explain to humans? → Linear/Logistic Regression or Decision Tree
  - Maximum accuracy on tabular data? → Gradient Boosted Trees
  - Very small dataset (<500 rows)? → Linear models (less overfit)
  - Very high-dimensional/sparse? → Linear with regularization or SVM
  - Continuous data arrival? → Online learning variants

Step 3: What's the data like?
  - Linear relationships? → Linear/Logistic Regression
  - Complex interactions? → Tree-based or Neural Networks
  - Mixed types (num + cat)? → Tree-based (no encoding hassle)
  - Need feature engineering budget? → Trees need less; Linear needs more
```

> **The golden rule**: The CONSTRAINTS (interpretability, data size, latency) narrow your choices. The DATA CHARACTERISTICS (linearity, dimensionality) finalize the pick.

---

## Sub-topic 2: Handling Data Imbalance

### What Is Data Imbalance?

When one class is significantly more frequent than another (e.g., 95% non-fraud, 5% fraud), models tend to predict the majority class and appear accurate but fail on the minority class.

### Methods to Mitigate Imbalance

| Method | Description | Pros | Cons |
| --- | --- | --- | --- |
| **Oversampling (SMOTE)** | Create synthetic minority samples | More data for minority class | Can overfit; synthetic data may not be realistic |
| **Undersampling** | Remove majority class samples | Faster training; balances classes | Loses information; smaller dataset |
| **Class weights** | Assign higher weight to minority class | No data modification needed | May not fully solve severe imbalance |
| **Threshold tuning** | Adjust classification threshold | Simple; preserves model | Requires careful calibration |
| **Ensemble methods** | Balanced Random Forest, EasyEnsemble | Combines under/oversampling | More complex |

### Class Weights in Practice

```python
from sklearn.linear_model import LogisticRegression

# Automatically balance class weights inversely proportional to frequency
model = LogisticRegression(class_weight='balanced')

# Or specify manually
model = LogisticRegression(class_weight={0: 1, 1: 10})
```

```python
from pyspark.ml.classification import LogisticRegression

# SparkML uses weightCol parameter
lr = LogisticRegression(weightCol='class_weight')
```

### Key Points

- **Accuracy is misleading** for imbalanced data — use precision, recall, F1, ROC/AUC instead
- **SMOTE** (Synthetic Minority Over-sampling Technique) creates synthetic samples by interpolating between existing minority samples
- **Class weights** tell the algorithm to penalize misclassification of the minority class more heavily
- Apply resampling **only to training data**, never to test/validation data

> **Exam trap**: Oversampling/undersampling must happen AFTER the train-test split, not before. Otherwise, synthetic samples may leak into the test set.

---

## Sub-topic 3: Estimators and Transformers (SparkML)

### Core Concepts

| Concept | Description | Method | Example |
| --- | --- | --- | --- |
| **Transformer** | Converts one DataFrame to another | `.transform(df)` | `VectorAssembler`, `StringIndexer` (fitted) |
| **Estimator** | Learns from data to produce a Transformer | `.fit(df)` → Transformer | `LogisticRegression`, `StringIndexer` (unfitted) |
| **Pipeline** | Chains Estimators and Transformers | `.fit(df)` → PipelineModel | Sequence of stages |
| **PipelineModel** | A fitted Pipeline (all stages are Transformers) | `.transform(df)` | Result of `pipeline.fit()` |

### How They Relate

```
Estimator.fit(training_df) → Transformer (Model)
Transformer.transform(df) → new DataFrame
Pipeline.fit(training_df) → PipelineModel
PipelineModel.transform(new_df) → predictions DataFrame
```

### Important Distinction

```python
from pyspark.ml.feature import StringIndexer

# StringIndexer is an ESTIMATOR (needs to learn categories from data)
indexer = StringIndexer(inputCol='color', outputCol='color_idx')

# After .fit(), it becomes a TRANSFORMER (StringIndexerModel)
indexer_model = indexer.fit(train_df)  # Learns: red=0, blue=1, green=2

# Now it can transform any DataFrame
result = indexer_model.transform(test_df)
```

```python
from pyspark.ml.feature import VectorAssembler

# VectorAssembler is a TRANSFORMER (no learning needed)
assembler = VectorAssembler(inputCols=['age', 'income'], outputCol='features')
result = assembler.transform(df)  # No .fit() needed!
```

> **Exam trap**: `VectorAssembler` is a Transformer (no `.fit()` required). `StringIndexer` is an Estimator (requires `.fit()` to learn mappings).

---

## Sub-topic 4: Developing a Training Pipeline

### SparkML Pipeline

```python
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression

# Define stages
indexer = StringIndexer(inputCol='category', outputCol='category_idx')
assembler = VectorAssembler(inputCols=['age', 'income', 'category_idx'], outputCol='raw_features')
scaler = StandardScaler(inputCol='raw_features', outputCol='features')
lr = LogisticRegression(featuresCol='features', labelCol='label')

# Create pipeline
pipeline = Pipeline(stages=[indexer, assembler, scaler, lr])

# Fit entire pipeline at once
pipeline_model = pipeline.fit(train_df)

# Predict on new data
predictions = pipeline_model.transform(test_df)
```

### Pipeline Benefits

1. **Reproducibility** — same preprocessing + model in one object
2. **No data leakage** — scaler/encoder fits only on training data
3. **Easy deployment** — save/load entire pipeline as one artifact
4. **Works with CrossValidator** — treats pipeline as a single estimator
5. **MLflow logging** — log the entire pipeline model

### sklearn Pipeline (Single-node)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression

preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['age', 'income']),
    ('cat', OneHotEncoder(), ['category'])
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression())
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

> **Key point**: A Pipeline is an Estimator. When you call `.fit()`, each stage is fit sequentially and the output feeds into the next stage.

---

## Sub-topic 5: Hyperparameter Tuning with Hyperopt

### What Is Hyperopt?

Hyperopt is a Python library for **Bayesian hyperparameter optimization**. It uses algorithms like Tree-structured Parzen Estimators (TPE) to efficiently search the hyperparameter space.

### Hyperopt's `fmin` Operation

```python
from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK

# 1. Define objective function (returns loss to MINIMIZE)
def objective(params):
    model = XGBClassifier(
        max_depth=int(params['max_depth']),
        learning_rate=params['learning_rate'],
        n_estimators=int(params['n_estimators'])
    )
    model.fit(X_train, y_train)
    accuracy = model.score(X_val, y_val)
    return {'loss': -accuracy, 'status': STATUS_OK}  # Negate because fmin MINIMIZES

# 2. Define search space
search_space = {
    'max_depth': hp.quniform('max_depth', 3, 15, 1),
    'learning_rate': hp.loguniform('learning_rate', -5, 0),  # exp(-5) to exp(0)
    'n_estimators': hp.quniform('n_estimators', 50, 500, 50)
}

# 3. Run optimization
best_params = fmin(
    fn=objective,           # Objective function
    space=search_space,     # Search space
    algo=tpe.suggest,       # Bayesian TPE algorithm
    max_evals=50,           # Number of trials
    trials=SparkTrials(parallelism=4)  # Parallelize on Spark!
)
```

### Key Hyperopt Concepts

| Concept | Description |
| --- | --- |
| `fmin` | Main function that minimizes the objective |
| `hp.uniform(name, low, high)` | Continuous uniform distribution |
| `hp.loguniform(name, low, high)` | Log-uniform (good for learning rates) |
| `hp.quniform(name, low, high, q)` | Quantized uniform (integers: round to q) |
| `hp.choice(name, options)` | Categorical choice from a list |
| `tpe.suggest` | Tree-structured Parzen Estimator (Bayesian) |
| `SparkTrials` | Distributes trials across Spark workers |
| `Trials` | Stores results locally (single-node) |
| `STATUS_OK` | Return status indicating successful trial |

### Critical Rules

1. **`fmin` always MINIMIZES** — negate accuracy/AUC, or use loss/error directly
2. **`hp.loguniform`** takes log-space bounds: `hp.loguniform('lr', log(0.001), log(1))` samples between 0.001 and 1
3. **`hp.quniform`** for integers — but cast to `int()` inside the objective
4. **`SparkTrials`** parallelizes single-node models across cluster workers
5. **Return a dict** with `'loss'` and `'status'` keys from the objective

> **Exam trap**: `fmin` returns the INDICES for `hp.choice`, not the values! Use `hyperopt.space_eval(space, best_params)` to get actual values.

---

## Sub-topic 6: Grid Search, Random Search, and Bayesian Search

### Comparison of Search Methods

| Method | Strategy | Pros | Cons |
| --- | --- | --- | --- |
| **Grid Search** | Exhaustive: tries all combinations | Guaranteed to find best in grid | Exponentially expensive; curse of dimensionality |
| **Random Search** | Random sampling from distributions | More efficient for high-dim; finds good solutions faster | No guarantee of best; not directed |
| **Bayesian (TPE)** | Uses prior results to guide search | Most efficient; focuses on promising regions | More complex; overhead per trial |

### Grid Search (SparkML)

```python
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

param_grid = ParamGridBuilder() \
    .addGrid(lr.regParam, [0.01, 0.1, 1.0]) \
    .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
    .build()

# Total models = 3 × 3 = 9 combinations
```

### Random Search (sklearn)

```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

param_distributions = {
    'max_depth': randint(3, 15),
    'learning_rate': uniform(0.01, 0.3),
    'n_estimators': randint(50, 500)
}

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_distributions,
    n_iter=20,  # Only try 20 random combinations
    cv=3,
    scoring='f1'
)
search.fit(X_train, y_train)
```

### When to Use Each

- **Grid Search**: Few hyperparameters (≤3), small grid, want exhaustive coverage
- **Random Search**: Many hyperparameters, limited compute budget, continuous parameters
- **Bayesian (Hyperopt)**: Expensive objective function, want maximum efficiency, have prior knowledge

> **Key insight**: Random search is provably more efficient than grid search when only a few hyperparameters actually matter (most real-world scenarios).

---

### Hyperopt vs Optuna: What's the Difference?

**Both solve the same problem**: find the best hyperparameters without exhaustive grid search. Both use Bayesian-style optimization. But they differ in HOW you write the code and WHERE they run best.

#### How They Work (Conceptually Identical)

Both do this loop:
1. Pick a set of hyperparameters to try (guided by past results)
2. Train a model with those hyperparameters
3. Evaluate the model, record the score
4. Use that score to pick smarter hyperparameters next time
5. Repeat until budget exhausted

The intelligence is in step 4 — both use Bayesian methods to focus on promising regions instead of searching randomly.

#### The Key API Difference

**Hyperopt** — you define the search space OUTSIDE the objective, then `fmin` passes sampled values IN:

```python
# Search space defined separately
space = {
    'max_depth': hp.quniform('max_depth', 3, 15, 1),
    'learning_rate': hp.loguniform('learning_rate', -5, 0)
}

# Objective receives pre-sampled params
def objective(params):
    model = XGBClassifier(max_depth=int(params['max_depth']), ...)
    ...
    return {'loss': -accuracy, 'status': STATUS_OK}

fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50)
```

**Optuna** — you define the search space INSIDE the objective using `trial.suggest_*`:

```python
# No separate space definition!
def objective(trial):
    # Space defined inline
    max_depth = trial.suggest_int('max_depth', 3, 15)
    lr = trial.suggest_float('learning_rate', 1e-5, 1.0, log=True)
    
    model = XGBClassifier(max_depth=max_depth, learning_rate=lr)
    ...
    return accuracy  # Optuna MAXIMIZES or MINIMIZES (you choose)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

#### Head-to-Head Comparison

| Property | Hyperopt | Optuna |
| --- | --- | --- |
| **Search space location** | Defined OUTSIDE objective (separate dict) | Defined INSIDE objective (trial.suggest_*) |
| **Optimization direction** | Always MINIMIZES (negate accuracy) | You choose: 'maximize' or 'minimize' |
| **Algorithm** | TPE (Tree-structured Parzen Estimators) | TPE + CMA-ES + more (pluggable samplers) |
| **Pruning (early stopping)** | Not built-in | Built-in: stops bad trials early (saves time) |
| **Databricks integration** | Native: `SparkTrials` parallelizes across cluster | No native Spark parallelism (use Joblib or manual) |
| **MLflow auto-logging** | Yes, with `SparkTrials` | Requires manual `mlflow.log_params()` |
| **Conditional params** | `hp.choice` with nested spaces (awkward) | Simple if/else inside objective (natural Python) |
| **Visualization** | Limited | Rich built-in: `optuna.visualization.*` |
| **Maturity** | Older, less actively developed | Newer, very actively maintained |

#### When to Use Which

| Scenario | Choose | Why |
| --- | --- | --- |
| **Databricks + distributed cluster** | Hyperopt | SparkTrials parallelizes across workers natively |
| **Databricks exam** | Hyperopt | It's the one tested on the certification |
| **Complex search spaces (conditional)** | Optuna | if/else inside objective is much cleaner |
| **Need early stopping of bad trials** | Optuna | Pruning kills unpromising trials mid-training |
| **Single-node experimentation** | Either works | Personal preference |
| **Production Databricks pipelines** | Hyperopt | Tighter MLflow + Spark integration |

#### Optuna's Killer Feature: Pruning

Imagine trial #23 is training an XGBoost with `max_depth=2, learning_rate=0.001`. After 10 boosting rounds, it's clearly terrible. Optuna can **stop it early** and move on to trial #24 instead of waiting for all 500 rounds to finish.

```python
def objective(trial):
    params = {...}
    for step in range(100):
        model.partial_fit(X_train, y_train)
        score = model.score(X_val, y_val)
        trial.report(score, step)  # Report intermediate result
        if trial.should_prune():   # Optuna decides: keep going or stop?
            raise optuna.TrialPruned()
    return score
```

Hyperopt has no equivalent — every trial runs to completion, even clearly hopeless ones.

#### For the Exam: Focus on Hyperopt

The Databricks ML Associate exam tests **Hyperopt** specifically because:
- It integrates with `SparkTrials` for distributed tuning
- It auto-logs to MLflow on Databricks
- It's the recommended tool in Databricks documentation

Know these Hyperopt-specific details:
- `fmin` always **minimizes** → negate metrics you want to maximize
- `hp.choice` returns **indices** not values → use `space_eval()` to decode
- `SparkTrials(parallelism=N)` distributes N trials across Spark workers
- Higher parallelism = faster but less Bayesian benefit (approaches random search)

> **Exam signal**: If the question says "efficiently search hyperparameters on Databricks" or "parallelize tuning across the cluster" → Hyperopt + SparkTrials is the answer.

---

## Sub-topic 7: Parallelizing Hyperparameter Tuning

### SparkTrials (Hyperopt)

```python
from hyperopt import SparkTrials

# Distribute single-node model training across Spark workers
spark_trials = SparkTrials(parallelism=8)  # 8 trials run simultaneously

best = fmin(
    fn=objective,
    space=search_space,
    algo=tpe.suggest,
    max_evals=100,
    trials=spark_trials
)
```

### How It Works

| Component | Role |
| --- | --- |
| **Driver** | Coordinates trials, runs Hyperopt's TPE algorithm |
| **Workers** | Each worker trains one model with one set of hyperparameters |
| **parallelism** | Max number of concurrent trials on workers |

### Parallelism Guidelines

| Parallelism | Behavior |
| --- | --- |
| `parallelism=1` | Fully sequential — best Bayesian optimization (each trial informs next) |
| `parallelism=max_evals` | Equivalent to random search (no Bayesian benefit) |
| **Recommended** | `parallelism` = number of workers; balance between speed and Bayesian efficiency |

### Key Points

1. **`SparkTrials`** is for parallelizing **single-node models** (sklearn, XGBoost) across Spark
2. **Do NOT use** `SparkTrials` with SparkML models (they already distribute training)
3. Higher parallelism = faster but less directed search (loses Bayesian advantage)
4. Each trial gets a **full worker** — set `parallelism` ≤ number of available workers
5. MLflow automatically logs each trial when using `SparkTrials`

> **Exam trap**: `SparkTrials` parallelizes single-node models. For SparkML distributed models, use `CrossValidator` with `parallelism` parameter instead.

---

## Sub-topic 8: Cross-Validation vs Train-Validation Split

### Train-Validation Split

```python
train_df, val_df = df.randomSplit([0.8, 0.2], seed=42)
model = estimator.fit(train_df)
predictions = model.transform(val_df)
```

### K-Fold Cross-Validation

```
Fold 1: [VAL] [Train] [Train] [Train] [Train]
Fold 2: [Train] [VAL] [Train] [Train] [Train]
Fold 3: [Train] [Train] [VAL] [Train] [Train]
Fold 4: [Train] [Train] [Train] [VAL] [Train]
Fold 5: [Train] [Train] [Train] [Train] [VAL]
```

### Comparison

| Property | Train-Validation Split | K-Fold Cross-Validation |
| --- | --- | --- |
| **Reliability** | Single estimate — high variance | Average of K estimates — more robust |
| **Data usage** | Wastes validation data for training | Every point used for both train and val |
| **Compute cost** | Trains 1 model | Trains K models per hyperparameter set |
| **Best for** | Large datasets; quick iteration | Small datasets; final model selection |
| **Overfitting to val** | Risk of overfitting to one split | Reduced risk (averages over splits) |

### Cross-Validation in SparkML

```python
from pyspark.ml.tuning import CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(metricName='areaUnderROC')

cv = CrossValidator(
    estimator=pipeline,
    estimatorParamMaps=param_grid,
    evaluator=evaluator,
    numFolds=5,            # K = 5
    parallelism=4          # Parallelize fold training
)

cv_model = cv.fit(train_df)
best_model = cv_model.bestModel
```

### Calculating Total Models Trained

```
Total models = (number of param combinations) × (number of folds)
```

**Example**: Grid has 3 × 3 = 9 combinations, using 5-fold CV:
```
Total = 9 × 5 = 45 models trained
```

> **Exam question pattern**: "How many models are trained?" → Count grid combinations × K folds.

### Cross-Validation in sklearn

```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Mean accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

---

## Sub-topic 9: Classification Metrics

### Confusion Matrix

```
                  Predicted Positive    Predicted Negative
Actual Positive       TP                     FN
Actual Negative       FP                     TN
```

### Key Metrics

| Metric | Formula | Answers the Question |
| --- | --- | --- |
| **Accuracy** | (TP + TN) / Total | What % of predictions are correct? |
| **Precision** | TP / (TP + FP) | Of predicted positives, what % are truly positive? |
| **Recall (Sensitivity)** | TP / (TP + FN) | Of actual positives, what % did we catch? |
| **F1 Score** | 2 × (Precision × Recall) / (Precision + Recall) | Harmonic mean of precision and recall |
| **Log Loss** | -mean(y·log(p) + (1-y)·log(1-p)) | How well-calibrated are probability estimates? |
| **ROC AUC** | Area under ROC curve | Discrimination ability across all thresholds |

### When to Use Each Metric

| Scenario | Best Metric | Why |
| --- | --- | --- |
| Balanced classes, general performance | **Accuracy** | Simple and interpretable |
| Imbalanced classes | **F1, Precision, Recall** | Accuracy is misleading |
| Minimize false positives (spam filter) | **Precision** | Don't want legitimate email in spam |
| Minimize false negatives (disease detection) | **Recall** | Don't want to miss sick patients |
| Balance precision and recall | **F1 Score** | Harmonic mean penalizes extremes |
| Compare models across thresholds | **ROC AUC** | Threshold-independent |
| Well-calibrated probabilities needed | **Log Loss** | Penalizes confident wrong predictions |

### ROC Curve Explained

- **X-axis**: False Positive Rate (FPR) = FP / (FP + TN)
- **Y-axis**: True Positive Rate (TPR) = Recall = TP / (TP + FN)
- **AUC = 0.5**: Random guessing (diagonal line)
- **AUC = 1.0**: Perfect classifier
- **Higher AUC** = better discrimination between classes

> **Exam trap**: ROC AUC measures discrimination across ALL thresholds. A model with AUC=0.9 isn't necessarily better at your chosen threshold — it's better overall.

---

## Sub-topic 10: Regression Metrics

### Key Metrics

| Metric | Formula | Interpretation |
| --- | --- | --- |
| **MSE** | mean((y - ŷ)²) | Average squared error; penalizes large errors |
| **RMSE** | √MSE | Same unit as target; most interpretable |
| **MAE** | mean(abs(y - ŷ)) | Average absolute error; robust to outliers |
| **R² (R-squared)** | 1 - (SS_res / SS_tot) | Proportion of variance explained (0 to 1) |
| **Adjusted R²** | Accounts for number of features | Penalizes adding useless features |

### When to Use Each Metric

| Scenario | Best Metric | Why |
| --- | --- | --- |
| General regression performance | **RMSE** | Same unit as target, penalizes big errors |
| Outliers present in data | **MAE** | Less sensitive to outliers than RMSE |
| Comparing models on different scales | **R²** | Normalized 0-1, scale-independent |
| Optimization objective | **MSE** | Differentiable; used in loss functions |
| Feature selection | **Adjusted R²** | Penalizes unnecessary features |

### Key Relationships

- **RMSE ≥ MAE** always (by Cauchy-Schwarz inequality)
- **R² = 1**: Perfect predictions
- **R² = 0**: Model no better than predicting the mean
- **R² < 0**: Model is WORSE than predicting the mean (very bad)
- **MSE/RMSE**: Sensitive to outliers (squaring amplifies large errors)
- **MAE**: Robust to outliers (no squaring)

> **Exam trap**: R² can be negative! This means the model performs worse than simply predicting the mean of the target variable.

---

## Sub-topic 11: Choosing the Right Metric

### Decision Framework

| Question to Ask | If Yes → Use |
| --- | --- |
| Is it a classification problem? | Precision, Recall, F1, ROC AUC, Log Loss |
| Is it a regression problem? | RMSE, MAE, R² |
| Are classes imbalanced? | F1, Precision, Recall (NOT Accuracy) |
| Is cost of false positive high? | Precision |
| Is cost of false negative high? | Recall |
| Need threshold-independent comparison? | ROC AUC |
| Need probability calibration? | Log Loss |
| Are there outliers in the target? | MAE (not RMSE) |
| Comparing models with different # features? | Adjusted R² |

> **Exam tip**: The scenario will describe a business context. Map the business cost to the metric:
> - "Missing a fraud case is very costly" → Maximize **Recall**
> - "Sending too many false alerts is costly" → Maximize **Precision**
> - "Need balanced performance" → Maximize **F1**

---

## Sub-topic 12: Log Transformations and Metrics

### Why Log-Transform the Target?

- Skewed distributions (e.g., housing prices, salary) → log makes them more normal
- Stabilizes variance (heteroscedasticity)
- Models train better on normally distributed targets

### The Exponentiation Requirement

```python
import numpy as np

# Training: log-transform the target
y_train_log = np.log(y_train)
model.fit(X_train, y_train_log)

# Prediction: model outputs log-scale predictions
y_pred_log = model.predict(X_test)

# MUST exponentiate before calculating metrics or interpreting!
y_pred = np.exp(y_pred_log)

# Now calculate metrics in ORIGINAL scale
from sklearn.metrics import mean_squared_error
rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # Original scale!
```

### Critical Rules

1. **Always exponentiate** predictions before computing metrics (RMSE, MAE) against original targets
2. **Always exponentiate** before presenting predictions to stakeholders
3. Metrics computed in log-space are NOT interpretable in original units
4. `np.exp(log_prediction)` converts back to original scale
5. If you trained on `log(y)`, predictions are in `log(y)` space — you MUST convert back

> **Exam trap**: If a model was trained on log-transformed targets, you MUST exponentiate (`np.exp()`) the predictions BEFORE calculating RMSE/MAE or interpreting the output. Metrics in log-space are meaningless for business interpretation.

---

## Sub-topic 13: Bias-Variance Tradeoff

### Core Concepts

| Concept | Definition | Symptom |
| --- | --- | --- |
| **Bias** | Error from overly simple assumptions | Underfitting: poor on training AND test |
| **Variance** | Error from sensitivity to training data fluctuations | Overfitting: great on training, poor on test |
| **Irreducible Error** | Noise in the data that no model can capture | — |

```
Total Error = Bias² + Variance + Irreducible Error
```

### Model Complexity Spectrum

```
Simple Model ←————————————————————→ Complex Model
(High Bias, Low Variance)          (Low Bias, High Variance)
Underfitting                        Overfitting
Linear Regression                   Deep Neural Network
Shallow Tree (depth=1)              Deep Tree (depth=50)
High regularization                 No regularization
```

### Detecting Bias vs Variance

| Symptom | Diagnosis | Fix |
| --- | --- | --- |
| High training error, high test error | **High Bias** (Underfitting) | More complex model; more features; less regularization |
| Low training error, high test error | **High Variance** (Overfitting) | More data; regularization; simpler model; dropout |
| Low training error, low test error | **Good fit** | — |
| Gap between train/test error growing | **Increasing Variance** | Regularize; reduce model capacity |

### How Hyperparameters Affect Complexity

| Hyperparameter | Increasing It → |
| --- | --- |
| `max_depth` (trees) | More complex → more variance |
| `n_estimators` (forests) | More complex but with diminishing returns |
| `learning_rate` (boosting) | Higher = faster learning, risk of overfitting |
| `regularization` (lambda/alpha) | Less complex → more bias |
| `min_samples_leaf` (trees) | Less complex → more bias |
| `polynomial degree` | More complex → more variance |

### Regularization

| Type | Name | Effect | SparkML Parameter |
| --- | --- | --- | --- |
| L1 | Lasso | Shrinks some coefficients to exactly 0 (feature selection) | `elasticNetParam=1.0` |
| L2 | Ridge | Shrinks all coefficients toward 0 (no elimination) | `elasticNetParam=0.0` |
| Elastic Net | Combination | Mix of L1 and L2 | `0 < elasticNetParam < 1` |

> **Exam tip**: The question may describe a model's performance on train vs test. If training performance is great but test is poor → overfitting → reduce complexity, add regularization, get more data.

---

## Quick Reference: Common Exam Traps

1. **`fmin` always MINIMIZES** — return negative accuracy: `{'loss': -accuracy, 'status': STATUS_OK}`
2. **Total models in CV + Grid** = (grid combinations) × (K folds)
3. **`SparkTrials`** parallelizes single-node models; do NOT use with SparkML
4. **VectorAssembler** is a Transformer (no `.fit()`); **StringIndexer** is an Estimator (needs `.fit()`)
5. **Pipeline.fit()** → PipelineModel; Pipeline is an Estimator, PipelineModel is a Transformer
6. **Exponentiate predictions** before computing metrics on log-transformed targets
7. **R² can be negative** — means model is worse than predicting the mean
8. **RMSE ≥ MAE** always
9. **Imbalanced data** → don't use accuracy; use F1, Precision, Recall, ROC AUC
10. **High bias** = underfitting (train error high); **High variance** = overfitting (gap between train/test)
11. **Random search** is more efficient than grid search when few hyperparameters matter
12. **Cross-validation** uses ALL data for training and validation; better for small datasets
13. **Oversampling/undersampling** — apply ONLY to training data, never test
14. **`hp.choice` returns index** — use `space_eval()` to get actual values
15. **Recall** = "catch all positives" (disease, fraud); **Precision** = "don't cry wolf" (spam, alerts)
