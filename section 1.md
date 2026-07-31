%md
# Section 1: Databricks Machine Learning - Complete Study Guide

---

## Sub-topic 1: MLOps Best Practices

### What is MLOps?

MLOps (Machine Learning Operations) applies DevOps principles to ML systems. It bridges the gap between model development and production deployment.

### Key Stages of the ML Lifecycle

| Stage | Activities | Key Tools |
| --- | --- | --- |
| Data Preparation | Feature engineering, data validation | Feature Store, Delta Lake |
| Model Development | Training, tuning, evaluation | AutoML, MLflow Tracking |
| Model Validation | Testing, comparison, approval | MLflow Model Registry |
| Deployment | Serving, monitoring, retraining | Model Serving, Lakehouse Monitoring |

### MLOps Best Practices

1. **Version everything** — code, data, models, and configurations
2. **Automate pipelines** — training, validation, and deployment should be repeatable
3. **Monitor in production** — track data drift, model performance degradation
4. **Separate environments** — dev/staging/prod with promotion gates
5. **Reproducibility** — any model should be re-trainable from logged parameters + data version
6. **Governance** — access control, lineage tracking, audit trails (Unity Catalog)

### Promote Code vs Promote Models

| Approach | When to Use | Advantages |
| --- | --- | --- |
| **Promote Models** | Stable pipelines, frequent retraining | Faster iteration; model is the artifact |
| **Promote Code** | Complex pipelines, strict governance | Full reproducibility; retrain in target env |

> **Promote code** = move training code to production and retrain there.  
> **Promote models** = train in dev, register the model artifact, deploy to production.

- **Promote code is preferred** when: data access differs across environments, strict audit/compliance needed, training must be reproducible end-to-end
- **Promote models is preferred** when: training is expensive (GPU days), model is validated artifact, quick iteration needed

---

## Sub-topic 2: ML Runtimes

### What Are Databricks ML Runtimes?

| Runtime | Includes | Use Case |
| --- | --- | --- |
| Standard Runtime | Spark, Delta Lake, basic libraries | General data engineering |
| **ML Runtime** | Standard + ML libraries pre-installed | ML development |
| ML Runtime (GPU) | ML Runtime + GPU drivers + CUDA | Deep learning, GPU training |

### Advantages of ML Runtimes

1. **Pre-installed libraries** — scikit-learn, XGBoost, LightGBM, TensorFlow, PyTorch, Hugging Face, MLflow, etc.
2. **Tested compatibility** — all library versions are tested together (no dependency conflicts)
3. **MLflow pre-configured** — autologging enabled, experiment tracking ready
4. **Optimized performance** — libraries compiled with optimizations for the cluster hardware
5. **No setup time** — no `%pip install` needed for common ML packages
6. **GPU support** — GPU runtimes include CUDA, cuDNN, NCCL for distributed training

### Key Point
> ML Runtimes reduce setup friction, ensure compatibility, and provide optimized versions of popular ML libraries. They do NOT change how Spark works — they add ML tooling on top.

---

## Sub-topic 3: AutoML

### What AutoML Does

| Capability | Description |
| --- | --- |
| **Algorithm selection** | Tries multiple model types (Random Forest, XGBoost, LightGBM, Linear models) |
| **Feature selection** | Automatically handles feature preprocessing and selection |
| **Hyperparameter tuning** | Searches optimal parameters for each algorithm |
| **Data splitting** | Handles train/validation/test splitting |
| **Evaluation** | Ranks models by the primary metric |

### How AutoML Facilitates Model/Feature Selection

- AutoML generates **separate notebook for each trial** — fully transparent and editable
- It performs **automatic feature engineering**: one-hot encoding, imputation, standardization
- It tries **multiple algorithms** and selects the best based on the evaluation metric
- It produces a **leaderboard** ranking all models

### Advantages of AutoML

1. **Glass-box approach** — every trial produces a viewable, editable notebook (not a black box)
2. **Baseline generation** — quickly establishes a performance baseline
3. **Best practices built-in** — proper cross-validation, data splitting, encoding
4. **MLflow integration** — every trial automatically logged with parameters, metrics, artifacts
5. **Editable notebooks** — generated code can be customized and extended
6. **Time-saving** — eliminates boilerplate code for common ML workflows

### AutoML API

```python
from databricks import automl

# Classification
summary = automl.classify(train_df, target_col='label', timeout_minutes=30)

# Regression
summary = automl.regress(train_df, target_col='price', timeout_minutes=30)

# Forecasting
summary = automl.forecast(train_df, target_col='sales', time_col='date', timeout_minutes=30)
```

### What AutoML Returns

```python
summary.best_trial          # Best trial info
summary.best_trial.model_path  # Path to best model
summary.best_trial.metrics  # Metrics of best model
```

> AutoML does NOT deploy models. It trains, evaluates, and logs them. Deployment is a separate step.

---

## Sub-topic 4: Feature Store in Unity Catalog

### Feature Store Concepts

| Concept | Description |
| --- | --- |
| **Feature Table** | A Delta table registered with primary key(s) for feature lookup |
| **Primary Key** | Column(s) that uniquely identify each entity (e.g., `customer_id`) |
| **Feature Lookup** | Mechanism to join features to training/inference data by key |
| **Timestamp Key** | Optional; enables point-in-time lookups (prevents data leakage) |

### Unity Catalog Feature Store vs Workspace Feature Store

| Property | Unity Catalog (Account-Level) | Workspace-Level (Legacy) |
| --- | --- | --- |
| **Scope** | Shared across workspaces | Single workspace only |
| **Governance** | Full UC permissions (GRANT/REVOKE) | Workspace ACLs |
| **Discoverability** | Searchable across the organization | Limited to one workspace |
| **Lineage** | Full lineage tracking in UC | Limited lineage |
| **Client** | `FeatureEngineeringClient` | `FeatureStoreClient` (deprecated) |
| **Collaboration** | Cross-team feature sharing | Team silos |

> **Key benefit**: Unity Catalog Feature Store enables **cross-workspace feature sharing** with centralized governance.

### Creating a Feature Table

```python
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Create feature table from an existing DataFrame
fe.create_table(
    name='catalog.schema.customer_features',  # Unity Catalog path
    primary_keys=['customer_id'],
    df=feature_df,
    description='Customer demographic features'
)
```

> You MUST provide `primary_keys`. Without them, it's just a regular Delta table.

### Writing Data to a Feature Table

```python
# Write (merge/upsert) new data into existing feature table
fe.write_table(
    name='catalog.schema.customer_features',
    df=new_features_df,
    mode='merge'  # 'merge' (upsert) or 'overwrite'
)
```

| Mode | Behavior |
| --- | --- |
| `'merge'` | Upsert: update existing rows by PK, insert new ones |
| `'overwrite'` | Replace all data in the table |

### Training with Feature Store

```python
from databricks.feature_engineering import FeatureLookup

# Define feature lookups
feature_lookups = [
    FeatureLookup(
        table_name='catalog.schema.customer_features',
        feature_names=['age', 'income', 'tenure'],
        lookup_key='customer_id'
    )
]

# Create training set (joins features to label DataFrame)
training_set = fe.create_training_set(
    df=label_df,              # DataFrame with labels + lookup keys
    feature_lookups=feature_lookups,
    label='churn',
    exclude_columns=['customer_id']  # Don't use keys as features
)

training_df = training_set.load_df()
```

### Scoring with Feature Store

```python
# Batch scoring — automatically looks up features by key
predictions = fe.score_batch(
    model_uri='models:/my_model@champion',
    df=scoring_df  # Only needs the lookup keys!
)
```

> **Key insight**: At inference time, you only provide the keys (e.g., `customer_id`). The Feature Store automatically joins the latest features.

### Online vs Offline Feature Tables

| Property | Offline (Batch) | Online (Real-time) |
| --- | --- | --- |
| **Storage** | Delta Lake | Low-latency store (DynamoDB, CosmosDB) |
| **Latency** | Seconds to minutes | Milliseconds |
| **Use case** | Batch inference, training | Real-time model serving |
| **Freshness** | Updated periodically (scheduled) | Synced from offline or updated in real-time |
| **Access** | Spark queries | Key-value lookups via serving endpoint |

> Online tables are **published from offline tables** for low-latency serving. They are NOT a replacement — they complement offline tables.

---

## Sub-topic 5: MLflow Tracking

### MLflow Components

| Component | Purpose |
| --- | --- |
| **Tracking** | Log parameters, metrics, artifacts, models |
| **Models** | Package models in a standard format |
| **Model Registry** | Version, stage, and manage model lifecycle |
| **Projects** | Package code for reproducible runs |

### Manually Logging in MLflow

```python
import mlflow

with mlflow.start_run() as run:
    # Log parameters (hyperparameters, config)
    mlflow.log_param('learning_rate', 0.01)
    mlflow.log_param('max_depth', 5)
    
    # Log metrics (evaluation results)
    mlflow.log_metric('rmse', 0.85)
    mlflow.log_metric('r2', 0.92)
    
    # Log artifacts (files: plots, data, configs)
    mlflow.log_artifact('/path/to/plot.png')
    
    # Log the model
    mlflow.sklearn.log_model(model, 'model')
```

### What You Can Log

| Method | What It Logs | Example |
| --- | --- | --- |
| `log_param(key, value)` | Single hyperparameter | `log_param('n_estimators', 100)` |
| `log_params(dict)` | Multiple parameters | `log_params({'lr': 0.01, 'depth': 5})` |
| `log_metric(key, value, step)` | Single metric (optional step) | `log_metric('loss', 0.5, step=1)` |
| `log_metrics(dict)` | Multiple metrics | `log_metrics({'rmse': 1.2, 'mae': 0.8})` |
| `log_artifact(path)` | Any file | `log_artifact('confusion_matrix.png')` |
| `log_model(model, name)` | Serialized model | `mlflow.sklearn.log_model(model, 'model')` |

### Finding the Best Run (MLflow Client API)

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Search runs, order by metric
runs = client.search_runs(
    experiment_ids=['12345'],
    order_by=['metrics.rmse ASC'],  # ASC for minimize, DESC for maximize
    max_results=1
)

best_run = runs[0]
best_run_id = best_run.info.run_id
best_rmse = best_run.data.metrics['rmse']
```

> **`order_by`** accepts SQL-like syntax: `'metrics.<name> ASC'` or `'metrics.<name> DESC'`

### MLflow UI Information Available

| Section | What You Can See |
| --- | --- |
| **Experiments** | All runs, comparison charts, search/filter |
| **Run Details** | Parameters, metrics, artifacts, model, tags, git info |
| **Model artifacts** | Model signature, input example, requirements |
| **Metric history** | Step-by-step metric plots (e.g., loss over epochs) |

---

## Sub-topic 6: Model Registry in Unity Catalog

### Registering a Model

```python
import mlflow

# Set the registry to Unity Catalog
mlflow.set_registry_uri('databricks-uc')

# Register during logging
with mlflow.start_run():
    mlflow.sklearn.log_model(
        model,
        artifact_path='model',
        registered_model_name='catalog.schema.my_model'  # 3-level namespace!
    )

# Or register an existing run's model
result = mlflow.register_model(
    model_uri=f'runs:/{run_id}/model',
    name='catalog.schema.my_model'
)
```

> In Unity Catalog, model names use **3-level namespace**: `catalog.schema.model_name`

### Unity Catalog Registry vs Workspace Registry

| Property | Unity Catalog Registry | Workspace Registry (Legacy) |
| --- | --- | --- |
| **Namespace** | `catalog.schema.model` (3-level) | `model_name` (flat) |
| **Governance** | UC permissions (GRANT/REVOKE) | Workspace ACLs |
| **Cross-workspace** | Shared across workspaces | Single workspace only |
| **Lineage** | Full lineage in UC | Limited |
| **Lifecycle** | Aliases (champion/challenger) | Stages (Staging/Production) |
| **Discovery** | Searchable in UC Explorer | Workspace model registry UI |

> **Key difference**: UC Registry uses **aliases** (not stages). Workspace registry used stages like "Staging", "Production" — these are deprecated.

### Model Aliases (Champion/Challenger)

```python
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Set alias on a model version
client.set_registered_model_alias(
    name='catalog.schema.my_model',
    alias='champion',
    version=3
)

# Load model by alias
model = mlflow.pyfunc.load_model('models:/catalog.schema.my_model@champion')

# Promote challenger to champion (just reassign the alias)
client.set_registered_model_alias(
    name='catalog.schema.my_model',
    alias='champion',
    version=5  # New version becomes champion
)
```

> **Promoting a model** = reassigning the `champion` alias to a new version. The old version is NOT deleted — it just loses the alias.

### Model Tags

```python
# Set a tag on a model version
client.set_model_version_tag(
    name='catalog.schema.my_model',
    version=3,
    key='validation_status',
    value='approved'
)

# Remove a tag
client.delete_model_version_tag(
    name='catalog.schema.my_model',
    version=3,
    key='validation_status'
)
```

> Tags are key-value metadata for organizing and filtering models. They do NOT affect model behavior.

---

## Quick Reference: Common Exam Traps

1. **Feature Store client**: Use `FeatureEngineeringClient` (not `FeatureStoreClient`) for Unity Catalog
2. **Model registry namespace**: UC uses 3-level (`catalog.schema.model`), workspace uses flat name
3. **Aliases vs Stages**: UC registry uses aliases (`@champion`); stages are deprecated/workspace-only
4. **AutoML output**: Generates editable notebooks (glass-box), NOT black-box models
5. **`create_table` requires `primary_keys`** — without them it's just a Delta table
6. **`score_batch` only needs lookup keys** — features are joined automatically
7. **Online tables** are for real-time serving; offline tables are for training/batch inference
8. **`search_runs` with `order_by`** is the correct way to find the best run (not manual iteration)
9. **Promote code** = retrain in production env; **Promote models** = deploy the trained artifact
10. **ML Runtime advantage** = pre-installed, tested, compatible ML libraries (not faster Spark)
11. **`log_model`** logs the serialized model; **`register_model`** adds it to the registry
12. **`set_registered_model_alias`** promotes a model; there is no `promote_model()` method
13. **`write_table` with `mode='merge'`** performs upsert (update + insert) by primary key
14. **Feature lookups at inference** only require the key columns — NOT the feature values
