%md
# Section 4: Model Deployment - Complete Study Guide

---

## Sub-topic 1: Model Serving Approaches — Batch, Real-time, and Streaming

### Overview of Serving Approaches

| Approach | Latency | Throughput | Use Case |
| --- | --- | --- | --- |
| **Batch** | Minutes to hours | Very high (millions of rows) | Scheduled predictions, reports, offline recommendations |
| **Real-time** | Milliseconds | Low to moderate (per-request) | Fraud detection, chatbots, dynamic pricing |
| **Streaming** | Seconds to minutes | Moderate to high (continuous) | IoT scoring, real-time dashboards, event-driven predictions |

### Batch Inference

| Property | Description |
| --- | --- |
| **When to use** | Predictions needed on a schedule (daily, hourly), not per-request |
| **How it works** | Load model → apply to entire DataFrame → write results to table |
| **Advantages** | Simple, cost-effective, leverages Spark parallelism, no endpoint needed |
| **Disadvantages** | Not real-time; predictions may be stale by the time they are consumed |
| **Typical tools** | `mlflow.pyfunc.spark_udf()`, pandas UDF, `fe.score_batch()` |

### Real-time Inference (Model Serving)

| Property | Description |
| --- | --- |
| **When to use** | Predictions needed immediately per request (sub-second latency) |
| **How it works** | Model deployed to REST endpoint → clients send JSON requests → get JSON responses |
| **Advantages** | Low latency, always-on, supports interactive applications |
| **Disadvantages** | More expensive (always-on compute), requires endpoint management, rate limits |
| **Typical tools** | Databricks Model Serving, custom containers |

### Streaming Inference

| Property | Description |
| --- | --- |
| **When to use** | Continuous stream of data needs predictions as it arrives |
| **How it works** | Model applied as a UDF on a streaming DataFrame; results written to Delta |
| **Advantages** | Near real-time, integrates with Delta Live Tables, handles late/out-of-order data |
| **Disadvantages** | More complex than batch, requires streaming infrastructure |
| **Typical tools** | Structured Streaming + `mlflow.pyfunc.spark_udf()`, Delta Live Tables |

### Decision Framework

```
Do you need predictions within milliseconds per request?
  YES → Real-time (Model Serving endpoint)
  NO →
    Is data arriving continuously as a stream?
      YES → Streaming inference
      NO → Batch inference (scheduled job)
```

### Key Comparison Table

| Factor | Batch | Real-time | Streaming |
| --- | --- | --- | --- |
| **Latency** | High (minutes–hours) | Very low (ms) | Low (seconds) |
| **Cost model** | Pay per job run | Pay for always-on endpoint | Pay for streaming cluster |
| **Scalability** | Scales via Spark | Scales via endpoint concurrency | Scales via Spark Streaming |
| **Data freshness** | Stale until next run | Always current | Near real-time |
| **Complexity** | Low | Medium | Medium–High |
| **Feature lookup** | Direct table join or Feature Store | Online Feature Store / request payload | Feature Store + streaming join |

> **Exam tip**: Batch is simplest and cheapest for scheduled workloads. Real-time is for interactive, per-request scenarios. Streaming is for continuous data pipelines.

---

## Sub-topic 2: Deploying a Custom Model to an Endpoint

### What Is a Model Serving Endpoint?

A managed REST API that hosts your model for real-time inference. Databricks handles:
- Auto-scaling (scale to zero supported)
- Load balancing
- Model versioning
- Authentication & access control

### Steps to Deploy a Custom Model

1. **Log the model** to MLflow (with signature and input example)
2. **Register the model** in Unity Catalog
3. **Create a serving endpoint** pointing to the registered model version/alias

### Logging a Model with Signature

```python
import mlflow
from mlflow.models.signature import infer_signature

# Train your model
model.fit(X_train, y_train)
predictions = model.predict(X_train)

# Infer signature (input schema + output schema)
signature = infer_signature(X_train, predictions)

# Log with signature and input example
with mlflow.start_run():
    mlflow.sklearn.log_model(
        model,
        artifact_path='model',
        signature=signature,
        input_example=X_train[:5],  # Small sample for documentation
        registered_model_name='catalog.schema.my_model'
    )
```

> **Why signature matters**: Model Serving uses the signature to validate incoming requests and generate the endpoint's API documentation.

### Custom PyFunc Models

When your model needs custom logic (pre/post-processing, multi-model ensemble, business rules):

```python
import mlflow.pyfunc

class CustomModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        """Load artifacts (called once when model is loaded)"""
        import pickle
        with open(context.artifacts['preprocessor'], 'rb') as f:
            self.preprocessor = pickle.load(f)
        # Load underlying model
        self.model = mlflow.sklearn.load_model(context.artifacts['model_path'])

    def predict(self, context, model_input):
        """Generate predictions (called per request)"""
        processed = self.preprocessor.transform(model_input)
        predictions = self.model.predict(processed)
        return predictions

# Log custom model
with mlflow.start_run():
    mlflow.pyfunc.log_model(
        artifact_path='model',
        python_model=CustomModel(),
        artifacts={
            'preprocessor': '/path/to/preprocessor.pkl',
            'model_path': f'runs:/{run_id}/sklearn_model'
        },
        registered_model_name='catalog.schema.custom_model'
    )
```

### Creating a Serving Endpoint (SDK)

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, ServedEntityInput
)

w = WorkspaceClient()

endpoint = w.serving_endpoints.create_and_wait(
    name='my-model-endpoint',
    config=EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name='catalog.schema.my_model',  # UC model path
                entity_version='1',                      # or use scale_to_zero
                workload_size='Small',                   # Small/Medium/Large
                scale_to_zero_enabled=True               # Cost optimization
            )
        ]
    )
)
```

> **Key point**: `entity_name` uses the 3-level UC namespace. `scale_to_zero_enabled=True` means you only pay when requests arrive.

---

## Sub-topic 3: Batch Inference with Pandas

### Using `mlflow.pyfunc.spark_udf()` for Distributed Batch Inference

```python
import mlflow

# Load model as a Spark UDF (distributes inference across cluster)
model_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri='models:/catalog.schema.my_model@champion'
)

# Apply to Spark DataFrame
predictions_df = spark_df.withColumn(
    'prediction',
    model_udf(*feature_columns)  # Pass feature columns as arguments
)

# Write results
predictions_df.write.mode('overwrite').saveAsTable('catalog.schema.predictions')
```

### Using Pandas for Single-Node Batch Inference

```python
import mlflow
import pandas as pd

# Load model as a generic Python function
model = mlflow.pyfunc.load_model('models:/catalog.schema.my_model@champion')

# Load data as pandas DataFrame
pdf = spark.table('catalog.schema.scoring_data').toPandas()

# Predict
pdf['prediction'] = model.predict(pdf[feature_columns])

# Convert back to Spark and save
result_df = spark.createDataFrame(pdf)
result_df.write.mode('overwrite').saveAsTable('catalog.schema.predictions')
```

### Using Feature Store for Batch Scoring

```python
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# Only pass the lookup keys — features are joined automatically
predictions = fe.score_batch(
    model_uri='models:/catalog.schema.my_model@champion',
    df=scoring_df  # DataFrame with primary key columns only
)
```

### When to Use Each Approach

| Method | When to Use |
| --- | --- |
| `spark_udf()` | Large datasets, distributed cluster available, Spark-native pipeline |
| Pandas `model.predict()` | Small datasets, single-node, quick testing |
| `fe.score_batch()` | Model was trained with Feature Store lookups |

> **Exam tip**: `mlflow.pyfunc.spark_udf()` is the standard way to perform scalable batch inference on Spark DataFrames. For Feature Store models, always use `score_batch()` — it handles feature joins automatically.

---

## Sub-topic 4: Streaming Inference with Delta Live Tables

### How Streaming Inference Works

Streaming inference applies a model to data as it arrives — rather than waiting for a batch job to run. In Databricks, this is typically done by:

1. Reading a **streaming source** (Delta table, Kafka, Auto Loader)
2. Applying the model as a **UDF** on the streaming DataFrame
3. Writing results to a **streaming sink** (Delta table)

### Streaming Inference with Structured Streaming

```python
import mlflow

# Load model as Spark UDF
model_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri='models:/catalog.schema.my_model@champion'
)

# Read streaming source
streaming_df = spark.readStream.table('catalog.schema.incoming_events')

# Apply model
scored_df = streaming_df.withColumn(
    'prediction',
    model_udf(*feature_columns)
)

# Write to streaming sink
scored_df.writeStream \
    .outputMode('append') \
    .option('checkpointLocation', '/checkpoints/scoring') \
    .toTable('catalog.schema.scored_events')
```

### Streaming Inference with Delta Live Tables (DLT)

In DLT, streaming inference is performed by defining a **streaming table** that applies a model UDF:

```python
import dlt
import mlflow

model_udf = mlflow.pyfunc.spark_udf(
    spark,
    model_uri='models:/catalog.schema.my_model@champion'
)

@dlt.table(
    comment='Scored events with real-time predictions'
)
def scored_events():
    return (
        dlt.read_stream('raw_events')
        .withColumn('prediction', model_udf(*feature_columns))
    )
```

### Benefits of DLT for Streaming Inference

| Benefit | Description |
| --- | --- |
| **Declarative** | Define WHAT, not HOW — DLT manages the streaming mechanics |
| **Built-in monitoring** | Data quality expectations, pipeline health |
| **Automatic checkpointing** | No manual checkpoint management |
| **Lineage** | Full data lineage from source → prediction table |
| **Error handling** | Auto-retry, dead-letter queues, expectations for bad records |

### Streaming vs Batch vs Real-time for Inference

| Question | Batch | Streaming | Real-time |
| --- | --- | --- | --- |
| Is the data already in a table? | Yes | No (continuous) | No (per request) |
| Are predictions consumed by another pipeline? | Often | Yes | Rarely |
| Latency acceptable | Minutes–hours | Seconds | Milliseconds |
| Who triggers it? | Scheduler | Data arrival | Client request |

> **Key insight**: DLT simplifies streaming inference by managing state, checkpoints, and retries. You only define the transformation logic (including the model UDF).

---

## Sub-topic 5: Deploying and Querying a Model for Real-time Inference

### End-to-End Deployment Flow

```
1. Log model to MLflow (with signature)
2. Register model in Unity Catalog
3. Set alias (e.g., @champion)
4. Create serving endpoint
5. Query endpoint via REST API
```

### Querying a Serving Endpoint

#### Using the Databricks SDK (Python)

```python
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

response = w.serving_endpoints.query(
    name='my-model-endpoint',
    dataframe_records=[
        {'feature1': 1.5, 'feature2': 'A', 'feature3': 100},
        {'feature1': 2.3, 'feature2': 'B', 'feature3': 200}
    ]
)

print(response.predictions)
```

#### Using REST API (curl)

```bash
curl -X POST \
  https://<workspace-url>/serving-endpoints/my-model-endpoint/invocations \
  -H 'Authorization: Bearer <token>' \
  -H 'Content-Type: application/json' \
  -d '{
    "dataframe_records": [
      {"feature1": 1.5, "feature2": "A", "feature3": 100}
    ]
  }'
```

#### Using REST API (Python requests)

```python
import requests
import json

url = f'https://<workspace-url>/serving-endpoints/my-model-endpoint/invocations'
headers = {
    'Authorization': f'Bearer {token}',
    'Content-Type': 'application/json'
}
payload = {
    'dataframe_records': [
        {'feature1': 1.5, 'feature2': 'A', 'feature3': 100}
    ]
}

response = requests.post(url, headers=headers, json=payload)
predictions = response.json()['predictions']
```

### Request/Response Formats

| Format | Request Key | Example |
| --- | --- | --- |
| DataFrame records | `dataframe_records` | `[{"col1": val1, "col2": val2}, ...]` |
| DataFrame split | `dataframe_split` | `{"columns": [...], "data": [[...], ...]}` |
| Instances | `instances` | `[[val1, val2], [val3, val4]]` |
| Inputs | `inputs` | `{"col1": [v1, v2], "col2": [v3, v4]}` |

> **Most common format**: `dataframe_records` — a list of dictionaries where each dict is one row.

### Endpoint Configuration Options

| Setting | Description |
| --- | --- |
| `workload_size` | Small / Medium / Large (determines compute) |
| `scale_to_zero_enabled` | If True, endpoint scales down when idle (cost saving) |
| `workload_type` | CPU (default) or GPU |
| `entity_version` | Specific model version number |
| `entity_name` | UC model path (`catalog.schema.model`) |

---

## Sub-topic 6: Traffic Splitting Between Endpoints (A/B Testing)

### What Is Traffic Splitting?

Traffic splitting routes a percentage of incoming requests to different model versions. This enables:
- **A/B testing** — compare a new model against the current champion
- **Canary deployments** — gradually roll out a new model
- **Shadow mode** — send traffic to both but only return champion's response

### Configuring Traffic Split

```python
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, ServedEntityInput, TrafficConfig, Route
)

w = WorkspaceClient()

# Update endpoint with traffic split
w.serving_endpoints.update_config_and_wait(
    name='my-model-endpoint',
    served_entities=[
        ServedEntityInput(
            name='champion',
            entity_name='catalog.schema.my_model',
            entity_version='3',
            workload_size='Small',
            scale_to_zero_enabled=True
        ),
        ServedEntityInput(
            name='challenger',
            entity_name='catalog.schema.my_model',
            entity_version='5',
            workload_size='Small',
            scale_to_zero_enabled=True
        )
    ],
    traffic_config=TrafficConfig(
        routes=[
            Route(served_model_name='champion', traffic_percentage=90),
            Route(served_model_name='challenger', traffic_percentage=10)
        ]
    )
)
```

### Traffic Split Strategies

| Strategy | Split Example | Purpose |
| --- | --- | --- |
| **Canary** | 95% champion / 5% challenger | Safe rollout, detect issues early |
| **A/B Test** | 50% / 50% | Statistical comparison of two models |
| **Gradual Rollout** | 90→70→50→0 champion | Progressive confidence building |
| **Blue/Green** | 100% → 0% (instant switch) | Zero-downtime deployment |

### Key Points

- Traffic percentages **must sum to 100%**
- Each served entity has its own `name` for routing
- You can have **multiple model versions** on the same endpoint
- Traffic split is configured at the **endpoint level**, not the model level
- Requests are routed randomly based on percentage — the client does NOT choose

> **Exam tip**: Traffic splitting is the mechanism for A/B testing models in production. The endpoint handles routing — clients send requests to a single URL and are unaware of the split.

---

## Quick Reference: Common Exam Traps

1. **`spark_udf()` for batch inference** — the standard way to distribute model predictions across a Spark cluster
2. **Batch inference does NOT require a serving endpoint** — you load the model and apply it directly
3. **`score_batch()` only needs key columns** — features are joined automatically from Feature Store
4. **Real-time requires a serving endpoint** — there is no way around this for sub-second latency
5. **Streaming inference uses `spark_udf()` on a streaming DataFrame** — same UDF, different read mode
6. **DLT handles checkpointing automatically** — you don't manage streaming state manually
7. **Traffic split percentages must sum to 100%** — Databricks enforces this
8. **`scale_to_zero_enabled=True`** means zero cost when idle but cold-start latency on first request
9. **`dataframe_records`** is the most common request format for serving endpoints
10. **Signature is required** for Model Serving to validate inputs — always use `infer_signature()`
11. **Custom PyFunc models** need `load_context()` and `predict()` methods
12. **A/B testing = traffic split** — clients don't know which model they hit; the endpoint routes randomly
13. **Streaming inference latency** is seconds (not milliseconds like real-time serving)
14. **Batch is cheapest** — no always-on infrastructure; just a scheduled job
