# Lilac + Eval Protocol: Diverse Dataset Curation

Use [Lilac](https://lilacml.com/) to automatically cluster and sample diverse examples from your LLM traces for evaluation.

**This is an example showing how to use Eval Protocol's utilities with Lilac for intelligent data curation.**

## What This Does

When evaluating LLMs, running on *all* your production traces is expensive and often redundant—many queries are semantically similar. This integration:

1. **Pulls traces** from Langfuse (or any supported observability platform)
2. **Clusters** them semantically using embeddings + HDBSCAN
3. **Samples** diverse examples from each cluster
4. **Evaluates** the representative subset

**Result:** Instead of evaluating 1000 similar traces, you evaluate 30 diverse ones that cover all query types.

```
100 traces → Lilac clustering → 6 semantic groups → 12 diverse samples
```

## Quick Start

### 1. Setup

```bash
# Clone this repo
git clone <repo-url>
cd lilac-eval-example

# Run setup script (creates venv and installs everything)
./setup.sh
```

### 2. Configure API Keys

```bash
cp env.template .env
# Edit .env with your keys
```

### 3. Run

```bash
source .venv/bin/activate
pytest test_lilac_preprocessing.py -v -s
```

---

## How Lilac Integration Works

### The Integration Point

The key is the `preprocess_fn` parameter in `DynamicDataLoader`. This function receives ALL loaded rows and returns a filtered/transformed subset:

```python
@evaluation_test(
    data_loaders=DynamicDataLoader(
        generators=[langfuse_traces_generator],
        preprocess_fn=lilac_cluster_and_sample,  # ← Your Lilac logic here!
    ),
    ...
)
def test_my_evaluation(row: EvaluationRow) -> EvaluationRow:
    return evaluate(row)
```

### The Preprocessing Pipeline

```python
def lilac_cluster_and_sample(rows: List[EvaluationRow]) -> List[EvaluationRow]:
    """
    1. Convert to DataFrame (for Lilac compatibility)
    2. Create Lilac dataset
    3. Cluster on user queries
    4. Sample from each cluster
    5. Convert back to EvaluationRows
    """
    import lilac as ll
    
    # Step 1: Convert to DataFrame using eval-protocol utility
    df = evaluation_rows_to_dataframe(rows)
    df["user_query"] = df["messages_json"].apply(extract_first_user_message)
    
    # Step 2: Create Lilac dataset
    config = ll.DatasetConfig(
        namespace="local",
        name="my_dataset",
        source=ll.PandasSource(df),
    )
    dataset = ll.create_dataset(config)
    
    # Step 3: Cluster (Lilac handles embedding + UMAP + HDBSCAN)
    dataset.cluster("user_query")
    
    # Step 4: Sample diverse examples from each cluster
    df = dataset.to_pandas(include_signals=True)
    # ... sampling logic per cluster ...
    
    # Step 5: Convert back using eval-protocol utility
    return dataframe_to_evaluation_rows(df)
```

### Clustering Pipeline (What Lilac Does)

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│ User Queries │ ──▶ │  Embed with  │ ──▶ │    UMAP     │ ──▶ │   HDBSCAN    │
│  (text)      │     │ Transformers │     │ (dim reduce)│     │ (clustering) │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
                                                                    │
                                                                    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────────────────────┐
│   Output:   │ ◀── │ Sample N per │ ◀── │ Clusters with auto-generated   │
│ Diverse Set │     │   cluster    │     │ titles (via LLM, optional)     │
└─────────────┘     └──────────────┘     └─────────────────────────────────┘
```

1. **Embeds** each user query using sentence transformers (`jina-embeddings-v2-small-en`)
2. **Reduces dimensions** with UMAP (512 → 5 dimensions)
3. **Clusters** with HDBSCAN (automatically determines cluster count)
4. **Names clusters** using an LLM (optional, requires `API_MODEL` env var)
5. **Samples** N examples from each cluster for diversity

### Example Output

```
============================================================
🌸 LILAC PREPROCESSING
============================================================
📥 Input: 100 rows

🧮 Clustering user queries...
   Method: Embed → UMAP → HDBSCAN
   Cluster naming: LLM (gpt-4o-mini)

📊 Found 6 clusters:
--------------------------------------------------
   Cluster 0 "Account Management Requests": 14 items
      e.g., "Update phone number on account"
   Cluster 1 "Order Returns and Refunds": 26 items
      e.g., "ORD-54656 shipping status?"
   Cluster 2 "Customer Service Inquiries": 17 items
      e.g., "Recovery options change"

✅ Output: 12 diverse samples
   Strategy: 2 per cluster, max 30 total
============================================================
```

---

## Eval Protocol Utilities Used

This example uses several Eval Protocol utilities that enable the Lilac integration:

### 1. DataFrame Conversion - Bridge Between EvaluationRows and Pandas

```python
from eval_protocol.adapters.lilac import (
    evaluation_rows_to_dataframe,
    dataframe_to_evaluation_rows,
)

# Convert EvaluationRows → DataFrame (for Lilac/pandas processing)
df = evaluation_rows_to_dataframe(rows)

# ... do clustering, filtering, transformations with pandas/Lilac ...

# Convert DataFrame → EvaluationRows (back to eval-protocol format)
filtered_rows = dataframe_to_evaluation_rows(df)
```

### 2. Trace Adapters - Pull Data from Observability Platforms

```python
from eval_protocol import create_langfuse_adapter

# Create adapter for your platform
adapter = create_langfuse_adapter()

# Pull traces and convert to EvaluationRows
rows = adapter.get_evaluation_rows(
    limit=100,              # How many traces
    hours_back=168,         # Time window (7 days)
    include_tool_calls=True # Include function calls
)
```

**Supported platforms:**
- `create_langfuse_adapter()` - Langfuse
- `create_langsmith_adapter()` - LangSmith  
- `create_braintrust_adapter()` - Braintrust
- `create_fireworks_tracing_adapter()` - Fireworks Tracing

### 3. DynamicDataLoader - Flexible Data Loading with Preprocessing

```python
from eval_protocol import DynamicDataLoader

data_loader = DynamicDataLoader(
    generators=[my_data_generator],     # Functions that return EvaluationRows
    preprocess_fn=my_preprocess_fn,     # Transform rows before evaluation
)
```

---

## Configuration

Edit these constants in `test_lilac_preprocessing.py`:

```python
SAMPLES_PER_CLUSTER = 2   # How many samples from each cluster
MAX_TOTAL_SAMPLES = 30    # Cap on total output rows
LANGFUSE_LIMIT = 100      # How many traces to pull from Langfuse
```

---

## Bring Your Own Data

### From Different Trace Sources

```python
# Langfuse
adapter = create_langfuse_adapter()
rows = adapter.get_evaluation_rows(limit=100)

# LangSmith
adapter = create_langsmith_adapter()
rows = adapter.get_evaluation_rows(limit=100)

# From a JSONL file
def load_from_file():
    with open("traces.jsonl") as f:
        return [EvaluationRow.from_dict(json.loads(line)) for line in f]
```

### Custom Preprocessing (Without Lilac)

You can write any preprocessing logic:

```python
def my_custom_preprocess(rows: List[EvaluationRow]) -> List[EvaluationRow]:
    # Filter by length
    rows = [r for r in rows if len(r.last_user_message().content) > 10]
    
    # Deduplicate
    seen = set()
    unique = []
    for r in rows:
        key = r.last_user_message().content[:100]
        if key not in seen:
            seen.add(key)
            unique.append(r)
    
    # Random sample
    return random.sample(unique, min(50, len(unique)))
```

---

## Advanced Lilac Features

Beyond clustering, Lilac offers:

### Semantic Search
```python
dataset.search("user_query", "password reset", limit=10)
```

### Concept Detection
```python
from lilac.signals import PIISignal
dataset.compute_signal(PIISignal(), "user_query")
```

### Lilac Web UI
```python
import lilac as ll
ll.start_server()  # Interactive UI at localhost:5432
```

📚 **Full Lilac Documentation:** [https://docs.lilacml.com/](https://docs.lilacml.com/)

---

## API Keys Reference

**Required:**
```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...      # Langfuse public key
LANGFUSE_SECRET_KEY=sk-lf-...      # Langfuse secret key
LANGFUSE_HOST=https://cloud.langfuse.com
FIREWORKS_API_KEY=fw_...           # For model evaluation
```

**Optional (for LLM cluster naming):**
```bash
OPENAI_API_KEY=sk-...              # OpenAI API key
API_MODEL=gpt-4o-mini              # Model for naming clusters
```

---

## Troubleshooting

### Test skipped with "No Langfuse credentials"
Ensure `.env` has valid `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`.

### "you must provide a model parameter"
Set `API_MODEL` for LLM cluster naming, or it falls back to generic names.

### "HDBSCAN: X noise points"
Normal! Uncertain points are assigned to nearest cluster automatically.

### Slow first run
First run downloads embedding model (~400MB). Subsequent runs use cache.

---

## Requirements

- Python 3.10+
- ~2GB disk space for embedding model cache
- API keys for trace source + evaluation model

## License

MIT
