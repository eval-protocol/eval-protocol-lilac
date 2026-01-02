"""
Lilac + Eval Protocol: Diverse Dataset Curation

This test demonstrates using Lilac to:
1. Pull production traces from Langfuse
2. Cluster them semantically using HDBSCAN
3. Sample diverse examples from each cluster
4. Evaluate the representative subset using LLM-as-judge

Run with:
    pytest test_lilac_preprocessing.py -v -s

Prerequisites:
    pip install 'eval-protocol[lilac,langfuse]'

Environment variables:
    Required:
    - LANGFUSE_PUBLIC_KEY
    - LANGFUSE_SECRET_KEY
    - FIREWORKS_API_KEY

    Optional (for LLM-based cluster naming):
    - OPENAI_API_KEY (or use FIREWORKS_API_KEY with OPENAI_API_BASE)
    - API_MODEL (e.g., 'gpt-4o-mini')
    - OPENAI_API_BASE (for non-OpenAI providers)
    
    Optional (for LLM judge):
    - JUDGE_MODEL (default: fireworks_ai/accounts/fireworks/models/llama-v3p1-70b-instruct)
"""

import json
import os
from pathlib import Path
from typing import List

import pandas as pd
import pytest
from dotenv import load_dotenv

# Load .env file if present
load_dotenv()

from eval_protocol import (
    DynamicDataLoader,
    EvaluateResult,
    EvaluationRow,
    MetricResult,
    SingleTurnRolloutProcessor,
    create_langfuse_adapter,
    evaluation_test,
)
from eval_protocol.adapters.lilac import (
    dataframe_to_evaluation_rows,
    evaluation_rows_to_dataframe,
)


# =============================================================================
# Configuration - Adjust these for your use case
# =============================================================================

SAMPLES_PER_CLUSTER = 2   # How many samples to take from each cluster
MAX_TOTAL_SAMPLES = 30    # Cap on total output rows
LANGFUSE_LIMIT = 100      # How many traces to pull from Langfuse


# =============================================================================
# Data Generator - Pull traces from Langfuse
# =============================================================================


def langfuse_traces_generator():
    """
    Pull recent traces from Langfuse.
    
    Customize this to filter by tags, user_id, session, etc.
    See: https://langfuse.com/docs/query-traces
    """
    adapter = create_langfuse_adapter()
    return adapter.get_evaluation_rows(
        limit=LANGFUSE_LIMIT,
        hours_back=2160,  # Last 90 days
        include_tool_calls=True,
        sleep_between_gets=0.2,
    )


# =============================================================================
# Helper: Extract text for clustering
# =============================================================================


def extract_first_user_message(messages_json: str) -> str:
    """Extract the first user message from a conversation for clustering."""
    try:
        messages = json.loads(messages_json)
        for msg in messages:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content
                elif isinstance(content, list):
                    # Handle multi-part content (e.g., text + images)
                    return " ".join(
                        p.get("text", "") if isinstance(p, dict) else str(p)
                        for p in content
                    )
        return ""
    except (json.JSONDecodeError, TypeError):
        return ""


# =============================================================================
# Lilac Preprocessing Function (THE KEY INTEGRATION)
# =============================================================================


def lilac_cluster_and_sample(rows: List[EvaluationRow]) -> List[EvaluationRow]:
    """
    Use Lilac to cluster user queries and sample diverse examples.
    
    This is the preprocessing function that gets passed to DynamicDataLoader.
    
    Pipeline:
    1. Convert EvaluationRows to DataFrame
    2. Create Lilac dataset
    3. Cluster on user_query field
    4. Sample N examples from each cluster
    5. Convert back to EvaluationRows
    
    Args:
        rows: List of EvaluationRow objects from the data generator
        
    Returns:
        Filtered list of diverse EvaluationRow objects
    """
    import lilac as ll
    
    if not rows:
        return rows
    
    print(f"\n{'='*60}")
    print(f"🌸 LILAC PREPROCESSING")
    print(f"{'='*60}")
    print(f"📥 Input: {len(rows)} rows")
    
    # Set up Lilac project directory
    project_dir = os.path.expanduser("~/lilac_eval_project")
    ll.set_project_dir(project_dir)
    
    # Convert EvaluationRows to DataFrame
    df = evaluation_rows_to_dataframe(rows)
    df["user_query"] = df["messages_json"].apply(extract_first_user_message)
    
    # Clean up any existing dataset
    try:
        ll.get_dataset("local", "langfuse_traces_temp").delete()
    except Exception:
        pass
    
    # Create Lilac dataset
    config = ll.DatasetConfig(
        namespace="local",
        name="langfuse_traces_temp",
        source=ll.PandasSource(df),
    )
    dataset = ll.create_dataset(config)
    
    # Check if LLM naming is available
    has_api_key = os.environ.get("OPENAI_API_KEY") is not None
    has_api_model = os.environ.get("API_MODEL") is not None
    use_llm_naming = has_api_key and has_api_model
    
    print(f"\n🧮 Clustering user queries...")
    print(f"   Method: Embed → UMAP → HDBSCAN")
    
    if use_llm_naming:
        print(f"   Cluster naming: LLM ({os.environ.get('API_MODEL')})")
        dataset.cluster("user_query")
    else:
        print(f"   Cluster naming: Generic (set API_MODEL for LLM naming)")
        dataset.cluster(
            "user_query",
            topic_fn=lambda docs: "Untitled",
            category_fn=lambda titles: "General",
        )
    
    # Get DataFrame with cluster signals
    df = dataset.to_pandas(include_signals=True)
    
    # Extract cluster info from nested column
    cluster_col = "user_query__cluster"
    if cluster_col in df.columns:
        df["cluster_id"] = df[cluster_col].apply(
            lambda x: x.get("cluster_id") if isinstance(x, dict) else None
        )
        df["cluster_title"] = df[cluster_col].apply(
            lambda x: x.get("cluster_title") if isinstance(x, dict) else None
        )
    
    # Sample from clusters
    if "cluster_id" in df.columns:
        cluster_ids = sorted(df["cluster_id"].dropna().unique())
        
        print(f"\n📊 Found {len(cluster_ids)} clusters:")
        print("-" * 50)
        
        for cid in cluster_ids:
            cluster_df = df[df["cluster_id"] == cid]
            title = cluster_df["cluster_title"].iloc[0] if "cluster_title" in cluster_df.columns else None
            title_str = f'"{title}"' if title and title != "Untitled" else ""
            example = cluster_df["user_query"].iloc[0]
            example = example[:50] + "..." if len(example) > 50 else example
            print(f"   Cluster {int(cid)} {title_str}: {len(cluster_df)} items")
            print(f"      e.g., \"{example}\"")
        
        # Sample from each cluster
        sampled_dfs = []
        for cluster_id in cluster_ids:
            cluster_df = df[df["cluster_id"] == cluster_id]
            n = min(SAMPLES_PER_CLUSTER, len(cluster_df))
            sampled_dfs.append(cluster_df.sample(n=n, random_state=42))
        
        df = pd.concat(sampled_dfs, ignore_index=True)
        
        # Apply max cap
        if len(df) > MAX_TOTAL_SAMPLES:
            print(f"\n⚠️  Capping from {len(df)} to {MAX_TOTAL_SAMPLES}")
            df = df.sample(n=MAX_TOTAL_SAMPLES, random_state=42)
    
    # Convert back to EvaluationRows
    result_rows = dataframe_to_evaluation_rows(df)
    
    print(f"\n✅ Output: {len(result_rows)} diverse samples")
    print(f"   Strategy: {SAMPLES_PER_CLUSTER} per cluster, max {MAX_TOTAL_SAMPLES} total")
    print(f"{'='*60}\n")
    
    # NOTE: Dataset is kept for visualization in Lilac UI
    # Run: lilac start ~/lilac_eval_project --port 5433
    # Then open http://localhost:5433 to explore clusters
    
    return result_rows


# =============================================================================
# Evaluation Function - LLM as Judge
# =============================================================================

# Judge model configuration
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "fireworks_ai/accounts/fireworks/models/deepseek-v3p2")

JUDGE_PROMPT = """You are evaluating the quality of an AI assistant's response to a user query.

## User Query
{user_query}

## Assistant Response
{assistant_response}

## Evaluation Criteria
Rate the response on a scale of 1-5:
- 5: Excellent - Fully addresses the query, accurate, helpful, well-structured
- 4: Good - Addresses the query well with minor issues
- 3: Acceptable - Partially addresses the query, some gaps
- 2: Poor - Misses key aspects, unhelpful, or confusing
- 1: Very Poor - Completely off-topic, incorrect, or harmful

## Your Response
Respond in this exact format:
SCORE: [1-5]
REASON: [One sentence explaining your score]
"""


def llm_judge_evaluate(row: EvaluationRow) -> EvaluationRow:
    """
    Use an LLM to judge the quality of the assistant's response.
    
    This evaluator:
    1. Extracts the user query and assistant response from the trace
    2. Asks a judge LLM to rate the response quality (1-5)
    3. Returns the score and reasoning
    """
    import litellm
    
    # Get user query (last user message)
    user_messages = row.get_user_messages()
    user_msg = user_messages[-1] if user_messages else None
    user_query = ""
    if user_msg:
        content = user_msg.content or ""
        if isinstance(content, list):
            user_query = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in content
            )
        else:
            user_query = content
    
    # Get assistant response
    assistant_msg = row.last_assistant_message()
    assistant_response = ""
    if assistant_msg:
        content = assistant_msg.content or ""
        if isinstance(content, list):
            assistant_response = " ".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in content
            )
        else:
            assistant_response = content
        
        # Include tool calls if any
        if assistant_msg.tool_calls:
            tool_calls_str = "\n".join(
                f"- Called tool: {tc.function.name}({tc.function.arguments})"
                for tc in assistant_msg.tool_calls
            )
            assistant_response += f"\n\n[Tool Calls]\n{tool_calls_str}"
    
    # Handle missing data
    if not user_query:
        row.evaluation_result = EvaluateResult(
            score=0.0,
            is_score_valid=False,
            reason="No user query found in trace",
            metrics={},
        )
        return row
    
    if not assistant_response:
        row.evaluation_result = EvaluateResult(
            score=0.0,
            is_score_valid=False,
            reason="No assistant response found in trace",
            metrics={},
        )
        return row
    
    # Call judge LLM
    try:
        judge_response = litellm.completion(
            model=JUDGE_MODEL,
            messages=[{
                "role": "user",
                "content": JUDGE_PROMPT.format(
                    user_query=user_query[:2000],  # Truncate if too long
                    assistant_response=assistant_response[:2000],
                )
            }],
            temperature=0.0,
            max_tokens=200,
        )
        
        judge_text = judge_response.choices[0].message.content or ""
        
        # Parse score and reason
        score = 3.0  # Default
        reason = "Could not parse judge response"
        
        for line in judge_text.strip().split("\n"):
            if line.startswith("SCORE:"):
                try:
                    score = float(line.replace("SCORE:", "").strip())
                    score = max(1.0, min(5.0, score))  # Clamp to 1-5
                except ValueError:
                    pass
            elif line.startswith("REASON:"):
                reason = line.replace("REASON:", "").strip()
        
        # Normalize score to 0-1 range
        normalized_score = (score - 1) / 4.0
        
        row.evaluation_result = EvaluateResult(
            score=normalized_score,
            is_score_valid=True,
            reason=f"[{score}/5] {reason}",
            metrics={
                "quality": MetricResult(
                    score=normalized_score,
                    is_score_valid=True,
                    reason=f"LLM judge score: {score}/5",
                ),
            },
        )
        
    except Exception as e:
        row.evaluation_result = EvaluateResult(
            score=0.0,
            is_score_valid=False,
            reason=f"Judge LLM error: {str(e)[:100]}",
            metrics={},
        )
    
    return row


# =============================================================================
# THE TEST - Lilac preprocessing via DynamicDataLoader.preprocess_fn
# =============================================================================


@pytest.mark.skipif(
    not os.environ.get("LANGFUSE_PUBLIC_KEY"),
    reason="Requires LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY"
)
@evaluation_test(
    data_loaders=DynamicDataLoader(
        generators=[langfuse_traces_generator],
        preprocess_fn=lilac_cluster_and_sample,  # ← LILAC INTEGRATION
    ),
    rollout_processor=SingleTurnRolloutProcessor(),
    completion_params=[{
        "model": "fireworks_ai/accounts/fireworks/models/deepseek-v3p2",
        "temperature": 0.0,
        "max_tokens": 1000,
    }],
    mode="pointwise",
    max_concurrent_rollouts=10,
)
def test_diverse_langfuse_traces(row: EvaluationRow) -> EvaluationRow:
    """
    Evaluate a diverse sample of Langfuse traces using LLM-as-judge.
    
    This test:
    1. Pulls traces from Langfuse (via langfuse_traces_generator)
    2. Clusters and samples with Lilac (via preprocess_fn)
    3. Runs each through the model (via rollout_processor)
    4. Uses an LLM judge to evaluate response quality (this function)
    """
    return llm_judge_evaluate(row)

