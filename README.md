# KG-Guided Codegen: Dynamic Analysis Pipeline

This pipeline traces Python execution to build dynamic call graphs and uses the **Tarantula** fault localization metric to identify suspicious code.

## Usage Guide

### 1. Run Dynamic Tracing
Executes tests and captures call relationships, execution counts, and arguments.
```bash
uv run -m src.program_analysis.dynamic_call_graph
```
*Output: `artifacts/demo_call_graph_dynamic_with_suspiciousness.json`*

### 2. Generate Visualization
Converts the captured data into an interactive HTML dashboard.
```bash
uv run python src/program_analysis/visualize_call_graph.py
```
*Output: `artifacts/call_graph_visualization.html`*

---

## Self-Correcting Code Generation Agent

A LangGraph-based agent that generates code using a local Ollama model with a multi-stage reasoning loop.

### How it Works
The agent iterates through:
`Hypothesis` → `Code Generation` → `Evidence Collection` → `Evaluation` → `Reflection` (if confidence < 7/10).

### Usage
1. **Start Ollama**: Ensure `ollama serve` is running.
2. **Run Agent**:
```bash
uv run python src/main.py
```

### Configuration
- **Model**: Change the model in `src/agent/tools.py` (defaults to `gemma3:12b`).
- **Prompting**: System instructions are located in `src/agent/prompts.py`.

---

## Tarantula Metric

Tarantula is a statistical fault localization technique that assigns a suspiciousness score to each function based on its execution frequency in passing and failing tests.

### Mathematical Definition

The suspiciousness of a node $n$ is calculated as:

$$
Suspiciousness(n) = \frac{\frac{failed(n)}{total\_failed}}{\frac{passed(n)}{total\_passed} + \frac{failed(n)}{total\_failed}}
$$

**Where:**
- $failed(n)$: Number of failing tests that executed node $n$.
- $passed(n)$: Number of passing tests that executed node $n$.
- $total\_failed$: Total number of failing tests in the suite.
- $total\_passed$: Total number of passing tests in the suite.

### Interpretation
- **1.0**: High suspiciousness (node executed exclusively by failing tests).
- **0.5**: Neutral (node executed proportionally by both passing and failing tests).
- **0.0**: Low suspiciousness (node executed exclusively by passing tests).
