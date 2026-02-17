# Agent Architecture: Research-Focused Documentation

This document provides a research-oriented description of the fault localization agent. It is intended for developers and researchers who need to understand the theoretical foundations, Bayesian calculations, and design assumptions.

---

## 1. Overview & Motivation

Standard LLMs struggle with software debugging because they lack a reliable signal to converge toward the root cause of a defect. Without formal feedback, they tend to use a "hit-and-trial" approach with unpredictable results. This system introduces a **Debugging Agent** — Bayesian-guided fault localization that iteratively inspects and narrows down root causes.

The key insight is to combine **formal program analysis** (call graphs, coverage, test outcomes) with **LLM-based reasoning** (reflection, qualitative decisions) in a principled way.

---

## 2. Debugging Agent

### 2.1 High-Level Flow

```
START → initialize_debugging_scores → select_target_node → generate_inspection_patch
                                                                    ↓
                                    ←────────────────── execute_inspection
                                                                    ↓
                                    update_suspiciousness_and_reflect
                                                                    ↓
                                    [if any confidence_score >= 0.9] → END
                                    [else] → select_target_node (loop)
```

The agent repeatedly selects the most suspicious node, generates an inspection patch (assertions/validations), runs tests, and updates confidence scores via a Bayesian update. It terminates when some node's `confidence_score` exceeds `CONFIDENCE_THRESHOLD` (0.9).

### 2.2 Input Requirements

- **Call graph**: Dynamic call graph with nodes (FQN, file, start_line, end_line) and edges.
- **Suspiciousness scores**: Tarantula-based scores from test pass/fail and coverage.
- **Test infrastructure**: Ability to run tests (locally or in Docker) and capture stdout/stderr/exit code.

---

## 3. Bayesian Calculations

### 3.1 Notation

- **B**: The target node is buggy.
- **~B**: The target node is not buggy.
- **E**: The observed evidence (execution result of the inspection patch).
- **P(B)**: Prior probability that the node is buggy (`confidence_score`).
- **P(E|B)**: Likelihood of E given the node is buggy.
- **P(E|~B)**: Likelihood of E given the node is not buggy.
- **P(B|E)**: Posterior probability after observing E.

### 3.2 Bayes’ Theorem

$$
P(B|E) = \frac{P(E|B) \cdot P(B)}{P(E|B) \cdot P(B) + P(E|\neg B) \cdot P(\neg B)}
$$

In code:
```
denominator = (p_e_given_buggy * prior) + (p_e_given_not_buggy * (1 - prior))
posterior = (p_e_given_buggy * prior) / denominator
```

### 3.3 Prior: Confidence Score Initialization

Before the debugging loop, `confidence_score` is derived from Tarantula `suspiciousness`:

1. For each node: `raw_score = max(0.01, suspiciousness)`
2. Normalize so that the sum over all nodes is 1.0:
   - `confidence_score[node] = raw_score[node] / sum(raw_scores)`

So the prior for each node is a **normalized version** of Tarantula, with a small floor (0.01) to ensure non-zero probability.

### 3.4 Tarantula Formula (Suspiciousness)

Tarantula is computed from dynamic coverage and test results:

- `failed_s` = number of failed tests that executed this node  
- `passed_s` = number of passed tests that executed this node  
- `total_failed` = total number of failed tests  
- `total_passed` = total number of passed tests  

$$
\text{Tarantula}(s) = \frac{\text{failed}_s / \text{total\_failed}}{\frac{\text{passed}_s}{\text{total\_passed}} + \frac{\text{failed}_s}{\text{total\_failed}}}
$$

Nodes executed mainly by failing tests get higher scores; nodes executed mostly by passing tests get lower scores.

### 3.5 Evidence Signals

The Bayesian update uses three formal signals:

| Signal | Meaning |
|--------|---------|
| `is_covered` | The heartbeat `--- INSPECTION_START: {target_fqn} ---` appears in stdout. Indicates the target function was executed. |
| `has_target_assertion` | An `AssertionError` appears in the traceback and is located in the target file (not a callee). |
| `is_failure` | Test run did not succeed (`Exit Code: 0` not in result). |

The LLM reflection yields a binary decision:
- `CONFIRMED_BUGGY`
- `CONFIRMED_NOT_BUGGY`

### 3.6 Likelihood Tables

The likelihoods P(E|B) and P(E|~B) are set per outcome category. These values encode assumptions about how often we would see this evidence if the node were buggy vs. not buggy.

| Outcome | P(E\|B) | P(E\|~B) | Condition |
|---------|---------|----------|-----------|
| **TARGET_ASSERTION_FAILED** | 0.95 | 0.05 | Assertion failed in target; LLM agrees or neutral |
| **TARGET_ASSERTION_FAILED_BUT_LLM_DISAGREES** | 0.60 | 0.40 | Assertion failed in target; LLM says CONFIRMED_NOT_BUGGY |
| **COVERED_AND_PASSED** | 0.10 | 0.90 | Executed and tests passed |
| **COLLATERAL_FAILURE_LLM_SUSPICIOUS** | 0.70 | 0.30 | Failure elsewhere; LLM says CONFIRMED_BUGGY |
| **COLLATERAL_FAILURE_LLM_INNOCENT** | 0.40 | 0.60 | Failure elsewhere; LLM says CONFIRMED_NOT_BUGGY |
| **COLLATERAL_FAILURE** | 0.60 | 0.40 | Failure elsewhere; LLM neutral |
| **NO_COVERAGE** | 0.50 | 0.50 | Target not executed (uninformative) |
| **INCONCLUSIVE** | 0.50 | 0.50 | Fallback |

### 3.7 Posterior Clamping

The posterior is clamped to `[0.01, 0.99]` so no node becomes absolutely certain or impossible, preserving future updates.

---

## 4. Design Assumptions

### 4.1 LLM as Qualitative Guide

- The LLM is used **after** running tests to interpret execution results.
- Its role is to resolve ambiguous cases (e.g., collateral failures) and to apply the “Golden Rule of Callees.”
- Likelihoods are tuned so LLM agreement/disagreement adjusts the strength of evidence.

### 4.2 Golden Rule of Callees

> If the target function fails because a function it **calls** (a callee) returned a wrong value, but the target function’s own logic is correct and it used that value correctly, then the target function is **CONFIRMED_NOT_BUGGY**.

This prevents blaming a function for bugs in its dependencies and keeps localization focused on the actual faulty implementation.

### 4.3 Single-Bug Hypothesis

The current model treats the problem as **one primary bug** per debugging run. Multi-fault scenarios would require extensions (e.g., multiple high-confidence nodes or different stopping logic).

### 4.4 Coverage Assumptions

- Dynamic tracing is assumed to capture executed nodes reasonably well.
- Test selection for each node is done by finding callers that look like tests (`test_` prefix, `tests/` in path).
- If no such tests are found, a fallback global test command is used.

### 4.5 Inspection Patch Design

- The patch **adds** assertions and a heartbeat print; it does not remove logic.
- The heartbeat ensures we can tell if the target was executed.
- Assertions validate invariants; failures indicate violations in the target’s behavior or in its dependencies.

---

## 5. Node Selection

### 5.1 Valid Nodes

Only “inspectable” nodes are considered:

- Must have `file` and `start_line`
- Must not be compiler-generated (e.g., `<listcomp>`, `<lambda>`, `<genexpr>`)
- FQN must not contain `.<` or start with `<`

### 5.2 Selection Rule

Among valid nodes, the agent selects the one with the **maximum `confidence_score`**. This implements a greedy strategy: inspect the current best candidate first.

---

## 6. Implementation Notes

### 6.1 State

- **DebuggingState**: `call_graph`, `target_node`, `inspection_patch`, `original_source`, `execution_result`, `reflection`, `history`, `llm_calls`, and optional Docker-related fields.

### 6.2 History and Reversibility

- After each inspection run, the original source is **restored** before the next iteration.
- Patches are applied only during execution; the codebase is not permanently modified by the agent.

### 6.3 Docker Support

- Tests can run in a Docker container, with path mapping between host and container (`host_workspace`, `container_workspace`).
- This supports consistent, isolated execution environments.

---

## 7. References & Related Work

- **Tarantula**: Jones, Harrold, Stasko (2002) — spectrum-based fault localization.
- **Bayesian fault localization**: Approaches that use priors and test outcomes to update fault probability.
- **LangGraph**: Framework for building multi-step, stateful agent workflows.

---

## 8. Summary

This agent system combines:

1. **Coverage-based suspiciousness** (Tarantula) for initial priors.
2. **Bayesian updates** driven by formal execution signals and LLM interpretation.
3. **Targeted inspection** via assertion-rich patches and a heartbeat for coverage.

The Bayesian formulation provides a formal update rule that, together with the Golden Rule of Callees and the likelihood tables, guides the agent toward the most likely buggy node while avoiding common pitfalls of purely heuristic or purely LLM-based debugging.
