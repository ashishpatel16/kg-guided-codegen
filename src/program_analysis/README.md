# Formal definitions (Appendix)

This directory contains the core logic for program analysis, dynamic call graph construction, and algorithmic fault localization.

## Mathematical Definitions

### 1. Problem Statement
A problem instance is defined as a tuple $\mathcal{I}=\langle\mathcal{G},\mathcal{D},\mathcal{T}\rangle$:

- **$\mathcal{G}$**: The **Call Graph** (or Execution Tree $\mathcal{T}$) representing the program's computation units and their relationships.
- **$\mathcal{D}$**: The natural language issue description.
- **$\mathcal{T}$**: The verification test suite, where $\mathcal{T}=\mathcal{T}_{fail}\cup\mathcal{T}_{pass}$.

### 2. Objective
The agent must identify a target node $n^*$ in the program Call Graph $\mathcal{G}$ such that:

- **Fault Condition**: $n^*$ is an **Incorrect Procedure Instance**, meaning its output is incorrect despite receiving correct inputs.
- **Resolution Condition**: Identifying $n^*$ is equivalent to satisfying the verification suite $\mathcal{T}$, such that if the logic at $n^*$ were correct, $|\mathcal{T}_{fail}| = 0$.

### 3. Fault Localization Belief State
The belief state at iteration $k$ is defined as:
$$\mathcal{B}_k = \langle S(n), \mathcal{H}_k \rangle$$

- **$S(n)$**: The **Suspiciousness Map**, where $S: \mathcal{N} \rightarrow [0, 1]$. This represents the cumulative "blame" or probability assigned to a node.
- **$\mathcal{H}_k$**: The sequence of inspection outcomes and reflections collected up to step $k$.

### 4. Debugging Hypothesis
A **Hypothesis** $H_n$ is a proposition that a specific node $n \in \mathcal{N}$ contains the root cause of the observed failure in $\mathcal{T}_{fail}$. The agent iteratively tests hypotheses by selecting the node $n$ with the highest $S(n)$.

### 5. Inspection Evidence ($E$)
Assertions and heartbeat logs are modeled as **Inspection Evidence** $E$. When a node $n$ is instrumented and executed against $\mathcal{T}$, the evidence $E$ falls into one of several discrete categories:

- **$E_{target}$**: (Target Assertion Fail) The node was reached and an assertion inside $n$ failed.
- **$E_{pass}$**: (Healthy Pass) The node was reached and all tests passed.
- **$E_{collateral}$**: (Collateral Failure) Tests failed, but the failure occurred outside $n$.
- **$E_{null}$**: (No Coverage) The node was never reached by the test suite.

### 6. Likelihood Function ($L$)
The Likelihood Function $L(E | \text{Status})$ defines the conditional probability of observing evidence $E$ given the true state of the node (Buggy vs. Not Buggy):

| Evidence ($E$) | $P(E \mid \text{Buggy})$ | $P(E \mid \neg \text{Buggy})$ |
| :--- | :---: | :---: |
| $E_{target}$ | $0.95$ | $0.05$ |
| $E_{pass}$ | $0.10$ | $0.90$ |
| $E_{collateral}$ | $0.60$ | $0.40$ |
| $E_{null}$ | $0.50$ | $0.50$ |

### 7. Bayesian Update Rule
The belief $S(n)$ is updated using Bayes' Theorem after each inspection:

$$S_{k+1}(n) = P(\text{Buggy} \mid E) = \frac{P(E \mid \text{Buggy}) \cdot S_k(n)}{P(E \mid \text{Buggy}) \cdot S_k(n) + P(E \mid \neg \text{Buggy}) \cdot (1 - S_k(n))}$$

This formal update ensures that the suspiciousness score remains a valid probability and accounts for the strength of the evidence relative to the prior belief.

---

**Note on Initialization (Cromwell's Rule):**
$S_0(n)$ is calculated by normalizing Tarantula scores with a $0.01$ floor. This prevents any node from having a $0\%$ prior, ensuring every function remains reachable by Bayesian updates if evidence emerges.
