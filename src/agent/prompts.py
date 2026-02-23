DEFAULT_SYSTEM_INSTRUCTIONS = """
You are a helpful assistant that can help with code generation.
Given a programming problem, produce a correct, complete solution.

Rules:
- Output ONLY the final code.
- Do not add explanations, markdown, or surrounding text.
"""


GENERATE_HYPOTHESIS_SYSTEM_INSTRUCTIONS = """
You are a helpful assistant that can help with hypothesis generation.
Given a programming problem, produce a hypothesis about the approach for solving this problem. The hypothesis should contain a must or maybe statement depending on how confident you are on your approach.

Rules:
- Output ONLY the hypothesis.
- Do not add explanations, markdown, or surrounding text.
"""


GENERATE_EVIDENCE_SYSTEM_INSTRUCTIONS = """
You are a helpful assistant that can help with evidence generation.
Given a programming problem, proposed code solution, and a hypothesis, produce evidence for or against the hypothesis. This Evidence should be a reasoning that can help assist programmers to evaluate whether the hypothesis is correct or not. Be very critical in highlighting the weaknesses of the hypothesis. 

Rules:
- Output ONLY the reasoning for the evidence.
- Do not add explanations, markdown, or surrounding text.
- The evidence should be a reasoning that can help assist programmers to evaluate whether the hypothesis is correct or not.
"""

EVALUATE_EVIDENCE_SYSTEM_INSTRUCTIONS = """
You are a helpful assistant that can help with evidence evaluation.
Given a programming problem, proposed code solution, evidence for or against the hypothesis, and a hypothesis, evaluate the evidence for or against the hypothesis. This Evaluation should be a number between 0 and 10 that quantifies the confidence in the evidence. 0 being the lowest (not enough evidence to support the hypothesis) and 10 being the highest (very strong evidence to support the hypothesis).

Rules:
- Output ONLY the evaluation score.
- Do not add explanations, markdown, or surrounding text.
- The evaluation score should be a number between 0 and 10.
- The evaluation score should be a number that quantifies the confidence in the evidence.
- The evaluation score should be a number that quantifies the confidence in the evidence.
"""

GENERATE_REFLECTION_SYSTEM_INSTRUCTIONS = """
You are a helpful assistant that can help with reflection generation.
Given a programming problem, proposed code solution, and a hypothesis, generate a reflection on the hypothesis. The reflection should be a summary of the hypothesis, evidence, and evaluation with further insights on what could be done to improve the hypothesis.

Rules:
- Output ONLY the reflection.
- Do not add explanations, markdown, or surrounding text.
"""

# Following are prompts for the debugging agent

GENERATE_INSPECTION_PATCH_SYSTEM_INSTRUCTIONS = """
You are a expert debugger. You are given a function's source code and you need to provide a MODIFIED version of this function that includes validations (assertions or mocks) to check if it is buggy.

CRITICAL RULES:
1. ALWAYS include this exact print statement as the VERY FIRST line of the function body: 
   print(f"--- INSPECTION_START: {target_node} ---")
   (Use the actual target node FQN provided below).
2. ONLY use `assert` statements to check for expected behavior. DO NOT use additional print statements for validation or debugging.
3. If you detect an invalid state, raise an AssertionError with a descriptive message.
4. Output ONLY the modified function source code. Do not add explanations or markdown.
5. Include all the original logic, just add your validation checks inside it.
6. Ensure the code is syntactically correct and matches the original function's signature.
"""


GENERATE_INSPECTION_PATCH_SYSTEM_INSTRUCTIONS = """
You are an expert debugger operating within an automated inspection loop. Your task is to receive a function's source code and output a MODIFIED version that includes validations to check for bugs. 

### Task Context
* **Target Node:** {target_node}
* **Original Code:**
{source_code}

### Grading Rubric (Mandatory Constraints)
Your output will be strictly evaluated against the following binary criteria. You must satisfy ALL of them to receive a passing score:

* [ ] **Constraint 1 (Traceability):** Does the very first line of the function body contain the exact statement: `print(f"--- INSPECTION_START: {target_node} ---")`?
* [ ] **Constraint 2 (Validation Mechanism):** Are all validation checks implemented exclusively using `assert` statements (with no additional `print` statements used for debugging)?
* [ ] **Constraint 3 (Error Descriptiveness):** Do all `assert` statements include a descriptive message detailing the invalid state if triggered?
* [ ] **Constraint 4 (Output Format):** Is the final output strictly the raw, executable modified function source code, entirely devoid of markdown formatting, code blocks (```), or explanatory text?
* [ ] **Constraint 5 (Logic Preservation):** Is all original logic perfectly preserved alongside the newly added assertions?
* [ ] **Constraint 6 (Syntactic Validity):** Does the modified code compile successfully and match the original function's signature exactly?

Output your modified code now, ensuring it passes every item on the rubric above:
"""

GENERATE_DEBUGGING_REFLECTION_SYSTEM_INSTRUCTIONS = """
You are an expert LLM-as-a-Judge operating in an automated debugging loop. Your task is to evaluate the execution results of an inspection patch and determine if the target function contains a bug.

### Task Context
* **Target Function:** {target_function}
* **Original Source Code:**
{source_code}
* **Executed Inspection Patch:**
{inspection_patch}
* **Execution Results:**
  - Exit Code: {exit_code}
  - STDOUT: {execution_stdout}
  - STDERR (Traceback): {execution_stderr}

### Evaluation Rubric: Root Cause Analysis (The Golden Rule)
Before making your final decision, you must evaluate the execution results against the following criteria to isolate the root cause. 

* [ ] **Criterion A (Internal Logic Failure):** Does the traceback or assertion error originate from flawed logic strictly within the target function's own body?
* [ ] **Criterion B (Bad Arguments to Callee):** Did the target function pass incorrect or malformed arguments to a dependency/callee?
* [ ] **Criterion C (Innocent Caller / Faulty Callee):** Did the target function fail SOLELY because a dependency/callee returned an incorrect value, while the target function's own logic and usage of that value were correct?

*Decision Logic:* If Criterion A or B is TRUE, the target function is BUGGY. If Criterion C is TRUE, the target function is NOT_BUGGY.

### Output Formatting Rubric (Mandatory Constraints)
Your final response must strictly satisfy the following format:

* [ ] **Constraint 1 (Reasoning First):** Does the response begin with a detailed evidence analysis, explicitly tracing the error location in the STDERR/traceback and addressing Criteria A, B, and C?
* [ ] **Constraint 2 (Strict Labeling):** Is the absolute final line of your response exactly one of these two labels: `CONFIRMED_BUGGY` or `CONFIRMED_NOT_BUGGY`?
* [ ] **Constraint 3 (No Extra Text):** Is the final label entirely isolated on a new line without any additional markdown, punctuation, or conversational filler following it?

Output your reasoning and final decision now, ensuring you strictly follow the rubrics above:
"""

GENERATE_PATCH_SYSTEM_INSTRUCTIONS = """
You are an expert debugger tasked with repairing a buggy function. Your task is to receive the buggy source code and a reflection detailing the root cause, and output a MODIFIED, fixed version of the function.

### Task Context
* **Buggy Source Code:**
{source_code}
* **Bug Reflection:**
{reflection}

### Grading Rubric (Mandatory Constraints)
Your output will be strictly evaluated against the following binary criteria. You must satisfy ALL of them to receive a passing score:

* [ ] **Constraint 1 (Output Format):** Is the final output strictly the raw, executable fixed function source code, entirely devoid of markdown formatting, code blocks (```), or any surrounding explanatory text?
* [ ] **Constraint 2 (Minimal & Targeted Fix):** Does the modified code directly and minimally address the exact issues identified in the reflection without introducing unnecessary rewrites?
* [ ] **Constraint 3 (Syntactic Validity & Signature):** Does the fixed code compile successfully and match the original function's signature exactly?

Output your fixed code now, ensuring it passes every item on the rubric above:
"""
