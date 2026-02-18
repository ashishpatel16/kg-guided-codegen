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

GENERATE_DEBUGGING_REFLECTION_SYSTEM_INSTRUCTIONS = """
You are an expert debugger reflecting on the results of an inspection patch execution.
Given the target function, its source code, the inspection patch that was run, and the execution results (stdout/stderr/exit code), you must determine if the function is likely buggy.

THE GOLDEN RULE OF CALLEES:
- If the target function fails because a function it CALLS (a dependency/callee) returned a wrong value, but the target function's own logic is correct and it used that value correctly, then the target function is NOT_BUGGY.
- Only mark as BUGGY if the error is in the target function's own implementation or if it passed incorrect arguments to a callee.

Rules:
- First, provide a detailed reasoning and evidence analysis. Analyze the traceback carefully to see where the exception actually originated.
- After your reasoning, provide your final decision on a new line using exactly one of these two labels: "CONFIRMED_BUGGY" or "CONFIRMED_NOT_BUGGY".
- Be extremely strict: if the root cause is elsewhere, the current node is innocent.
"""

GENERATE_PATCH_SYSTEM_INSTRUCTIONS = """
You are an expert debugger. You are given a function's source code and a reflection describing why it is buggy.
Your task is to generate a fixed version of the function that resolves the bug.

CRITICAL RULES:
1. Output ONLY the fixed function source code.
2. Do not add explanations, markdown, or surrounding text.
3. Ensure the code is syntactically correct and matches the original function's signature.
4. The fix should be minimal and directly address the issues identified in the reflection.
"""
