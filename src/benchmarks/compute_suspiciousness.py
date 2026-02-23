import os
import sys
import logging
import inspect
import re

# Ensure the project root is in sys.path so we can import src modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import necessary modules
from src.program_analysis.dynamic_call_graph import trace_repo, compute_suspiciousness_scores, DynamicCallGraphTracer
from src.program_analysis.suspiciousness_controller import SuspiciousnessController
from src.agent.tools import get_default_llm_connector, get_function_source

# Suppress excessive logging
logging.getLogger("src.program_analysis.dynamic_call_graph").setLevel(logging.WARNING)
logging.getLogger("src.program_analysis.suspiciousness_controller").setLevel(logging.INFO)

def print_block():
    print("\n" + "#"*80)

def main():
    # Define paths relative to the project root
    target_repo_path = os.path.join(project_root, "src/benchmarks/exp/demo")
    trace_scripts = ["test_demo.py"]

    print(f"Tracing repo at: {target_repo_path}")
    print(f"Executing scripts: {trace_scripts}")

    if not os.path.exists(target_repo_path):
        print(f"Error: Target path does not exist: {target_repo_path}")
        return

    # 1. Run initial trace
    try:
        call_graph = trace_repo(
            repo_root=target_repo_path,
            scripts=trace_scripts,
            test_mode=True
        )
    except Exception as e:
        print(f"Error tracing repo: {e}")
        return

    print("--- Initial Suspiciousness Scores ---")
    suspicious_nodes = []
    for node in call_graph.nodes:
        if node.suspiciousness >= 0:
            suspicious_nodes.append((node.fqn, node.suspiciousness))
    
    suspicious_nodes.sort(key=lambda x: x[1], reverse=True)
    
    
    top_3 = []
    if not suspicious_nodes:
        print("No nodes found.")
        return
    else:
        print(f"{'Function Name':<40} | {'Score':<10}")
        print("-" * 55)
        count = 0
        for fqn, score in suspicious_nodes:
            if score > 0:
                print(f"{fqn:<40} | {score:.4f}")
                if count < 3:
                    top_3.append(fqn)
                    count += 1
            else:
                 pass
        print("-" * 55)
    
    if not top_3:
        print("No suspicious nodes found to analyze.")
        return

    print(f"\nTop 3 Suspicious Nodes: {top_3}")
    
    # 2. Setup SuspiciousnessController
    test_to_nodes = {}
    if call_graph.coverage_matrix:
        for node_fqn, tests in call_graph.coverage_matrix.items():
            for test in tests:
                if test not in test_to_nodes:
                    test_to_nodes[test] = set()
                test_to_nodes[test].add(node_fqn)
                
    test_results = getattr(call_graph, "test_results", {})
    
    try:
        llm = get_default_llm_connector()
    except Exception as e:
        print(f"Error: Could not initialize LLM ({e}). Cannot proceed without mocking.")
        return

    controller = SuspiciousnessController(
        node_execution_map=test_to_nodes,
        test_results=test_results,
        llm_connector=llm
    )

    # 3. Add more tests to top 3
   
    print("--- Generating and Running New Tests ---")
    
    if target_repo_path not in sys.path:
        sys.path.insert(0, target_repo_path)
    
    for node_fqn in top_3:
        print_block()
        print(f"\nTargeting Node: {node_fqn}")
        
        test_code = None
        
        # Try controller first
        try:
            cg_dict = call_graph.model_dump()
            test_code = controller.generate_test_to_disambiguate(node_fqn, cg_dict)
        except Exception as e:
            print(f"Controller Generation failed: {e}")

        # Check if controller declined
        if not test_code or "not ambiguous" in test_code:
            print(f"Controller declined (not ambiguous). Falling back to manual generation...")
            
            # Find node source
            node_data = next((n for n in call_graph.nodes if n.fqn == node_fqn), None)
            source_code = ""
            if node_data:
                try:
                    source_code = get_function_source(node_data.model_dump())
                except:
                    pass
            
            prompt = f"""
            You are an expert tester. Write a pytest test case that exercises the following function:
            
            Function: {node_fqn}
            Source:
            ```python
            {source_code}
            ```
            
            INSTRUCTIONS:
            1. Write a complete test function starting with `test_`.
            2. The test should import the necessary function from `demo`.
            3. Call the function with valid arguments that are likely to pass if the code is correct.
            4. Add an assertion.
            5. Output ONLY the python code in markdown blocks.
            """
            
            try:
                raw = llm.generate(prompt)
                m = re.search(r"```python\s*([\s\S]*?)\s*```", raw)
                if m:
                    test_code = m.group(1).strip()
                else:
                    test_code = raw.strip()
            except Exception as e:
                print(f"Fallback generation failed: {e}")
                continue

        print(f"Generated Test Code:\n{test_code}")
        print_block()

        # Prepare execution environment
        global_scope = {}
        
        try:
            exec(test_code, global_scope)
        except Exception as e:
            print(f"Failed to compile/exec generated code: {e}")
            continue
            
        # Find the test function
        test_func_name = None
        test_func = None
        for name, obj in global_scope.items():
            if name.startswith("test_") and callable(obj):
                test_func_name = name
                test_func = obj
                break
        
        if not test_func:
            print("No test function found in executed code.")
            continue
            
        print(f"Executing {test_func_name} with tracer...")
        
        tracer = DynamicCallGraphTracer(repo_root=target_repo_path, include_external=False)
        
        try:
            passed = tracer.run_test_and_track(test_func_name, test_func)
        except Exception as e:
            print(f"Error running test with tracer: {e}")
            passed = False
            
        executed_nodes = tracer._node_execution_map.get(test_func_name, set())
        print(f"Result: PASSED={passed}, Covered {len(executed_nodes)} nodes")
        
        unique_test_name = f"generated_{node_fqn.split('.')[-1]}_{test_func_name}"
        controller.add_test_case(unique_test_name, executed_nodes, passed)


    # 4. Recompute Scores
    print("\n--- Recomputing Suspiciousness Scores ---")
    
    updated_call_graph = compute_suspiciousness_scores(
        call_graph,
        controller.test_results,
        controller.node_execution_map
    )

    new_scores = {n.fqn: n.suspiciousness for n in updated_call_graph.nodes}
    
    print(f"{'Function Name':<40} | {'Old Score':<10} | {'New Score':<10} | {'Change':<10}")
    print("-" * 80)
    
    combined_list = set([x[0] for x in suspicious_nodes])
    combined_sorted = sorted(combined_list, key=lambda x: new_scores.get(x, 0), reverse=True)
    
    for fqn in combined_sorted:
        old = next((score for n, score in suspicious_nodes if n == fqn), 0.0)
        new = new_scores.get(fqn, 0.0)
        change = new - old
        
        if old > 0 or new > 0:
            print(f"{fqn:<40} | {old:.4f}     | {new:.4f}     | {change:+.4f}")

    print("-" * 80)

if __name__ == "__main__":
    main()
