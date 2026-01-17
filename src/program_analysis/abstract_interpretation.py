import ast
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node
from src.program_analysis.file_utils import dump_to_json


import networkx as nx
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph


def visualize_call_graph(graph):
    plt.figure(figsize=(8, 6))

    # Calculate levels for hierarchy
    # 1. Find roots (nodes with 0 in-degree)
    roots = [n for n, d in graph.in_degree() if d == 0]
    # If no roots (cycle everywhere), pick an arbitrary one
    if not roots and len(graph) > 0:
        roots = [list(graph.nodes())[0]]

    # 2. Compute shortest path levels (BFS) - safe against cycles
    levels = {}
    queue = []

    for root in roots:
        levels[root] = 0
        queue.append((root, 0))

    visited = set(roots)

    while queue:
        node, depth = queue.pop(0)
        for neighbor in graph.successors(node):
            if neighbor not in visited:
                levels[neighbor] = depth + 1
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))

    # Assign unreachables / missed nodes
    for node in graph.nodes():
        if node not in levels:
            levels[node] = 0

    # Add level as node attribute
    nx.set_node_attributes(graph, levels, "level")

    # Use multipartite layout
    pos = nx.multipartite_layout(graph, subset_key="level", align="horizontal")

    # Flip x and y to make it vertical (Top-Down)
    new_pos = {}
    for node, (x, y) in pos.items():
        new_pos[node] = (y, -x)

    nx.draw(
        graph,
        new_pos,
        with_labels=True,
        node_size=2000,
        node_color="lightblue",
        edge_color="gray",
        arrows=True,
        font_size=10,
        font_weight="bold",
    )
    plt.savefig("artifacts/call_graph.png")
    plt.show()


def convert_graph_to_json(graph: nx.DiGraph) -> dict:
    """Converts a networkx DiGraph to a dictionary with 'nodes' and 'edges' keys containing attributes."""
    nodes = []
    for node, data in graph.nodes(data=True):
        node_data = {"id": node}
        node_data.update(data)
        nodes.append(node_data)

    edges = []
    for u, v, data in graph.edges(data=True):
        edge_data = {"source": u, "target": v}
        edge_data.update(data)
        edges.append(edge_data)

    return {
        "nodes": nodes,
        "edges": edges,
    }


def get_function_calls(node: Node, source_code: str):
    calls = []
    if node.type == "call":
        func_node = node.child_by_field_name("function")
        if func_node:
            line_number = func_node.start_point[0] + 1
            if func_node.type == "identifier":
                call_name = source_code[
                    func_node.start_byte : func_node.end_byte
                ].decode("utf-8")
                calls.append((call_name, line_number))

            # Handle attribute access: obj.method()
            elif func_node.type == "attribute":
                attr_node = func_node.child_by_field_name("attribute")
                if attr_node:
                    call_name = source_code[
                        attr_node.start_byte : attr_node.end_byte
                    ].decode("utf-8")
                    calls.append((call_name, line_number))

    for child in node.children:
        calls.extend(get_function_calls(child, source_code))
    return calls


def find_functions(
    node: Node, source_code: str, functions: dict, graph: nx.DiGraph, scope: str = ""
):
    if node.type == "class_definition":
        class_name_node = node.child_by_field_name("name")
        class_name = source_code[
            class_name_node.start_byte : class_name_node.end_byte
        ].decode("utf-8")
        new_scope = f"{scope}{class_name}." if scope else f"{class_name}."

        # Add class node itself? For now, let's just use it for scoping methods.
        # Recursively find methods
        for child in node.children:
            find_functions(child, source_code, functions, graph, new_scope)

    elif node.type == "function_definition":
        name_node = node.child_by_field_name("name")
        func_name_raw = source_code[name_node.start_byte : name_node.end_byte].decode(
            "utf-8"
        )
        func_name = f"{scope}{func_name_raw}"

        # Extract metadata
        # 1. Parameters
        params_node = node.child_by_field_name("parameters")
        params = (
            source_code[params_node.start_byte : params_node.end_byte].decode("utf-8")
            if params_node
            else "()"
        )

        # 2. Docstring (first expression statement in body which is a string)
        docstring = ""
        body_node = node.child_by_field_name("body")
        if body_node:
            for child in body_node.children:
                if child.type == "expression_statement":
                    # strictly check if it looks like a string
                    first_child = child.children[0]
                    if first_child.type == "string":
                        docstring = source_code[
                            first_child.start_byte : first_child.end_byte
                        ].decode("utf-8")
                        # cleaning quotes could be done here, but raw is fine for now
                        break
                elif child.type == "comment":
                    continue
                else:
                    # Statement that is not a docstring, stop looking
                    break

        # 3. Location
        start_line = node.start_point[0] + 1  # 0-indexed to 1-indexed
        end_line = node.end_point[0] + 1

        # 4. Source Code
        code_content = source_code[node.start_byte : node.end_byte].decode("utf-8")

        metadata = {
            "type": "method" if scope else "function",
            "parameters": params,
            "docstring": docstring,
            "start_line": start_line,
            "end_line": end_line,
            "code": code_content,
        }

        functions[func_name] = node
        graph.add_node(func_name, **metadata)

    else:
        # Recurse for other blocks (like checks, loops within top level, though rare for definitions)
        for child in node.children:
            find_functions(child, source_code, functions, graph, scope)


def get_calls_in_scope(node: Node, source_code: str):
    """Recursively finds calls but stops at function/class definition boundaries."""
    calls = []

    # Check if current node is a call
    if node.type == "call":
        func_node = node.child_by_field_name("function")
        if func_node:
            line_number = func_node.start_point[0] + 1
            if func_node.type == "identifier":
                call_name = source_code[
                    func_node.start_byte : func_node.end_byte
                ].decode("utf-8")
                calls.append((call_name, line_number))
            # Handle attribute access: obj.method()
            elif func_node.type == "attribute":
                # Capture full attribute chain e.g. "found.append" or "self.method"
                call_name = source_code[
                    func_node.start_byte : func_node.end_byte
                ].decode("utf-8")
                calls.append((call_name, line_number))

    # Recurse children, but STOP if entering a new scope definition
    for child in node.children:
        if child.type in ("function_definition", "class_definition"):
            continue
        calls.extend(get_calls_in_scope(child, source_code))

    return calls


def build_call_graph(file_path: str) -> nx.DiGraph:
    with open(file_path, "rb") as f:
        code = f.read()

    source_code = code

    PY_LANGUAGE = Language(tspython.language())
    parser = Parser(PY_LANGUAGE)
    tree = parser.parse(code)

    graph = nx.DiGraph()
    functions = {}

    find_functions(tree.root_node, source_code, functions, graph)

    # Handle top-level calls (script body)
    # We create a pseudo node "__main__" representing the module body
    top_level_calls = get_calls_in_scope(tree.root_node, source_code)

    if top_level_calls:
        main_node_name = "__main__"
        # Only add __main__ if it makes calls or is relevant?
        # Yes, let's add it.
        metadata = {
            "type": "module",
            "parameters": "",
            "docstring": "",
            "start_line": 1,
            "end_line": len(code.splitlines()),
        }
        graph.add_node(main_node_name, **metadata)
        # Add edges for top level calls
        for call_name, line_no in top_level_calls:
            # We add edges to discovered functions, OR external/imported calls logic?
            # Existing logic only adds edges to KNOWN functions in the file.
            # But top level often calls imports (like in main.py: configure_logging, print, etc.)
            # If we want to see 'one_shot_codegen_agent.invoke' we need to allow edges to unknown targets?
            # User request "store more aspects" -> maybe we should capture external calls too now?
            # For consistency with previous step, let's stick to "known targets" + "local definitions".
            # BUT for main.py, everything is imported. If we filter strictly, we get empty edges.
            # Let's relax the filter for __main__ or generally?

            # For now, let's allow edges to ANYTHING from __main__ to show the script activity?
            # Or keep strict?
            # Given main.py calls imported things, showing them is valuable.
            # Let's add them as external nodes?

            if call_name in functions:
                graph.add_edge(main_node_name, call_name, lineno=line_no)
            else:
                # It's an external call. Let's add it to graph to be useful for main.py
                if call_name not in graph:
                    graph.add_node(call_name, type="external")
                graph.add_edge(main_node_name, call_name, lineno=line_no)

    for func_name, node in functions.items():
        # Determine current class context if any (from func_name prefix)
        # simplistic: if func_name has dots, scope is everything before last dot
        current_scope = func_name.rsplit(".", 1)[0] if "." in func_name else ""

        # Use existing logic for internal function bodies
        # Note: We should probably use get_calls_in_scope here too to avoid double counting nested funcs?
        # get_function_calls WAS recursive.
        # find_functions iterates all definitions.
        # If we use get_function_calls on outer function, it traverses inner function.
        # The inner function also gets its own node.
        # So outer->inner_call edges are created.
        # BUT outer->inner_body_call edges are ALSO created?
        # Yes, get_function_calls is fully recursive.
        # We should replace it with get_calls_in_scope to be correct!

        calls = get_calls_in_scope(
            node, source_code
        )  # Swapping to safer scoped call finder
        for call_name, line_no in calls:
            target = None

            # Handle self.method calls by checking against current_scope
            search_name = call_name
            if search_name.startswith("self."):
                search_name = search_name.split(".", 1)[1]

            # 1. Direct match
            if call_name in functions:
                target = call_name
            # 2. Scoped match (using search_name to handle 'self')
            elif current_scope and f"{current_scope}.{search_name}" in functions:
                target = f"{current_scope}.{search_name}"

            # IF target found (internal), add edge
            if target:
                graph.add_edge(func_name, target, lineno=line_no)
            # ELSE (external), should we add it?
            # For rich graphs, yes.
            else:
                if call_name not in graph:
                    graph.add_node(call_name, type="external")
                graph.add_edge(func_name, call_name, lineno=line_no)

    return graph


def build_ast_from_file(file_path: str):
    """Parses a Python file and returns its AST."""
    with open(file_path, "r") as file:
        code = file.read()

    # Parse the code into an AST
    tree = ast.parse(code)
    return tree


if __name__ == "__main__":
    example_file = "src/program_analysis/demo.py"
    # build_ast_from_file_tree_sitter(example_file)

    print("--- Building Call Graph ---")
    call_graph = build_call_graph(example_file)
    graph_json = convert_graph_to_json(call_graph)
    print(graph_json)

    dump_to_json("call_graph.json", graph_json)

    visualize_call_graph(call_graph)
