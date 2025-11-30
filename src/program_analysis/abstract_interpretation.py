import ast
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node
from src.program_analysis.file_utils import dump_to_json

import networkx as nx
import matplotlib.pyplot as plt
from networkx.readwrite import json_graph

# with PyCallGraph(output=GraphvizOutput()):
#     find_primes_between(10, 20)


def visualize_call_graph(graph):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(graph)
    nx.draw(graph, pos, with_labels=True, node_color="lightblue", edge_color="gray")
    plt.show()


def convert_graph_to_json(graph: nx.DiGraph) -> dict:
    """Converts a networkx DiGraph to a dictionary with 'nodes' and 'edges' keys."""
    return {
        "nodes": list(graph.nodes()),
        "edges": [list(edge) for edge in graph.edges()],
    }


def build_ast_from_file(file_path: str):
    """Parses a Python file and returns its AST."""
    with open(file_path, "r") as file:
        code = file.read()

    # Parse the code into an AST
    tree = ast.parse(code)
    return tree


def get_function_calls(node: Node, source_code: str):
    calls = []
    if node.type == "call":
        call_name = source_code[node.start_byte : node.end_byte]
        calls.append(call_name)
    for child in node.children:
        calls.extend(get_function_calls(child, source_code))
    return calls


def find_functions(node: Node, source_code: str, functions: dict, graph: nx.DiGraph):
    if node.type == "function_definition":
        func_name = source_code[
            node.child_by_field_name("name")
            .start_byte : node.child_by_field_name("name")
            .end_byte
        ]
        functions[func_name] = node
        graph.add_node(func_name)
    for child in node.children:
        find_functions(child, source_code, functions, graph)


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

    for func_name, node in functions.items():
        calls = get_function_calls(node, source_code)
        for call in calls:
            graph.add_edge(func_name, call)

    return graph


if __name__ == "__main__":
    example_file = "program_analysis/demo.py"
    # build_ast_from_file_tree_sitter(example_file)

    print("--- Building Call Graph ---")
    call_graph = build_call_graph(example_file)
    graph_json = convert_graph_to_json(call_graph)
    print(graph_json)

    # dump_to_json("call_graph.json", graph_json)

    visualize_call_graph(call_graph)
