import os
from typing import Dict

import networkx as nx
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node

from src.program_analysis.indexing_utils import RepoIndexer, ImportResolver
from src.program_analysis.models import CallGraphNode, CallGraphEdge, CallGraph

# Re-using tree-sitter setup from static_call_graph.py
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)

class CallGraphBuilder:
    def __init__(self, repo_root: str):
        self.repo_root = os.path.abspath(repo_root)
        self.indexer = RepoIndexer(self.root_dir)
        self.resolver = ImportResolver(self.root_dir)
        self.graph = nx.DiGraph()

    @property
    def root_dir(self):
        return self.repo_root

    def export_graph(self) -> CallGraph:
        """
        Exports the NetworkX graph to the CallGraph Pydantic model.
        """
        nodes = []
        for fqn, data in self.graph.nodes(data=True):
            nodes.append(CallGraphNode(
                fqn=fqn,
                type=data.get("type", "external"),
                file=data.get("file", ""),
                start_line=data.get("start_line", 0),
                end_line=data.get("end_line", 0),
                analysis_type="static"
            ))

        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append(CallGraphEdge(
                source=u,
                target=v,
                analysis_type="static"
            ))

        return CallGraph(
            static=True,
            dynamic=False,
            nodes=nodes,
            edges=edges
        )

    def build_call_graph(self) -> CallGraph:
        """Builds the static call graph and returns it as a CallGraph model."""
        self.build()
        return self.export_graph()

    def build(self):
        self.indexer.index_repo()
        # Add all defined nodes first
        for fqn, meta in self.indexer.index.items():
            self.graph.add_node(fqn, **meta)

        # Now parse every file again to find calls
        for root, _, files in os.walk(self.repo_root):
            for file in files:
                if file.endswith(".py"):
                    self._process_file(os.path.join(root, file))
        
        return self.graph

    def _module_fqn_from_file(self, file_path: str) -> str:
        rel_path = os.path.relpath(file_path, self.repo_root)
        module_path = rel_path.replace(".py", "").replace(os.sep, ".")
        if module_path.endswith(".__init__"):
            module_path = module_path[:-9]
        return module_path

    def _ensure_node(
        self,
        fqn: str,
        *,
        type: str = "external",
        file: str = "",
        start_line: int = 0,
        end_line: int = 0,
    ) -> None:
        if fqn in self.graph.nodes:
            # Fill missing metadata opportunistically (prefer existing).
            data = self.graph.nodes[fqn]
            data.setdefault("type", type)
            data.setdefault("file", file)
            data.setdefault("start_line", start_line)
            data.setdefault("end_line", end_line)
            return
        self.graph.add_node(
            fqn,
            type=type,
            file=file,
            start_line=start_line,
            end_line=end_line,
        )

    def _process_file(self, file_path: str):
        # 1. Resolve imports for this file
        imports = self.resolver.resolve_imports(file_path)
        
        # 2. Determine current module FQN
        module_path = self._module_fqn_from_file(file_path)

        # Ensure module node exists (captures top-level calls)
        try:
            with open(file_path, "rb") as f:
                code = f.read()
            end_line = code.count(b"\n") + 1 if code else 0
        except Exception:
            end_line = 0
        self._ensure_node(
            module_path,
            type="module",
            file=file_path,
            start_line=1,
            end_line=end_line,
        )

        # 3. Find calls using tree-sitter
        if "code" not in locals():
            with open(file_path, "rb") as f:
                code = f.read()
        tree = parser.parse(code)
        
        # We need a way to track the "Current Scope" (Function/Class) so we adding edges from Correct Source
        self._find_calls_in_scope(tree.root_node, code, imports, module_path)

    def _find_calls_in_scope(self, node: Node, source: bytes, imports: Dict[str, str], current_scope: str):
        # Update scope if we entered a function/class
        new_scope = current_scope
        
        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")
            new_scope = f"{current_scope}.{name}"
        elif node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            name = source[name_node.start_byte : name_node.end_byte].decode("utf-8")
            new_scope = f"{current_scope}.{name}"
        
        # If it's a call, resolve and add edge
        if node.type == "call":
            self._handle_call(node, source, imports, new_scope)

        # Recurse
        for child in node.children:
            self._find_calls_in_scope(child, source, imports, new_scope)

    def _handle_call(self, node: Node, source: bytes, imports: Dict[str, str], source_fqn: str):
        func_node = node.child_by_field_name("function")
        if not func_node:
            return

        call_text = source[func_node.start_byte : func_node.end_byte].decode("utf-8")
        
        # Heuristic resolution
        target_fqn = None
        
        parts = call_text.split(".")
        root_name = parts[0]
        
        if root_name in imports:
            # Resolved import
            imported_fqn = imports[root_name] # e.g. "tree_sitter.Node" or "os"
            # Reassemble
            if len(parts) > 1:
                target_fqn = f"{imported_fqn}.{'.'.join(parts[1:])}"
            else:
                target_fqn = imported_fqn
        else:
            # Case 2: Local method/function in same module or Class method (self)
            
            # Subcase 2a: 'self.method()'
            if root_name == "self":
                if "." in source_fqn:
                    parent_scope = source_fqn.rsplit(".", 1)[0]
                    # We expect parent_scope to be a Class
                    # Construct candidate: parent_scope.parts[1] (method name)
                    if len(parts) > 1:
                        method_name = parts[1]
                        candidate = f"{parent_scope}.{method_name}"
                        if candidate in self.indexer.index:
                            target_fqn = candidate

            else:
                # Subcase 2b: Sibling in same module (e.g. helper())
                
                scope_parts = source_fqn.split(".")
                # Try all parent scopes
                for i in range(len(scope_parts), 0, -1):
                    prefix = ".".join(scope_parts[:i])
                    candidate = f"{prefix}.{root_name}"
                    if candidate in self.indexer.index:
                        target_fqn = candidate
                        break

        
        # Ensure source exists (e.g., module-level calls use the module node)
        if source_fqn not in self.graph.nodes:
            self._ensure_node(source_fqn, type="external")

        # Final Verification: Is target_fqn in our index?
        if target_fqn and target_fqn in self.indexer.index:
            self.graph.add_edge(source_fqn, target_fqn)
            return

        # Maximal graph: keep unresolved/imported targets as "external" nodes.
        if target_fqn:
            self._ensure_node(target_fqn, type="external")
            self.graph.add_edge(source_fqn, target_fqn)

    


class GraphSlicer:
    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def slice_for_test(self, test_file_path: str) -> nx.DiGraph:
        """
        Returns a subgraph reachable from all test functions in the given file.
        """
        # 1. Identify test functions in the graph for this file
        # The indexer stores 'file' in node attributes.
        test_nodes = []
        for node, data in self.graph.nodes(data=True):
            if data.get("file") == test_file_path:
                # Heuristic: starts with test_
                if node.split(".")[-1].startswith("test_"):
                    test_nodes.append(node)
        
        if not test_nodes:
            print(f"No test functions found in {test_file_path}")
            return nx.DiGraph()

        # 2. Perform BFS/DFS traversal
        reachable_nodes = set()
        for start_node in test_nodes:
            reachable_nodes.add(start_node)
            descendants = nx.descendants(self.graph, start_node)
            reachable_nodes.update(descendants)
            
        # 3. Create subgraph
        return self.graph.subgraph(reachable_nodes).copy()

def build_static_call_graph(repo_root: str) -> CallGraph:
    """Builds a static call graph for a repository and returns the CallGraph model."""
    builder = CallGraphBuilder(repo_root)
    return builder.build_call_graph()

if __name__ == "__main__":
    import json
    repo_root = "/Users/ashish/master-thesis/kg-guided-codegen/src/benchmarks/exp/demo"

    g = build_static_call_graph(str(repo_root))
    # print(g, type(g))

    # Export as JSON
    with open("artifacts/demo_call_graph_new.json", "w") as f:
        json.dump(g.model_dump(), f, indent=4)
        print(f"Saved to artifacts/demo_call_graph_new.json")

    # print(f"static={g.static} dynamic={g.dynamic}")
    # print(f"nodes={len(g.nodes)} edges={len(g.edges)}")
    # print("node types:", dict(Counter(n.type for n in g.nodes)))

    # # Show a few sample nodes/edges
    # for n in sorted(g.nodes, key=lambda n: n.fqn)[:10]:
    #     print(f"NODE {n.type}: {n.fqn} ({Path(n.file).name}:{n.start_line}-{n.end_line})")

    for e in g.edges[:10]:
        print(f"EDGE: {e.source} -> {e.target}")