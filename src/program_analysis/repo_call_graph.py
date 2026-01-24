import os
import glob
import ast
import networkx as nx
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node
from typing import Dict, List, Tuple, Optional, Set

# Re-using tree-sitter setup from static_call_graph.py
PY_LANGUAGE = Language(tspython.language())
parser = Parser(PY_LANGUAGE)


class RepoIndexer:
    """
    Scans a repository to index all classes and functions.
    Maps FullyQualifiedName -> (FilePath, NodeInfo)
    """
    def __init__(self, root_dir: str):
        self.root_dir = os.path.abspath(root_dir)
        self.index: Dict[str, Dict] = {}  # FQN -> {path, line, type, ...}

    def index_repo(self):
        """Walks the repo and builds the index. Includes anyfile that ends in .py extension"""
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    self._index_file(full_path)
        return self.index

    def _index_file(self, file_path: str):
        print(f"Indexing {file_path}..")
        rel_path = os.path.relpath(file_path, self.root_dir)
        module_path = rel_path.replace(".py", "").replace(os.sep, ".")

        # Handle __init__
        if module_path.endswith(".__init__"):
            module_path = module_path[:-9]
        
        try:
            with open(file_path, "rb") as f:
                code = f.read()
            tree = parser.parse(code)
            self._find_definitions(tree.root_node, code, module_path, file_path)
        except Exception as e:
            print(f"Failed to index {file_path}: {e}")

    def _find_definitions(self, node: Node, source_code: bytes, scope: str, file_path: str):
        print(f"Processing node: {node.type} \n {node.text.decode('utf-8')}\n")    
        if node.type == "class_definition":
            name_node = node.child_by_field_name("name")
            name = source_code[name_node.start_byte : name_node.end_byte].decode("utf-8")
            fqn = f"{scope}.{name}"
            
            self.index[fqn] = {
                "type": "class_definition",
                "file": file_path,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1
            }
            
            # Recurse for methods
            for child in node.children:
                self._find_definitions(child, source_code, fqn, file_path)

        elif node.type == "function_definition":
            name_node = node.child_by_field_name("name")
            name = source_code[name_node.start_byte : name_node.end_byte].decode("utf-8")
            fqn = f"{scope}.{name}"
            
            self.index[fqn] = {
                "type": "function_definition",
                "file": file_path,
                "start_line": node.start_point[0] + 1,
                "end_line": node.end_point[0] + 1
            }
            # TODO: Currently recursing into functions for definitions is not implemented, TBD

        else:
            for child in node.children:
                self._find_definitions(child, source_code, scope, file_path)


class ImportResolver:
    """
    Resolves imports in a file to map local names to Fully Qualified Names.
    """
    def __init__(self, repo_root: str):
        self.repo_root = repo_root

    def resolve_imports(self, file_path: str) -> Dict[str, str]:
        """
        Returns a dict: LocalName -> FullyQualifiedName
        e.g. "Node" -> "tree_sitter.Node"
        """
        resolved = {}
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                tree = ast.parse(f.read())
        except Exception:
            return {}

        rel_dir = os.path.dirname(os.path.relpath(file_path, self.repo_root))
        base_package = rel_dir.replace(os.sep, ".")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # import os -> os: os
                    # import os as o -> o: os
                    target = alias.name
                    local = alias.asname if alias.asname else alias.name
                    resolved[local] = target
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                # Handle relative imports
                if node.level > 0:
                    # simplistic relative import handling
                    # level 1 = ., level 2 = ..
                    parts = base_package.split(".")
                    if node.level > len(parts) + 1:
                        # Error or too far back
                        resolved_module = "" # fallback
                    else:
                        # Strip last (level-1) parts
                        # e.g. pkg.sub, level 1 (.) -> pkg.sub
                        # e.g. pkg.sub, level 2 (..) -> pkg
                        if node.level == 1:
                            parent = parts
                        else:
                            parent = parts[:-(node.level-1)]
                        resolved_module = ".".join(parent)
                        if module:
                            resolved_module += f".{module}"
                else:
                    resolved_module = module

                for alias in node.names:
                    # from X import Y
                    name = alias.name
                    local = alias.asname if alias.asname else name
                    if name == "*":
                        continue # Can't resolve star imports easily without index check
                    
                    fqn = f"{resolved_module}.{name}" if resolved_module else name
                    resolved[local] = fqn
        
        return resolved


class CallGraphBuilder:
    def __init__(self, repo_root: str):
        self.repo_root = os.path.abspath(repo_root)
        self.indexer = RepoIndexer(self.root_dir)
        self.resolver = ImportResolver(self.root_dir)
        self.graph = nx.DiGraph()

    @property
    def root_dir(self):
        return self.repo_root

    def build(self):
        print("Indexing repo...")
        self.indexer.index_repo()

        print(self.indexer.index)
        
        print("Building graph...")
        # Add all defined nodes first
        for fqn, meta in self.indexer.index.items():
            self.graph.add_node(fqn, **meta)

        # Now parse every file again to find calls
        for root, _, files in os.walk(self.repo_root):
            for file in files:
                if file.endswith(".py"):
                    self._process_file(os.path.join(root, file))
        
        return self.graph

    def _process_file(self, file_path: str):
        # 1. Resolve imports for this file
        imports = self.resolver.resolve_imports(file_path)
        
        # 2. Determine current module FQN
        rel_path = os.path.relpath(file_path, self.repo_root)
        module_path = rel_path.replace(".py", "").replace(os.sep, ".")
        if module_path.endswith(".__init__"):
            module_path = module_path[:-9]

        # 3. Find calls using tree-sitter
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
        
        # Case 1: Direct import match (e.g. usage of `Node` where `from tree_sitter import Node`)
        # logic: split call_text by dot. 
        # e.g. "os.path.join" -> check "os" in imports
        # e.g. "Node" -> check "Node" in imports
        
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
                # source_fqn: package.module.Class.method
                # We want: package.module.Class.target_method
                
                # Check if we are inside a class (heuristic: 2 dots or check index type)
                # Let's try to extract class scope from source_fqn
                # source_fqn = src.utils.Worker.do_work
                # parent = src.utils.Worker
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
                
                # TODO: Intra-module resolution implemented via heuristic.
                # DONE:
                # - Reconstructs module path by stripping last parts of source_fqn and checking against index.
                # MISSING:
                # - Explicitly passing 'module_path' from _process_file would be robust.
                # - Handling inner functions/closures (not indexed currently).
                # - Handling dynamic resolution/aliasing within module.

                
                scope_parts = source_fqn.split(".")
                # Try all parent scopes
                for i in range(len(scope_parts), 0, -1):
                    prefix = ".".join(scope_parts[:i])
                    candidate = f"{prefix}.{root_name}"
                    if candidate in self.indexer.index:
                        target_fqn = candidate
                        break

        
        # Final Verification: Is target_fqn in our index?
        if target_fqn and target_fqn in self.indexer.index:
            self.graph.add_edge(source_fqn, target_fqn)
        else:
            # Maybe add as external definition?
            # self.graph.add_edge(source_fqn, target_fqn or call_text, type="external")
            pass
            
            # IMPROVEMENT: For now, let's just create the edge if we have a strong candidate
            # or if it's explicitly explicitly resolved.
            if target_fqn:
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


def get_repo_call_graph(repo_root: str) -> nx.DiGraph:
    builder = CallGraphBuilder(repo_root)
    return builder.build()

if __name__ == "__main__":
    repo_root = "/Users/ashish/master-thesis/kg-guided-codegen/src/benchmarks/exp/demo"
    # print(list(os.walk(repo_root))) # Returns dir_path, dir_name, file_names
    indexer = RepoIndexer(repo_root)
    indexer.index_repo()
    print(indexer.index)
    # graph = get_repo_call_graph(repo_root)
    # print(f"Full graph nodes: {graph.number_of_nodes()}, edges: {graph.number_of_edges()}")
    
    # # Save full graph
    # nx.write_gexf(graph, "artifacts/black_1_full.gexf")
    
    # # Slice for a test
    # test_file = "tests/test_black.py"
    # abs_test_file = str(Path(repo_root) / test_file)
    # slicer = GraphSlicer(graph)
    # sliced_graph = slicer.slice_for_test(abs_test_file)
    # print(f"Sliced graph nodes: {sliced_graph.number_of_nodes()}, edges: {sliced_graph.number_of_edges()}")
    
    # # Save sliced graph
    # nx.write_gexf(sliced_graph, "artifacts/black_1_sliced.gexf")