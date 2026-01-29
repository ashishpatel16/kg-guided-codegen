import ast
import os
from typing import Dict

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node

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
