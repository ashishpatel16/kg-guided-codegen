import sys
import inspect
import os
import json

class DebuggingNode:
    def __init__(self, function_name, args, parent=None, node_id=None):
        self.id = node_id
        self.function_name = function_name
        self.args = args
        self.return_value = None
        self.children = []
        self.parent = parent

    def add_child(self, child):
        self.children.append(child)

    def set_return_value(self, value):
        self.return_value = value

    def to_dict(self):
        return {
            "id": self.id,
            "function": self.function_name,
            "args": {k: repr(v) for k, v in self.args.items()},
            "return_value": repr(self.return_value),
            "children": [child.to_dict() for child in self.children]
        }

    def __repr__(self):
        args_str = ", ".join(f"{k}={v!r}" for k, v in self.args.items())
        return f"[{self.id}] {self.function_name}({args_str}) -> {self.return_value!r}"

class AlgorithmicDebugger:
    def __init__(self):
        self.node_counter = 0
        self.root = DebuggingNode("root", {}, None, self._next_id())
        self.current_node = self.root
        self.stack = []
        self.nodes_by_id = {self.root.id: self.root}

    def _next_id(self):
        self.node_counter += 1
        return f"node_{self.node_counter}"

    def trace_calls(self, frame, event, arg):
        if event == 'call':
            code = frame.f_code
            function_name = code.co_name
            
            # Get arguments
            arg_info = inspect.getargvalues(frame)
            args = {arg: arg_info.locals[arg] for arg in arg_info.args}
            
            new_id = self._next_id()
            new_node = DebuggingNode(function_name, args, self.current_node, new_id)
            self.nodes_by_id[new_id] = new_node
            
            self.current_node.add_child(new_node)
            self.stack.append(self.current_node)
            self.current_node = new_node
            return self.trace_calls
            
        elif event == 'return':
            self.current_node.set_return_value(arg)
            if self.stack:
                self.current_node = self.stack.pop()
            return self.trace_calls

    def execute(self, file_path):
        self.node_counter = 0
        self.root = DebuggingNode("root", {}, None, self._next_id())
        self.nodes_by_id = {self.root.id: self.root}
        self.current_node = self.root
        self.stack = []
        
        abs_path = os.path.abspath(file_path)
        directory = os.path.dirname(abs_path)
        
        if directory not in sys.path:
            sys.path.insert(0, directory)
            
        with open(abs_path, 'r') as f:
            script = f.read()
            
        code = compile(script, abs_path, 'exec')
        
        sys.settrace(self.trace_calls)
        
        try:
            globs = {
                '__name__': '__main__',
                '__file__': abs_path,
                '__builtins__': __builtins__,
            }
            exec(code, globs)
        except Exception as e:
            print(f"Execution failed: {e}")
        finally:
            sys.settrace(None)
            if directory in sys.path:
                sys.path.remove(directory)

    def print_tree(self, node=None, indent=0):
        if node is None:
            node = self.root
            
        if node == self.root:
            for child in node.children:
                self.print_tree(child, indent)
            return

        print("  " * indent + str(node))
        for child in node.children:
            self.print_tree(child, indent + 2)

    def get_trace_json(self):
        return json.dumps([c.to_dict() for c in self.root.children], indent=2)

    def get_node(self, node_id):
        return self.nodes_by_id.get(node_id)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        debugger = AlgorithmicDebugger()
        debugger.execute(sys.argv[1])
        if "--json" in sys.argv:
            print(debugger.get_trace_json())
        else:
            debugger.print_tree()
    else:
        print("Usage: python algorithmic_debugger.py <file_to_debug> [--json]")
