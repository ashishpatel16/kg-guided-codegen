import json
import os
import sys
from typing import Dict, Any, List

def get_source_snippet(file_path: str, start_line: int, end_line: int) -> str:
    if not file_path or not os.path.exists(file_path):
        return ""
    try:
        with open(file_path, "r") as f:
            lines = f.readlines()
            # If end_line is 0 or invalid, try to find the end of the function or just take a snippet
            if end_line <= 0:
                # Heuristic: find the end of the function by looking at indentation
                # or just take 30 lines for now.
                target_lines = lines[max(0, start_line-1) : start_line + 30]
            else:
                target_lines = lines[max(0, start_line-1) : end_line]
            return "".join(target_lines)
    except Exception:
        return ""

def generate_html(json_data: Dict[str, Any], output_path: str):
    nodes = json_data.get("nodes", [])
    edges = json_data.get("edges", [])
    
    # Add source code to nodes
    for node in nodes:
        node["code"] = get_source_snippet(node.get("file", ""), node.get("start_line", 0), node.get("end_line", 0))

    # Sort nodes by suspiciousness descending
    nodes.sort(key=lambda x: x.get("suspiciousness", 0) or 0, reverse=True)

    # Group nodes by file for visual clustering
    file_groups = {}
    for n in nodes:
        abs_path = n.get("file", "")
        if abs_path:
            file_name = os.path.basename(abs_path)
        else:
            # Fallback for nodes without file info (e.g., some external/module nodes)
            parts = n["fqn"].split('.')
            file_name = parts[0] + ".py" if len(parts) > 1 else "external"
            
        if file_name not in file_groups:
            file_groups[file_name] = []
        file_groups[file_name].append(n)

    # Prepare Mermaid graph with color coding and Left-to-Right layout
    mermaid_lines = ["flowchart LR"]
    node_styles = []
    
    def sanitize(s: str) -> str:
        return "".join(c if c.isalnum() else "_" for c in s)

    # 1. Define Subgraphs (Files)
    for file_idx, (file_name, group_nodes) in enumerate(file_groups.items()):
        # Sanitize subgraph ID to be valid CSS/Mermaid identifier
        subgraph_id = f"file_{file_idx}"
        mermaid_lines.append(f"    subgraph {subgraph_id} [\"{file_name}\"]")
        for n in group_nodes:
            # Use the node's index in the original list as its ID for robust lookup
            try:
                node_idx = nodes.index(n)
                node_id = f"node_{node_idx}"
            except ValueError:
                node_id = "node_" + sanitize(n["fqn"])
            
            # Escape characters that Mermaid might choke on even in quotes
            label = n["fqn"].replace("\"", "'").replace("<", "(").replace(">", ")")
            mermaid_lines.append(f"        {node_id}[\"{label}\"]")
            
            # Prepare styling
            score = n.get("suspiciousness", 0) or 0
            r = int(34 + (239 - 34) * score)
            g = int(197 + (68 - 197) * score)
            b = int(94 + (68 - 94) * score)
            hex_color = f"#{r:02x}{g:02x}{b:02x}"
            node_styles.append(f"    style {node_id} fill:{hex_color},stroke:#333,stroke-width:1px,color:#fff")
        mermaid_lines.append("    end")
        # Style the subgraph itself
        node_styles.append(f"    style {subgraph_id} fill:#fef9c3,stroke:#facc15,stroke-width:2px,stroke-dasharray: 5 5")

    # 2. Define Edges (Calls)
    fqn_to_idx = {n["fqn"]: i for i, n in enumerate(nodes)}
    for edge in edges:
        source_idx = fqn_to_idx.get(edge["source"])
        target_idx = fqn_to_idx.get(edge["target"])
        
        if source_idx is not None and target_idx is not None:
            mermaid_lines.append(f"    node_{source_idx} --> node_{target_idx}")
    
    mermaid_graph = "\n".join(mermaid_lines + node_styles)

    html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Call Graph Analysis Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/svg-pan-zoom@3.6.1/dist/svg-pan-zoom.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <!-- Prism.js for syntax highlighting -->
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism.min.css" rel="stylesheet" />
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
    <style>
        :root {{
            --bg-color: #f8fafc;
            --sidebar-bg: #ffffff;
            --card-bg: #ffffff;
            --text-color: #0f172a;
            --text-muted: #64748b;
            --border-color: #e2e8f0;
            --primary: #2563eb;
            --primary-light: #dbeafe;
        }}
        
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-color);
            margin: 0;
            display: flex;
            flex-direction: column;
            height: 100vh;
            overflow: hidden;
        }}

        header {{
            padding: 1rem 1.5rem;
            background-color: var(--sidebar-bg);
            border-bottom: 1px solid var(--border-color);
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 10;
        }}

        .container {{
            display: flex;
            flex: 1;
            overflow: hidden;
        }}

        .sidebar {{
            width: 350px;
            background-color: var(--sidebar-bg);
            border-right: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            overflow: hidden;
            z-index: 5;
        }}

        .search-box {{
            padding: 1.25rem 1rem;
            border-bottom: 1px solid var(--border-color);
        }}

        .search-box input {{
            width: 100%;
            padding: 0.6rem 0.75rem;
            background: #f1f5f9;
            border: 1px solid var(--border-color);
            color: var(--text-color);
            border-radius: 6px;
            box-sizing: border-box;
            font-size: 0.9rem;
            transition: border-color 0.2s, box-shadow 0.2s;
        }}

        .search-box input:focus {{
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px var(--primary-light);
        }}

        .node-list {{
            flex: 1;
            overflow-y: auto;
        }}

        .node-item {{
            padding: 1rem;
            border-bottom: 1px solid #f1f5f9;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .node-item:hover {{
            background: #f8fafc;
        }}

        .node-item.active {{
            background: #eff6ff;
            border-left: 4px solid var(--primary);
        }}

        .node-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.35rem;
        }}

        .node-fqn {{
            font-weight: 500;
            font-size: 0.875rem;
            word-break: break-all;
            color: #1e293b;
        }}

        .suspiciousness-badge {{
            padding: 2px 8px;
            border-radius: 6px;
            font-size: 0.7rem;
            font-weight: 600;
            color: white;
            min-width: 44px;
            text-align: center;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
        }}

        .node-meta {{
            font-size: 0.75rem;
            color: var(--text-muted);
            display: flex;
            gap: 8px;
        }}

        .main-content {{
            flex: 1;
            position: relative;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            background: #fcfcfd;
        }}

        .graph-container {{
            flex: 1;
            position: relative;
            cursor: grab;
        }}

        .graph-container:active {{
            cursor: grabbing;
        }}

        #mermaidGraph {{
            width: 100%;
            height: 100%;
        }}

        .details-overlay {{
            position: absolute;
            top: 1rem;
            right: 1rem;
            width: 500px;
            max-height: calc(100% - 2rem);
            background: rgba(255, 255, 255, 0.98);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            overflow-y: auto;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
            z-index: 100;
            transition: transform 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        }}

        .details-overlay.hidden {{
            transform: translateX(550px);
        }}

        .controls {{
            position: absolute;
            bottom: 1.5rem;
            left: 1.5rem;
            display: flex;
            gap: 0.5rem;
            z-index: 50;
            background: white;
            padding: 4px;
            border-radius: 8px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            border: 1px solid var(--border-color);
        }}

        .control-btn {{
            background: white;
            border: 1px solid transparent;
            color: var(--text-color);
            width: 36px;
            height: 36px;
            border-radius: 6px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.1rem;
            transition: all 0.2s;
        }}

        .control-btn:hover {{
            background: #f1f5f9;
            border-color: var(--border-color);
        }}

        .metrics-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1rem;
            margin-bottom: 1.5rem;
        }}

        .metric-card {{
            background: #f8fafc;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #f1f5f9;
        }}

        .metric-label {{
            font-size: 0.65rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        }}

        .metric-value {{
            font-size: 1.1rem;
            font-weight: 700;
            margin-top: 0.25rem;
            color: #1e293b;
        }}

        pre {{
            background: #f1f5f9;
            padding: 1rem;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 0.75rem;
            margin-top: 0.5rem;
            border: 1px solid #e2e8f0;
            line-height: 1.5;
        }}

        /* Code block styling */
        .code-container {{
            margin-top: 1rem;
            position: relative;
        }}

        .code-container pre {{
            margin: 0;
            max-height: 400px;
        }}

        h2 {{ margin: 0 0 1.25rem 0; font-size: 1.25rem; font-weight: 600; color: #1e293b; }}
        h3 {{ font-size: 0.8rem; margin: 1.5rem 0 0.75rem 0; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; font-weight: 600; }}

        .close-btn {{
            position: absolute;
            top: 1rem;
            right: 1rem;
            background: #f1f5f9;
            border: none;
            color: var(--text-muted);
            cursor: pointer;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.2rem;
            transition: all 0.2s;
        }}
        
        .close-btn:hover {{
            background: #e2e8f0;
            color: var(--text-color);
        }}

        ::-webkit-scrollbar {{
            width: 6px;
        }}
        ::-webkit-scrollbar-track {{
            background: transparent;
        }}
        ::-webkit-scrollbar-thumb {{
            background: #e2e8f0;
            border-radius: 10px;
        }}
        ::-webkit-scrollbar-thumb:hover {{
            background: #cbd5e1;
        }}
    </style>
</head>
<body>
    <header>
        <div style="display:flex; align-items:center; gap:12px;">
            <div style="background: var(--primary); padding: 8px; border-radius: 8px;">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2v8"/><path d="m16 6-4 4-4-4"/><rect width="20" height="8" x="2" y="14" rx="2"/></svg>
            </div>
            <h1 style="margin:0; font-size:1.1rem; font-weight:600;">Call Graph Explorer</h1>
        </div>
        <div class="stats" style="font-size:0.8rem; font-weight:500; color:var(--text-muted)">
            <span style="background:#f1f5f9; padding:4px 10px; border-radius:20px;">{len(nodes)} Nodes</span>
            <span style="background:#f1f5f9; padding:4px 10px; border-radius:20px; margin-left:8px;">{len(edges)} Edges</span>
        </div>
    </header>

    <div class="container">
        <div class="sidebar">
            <div class="search-box">
                <input type="text" id="nodeSearch" placeholder="Filter functions or modules...">
            </div>
            <div class="node-list" id="nodeList"></div>
        </div>

        <div class="main-content">
            <div class="graph-container" id="graphContainer">
                <div class="mermaid" id="mermaidGraph">
                    {mermaid_graph}
                </div>
            </div>

            <div class="controls">
                <button class="control-btn" id="zoomIn" title="Zoom In">+</button>
                <button class="control-btn" id="zoomOut" title="Zoom Out">−</button>
                <button class="control-btn" id="resetZoom" title="Reset View">⟲</button>
            </div>

            <div id="nodeDetails" class="details-overlay hidden">
                <button class="close-btn" id="closeDetails" title="Close">×</button>
                <h2 id="detailFQN">Node FQN</h2>
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Suspiciousness</div>
                        <div id="detailSuspiciousness" class="metric-value">0.00</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Calls</div>
                        <div id="detailExecutions" class="metric-value">0</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Avg Duration</div>
                        <div id="detailDuration" class="metric-value">0ms</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Type</div>
                        <div id="detailType" class="metric-value">-</div>
                    </div>
                </div>
                
                <h3>Location</h3>
                <p id="detailFile" class="node-meta" style="word-break:break-all; font-family:monospace; color:#475569;"></p>

                <div id="sourceCodeSection">
                    <h3>Source Code</h3>
                    <div class="code-container">
                        <pre><code id="detailCode" class="language-python"># Code will appear here</code></pre>
                    </div>
                </div>

                <div id="executionLogsSection">
                    <h3>Recent Executions (Last 5)</h3>
                    <div id="executionLogs"></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const nodes = {json.dumps(nodes)};
        let panZoomInstance = null;

        function getSuspiciousColor(score) {{
            if (score === null || score < 0) return '#94a3b8';
            const r = Math.floor(34 + (239 - 34) * score);
            const g = Math.floor(197 + (68 - 197) * score);
            const b = Math.floor(94 + (68 - 94) * score);
            return `rgb(${{r}}, ${{g}}, ${{b}})`;
        }}

        function renderNodeList(filter = '') {{
            const list = document.getElementById('nodeList');
            list.innerHTML = '';
            
            const filteredNodes = nodes.filter(n => 
                n.fqn.toLowerCase().includes(filter.toLowerCase()) || 
                n.file.toLowerCase().includes(filter.toLowerCase())
            ).sort((a, b) => (b.suspiciousness || 0) - (a.suspiciousness || 0));

            filteredNodes.forEach(node => {{
                const div = document.createElement('div');
                div.className = 'node-item';
                div.onclick = () => {{
                    showDetails(node);
                }};
                
                const score = node.suspiciousness || 0;
                const color = getSuspiciousColor(score);

                div.innerHTML = `
                    <div class="node-header">
                        <span class="node-fqn">${{node.fqn}}</span>
                        <span class="suspiciousness-badge" style="background-color: ${{color}}">
                            ${{score.toFixed(3)}}
                        </span>
                    </div>
                    <div class="node-meta">
                        <span>${{node.type}}</span>
                        <span>•</span>
                        <span>${{node.execution_count}} calls</span>
                    </div>
                `;
                list.appendChild(div);
            }});
        }}

        function showDetails(node) {{
            const details = document.getElementById('nodeDetails');
            details.classList.remove('hidden');

            document.getElementById('detailFQN').innerText = node.fqn;
            document.getElementById('detailSuspiciousness').innerText = (node.suspiciousness || 0).toFixed(4);
            document.getElementById('detailSuspiciousness').style.color = getSuspiciousColor(node.suspiciousness || 0);
            document.getElementById('detailExecutions').innerText = node.execution_count;
            document.getElementById('detailDuration').innerText = (node.avg_duration * 1000).toFixed(2) + 'ms';
            document.getElementById('detailType').innerText = node.type;
            document.getElementById('detailFile').innerText = `${{node.file}}:${{node.start_line}}`;

            const codeElement = document.getElementById('detailCode');
            if (node.code) {{
                codeElement.textContent = node.code;
                Prism.highlightElement(codeElement);
                document.getElementById('sourceCodeSection').style.display = 'block';
            }} else {{
                document.getElementById('sourceCodeSection').style.display = 'none';
            }}

            const logs = document.getElementById('executionLogs');
            logs.innerHTML = '';
            
            if (node.executions && node.executions.length > 0) {{
                node.executions.slice(0, 5).forEach((ex, i) => {{
                    const log = document.createElement('pre');
                    log.innerHTML = `<strong>Ex #${{i+1}}</strong> [${{ex.timestamp.toFixed(3)}}s]\\n<strong>Args:</strong> ${{JSON.stringify(ex.args, null, 1)}}\\n<strong>Ret:</strong> ${{JSON.stringify(ex.return_value, null, 1)}}`;
                    logs.appendChild(log);
                }});
            }} else {{
                logs.innerHTML = '<p class="node-meta">No execution data recorded.</p>';
            }}
        }}

        // Initialization
        mermaid.initialize({{ 
            startOnLoad: true, 
            theme: 'default',
            flowchart: {{ 
                useMaxWidth: false,
                htmlLabels: true,
                curve: 'basis'
            }},
            themeVariables: {{
                primaryColor: '#ffffff',
                primaryTextColor: '#1e293b',
                primaryBorderColor: '#e2e8f0',
                lineColor: '#94a3b8',
                secondaryColor: '#f8fafc',
                tertiaryColor: '#f1f5f9'
            }}
        }});

        window.addEventListener('load', () => {{
            const checkMermaid = setInterval(() => {{
                const svg = document.querySelector('#mermaidGraph svg');
                if (svg) {{
                    clearInterval(checkMermaid);
                    svg.style.width = '100%';
                    svg.style.height = '100%';
                    
                    panZoomInstance = svgPanZoom(svg, {{
                        zoomEnabled: true,
                        controlIconsEnabled: false,
                        fit: true,
                        center: true,
                        minZoom: 0.1,
                        maxZoom: 10
                    }});

                    document.getElementById('zoomIn').onclick = () => panZoomInstance.zoomIn();
                    document.getElementById('zoomOut').onclick = () => panZoomInstance.zoomOut();
                    document.getElementById('resetZoom').onclick = () => {{
                        panZoomInstance.resetZoom();
                        panZoomInstance.center();
                    }};

                    // Add click handlers for mermaid nodes
                    const mermaidNodes = svg.querySelectorAll('.node');
                    mermaidNodes.forEach(node => {{
                        node.style.cursor = 'pointer';
                        node.onclick = () => {{
                            // Use the ID to find the node index
                            const nodeId = node.id;
                            // Mermaid sometimes prefixes IDs or adds suffixes
                            const match = nodeId.match(/node_([0-9]+)/);
                            if (match) {{
                                const index = parseInt(match[1]);
                                if (nodes[index]) {{
                                    showDetails(nodes[index]);
                                    return;
                                }}
                            }}
                            
                            // Fallback to label search if ID lookup fails
                            const label = node.querySelector('.nodeLabel')?.innerText;
                            if (label) {{
                                const foundNode = nodes.find(n => n.fqn === label || n.fqn.replace("<", "(").replace(">", ")") === label);
                                if (foundNode) showDetails(foundNode);
                            }}
                        }};
                    }});
                }}
            }}, 100);
        }});

        document.getElementById('nodeSearch').addEventListener('input', (e) => {{
            renderNodeList(e.target.value);
        }});

        document.getElementById('closeDetails').onclick = () => {{
            document.getElementById('nodeDetails').classList.add('hidden');
        }};

        renderNodeList();
    </script>
</body>
</html>
"""
    with open(output_path, "w") as f:
        f.write(html_template)
    print(f"Visualization generated at: {output_path}")

if __name__ == "__main__":
    json_path = "experiments/youtube-dl/call_graph_youtube-dl_1.json"
    if not os.path.exists(json_path):
        print(f"Error: {json_path} not found. Run the dynamic call graph script first.")
        sys.exit(1)
        
    with open(json_path, "r") as f:
        data = json.load(f)
    
    os.makedirs("artifacts", exist_ok=True)
    generate_html(data, "experiments/youtube-dl/call_graph.html")
