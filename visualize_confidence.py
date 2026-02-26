# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "matplotlib",
# ]
# ///

import json
import matplotlib.pyplot as plt
import argparse
import sys

def main():
    conf_file = "/Users/ashish/master-thesis/kg-guided-codegen/experiments/youtube-dl/confidence_evolution.json"
    parser = argparse.ArgumentParser(description="Visualize confidence evolution.")
    parser.add_argument("--input", "-i", type=str, default=conf_file, help="Path to confidence_evolution.json")
    parser.add_argument("--output", "-o", type=str, default="confidence_evolution.png", help="Output plot image file")
    parser.add_argument("--top-k", "-k", type=int, default=10, help="Number of nodes to highlight with separate colors")
    args = parser.parse_args()

    try:
        with open(args.input, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {args.input} not found.")
        sys.exit(1)
    
    if not data:
        print("Empty data.")
        return

    iterations = []
    # node_fqn -> list of scores aligned with iterations
    node_scores = {}
    
    # Initialize node_scores for all known nodes
    all_nodes = set()
    for entry in data:
        all_nodes.update(entry.get("scores", {}).keys())
    
    for node in all_nodes:
        node_scores[node] = []

    for entry in data:
        it = entry["iteration"]
        iterations.append(it)
        scores = entry.get("scores", {})
        for node in all_nodes:
            # fill 0.0 if not present in that iteration
            node_scores[node].append(scores.get(node, 0.0))

    # sort nodes by their max confidence throughout the evolution
    node_max_score = {node: max(scores) for node, scores in node_scores.items()}
    sorted_nodes = sorted(node_max_score.keys(), key=lambda n: node_max_score[n], reverse=True)

    top_nodes = sorted_nodes[:args.top_k]
    other_nodes = sorted_nodes[args.top_k:]

    plt.figure(figsize=(12, 8))
    
    # Plot other nodes with low alpha and grey color
    for node in other_nodes:
        if node_max_score[node] > 0.01: # only plot if it's meaningful
            plt.plot(iterations, node_scores[node], color='grey', alpha=0.1, linewidth=1)
            
    # Plot top nodes with labels
    for node in top_nodes:
        plt.plot(iterations, node_scores[node], label=node, linewidth=2, marker='o', markersize=4)

    plt.xlabel("Iteration")
    plt.ylabel("Confidence Score")
    plt.title("Evolution of Confidence Scores per Node")
    
    # Shrink current axis by 20% to put legend outside
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small', title="Top nodes")
    
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # ensure everything fits
    plt.savefig(args.output, bbox_inches="tight")
    print(f"Visualization saved to {args.output}")
    # try showing it, if not headless
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()
