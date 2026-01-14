from agent.graph import one_shot_codegen_agent
from agent.tools import configure_logging

configure_logging(level="INFO")

print(one_shot_codegen_agent.get_graph().draw_ascii())
problem_statement = """
Write a program that finds the minimum time required to deliver packages to a set of target cities and reach a final destination, given a limited battery range and specific time windows for delivery.

Input
You are given a weighted undirected graph with $N$ nodes (cities) and $M$ edges (roads).$N$ 
(Nodes): Up to 200.$K$ 
(Target Cities): Up to 16.$C$ 
(Charging Stations): A subset of $N$.
Battery Capacity: $B$ (max distance unit).

Rules
Movement: Traversing an edge takes time equal to its weight and consumes battery charge equal to its weight.
Battery: You cannot traverse an edge if your current charge is less than the edge weight. You start with full charge ($B$).
Recharging: Arriving at a Charging Station node instantly refills your battery to $B$.
Delivery Targets: You must visit every node in the set $K$.
Time Windows: Each target node $k \in K$ has a time window $[start_k, end_k]$. You can only "visit" (deliver to) the node if your total elapsed time $T$ satisfies $start_k \le T \le end_k$.If you arrive early ($T < start_k$), you must wait at that node until $T = start_k$.If you arrive late ($T > end_k$), the path is invalid.
End: After visiting all targets, you must end at Node $N-1$.

Output
Output the minimum total time to complete the route. If no valid route exists, output -1.
"""

result = one_shot_codegen_agent.invoke(
    {
        "problem": "Write a program that finds the minimum time required to deliver packages to a set of target cities and reach a final destination, given a limited battery range and specific time windows for delivery."
    }
)

print("=== GENERATED CODE ===")
print(result["generated_code"])
