import numpy as np
from collections import deque, defaultdict


def longest_hop_undirect(edge_index, num_nodes):
    edge_index = np.array(edge_index)
    if num_nodes is None:
        num_nodes = edge_index.max()

    adj_list = defaultdict(list)
    for u, v in edge_index.T:
        adj_list[u].append(v)
        adj_list[v].append(u)  # Assume undirected graph

    # Store longest hop for each node
    longest_hops = [-1] * num_nodes

    # Perform BFS for each node
    for start_node in range(num_nodes):
        visited = set()
        queue = deque([(start_node, 0)])  # (current_node, current_hop)

        while queue:
            current_node, current_hop = queue.popleft()

            if current_node in visited:
                continue
            visited.add(current_node)

            # Update the longest hop for the start_node
            longest_hops[start_node] = max(longest_hops[start_node], current_hop)

            # Enqueue all unvisited neighbors
            for neighbor in adj_list[current_node]:
                if neighbor not in visited:
                    queue.append((neighbor, current_hop + 1))

    return longest_hops

def longest_hop_direct0(edge_index, num_nodes):
    edge_index = np.array(edge_index)
    if num_nodes is None:
        num_nodes = edge_index.max()
    # Build directed adjacency list
    adj_list = defaultdict(list)
    for u, v in edge_index.T:
        adj_list[u].append(v)  # Only outgoing edges

    # Store longest hop for each node
    longest_hops = [-1] * num_nodes

    # Perform BFS for each node
    for start_node in range(num_nodes):
        visited = set()
        queue = deque([(start_node, 0)])  # (current_node, current_hop)

        while queue:
            current_node, current_hop = queue.popleft()

            if current_node in visited:
                continue
            visited.add(current_node)

            # Update the longest hop for the start_node
            longest_hops[start_node] = max(longest_hops[start_node], current_hop)

            # Enqueue outgoing neighbors only
            for neighbor in adj_list[current_node]:
                if neighbor not in visited:
                    queue.append((neighbor, current_hop + 1))

    return longest_hops

def longest_hop_direct(edge_index, num_nodes):
    edge_index = np.array(edge_index)
    if num_nodes is None:
        num_nodes = edge_index.max()

    # print("edge_index shape:", edge_index.shape)
    # print("Unique nodes in edge_index:", np.unique(edge_index))
    # print("Number of nodes:", num_nodes)
    # Build directed adjacency list
    adj_list = defaultdict(list)
    for u, v in edge_index.T:
        adj_list[u].append(v)  # Directed edges only

    # Store longest hop for each node
    longest_hops = [-1] * num_nodes  # -1 means not reachable

    # Perform BFS for each node
    for start_node in range(num_nodes):
        visited = set()
        queue = deque([(start_node, 0)])  # (current_node, current_hop)

        while queue:
            current_node, current_hop = queue.popleft()

            if current_node in visited:
                continue
            visited.add(current_node)

            # Update the longest hop for the start_node
            longest_hops[start_node] = max(longest_hops[start_node], current_hop)

            # Enqueue outgoing neighbors only
            for neighbor in adj_list[current_node]:
                if neighbor not in visited:
                    queue.append((neighbor, current_hop + 1))

    return longest_hops


# Example usage:
if __name__ == "__main__":
    # edge_index = np.array([[0, 0, 1, 2, 3],
    #                        [1, 2, 2, 3, 4]])
    #
    num_nodes = 200

    # edge_index = np.array([np.arange(num_nodes - 1), np.arange(1, num_nodes)])
    edge_index = np.array([np.arange(num_nodes), np.roll(np.arange(num_nodes), -1)])

    result = longest_hop_direct(edge_index, num_nodes)
    print("Longest hop for each node:", result)
