import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import networkx as nx
import matplotlib.pyplot as plt
import igraph

def q_learning(graph, num_episodes, learning_rate, discount_factor):
    Q = np.ones((len(graph), len(graph)))
    for episode in range(num_episodes):
        # Reset the graph for each episode
        current_graph = graph.copy()
        # Choose a random starting node
        current_node = np.random.randint(len(graph))
        while True:
            # Select an action (next node) based on exploration or exploitation strategy
            action = epsilon_greedy(Q[current_node])
            # Update the Q-table with Q-learning equation
            reward = -current_graph[current_node, action]  # Negative reward for selecting an edge
            Q[current_node, action] = (1 - learning_rate) * Q[current_node, action] + learning_rate * (
                        reward + discount_factor * np.min(Q[action]))
            # Remove the edge between the current node and the selected action
            current_graph[current_node, action] = 0
            current_graph[action, current_node] = 0
            # Move to the next node
            current_node = action
            # If all edges are removed, MST is found for this episode
            if np.sum(current_graph) == 0:
                break
    return Q

def epsilon_greedy(Q_values, epsilon=0.1):
    # Exploration or exploitation strategy
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.randint(len(Q_values))
    else:
        action = np.argmin(Q_values)  # Select the action with the minimum Q-value
    return action

def get_minimum_spanning_tree(Q_table):
    num_nodes = Q_table.shape[0]
    MST = np.zeros_like(Q_table)

    # Create a list of all nodes
    nodes = list(range(num_nodes))

    # Select the starting node
    start_node = nodes[0]

    # Create a set to keep track of visited nodes
    visited = set([start_node])

    # Set the maximum runtime to 5 minutes (300 seconds)
    max_runtime = 300
    start_time = time.time()

    # Iterate until all nodes are visited or the maximum runtime is reached
    while len(visited) < num_nodes and time.time() - start_time < max_runtime:
        min_weight = float('inf')
        min_edge = None

        # Find the minimum weight edge connecting a visited node and an unvisited node
        for node in visited:
            for neighbor in nodes:
                if neighbor not in visited and Q_table[node, neighbor] < min_weight:
                    min_weight = Q_table[node, neighbor]
                    min_edge = (node, neighbor)

        # Add the minimum weight edge to the MST
        if min_edge is not None:
            node1, node2 = min_edge
            MST[node1, node2] = 1
            visited.add(node2)

    return MST
def plot_graph(graph, MST, column_clusters, cluster_colors):
    # Create an igraph graph from the adjacency matrix
    g = igraph.Graph.Adjacency(graph.tolist())
    # Define layout for plotting
    layout = g.layout_reingold_tilford(mode="in", root=[0])
    # Plot the original graph
    fig, ax = plt.subplots()
    g.vs["color"] = [cluster_colors[i] for i in column_clusters]  # Set node color based on cluster
    g.es["color"] = "lightgray"  # Set edge color
    g.vs["label"] = range(len(graph))  # Set node labels
    g.es["width"] = 0.1  # Set edge width
    igraph.plot(g, target=ax, layout=layout, bbox=(300, 300))

    # Highlight edges from the minimum spanning tree
    for i in range(len(graph)):
        for j in range(len(graph)):
            if MST[i, j] == 1:
                g.add_edge(i, j)
    g.es.select(color="lightgray").delete()  # Delete non-tree edges
    g.es["color"] = "blue"  # Set edge color for tree edges
    g.vs['width'] = 0.5
    g.es["width"] = 1.0  # Set edge width for tree edges
    igraph.plot(g, target=ax, layout=layout, bbox=(300, 300))
def main():
    np.random.seed(0)
    start_time = time.time()
    dataset_path = input("Enter the dataset path: ")
    graph_data = pd.read_csv(dataset_path, index_col=False)
    counts = graph_data.values
    # Prompt the user for the number of clusters
    num_clusters = int(input("Enter the number of clusters: "))
    # Cluster columns using K-means
    kmeans = KMeans(n_clusters=num_clusters)
    column_clusters = kmeans.fit_predict(counts.T)
    # Create a new graph with cluster-wise counts
    cluster_counts = np.zeros((num_clusters, num_clusters))
    for i in range(counts.shape[1]):
        for j in range(i + 1, counts.shape[1]):
            cluster_counts[column_clusters[i], column_clusters[j]] += counts[i, j]
            cluster_counts[column_clusters[j], column_clusters[i]] += counts[j, i]

    learning_rate = 0.1
    discount_factor = 0.9
    num_episodes = 500
    Q_table = q_learning(cluster_counts, num_episodes, learning_rate, discount_factor)
    MST = get_minimum_spanning_tree(Q_table)

    # Define cluster colors for plotting
    cluster_colors = ["green", "blue", "red", 'pink', 'violet', 'gray', 'yellow', 'brown',
                      'magenta', 'orange']
    plot_graph(cluster_counts, MST, column_clusters, cluster_colors)
    plt.show()

    # Print cells in each cluster
    cluster_cells = {}
    for i, cluster_idx in enumerate(column_clusters):
        if cluster_idx not in cluster_cells:
            cluster_cells[cluster_idx] = []
        cluster_cells[cluster_idx].append(i)

    for cluster_idx, cells in cluster_cells.items():
        print(f"Cluster {cluster_idx}: Cells {cells}")

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time: {execution_time} seconds")

if __name__ == '__main__':
     main()
