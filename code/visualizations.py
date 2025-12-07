# %%
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


# %%
#Plot the convergence of the fitness and the population
def plot_convergence(avg_costs, best_costs, worst_costs,title):
    iters = range(1,len(avg_costs) + 1)

    plt.figure(figsize=(10,5))
    plt.plot(iters, best_costs, label="Best cost", linewidth=2)
    plt.plot(iters, avg_costs, label="Average cost", linestyle="--")
    plt.plot(iters, worst_costs, label="Worst cost", linestyle=":")

    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title(title)
    plt.grid(True)
    plt.xticks(iters)
    plt.legend(loc="upper right")
    plt.savefig(f"../results/{title}.png")


# %%
def plot_best_cost_comparison(aco_best, bf_best, title="ACO vs BF Best Cost Comparison"):

    #Calculate X axis
    max_iters = max(len(aco_best), len(bf_best))
    common_x = np.arange(1, max_iters + 1)

    aco_best = np.array(aco_best, dtype=float)
    aco_best_padded = np.pad(aco_best, (0, max_iters - len(aco_best)), 
                             constant_values=np.nan)
    
    bf_best = np.array(bf_best, dtype=float)
    bf_best_padded = np.pad(bf_best, (0, max_iters - len(bf_best)), 
                            constant_values=np.nan)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(common_x, aco_best_padded, label="ACO Best cost", linewidth=2)
    plt.plot(common_x, bf_best_padded, label="Brute Force Best cost", linewidth=2)

    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Best Cost")
    plt.xticks(common_x)
    plt.grid(True)
    plt.legend()

    #Allineate y values
    all_values = np.concatenate([aco_best, bf_best])
    plt.ylim(min(all_values) - 2, max(all_values) + 2)

    plt.tight_layout()
    plt.show()


# %%
def plot_tsp_graph(G, pos, best_path=None, title="TSP Graph"):
    
    plt.figure(figsize=(7,5))
    
    #Draw nodes and edges
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=800)
    nx.draw_networkx_edges(G, pos, edge_color='lightgray')

    
    #Add weight visualization
    edge_labels = {(u,v): G[u][v]['weight'] for u,v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title(title)
    plt.show()


# %%
def draw_tsp_path_clean(G, path, title="Shortest Path"):
    
    path_edges = []

    #Take edges from the path
    for i in range (len(path) - 1):
        path_edges.append((path[i],path[i+1]))

    #Create graph to be visualized
    H = nx.Graph()
    H.add_nodes_from(path)
    for u, v in path_edges:
        H.add_edge(u, v, weight=G[u][v]['weight'])

    #Circular layout for more readability
    pos = nx.circular_layout(H)

    plt.figure(figsize=(7, 7))

    #Draw nodes
    nx.draw_networkx_nodes(H, pos, node_size=1000, node_color="#90EE90", edgecolors="black")

    #Draw edges
    nx.draw_networkx_edges(H, pos, edgelist=path_edges, width=3, edge_color="red")

    #Label for the nodes
    nx.draw_networkx_labels(H, pos, font_size=14, font_weight="bold")

    #Weight for the edges
    edge_labels = {(u, v): G[u][v]['weight'] for u, v in path_edges}
    nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels, font_size=12)

    plt.title(title, fontsize=18)
    plt.axis("off")
    plt.tight_layout()
    plt.show()
