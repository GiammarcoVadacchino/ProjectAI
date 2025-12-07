# %%
import networkx as nx
import random


# %%
def create_random_graph(n_nodes=10, n_edges=15, seed=42, fully_connected=False):

    random.seed(seed)

    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))

    #Generate all possible oriented edges, 
    possible_edges = []
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j:
                possible_edges.append((i, j))



    #Build a fully connected graph, so (N-1)! combinations in the exhaustive search 
    if fully_connected:
        G = nx.complete_graph(n_nodes, create_using=nx.DiGraph())

        #Initialize the wheights and the pheromones
        for u, v in G.edges():
            weight = random.randint(1, 10)
            G[u][v]["weight"] = weight
            G[u][v]["pheromone"] = 0.1

        return G

    number_of_possible_edges = len(possible_edges)
    print(f"Number of possible edges: {number_of_possible_edges}")

    if n_edges > number_of_possible_edges:
        raise ValueError(f"Numero di archi troppo alto. Massimo per {n_nodes} nodi: {number_of_possible_edges}")

    #Sampling from all the possible edges, not guaranteed that the graph has a Hamiltonian path, but this bug is very very rare in the graphs that will be considered
    selected_edges = random.sample(possible_edges, n_edges)
    print(f"Number of sampled edges: {len(selected_edges)}")

    for u, v in selected_edges:
        weight = random.randint(1, 10)
        G.add_edge(u, v, weight=weight, pheromone=0.1)

    return G

