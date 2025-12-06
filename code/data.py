import networkx as nx
import random


def create_random_graph(n_nodes=10, n_edges=15, seed=42):

    #For reproducibility
    random.seed(seed)

    #Create an oriented grpah
    G = nx.DiGraph()
    G.add_nodes_from(range(n_nodes))

    #Generate al possible edges without duplicates
    possible_edges = []

    for i in range(n_nodes):
        for j in range(i+1,n_nodes):
            possible_edges.append((i,j))
            possible_edges.append((j,i))

    number_of_possible_edges = len(possible_edges)
    print(f"Number of possible edges: {number_of_possible_edges}")
    
    

    #Check if the number of edges given in input is greater than the actual number of possible edges given the number of nodes
    if n_edges > number_of_possible_edges:
        raise ValueError(f"Numero di archi troppo alto. Massimo per {n_nodes} nodi: {number_of_possible_edges}")

    #Sampling the edges from all possibole edges. 
    #NOTE: this can actually be a bug, if the selected edges doesn't build a graph with al the nodes, i have no safety, so i can sample different edges that are not connected and so is impossible to explore the graph 
    #TODO: need to solve this bug, maybe found a proof of this bug
    selected_edges = random.sample(possible_edges, n_edges)
    print(f"Number of sampled edges: {len(selected_edges)}")

    #Add weighted edges in the graph with initial value for the pheromone
    for u, v in selected_edges:
        weight = random.randint(1, 10)   # weight beetween 1 and 10
        G.add_edge(u, v, weight=weight, pheromone = 0.1) #NOTE: maybe is better to find an empirical rule for initialize the value of the pheromone

    return G