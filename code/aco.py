# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

import itertools
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import time


# +
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


    print(possible_edges)
    

    #Check if the number of edges given in input is greater than the actual number of possible edges given the number of nodes
    if n_edges > len(possible_edges):
        raise ValueError(f"Numero di archi troppo alto. Massimo per {n_nodes} nodi: {len(possible_edges)}")

    #Sampling the edges from all possibole edges. 
    #NOTE: this can actually be a bug, if the selected edges doesn't build a graph with al the nodes, i have no safety, so i can sample different edges that are not connected and so is impossible to explore the graph 
    #TODO: need to solve this bug, maybe found a proof of this bug
    selected_edges = random.sample(possible_edges, n_edges)

    #Add weighted edges in the graph with initial value for the pheromone
    for u, v in selected_edges:
        weight = random.randint(1, 10)   # weight beetween 1 and 10
        G.add_edge(u, v, weight=weight, pheromone = 0.1) #NOTE: maybe is better to find an empirical rule for initialize the value of the pheromone

    return G



# -

#Fitness function: calculate the length of the path
def path_length(G, path):
    length = 0

    for i in range (len(path) - 1):
        length += G[path[i]][path[i+1]]['weight']

    return length


# +
def ant_colony_optimization(G, n_ants=10, n_iters=30, alpha=1, beta=2, evaporation=0.4, Q=1):
    nodes = list(G.nodes())
    best_path = None
    best_cost = float('inf')
    
    for _ in range(n_iters):
        all_paths = []
        all_costs = []
        
        for _ in range(n_ants):
            path = [random.choice(nodes)]
            unvisited = set(nodes) - set(path)
            
            while unvisited:
                current = path[-1]
                reachable = list(set(G.neighbors(current)) & unvisited)
                if not reachable:
                    break
                probs = []
                for j in reachable:
                    tau = G[current][j]['pheromone'] ** alpha
                    eta = (1 / G[current][j]['weight']) ** beta
                    probs.append(tau * eta)
                probs = np.array(probs)
                probs /= probs.sum()
                
                next_node = random.choices(reachable, weights=probs, k=1)[0]
                path.append(next_node)
                unvisited.remove(next_node)
            
            if len(path) < len(nodes):
                continue
            
            cost = path_length(G, path)
            all_paths.append(path)
            all_costs.append(cost)
            
            if cost < best_cost:
                best_cost = cost
                best_path = path
        
        # Aggiorna feromoni
        for u,v in G.edges():
            G[u][v]['pheromone'] *= (1 - evaporation)
        
        for path, cost in zip(all_paths, all_costs):
            for i in range(len(path)-1):
                u,v = path[i], path[i+1]
                G[u][v]['pheromone'] += Q / cost
                
    return best_path, best_cost



#Exaustive research used for comparing the performance with ACO
#TODO: change this function in a way that only the permutations that correponds to a path are taken in consideration.
#NOTE: i can do this because a solution is in the research space if it is a possible path, so all the possibles paths are in the research space and the reaserch space is close.
def exhaustive_search(G):
    nodes = list(G.nodes())
    best_path = None
    best_cost = float('inf')
    permutations_of_nodes = itertools.permutations(nodes)

    for perm in  permutations_of_nodes:
        valid = True
        #Check if the permutation selected is a valid path or not
        for i in range(len(perm)-1):
            if not G.has_edge(perm[i], perm[i+1]):
                valid = False
                break

        #Used for skip the calcuation of the fitness for invalid path
        if not valid:
            continue
        
        #Calculate fitness
        cost = path_length(G, perm)
        if cost < best_cost:
            best_cost = cost
            best_path = perm

    return best_path, best_cost



# -

def plot_tsp_graph(G, pos, best_path=None, highlight_best_path=True, title="TSP Graph"):
    """
    Visualizza un grafo TSP con NetworkX.

    Parametri:
    - G: grafo NetworkX
    - pos: layout nodi (es. nx.spring_layout(G))
    - best_path: lista di nodi del percorso migliore (facoltativo)
    - highlight_best_path: se True evidenzia solo il percorso migliore
    - title: titolo del grafico
    """
    plt.figure(figsize=(7,5))
    
    # Disegna nodi e label
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=800)
    
    # Disegna archi
    if best_path is None or not highlight_best_path:
        # Mostra tutti gli archi normali
        nx.draw_networkx_edges(G, pos, edge_color='lightgray')
    else:
        # Mostra tutti gli archi in grigio
        nx.draw_networkx_edges(G, pos, edge_color='lightgray')
        # Evidenzia solo il percorso migliore
        path_edges = [(best_path[i], best_path[i+1]) for i in range(len(best_path)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
    
    # Etichette dei pesi
    edge_labels = {(u,v): G[u][v]['weight'] for u,v in G.edges()}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    
    plt.title(title)
    plt.show()



# +

def draw_tsp_path_clean(G, path, title="Shortest Path"):
    """
    Visualizza un percorso TSP in modo chiaro usando un layout circolare.
    Disegna solo gli archi e nodi del percorso.
    """

    # Estrai archi del percorso
    path_edges = [(path[i], path[i+1]) for i in range(len(path) - 1)]

    # Crea un sottografo minimale
    H = nx.Graph()
    H.add_nodes_from(path)
    for u, v in path_edges:
        H.add_edge(u, v, weight=G[u][v]['weight'])

    # Layout circolare per leggibilità
    pos = nx.circular_layout(H)

    plt.figure(figsize=(7, 7))

    # Disegna nodi
    nx.draw_networkx_nodes(H, pos, node_size=1000, node_color="#90EE90", edgecolors="black")

    # Disegna archi
    nx.draw_networkx_edges(H, pos, edgelist=path_edges, width=3, edge_color="red")

    # Etichette dei nodi
    nx.draw_networkx_labels(H, pos, font_size=14, font_weight="bold")

    # Etichette dei pesi degli archi
    edge_labels = {(u, v): G[u][v]['weight'] for u, v in path_edges}
    nx.draw_networkx_edge_labels(H, pos, edge_labels=edge_labels, font_size=12)

    plt.title(title, fontsize=18)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# -

def compare_tsp_algorithms(G, n_ants=10, n_iters=30):

    # ACO
    start_aco = time.time()
    best_path_aco, best_cost_aco = ant_colony_optimization(G, n_ants=n_ants, n_iters=n_iters)
    time_aco = time.time() - start_aco

    # Brute force
    start_brute = time.time()
    best_path_brute, best_cost_brute = exhaustive_search(G)
    time_brute = time.time() - start_brute

    # Risultati
    result = {
        "ACO": {"path": best_path_aco, "cost": best_cost_aco, "time": time_aco},
        "Brute Force": {"path": best_path_brute, "cost": best_cost_brute, "time": time_brute},
        "Speedup": time_brute/time_aco if time_aco>0 else None
    }
    return result


# +
n_nodes = 15
n_edges = 120
G = create_random_graph(n_nodes, n_edges)

# Confronto algoritmi
results = compare_tsp_algorithms(G, n_ants=20, n_iters=50)

print("=== Confronto TSP ACO vs Brute Force ===")
for algo, res in results.items():
    if algo != "Speedup":
        print(f"{algo}: best cost = {res['cost']}, best path = {res['path']}, time = {res['time']:.4f} s")
print(f"Speedup (Brute/ACO) ≈ {results['Speedup']:.2f}×")

# Visualizzazione del grafo
pos = nx.spring_layout(G, seed=42)
plot_tsp_graph(G,pos,title= f"TSP GRAPH ({n_nodes} nodes - {n_edges} edges)")
draw_tsp_path_clean(G, results["ACO"]["path"], title= "Shortest Path ACO")
draw_tsp_path_clean(G, results["Brute Force"]["path"], title= "Shortest Path (Brute Force)")


