# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import csv
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import time


# %%
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




# %%
#Fitness function: calculate the length of the path
def path_length(G, path):
    length = 0

    for i in range (len(path) - 1):
        length += G[path[i]][path[i+1]]['weight']

    return length


# %%
def ant_colony_optimization(G, n_ants=10, n_iters=30, alpha=1, beta=2, evaporation=0.4, Q=1):
    nodes = list(G.nodes())
    best_path = None
    best_cost = float('inf')


    
    for _ in range(n_iters):
        all_paths = []
        all_costs = []
        
        for _ in range(n_ants):

            #Randomly selecting a starting node for the ant
            path = [random.choice(nodes)]
            #Set of the unvisited nodes, at the end this must be empty, so the ant has visited all nodes
            unvisited = set(nodes) - set(path)
            
            #Iterate until the ant reach all the nodes once
            while unvisited:
                #Last node reached by the ant
                current = path[-1]
                #List of reachable nodes from currente position
                reachable = list(set(G.neighbors(current)) & unvisited)
                if not reachable:
                    #If there aren't nodes reachable the ant is stuck or it found a valid path
                    break

                probs = []

                #Calculate the distribution probability over the reachable nodes from current node 
                for j in reachable:
                    tau = G[current][j]['pheromone'] ** alpha
                    eta = (1 / G[current][j]['weight']) ** beta
                    probs.append(tau * eta)
                probs = np.array(probs)
                probs /= probs.sum() #normalize
                
                #Sampling the next node to visit by a probability distribution
                next_node = random.choices(reachable, weights=probs, k=1)[0]
                path.append(next_node) #add the sampled node to the path
                unvisited.remove(next_node) # remove the sampled node to the list of unvisited nodes
            

            #Used for skip the calculate of the fitness if the path is not completed
            if len(path) < len(nodes):
                continue
            
            #Calculate fitness for the path founded
            cost = path_length(G, path)
            all_paths.append(path)
            all_costs.append(cost)
            
            #Updating the best solution found
            if cost < best_cost:
                best_cost = cost
                best_path = path
        
        #Simple evapoation of pheromones with a parameter
        for u,v in G.edges():
            G[u][v]['pheromone'] *= (1 - evaporation)
        

        #Updating pheromone of edges and so the probability 
        for path, cost in zip(all_paths, all_costs):
            for i in range(len(path)-1):
                u,v = path[i], path[i+1]
                G[u][v]['pheromone'] += Q / cost
                
    return best_path, best_cost,n_iters * n_ants




#The idea for the exhaustive search is that we have to do a dfs for each node on the graph, this allows us to generate all the possible valid paths.
#I can do this because a solution is in the research space if it is a possible path, so all the possibles paths are in the research space and the reaserch space is close.
#This make the comparison more faster and maybe allows to use bigger graph for comparison.
#So i don't consider all the permutation of nodes for the reaserch space but only the all permutation of nodes that are valid path but without generating all possible permutations, instead i do a dfs for each node in the graph.
#With this method for exhaustive search i still check all the research space.
def dfs(G,start):

    nodes = list(G.nodes())
    best_cost = float('inf')
    best_path = None
    tested_path = 0


    #A list where each element is: set of starting point, cost of the path, set of visited nodes
    stack = [([start], 0, {start})]

    while stack:
        #Return the last element of the list in order to get the last updated infos
        path, cost, visited = stack.pop()
        #Get last element of the path, so the current node
        current = path[-1]

        #If all nodes are visited once
        if len(path) == len(nodes):
            tested_path += 1
            if cost < best_cost:
                best_cost = cost
                best_path = path
            continue
    
        #Checking for neighbors, so we build the path
        for next in G.neighbors(current):
            if next not in visited:
                weight = G[current][next]["weight"]
                #Updating infos
                stack.append((
                    path + [next],
                    cost + weight,
                    visited | {next} # Union beetween two sets
                ))


    return best_path,best_cost,tested_path

    
#Exaustive research used for comparing the performance with ACO
def exhaustive_search(G):
    nodes = list(G.nodes())
    gloabal_best_path = None
    global_best_cost = float('inf')
    global_tested_path = 0
    
    #Run a dfs for each node, 
    for node in nodes:

        local_best_path, local_best_cost,local_tested_path = dfs(G,node)
        global_tested_path += local_tested_path

        if local_best_path is not None and local_best_cost < global_best_cost:
            gloabal_best_path = local_best_path
            global_best_cost = local_best_cost

    return gloabal_best_path,global_best_cost, global_tested_path



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



# %%
def compare_tsp_algorithms(G, n_ants=10, n_iters=30):

    #Run aco
    start_aco = time.time()
    best_path_aco, best_cost_aco,tested_paths_aco = ant_colony_optimization(G, n_ants=n_ants, n_iters=n_iters)
    time_aco = time.time() - start_aco

    #Run brute force
    start_brute = time.time()
    best_path_brute, best_cost_brute, tested_paths_bf = exhaustive_search(G)
    time_brute = time.time() - start_brute

    print(f"Number of tested path with ACO: {tested_paths_aco}")
    print(f"Number of tested paths with Brute Force: {tested_paths_bf}")

    #Results
    result = {
        "ACO": {"path": best_path_aco, "cost": best_cost_aco, "time": time_aco,"tested_path": tested_paths_aco},
        "Brute Force": {"path": best_path_brute, "cost": best_cost_brute, "time": time_brute, "tested_path": tested_paths_bf},
        "Speedup": time_brute/time_aco if time_aco>0 else None
    }
    return result


# %%
n_nodes = 14
n_edges = 70
G = create_random_graph(n_nodes, n_edges)

#Run Comparison
results = compare_tsp_algorithms(G, n_ants=20, n_iters=50)

#Save comparison results in a csv file
filename = "../results/comparison.csv"
with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        
        writer.writerow([
            n_nodes,
            n_edges,
            f"{results['ACO']['time']:.3f}",
            f"{results['Brute Force']['time']:.3f}",
            f"{results['Speedup']:.3f}",
            results["ACO"]["tested_path"],
            results["Brute Force"]["tested_path"],
            results["ACO"]["cost"],
            results["Brute Force"]["cost"],
            results["ACO"]["path"],
            results["Brute Force"]["path"]
        ])
print(f"Risultati salvati in {filename}")

#Print stats
print("=== Confronto TSP ACO vs Brute Force ===")
for algo, res in results.items():
    if algo != "Speedup":
        print(f"{algo}: best cost = {res['cost']}, best path = {res['path']}, time = {res['time']:.4f} s")
print(f"Speedup (Brute/ACO) ≈ {results['Speedup']:.2f}×")

#Visualizations
pos = nx.spring_layout(G, seed=42)
plot_tsp_graph(G,pos,title= f"TSP GRAPH ({n_nodes} nodes - {n_edges} edges)")
draw_tsp_path_clean(G, results["ACO"]["path"], title= "Shortest Path ACO")
draw_tsp_path_clean(G, results["Brute Force"]["path"], title= "Shortest Path (Brute Force)")


