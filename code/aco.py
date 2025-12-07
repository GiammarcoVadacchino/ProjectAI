# %%
import numpy as np
import random


# %%
#Fitness function: calculate the length of the path
def path_length(G, path):
    length = 0

    for i in range (len(path) - 1):
        length += G[path[i]][path[i+1]]['weight']

    return length


# %%
"""
Classic ANt System: the goal is to update the pheromones of all edges that are crossed by all the ants.
So if an edge has a low cost, more pheromone is gonna be released on that edge, so this makes greater the probability that the edge is selected by other ants in later iterations.
"""

def update_pheromone_AS(G, paths, costs, Q=1, **kwargs):
    for path, cost in zip(paths, costs):
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            G[u][v]['pheromone'] += Q / cost # ph = Q / total cost of the path

# %%
"""
Elitist Ant System: it is like the classical update, but for each iteration the pheromones of the best path founded are increased by a factor.
So this accelerate the convergance, it increase the probability that in the later iterations the other ants follows these "elitist" path.
"""

def update_pheromone_EAS(G, paths, costs, best_path_global, best_cost_global, Q=1, elitist_factor=5, **kwargs):

    update_pheromone_AS(G, paths, costs, Q) # classic update

    #Increasing pheromones on the best path
    for i in range(len(best_path_global)-1):
        u, v = best_path_global[i], best_path_global[i+1]
        G[u][v]['pheromone'] += elitist_factor * Q / best_cost_global # classic update but multiplied by a costant 


# %%
"""
Rank-Based Ant System: the idea is to not update all the path founded by all the ants, but only the path that are in the top K paths, so the K path with lowest cost, so lowest fitness value.
So the best paths have more increment of pheromons compared to the lower paths in the top K, so if ad path has a better position in the top K received more pheromons respect to a path that is in a lower position in the top K.
It reduces the proability that paths with higher costs may influence the behavior of all the ants.
"""

def update_pheromone_Rank(G, paths, costs, Q=1, rank_k=3, **kwargs):

    sorted_idx = np.argsort(costs) # get ordered indexing of cost of the paths

    for rank, idx in enumerate(sorted_idx[:rank_k]):
        weight = rank_k - rank # costant factor, higher position in top K means higher factor of moltiplication
        path = paths[idx]
        cost = costs[idx]
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            G[u][v]['pheromone'] += weight * Q / cost

# %%

"""
Max-Min Ant System: only the best path update the pheromones, in this method are used a lower and upper bound for the pheromones to avoid the prominance of an edge, or "evitare stagnazione"??,
so the goal of these bounds is to try to stabilize the realase of the pheromones.
So the research is more stable, less risk of premature convergence on local best path.
"""

def update_pheromone_MMAS(G, paths, costs, Q=1, tau_min=0.1, tau_max=10, **kwargs):

    best_idx = np.argmin(costs) #index of minimum cost correlated to the best path
    path = paths[best_idx]
    cost = costs[best_idx]
    for i in range(len(path)-1):
        u, v = path[i], path[i+1]
        G[u][v]['pheromone'] += Q / cost 
        G[u][v]['pheromone'] = max(tau_min, min(G[u][v]['pheromone'], tau_max)) # apply upper and lower bound


# %%
"""
Best-Worst Ant System: the idea is to increase the pheromones on the best path and decrease the worst path.
Reduces the probability of selecting worst paths in the future iterations, increase the probability of the best paths maintaining a little bit of diversity.
"""

def update_pheromone_BWAS(G, paths, costs, Q=1, **kwargs):
    best_idx = np.argmin(costs)
    worst_idx = np.argmax(costs)
    
    #increase pheromones on the best path
    path = paths[best_idx]
    cost = costs[best_idx]
    for i in range(len(path)-1):
        u, v = path[i], path[i+1]
        G[u][v]['pheromone'] += Q / cost
    
    path_w = paths[worst_idx]
    cost_w = costs[worst_idx]

    #deacrease pheromones on the worst path
    for i in range(len(path_w)-1):
        u, v = path_w[i], path_w[i+1]
        G[u][v]['pheromone'] -= 0.1 * Q / cost_w
        G[u][v]['pheromone'] = max(0.01, G[u][v]['pheromone']) # avoid to get a negative quantity of pheromones


# %%
#mapping the name of the variants of ACO
STRATEGY_MAP = {
    "Basic ACO": update_pheromone_AS,
    "EAS": update_pheromone_EAS,
    "Rank": update_pheromone_Rank,
    "MMAS": update_pheromone_MMAS,
    "BWAS": update_pheromone_BWAS
}


# %%
#Run the ACO take the variant as a parameter
def ant_colony_optimization(G, n_ants=10, n_iters=30, alpha=1, beta=2, evaporation=0.4, Q=1, strategy_name="Basic ACO", **kwargs):

    #Select the strategy that is used to update pheromones
    if strategy_name not in STRATEGY_MAP:
        raise ValueError(f"Strategy '{strategy_name}' not recognized. Available strategies: {list(STRATEGY_MAP.keys())}")
    
    pheromone_update_func = STRATEGY_MAP[strategy_name]
    
    nodes = list(G.nodes())
    best_path_global = None
    best_cost_global = float('inf')
    avg_costs = []
    best_costs = []
    worst_costs = []


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
            

            #If no hamiltonian path 
            if len(path) < len(nodes):
                continue

            #Calculate cost of the hamiltonian path
            if len(path) == len(nodes):
                path.append(path[0])
                cost = path_length(G, path)
                all_paths.append(path)
                all_costs.append(cost)
                
            #Updating the best solution found
            if cost < best_cost_global:
                best_cost_global = cost
                best_path_global = path

        if len(all_costs) != 0:
            avg_costs.append(np.mean(all_costs))
            best_costs.append(min(all_costs))   
            worst_costs.append(max(all_costs))

        
        
        #Simple evapoation of pheromones with a costant parameter
        for u,v in G.edges():
            G[u][v]['pheromone'] *= (1 - evaporation)
        

        #Update pheromones if the are some path found
        if len(all_paths) > 0 and best_path_global is not None:
            #Add best_path_global and best_cost_global to kwargs for the strategies that needed
            kwargs['best_path_global'] = best_path_global
            kwargs['best_cost_global'] = best_cost_global

        #Update pheromones
        pheromone_update_func(G, all_paths, all_costs, Q=Q, **kwargs)
          
    return best_path_global, best_cost_global,n_iters * n_ants,avg_costs,best_costs,worst_costs
