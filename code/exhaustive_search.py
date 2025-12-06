# %%
import numpy as np


# %%
#The idea for the exhaustive search is that we have to do a dfs for each node on the graph, this allows us to generate all the possible valid paths.
#I can do this because a solution is in the research space if it is a possible path, so all the possibles paths are in the research space and the reaserch space is close.
#This make the comparison more faster and maybe allows to use bigger graph for comparison.
#So i don't consider all the permutation of nodes for the reaserch space but only the all permutation of nodes that are valid path but without generating all possible permutations, instead i do a dfs for each node in the graph.
#With this method for exhaustive search i still check all the research space.
#So a candidate solution is always a valid solution cause is an Hamiltonian path and so the i work with a close reaseach space, that avoid to introduce penalty term in the fitness function and avoid to increase the complexity of the ACO
def dfs(G,start):

    nodes = list(G.nodes())
    best_cost = float('inf')
    best_path = None
    tested_path = 0
    all_costs = []


    #A list where each element is: list of starting point, cost of the path, set of visited nodes
    stack = [([start], 0, {start})]

    while stack:
        #Return the last element of the list in order to get the last updated infos
        path, cost, visited = stack.pop()
        #Get last element of the path, so the current node
        current = path[-1]

        #If all nodes are visited once
        if len(path) == len(nodes):
            tested_path += 1
            all_costs.append(cost)
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


    return best_path,best_cost,tested_path,all_costs


# %%
#Exaustive research used for comparing the performance with ACO
def exhaustive_search(G):
    nodes = list(G.nodes())
    gloabal_best_path = None
    global_best_cost = float('inf')
    global_tested_path = 0
    avg_costs = []
    best_costs = []
    worst_costs = []
    
    #Run a dfs that find only Hamiltonian path for each node, 
    for node in nodes:

        local_best_path, local_best_cost,local_tested_path,all_costs = dfs(G,node)
        global_tested_path += local_tested_path
        avg_costs.append(np.mean(all_costs))
        best_costs.append(min(all_costs))
        worst_costs.append(max(all_costs))


        if local_best_path is not None and local_best_cost < global_best_cost:
            gloabal_best_path = local_best_path
            global_best_cost = local_best_cost

    return gloabal_best_path,global_best_cost, global_tested_path, avg_costs,best_costs,worst_costs
