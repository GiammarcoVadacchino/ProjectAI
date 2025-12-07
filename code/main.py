# %%
import csv
import networkx as nx
import time
from data import create_random_graph
from aco import ant_colony_optimization
from exhaustive_search import exhaustive_search
from visualizations import (
    plot_convergence,
    plot_best_cost_comparison
)
import pandas as pd

#NOTE: in this project i consider the Hamiltonian paths with the need to the return in the initial node (TSP)

# %%
#Run the comparison between ACO and brute force
def compare_tsp_algorithms(G, n_ants=10, n_iters=30, strategy_name="Basic ACO"):

    #Run aco
    start_aco = time.time()
    best_path_aco, best_cost_aco,tested_paths_aco,avg_costs,best_costs_aco,worst_costs = ant_colony_optimization(G, n_ants=n_ants, n_iters=n_iters, strategy_name=strategy_name)
    time_aco = time.time() - start_aco

    #Plot ACO convergence
    plot_convergence(avg_costs,best_costs_aco,worst_costs,title = f"ACO fitness convergence ({strategy_name})")

    #Run brute force
    start_brute = time.time()
    best_path_brute, best_cost_brute, tested_paths_bf,avg_costs,best_costs_bf,worst_costs = exhaustive_search(G)
    time_brute = time.time() - start_brute


    print(f"Number of tested path with ACO: {tested_paths_aco}")
    print(f"Number of tested paths with Brute Force: {tested_paths_bf}") # (N-1)! possible paths if the graph is fully connected and N is the number of the nodes

    #Results
    result = {
        "ACO": {"path": best_path_aco, "cost": best_cost_aco, "time": time_aco,"tested_path": tested_paths_aco},
        "Brute Force": {"path": best_path_brute, "cost": best_cost_brute, "time": time_brute, "tested_path": tested_paths_bf},
        "Speedup": time_brute/time_aco if time_aco>0 else None
    }
    return result


# %%
#Save comparison results in a csv file
def save_results(results,path):

    with open(path, mode="a", newline="") as f:
            writer = csv.writer(f)
            
            #write results in a csv file
            writer.writerow([
                results["strategy_name"],
                results["n_ants"],
                results["n_iters"],
                results["n_nodes"],
                results["n_edges"],
                results['ACO']['time'],
                results['Brute Force']['time'],
                results['Speedup'],
                results["ACO"]["tested_path"],
                results["Brute Force"]["tested_path"],
                results["ACO"]["cost"],
                results["Brute Force"]["cost"],
                results["random_graph"],
                results["fully_connected"],
                results["iterations"]
               
            ])
            
    print(f"Risultati salvati in {path}")


# %%

#Define the configurations of the tests that will be runned
test_configurations = [
    {"n_nodes": 8, "n_edges": 80, "n_ants": 8, "n_iters": 24, "seed": 42, "strategy_name": "Basic ACO","change_seed": False,"Fully_connected": True, "iterations": 10},
    {"n_nodes": 10, "n_edges": 50, "n_ants": 10, "n_iters": 30, "seed": 43, "strategy_name": "Basic ACO","change_seed": False,"Fully_connected": True, "iterations": 10},
    {"n_nodes": 12, "n_edges": 70, "n_ants": 12, "n_iters": 36, "seed": 44, "strategy_name": "Basic ACO","change_seed": False,"Fully_connected": True, "iterations": 1},
    {"n_nodes": 13, "n_edges": 70, "n_ants": 14, "n_iters": 42, "seed": 45, "strategy_name": "Basic ACO","change_seed": False,"Fully_connected": True,"iterations": 1},

    {"n_nodes": 8, "n_edges": 80, "n_ants": 8, "n_iters": 24, "seed": 42, "strategy_name": "EAS","change_seed": False,"Fully_connected": True, "iterations": 10},
    {"n_nodes": 10, "n_edges": 50, "n_ants": 10, "n_iters": 30, "seed": 43, "strategy_name": "EAS","change_seed": False,"Fully_connected": True, "iterations": 10},
    {"n_nodes": 12, "n_edges": 70, "n_ants": 12, "n_iters": 36, "seed": 44, "strategy_name": "EAS","change_seed": False,"Fully_connected": True, "iterations": 10},
    {"n_nodes": 13, "n_edges": 70, "n_ants": 14, "n_iters": 42, "seed": 45, "strategy_name": "EAS","change_seed": False,"Fully_connected": True,"iterations": 4},

    {"n_nodes": 8, "n_edges": 80, "n_ants": 8, "n_iters": 24, "seed": 42, "strategy_name": "Rank","change_seed": False,"Fully_connected": True, "iterations": 10},
    {"n_nodes": 10, "n_edges": 50, "n_ants": 10, "n_iters": 30, "seed": 43, "strategy_name": "Rank","change_seed": False,"Fully_connected": True, "iterations": 10},
    {"n_nodes": 12, "n_edges": 70, "n_ants": 12, "n_iters": 36, "seed": 44, "strategy_name": "Rank","change_seed": False,"Fully_connected": True, "iterations": 10},
    {"n_nodes": 13, "n_edges": 70, "n_ants": 14, "n_iters": 42, "seed": 45, "strategy_name": "Rank","change_seed": False,"Fully_connected": True,"iterations": 4},

    {"n_nodes": 8, "n_edges": 80, "n_ants": 8, "n_iters": 24, "seed": 42, "strategy_name": "MMAS","change_seed": False,"Fully_connected": True, "iterations": 10},
    {"n_nodes": 10, "n_edges": 50, "n_ants": 10, "n_iters": 30, "seed": 43, "strategy_name": "MMAS","change_seed": False,"Fully_connected": True, "iterations": 10},
    {"n_nodes": 12, "n_edges": 70, "n_ants": 12, "n_iters": 36, "seed": 44, "strategy_name": "MMAS","change_seed": False,"Fully_connected": True, "iterations": 10},
    {"n_nodes": 13, "n_edges": 70, "n_ants": 14, "n_iters": 42, "seed": 45, "strategy_name": "MMAS","change_seed": False,"Fully_connected": True,"iterations": 4},

    {"n_nodes": 8, "n_edges": 80, "n_ants": 8, "n_iters": 24, "seed": 42, "strategy_name": "BWAS","change_seed": False,"Fully_connected": True, "iterations": 10},
    {"n_nodes": 10, "n_edges": 50, "n_ants": 10, "n_iters": 30, "seed": 43, "strategy_name": "BWAS","change_seed": False,"Fully_connected": True, "iterations": 10},
    {"n_nodes": 12, "n_edges": 70, "n_ants": 12, "n_iters": 36, "seed": 44, "strategy_name": "BWAS","change_seed": False,"Fully_connected": True, "iterations": 10},
    {"n_nodes": 13, "n_edges": 70, "n_ants": 14, "n_iters": 42, "seed": 45, "strategy_name": "BWAS","change_seed": False,"Fully_connected": True,"iterations": 4},
]

#Run a single test
def run_single_test(config, test_number, total_tests):

    print(f"\n{'='*80}")
    print(f"Running Test {test_number}/{total_tests}")
    print(f"Configuration: {config}")
    print(f"{'='*80}\n")
    
    #Get parameters from the configuration
    n_nodes = config["n_nodes"]
    n_edges = config["n_edges"]
    n_ants = config["n_ants"]
    n_iters = config["n_iters"]
    seed = config["seed"]
    strategy_name = config["strategy_name"] 
    
    
    #Build the graph
    G = create_random_graph(n_nodes, n_edges, seed,fully_connected = config["Fully_connected"])

    # Run Comparison
    results = compare_tsp_algorithms(G, n_ants, n_iters, strategy_name)

    # Print stats
    print(f"\n=== Confronto TSP ACO ({strategy_name}) vs Brute Force ===")
    for algo, res in results.items():
        if algo != "Speedup":
            print(f"{algo}: best cost = {res['cost']}, best path = {res['path']}, time = {res['time']:.4f} s")
    print(f"Speedup (Brute/ACO) ≈ {results['Speedup']:.2f}×")

    results["strategy_name"] = strategy_name
    results["n_ants"] = n_ants
    results["n_iters"] = n_iters
    results["n_nodes"] = n_nodes
    results["n_edges"] = n_edges
    results["random_graph"] = config.get("change_seed")
    results["fully_connected"] = config["Fully_connected"]
    results["iterations"] = config["iterations"]

    # Store the results of the single run test in a csv file
    save_results(results,path = "../results/comparison.csv")


    return results




all_results = []

#Run all the tests with the configurations
for idx, config in enumerate(test_configurations, start=1):

    print(f"\n== RUNNING CONFIG {idx}/{len(test_configurations)} ==")

    base_seed = config["seed"]
    change_seed = config.get("change_seed", False)

    #Run a certain number of times a single test with the same configuration, used to obtain the mean and std for speedup and costs differences
    for run_id in range(1, config["iterations"]+1):

        #Apply the tests on random graphs or the same graph
        if change_seed:
            config["seed"] = base_seed + run_id  #chage the seed for randomness
        else:
            config["seed"] = base_seed 

        res = run_single_test(config, idx, len(test_configurations))
        all_results.append(res)

    # ---- FLATTEN RESULTS ----
    rows = []
    for r in all_results:
        rows.append({
            "ACO_cost": r["ACO"]["cost"],
            "ACO_time": r["ACO"]["time"],
            "BF_cost": r["Brute Force"]["cost"],
            "BF_time": r["Brute Force"]["time"],
            "Speedup": r["Speedup"],
            "strategy_name": r["strategy_name"],
            "n_ants": r["n_ants"],
            "n_iters": r["n_iters"],
            "n_nodes": r["n_nodes"],
            "n_edges": r["n_edges"],
            "tested_path": r["Brute Force"]["tested_path"]
        })

    df = pd.DataFrame(rows)

    #Calculate the mean and the std for the metrics
    numeric_cols = ["ACO_cost", "ACO_time", "Speedup","BF_time","BF_cost","tested_path"]
    summary = pd.DataFrame({
        "mean": df[numeric_cols].mean(),
        "std": df[numeric_cols].std()
    })

    
    summary_path = "../results/summary_results.csv"

    #Build the dict with all the values that will be stored
    summary_result_dict = {
        "strategy_name": config["strategy_name"],
        "n_ants": config["n_ants"],
        "n_iters": config["n_iters"],
        "n_nodes": config["n_nodes"],
        "n_edges": config["n_edges"],
        "ACO": {
            "time": f"{summary.loc['ACO_time', 'mean']:.3f} ± {summary.loc['ACO_time', 'std']:.3f}",
            "cost": f"{summary.loc['ACO_cost', 'mean']:.3f} ± {summary.loc['ACO_cost', 'std']:.3f}",
            "tested_path": config["n_ants"] * config["n_iters"],
        },
        "Brute Force": {
            "time": f"{summary.loc['BF_time', 'mean']:.3f} ± {summary.loc['BF_time', 'std']:.3f}",
            "cost": None,
            "tested_path": f"{summary.loc['tested_path', 'mean']:.3f} ± {summary.loc['tested_path', 'std']:.3f}",
        },
        "Speedup": f"{summary.loc['Speedup', 'mean']:.3f} ± {summary.loc['Speedup', 'std']:.3f}",
        "random_graph": change_seed,
        "fully_connected": config["Fully_connected"],
        "iterations": config["iterations"]
    }

    #save all avg results
    save_results(summary_result_dict, summary_path)

    all_results = []

