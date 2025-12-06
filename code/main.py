# %%
import csv
import networkx as nx
import time
from data import create_random_graph
from aco import ant_colony_optimization
from exhaustive_search import exhaustive_search
from visualizations import (
    plot_convergence,
    plot_best_cost_comparison,
    plot_tsp_graph,
    draw_tsp_path_clean
)
import pandas as pd


# %%
def compare_tsp_algorithms(G, n_ants=10, n_iters=30, strategy_name="Basic ACO"):

    #Run aco
    start_aco = time.time()
    best_path_aco, best_cost_aco,tested_paths_aco,avg_costs,best_costs_aco,worst_costs = ant_colony_optimization(G, n_ants=n_ants, n_iters=n_iters, strategy_name=strategy_name)
    time_aco = time.time() - start_aco

    plot_convergence(avg_costs,best_costs_aco,worst_costs,title = f"ACO fitness convergence ({strategy_name})")

    #Run brute force
    start_brute = time.time()
    best_path_brute, best_cost_brute, tested_paths_bf,avg_costs,best_costs_bf,worst_costs = exhaustive_search(G)
    time_brute = time.time() - start_brute

    plot_convergence(avg_costs,best_costs_bf,worst_costs,title = "Brute Force fitness convergence")

    plot_best_cost_comparison(best_costs_aco,best_costs_bf,title = f"ACO ({strategy_name}) vs BF Best Cost Comparison")

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
#Save comparison results in a csv file
def save_results(results,path):

    with open(path, mode="a", newline="") as f:
            writer = csv.writer(f)
            
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
                results["random_graph"],
                results["ACO"]["cost"],
                results["Brute Force"]["cost"]
               
            ])
            
    print(f"Risultati salvati in {path}")


# %%

# Definisci le configurazioni di test da eseguire
# Ogni configurazione è un dizionario con i parametri del test
test_configurations = [
    {"n_nodes": 10, "n_edges": 80, "n_ants": 15, "n_iters": 30, "seed": 42, "strategy_name": "MMAS","change_seed": True},
    {"n_nodes": 12, "n_edges": 50, "n_ants": 20, "n_iters": 40, "seed": 43, "strategy_name": "EAS"},
    {"n_nodes": 14, "n_edges": 70, "n_ants": 20, "n_iters": 50, "seed": 44, "strategy_name": "BWAS"},
    {"n_nodes": 14, "n_edges": 70, "n_ants": 20, "n_iters": 50, "seed": 44, "strategy_name": "MMAS"},
    {"n_nodes": 14, "n_edges": 70, "n_ants": 20, "n_iters": 50, "seed": 44, "strategy_name": "Rank"},
]


def run_single_test(config, test_number, total_tests):
    """
    Esegue un singolo test con la configurazione specificata
    
    Args:
        config: dizionario con i parametri del test
        test_number: numero del test corrente
        total_tests: numero totale di test
    """
    print(f"\n{'='*80}")
    print(f"Running Test {test_number}/{total_tests}")
    print(f"Configuration: {config}")
    print(f"{'='*80}\n")
    
    # Estrai parametri dalla configurazione
    n_nodes = config["n_nodes"]
    n_edges = config["n_edges"]
    n_ants = config["n_ants"]
    n_iters = config["n_iters"]
    seed = config["seed"]
    strategy_name = config.get("strategy_name", "Basic ACO")  # Default a "Basic ACO" se non specificato
    
    # Crea il grafo
    G = create_random_graph(n_nodes, n_edges, seed)

    # Run Comparison
    #NOTE: for randomly run the comparison i have to create a list of parameters for n_ants and n_iters and then iterate and call the function
    #NOTE: can be interested to run the ACO on really big graphs but the comparison i think is impossible in this case
    #TODO: add parallelization for speedup the runtime
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

    # Store the results in a csv file
    save_results(results,path = "../results/comparison.csv")

    # Visualizations
    #pos = nx.spring_layout(G, seed=seed)
    #plot_tsp_graph(G, pos, title=f"TSP GRAPH ({n_nodes} nodes - {n_edges} edges) - Test {test_number}")
    #draw_tsp_path_clean(G, results["ACO"]["path"], title=f"Shortest Path ACO ({strategy_name}) - Test {test_number}")
    #draw_tsp_path_clean(G, results["Brute Force"]["path"], title=f"Shortest Path (Brute Force) - Test {test_number}")

    return results


# Ciclo su tutte le configurazioni

all_results = []
number_of_iterations_per_test = 10

for idx, config in enumerate(test_configurations, start=1):

    print(f"\n== RUNNING CONFIG {idx}/{len(test_configurations)} ==")

    base_seed = config["seed"]
    change_seed = config.get("change_seed", False)

    # Esegui 30 test
    for run_id in range(1, number_of_iterations_per_test):

        # Gestione seed
        if change_seed:
            config["seed"] = base_seed + run_id  # cambia seed ad ogni run
        else:
            config["seed"] = base_seed  # seed fisso

        res = run_single_test(config, idx, len(test_configurations))
        res["test_number"] = run_id  # aggiungo questo campo
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
            "tested_path": r["Brute Force"]["tested_path"],
            "test_number": r["test_number"]
        })

    df = pd.DataFrame(rows)

    # ---- SUMMARY STATS ----
    numeric_cols = ["ACO_cost", "ACO_time", "Speedup","BF_time","BF_cost","tested_path"]
    summary = pd.DataFrame({
        "mean": df[numeric_cols].mean(),
        "std": df[numeric_cols].std()
    })

    # ---- SAVE SINGLE CONFIG SUMMARY USING save_results ----
    summary_path = "../results/summary_results.csv"

    summary_result_dict = {
        "strategy_name": config["strategy_name"],
        "n_ants": config["n_ants"],
        "n_iters": config["n_iters"],
        "n_nodes": config["n_nodes"],
        "n_edges": config["n_edges"],
        "ACO": {
            "time": f"{summary.loc['ACO_time', 'mean']:.3f} ± {summary.loc['ACO_time', 'std']:.3f}",
            "cost": None,
            "tested_path": config["n_ants"] * config["n_iters"],
        },
        "Brute Force": {
            "time": f"{summary.loc['BF_time', 'mean']:.3f} ± {summary.loc['BF_time', 'std']:.3f}",
            "cost": None,
            "tested_path": f"{summary.loc['tested_path', 'mean']:.3f} ± {summary.loc['tested_path', 'std']:.3f}",
        },
        "Speedup": f"{summary.loc['Speedup', 'mean']:.3f} ± {summary.loc['Speedup', 'std']:.3f}",
        "test_number": "MEAN",
        "random_graph": change_seed
    }

    save_results(summary_result_dict, summary_path)

    all_results = []

