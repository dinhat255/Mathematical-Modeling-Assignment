from src.PetriNet import PetriNet
from src.BFS import bfs_reachable
from src.DFS import dfs_reachable
from src.BDD import bdd_reachable
from src.DeadLock import check_deadlock
from src.Optimization import max_reachable_marking
import time
import tracemalloc
import numpy as np


def main():
    # === Task 1: Đọc PNML -> PetriNet ===
    # Single test file for all tasks
    pn = PetriNet.from_pnml("TestModel.pnml")
    print("=== Loaded Petri Net ===")
    print(pn)  # In I, O, M0 để kiểm tra dữ liệu đọc từ PNML

    # === Task 2: Explicit Reachability (BFS + DFS) ===
    print("\n=== Task 2: Explicit BFS ===")
    tracemalloc.start()
    start_time = time.time()
    reachable_bfs = bfs_reachable(pn)
    bfs_time = time.time() - start_time
    bfs_memory = tracemalloc.get_traced_memory()[1] / 1024  # Peak in KB
    tracemalloc.stop()

    print("BFS reachable markings:")
    for m in reachable_bfs:
        print(m)
    print(f"Total BFS: {len(reachable_bfs)}")
    print(f"BFS Time: {bfs_time:.6f} seconds")
    print(f"BFS Peak Memory: {bfs_memory:.2f} KB")

    print("\n=== Task 2: Explicit DFS ===")
    tracemalloc.start()
    start_time = time.time()
    reachable_dfs = dfs_reachable(pn)
    dfs_time = time.time() - start_time
    dfs_memory = tracemalloc.get_traced_memory()[1] / 1024  # Peak in KB
    tracemalloc.stop()

    print("DFS reachable markings:")
    for m in reachable_dfs:
        print(m)
    print(f"Total DFS: {len(reachable_dfs)}")
    print(f"DFS Time: {dfs_time:.6f} seconds")
    print(f"DFS Peak Memory: {dfs_memory:.2f} KB")

    # === Task 3: Symbolic BDD Reachability ===
    print("\n=== Task 3: Symbolic BDD Reachability ===")
    tracemalloc.start()
    start_time = time.time()
    bdd, count = bdd_reachable(pn)
    bdd_time = time.time() - start_time
    bdd_memory = tracemalloc.get_traced_memory()[1] / 1024  # Peak in KB
    tracemalloc.stop()

    print(f"Number of reachable markings (symbolic BDD): {count}")
    print(f"BDD Time: {bdd_time:.6f} seconds")
    print(f"BDD Peak Memory: {bdd_memory:.2f} KB")

    # Nếu muốn xem kích thước BDD
    try:
        print(f"BDD DAG size: {bdd.dag_size}")
    except AttributeError:
        pass

    # === Performance Comparison ===
    print("\n=== Performance Comparison (Explicit vs Symbolic) ===")
    print(f"{'Method':<10} {'Time (s)':<12} {'Memory (KB)':<15} {'Markings':<10}")
    print("-" * 50)
    print(f"{'BFS':<10} {bfs_time:<12.6f} {bfs_memory:<15.2f} {len(reachable_bfs):<10}")
    print(f"{'DFS':<10} {dfs_time:<12.6f} {dfs_memory:<15.2f} {len(reachable_dfs):<10}")
    print(f"{'BDD':<10} {bdd_time:<12.6f} {bdd_memory:<15.2f} {count:<10}")
    print(f"\nSpeedup (BFS vs BDD): {bfs_time/bdd_time:.2f}x")
    print(f"Memory Ratio (BFS vs BDD): {bfs_memory/bdd_memory:.2f}x")

    # === Task 4: Deadlock Detection (ILP + BDD) ===
    print("\n=== Task 4: Deadlock Detection (ILP + BDD) ===")
    deadlock = check_deadlock(pn, bdd)

    if deadlock:
        print("Result: Deadlock DETECTED.")
        # Map back to place names for clearer output
        dead_places = [pn.place_ids[i] for i, val in enumerate(deadlock) if val == 1]
        print(f"Deadlock State (Places with tokens): {dead_places}")
    else:
        print("Result: No deadlock found.")

    # === Task 5: Optimization (Maximize c^T M) ===
    print("\n=== Task 5: Optimization (Maximize c^T M over Reachable Set) ===")
    # Define cost vector c - prioritize process running states and resource utilization
    c = np.zeros(len(pn.place_ids))

    # Strategy: Maximize processes running + resources used
    for i, place_id in enumerate(pn.place_ids):
        if "Running" in place_id:
            c[i] = 10.0  # High value for running processes
        elif "Used" in place_id:
            c[i] = 5.0  # Medium value for resource utilization
        elif "HasR" in place_id:
            c[i] = 3.0  # Low value for processes holding resources

    print(f"Cost vector c (non-zero only):")
    for i, (place_id, cost) in enumerate(zip(pn.place_ids, c)):
        if cost > 0:
            print(f"  {place_id}: {cost}")

    best_marking, max_value = max_reachable_marking(pn.place_ids, bdd, c)

    if best_marking is not None:
        print(f"\nOptimal marking: {best_marking}")
        print(f"Maximum value c^T M: {max_value}")
        # Show which places have tokens
        optimal_places = [
            pn.place_ids[i] for i, val in enumerate(best_marking) if val == 1
        ]
        print(f"Places with tokens in optimal marking: {optimal_places}")
    else:
        print("No reachable marking found for optimization.")

    print("\nDone.")


if __name__ == "__main__":
    main()
