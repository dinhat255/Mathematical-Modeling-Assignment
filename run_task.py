#!/usr/bin/env python3
"""
Quick task runner for Mathematical Modeling Assignment
Usage: python run_task.py <task_number> [pnml_file]
Example: python run_task.py 1
         python run_task.py 2bfs
         python run_task.py 2dfs TestModel.pnml
"""

import sys
import time
import tracemalloc
import numpy as np
from src.PetriNet import PetriNet
from src.BFS import bfs_reachable
from src.DFS import dfs_reachable
from src.BDD import bdd_reachable
from src.DeadLock import check_deadlock
from src.Optimization import max_reachable_marking


def task1(pnml_file):
    """Task 1: Parse PNML and verify consistency"""
    print(f"\n=== Task 1: PNML Parser ===")
    pn = PetriNet.from_pnml(pnml_file)
    print(pn)
    return pn


def task2(pnml_file, method="bfs"):
    """Task 2: Explicit reachability (BFS or DFS)"""
    print(f"\n=== Task 2: Explicit Reachability ({method.upper()}) ===")
    pn = PetriNet.from_pnml(pnml_file)

    tracemalloc.start()
    start = time.time()

    if method.lower() == "bfs":
        markings = bfs_reachable(pn)
    else:
        markings = dfs_reachable(pn)

    elapsed = time.time() - start
    memory = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()

    print(f"Found {len(markings)} reachable markings")
    print(f"Time: {elapsed:.6f}s | Memory: {memory:.2f} KB")

    for i, m in enumerate(markings, 1):
        print(f"  {i}. {list(m)}")

    return pn, markings


def task3(pnml_file):
    """Task 3: Symbolic BDD reachability"""
    print(f"\n=== Task 3: Symbolic BDD Reachability ===")
    pn = PetriNet.from_pnml(pnml_file)

    tracemalloc.start()
    start = time.time()
    bdd, count = bdd_reachable(pn)
    elapsed = time.time() - start
    memory = tracemalloc.get_traced_memory()[1] / 1024
    tracemalloc.stop()

    print(f"Found {count} reachable markings")
    print(f"Time: {elapsed:.6f}s | Memory: {memory:.2f} KB")

    try:
        print(f"BDD DAG size: {bdd.dag_size}")
    except AttributeError:
        pass

    return pn, bdd, count


def task4(pnml_file):
    """Task 4: Deadlock detection"""
    print(f"\n=== Task 4: Deadlock Detection (ILP + BDD) ===")
    pn = PetriNet.from_pnml(pnml_file)
    bdd, _ = bdd_reachable(pn)

    deadlock = check_deadlock(pn, bdd)

    if deadlock:
        print("Result: DEADLOCK DETECTED")
        dead_places = [pn.place_ids[i] for i, val in enumerate(deadlock) if val == 1]
        print(f"Deadlock marking: {deadlock}")
        print(f"Places with tokens: {dead_places}")
    else:
        print("Result: NO DEADLOCK")

    return pn, deadlock


def task5(pnml_file):
    """Task 5: Optimization over reachable markings"""
    print(f"\n=== Task 5: Optimization (Maximize c^T M) ===")
    pn = PetriNet.from_pnml(pnml_file)
    bdd, _ = bdd_reachable(pn)

    # Define cost vector - prioritize running states
    c = np.zeros(len(pn.place_ids))
    for i, place_id in enumerate(pn.place_ids):
        if "Running" in place_id:
            c[i] = 10.0
        elif "Used" in place_id:
            c[i] = 5.0
        elif "HasR" in place_id:
            c[i] = 3.0

    print("Cost vector (non-zero only):")
    for i, (pid, cost) in enumerate(zip(pn.place_ids, c)):
        if cost > 0:
            print(f"  {pid}: {cost}")

    marking, value = max_reachable_marking(pn.place_ids, bdd, c)

    if marking:
        print(f"\nOptimal marking: {marking}")
        print(f"Maximum value: {value}")
        opt_places = [pn.place_ids[i] for i, val in enumerate(marking) if val == 1]
        print(f"Places with tokens: {opt_places}")
    else:
        print("No reachable marking found")

    return pn, marking, value


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print("\nAvailable tasks:")
        print("  1    - Parse PNML and verify consistency")
        print("  2bfs - Explicit BFS reachability")
        print("  2dfs - Explicit DFS reachability")
        print("  3    - Symbolic BDD reachability")
        print("  4    - Deadlock detection")
        print("  5    - Optimization")
        sys.exit(1)

    task = sys.argv[1].lower()
    pnml_file = sys.argv[2] if len(sys.argv) > 2 else "TestModel.pnml"

    print(f"Using PNML file: {pnml_file}")

    tasks = {
        "1": lambda: task1(pnml_file),
        "2bfs": lambda: task2(pnml_file, "bfs"),
        "2dfs": lambda: task2(pnml_file, "dfs"),
        "3": lambda: task3(pnml_file),
        "4": lambda: task4(pnml_file),
        "5": lambda: task5(pnml_file),
    }

    if task in tasks:
        tasks[task]()
    else:
        print(f"Unknown task: {task}")
        print("Valid tasks: 1, 2bfs, 2dfs, 3, 4, 5")
        sys.exit(1)


if __name__ == "__main__":
    main()
