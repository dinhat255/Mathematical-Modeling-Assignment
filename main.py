from PetriNet import PetriNet
from BFS import bfs_reachable
from DFS import dfs_reachable
from BDD import bdd_reachable
from DeadLock import check_deadlock

def main():
    # === Task 1: Đọc PNML -> PetriNet ===
    pn = PetriNet.from_pnml("SimpleMutex.pnml")
    print("=== Loaded Petri Net ===")
    print(pn)  # In I, O, M0 để kiểm tra dữ liệu đọc từ PNML

    # === Task 2: Explicit Reachability (BFS + DFS) ===
    print("\n=== Task 2: Explicit BFS ===")
    reachable_bfs = bfs_reachable(pn)
    print("BFS reachable markings:")
    for m in reachable_bfs:
        print(m)
    print("Total BFS:", len(reachable_bfs))

    print("\n=== Task 2: Explicit DFS ===")
    reachable_dfs = dfs_reachable(pn)
    print("DFS reachable markings:")
    for m in reachable_dfs:
        print(m)
    print("Total DFS:", len(reachable_dfs))

    # === Task 3: Symbolic BDD Reachability ===
    print("\n=== Task 3: Symbolic BDD Reachability ===")
    bdd, count = bdd_reachable(pn)

    print("Number of reachable markings (symbolic BDD):", count)

    # Nếu muốn xem kích thước BDD
    try:
        print("BDD DAG size:", bdd.dag_size)
    except AttributeError:
        pass

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

    print("Done.")

    


if __name__ == "__main__":
    main()
