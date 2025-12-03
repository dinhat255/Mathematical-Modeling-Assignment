import time
from typing import List, Optional
import pulp
from pyeda.inter import bddvar
from src.PetriNet import PetriNet


def check_deadlock(pn: PetriNet, reachable_bdd) -> Optional[List[int]]:
    """
    Task 4: Deadlock detection using ILP (PuLP) and BDD.

    Args:
        pn: PetriNet object (from Task 1)
        reachable_bdd: The BDD object representing reachable markings (from Task 3)

    Returns:
        List[int]: A deadlock marking (list of 0/1 for each place) if found.
        None: If no deadlock exists.
    """
    start_time = time.time()
    num_places = len(pn.place_ids)
    num_trans = len(pn.trans_ids)

    # 1. Initialize ILP Model using PuLP
    # LpMaximize or LpMinimize doesn't matter much for feasibility, usually Maximize total tokens
    prob = pulp.LpProblem("Deadlock_Detection", pulp.LpMaximize)

    # 2. Define Variables
    # M[p]: Token count at place p (Binary because 1-safe)
    # cat='Binary' ensures values are 0 or 1
    M = [pulp.LpVariable(f"M_{i}", cat="Binary") for i in range(num_places)]

    # Sigma[t]: Parikh vector (firing count). Integer, min=0
    Sigma = [
        pulp.LpVariable(f"Sigma_{i}", lowBound=0, cat="Integer")
        for i in range(num_trans)
    ]

    # 3. Add Constraint: State Equation (M = M0 + C * Sigma)
    # C = O - I
    C = pn.O - pn.I
    for p_idx in range(num_places):
        # PuLP syntax: sum([list of terms]) == value
        # Expression: M[p] - sum(C*Sigma) = M0[p]
        delta_sum = pulp.lpSum(
            [C[t_idx, p_idx] * Sigma[t_idx] for t_idx in range(num_trans)]
        )
        prob += (M[p_idx] == pn.M0[p_idx] + delta_sum), f"StateEq_Place_{p_idx}"

    # 4. Add Constraint: Dead Marking (Disable Condition)
    # A marking is dead if NO transition is enabled.
    # Transition t is disabled if Sum(tokens in inputs) <= |inputs| - 1
    for t_idx in range(num_trans):
        # Get indices of input places for transition t
        input_places = [p_idx for p_idx, val in enumerate(pn.I[t_idx]) if val > 0]

        if not input_places:
            # Source transition always enabled => No deadlock
            print(
                f"  [Info] Transition {pn.trans_ids[t_idx]} is a source. No deadlock possible."
            )
            return None

        # Constraint: Sum(M[p] for p in inputs) <= len(inputs) - 1
        prob += (
            pulp.lpSum([M[p_idx] for p_idx in input_places]) <= len(input_places) - 1
        ), f"Disable_Trans_{t_idx}"

    # 5. Iterative Solving (Hybrid Loop)
    iteration = 0
    print(f"  [Deadlock] Starting ILP(PuLP)+BDD search...")

    # Recreate BDD variables to match the mapping in Task 3
    bdd_place_vars = [bddvar(pid) for pid in pn.place_ids]

    while True:
        iteration += 1

        # Solve the ILP using default solver (usually CBC)
        # msg=False turns off solver logs
        status = prob.solve(pulp.PULP_CBC_CMD(msg=False))

        # Check status
        if status != pulp.LpStatusOptimal:
            # In PuLP, 'Optimal' means a feasible solution was found (even if just checking feasibility)
            print(
                f"  [Deadlock] ILP Infeasible. System is deadlock-free. (Time: {time.time() - start_time:.4f}s)"
            )
            return None

        # Extract Candidate Marking M_cand
        # pulp.value(var) gets the value
        m_cand = [int(pulp.value(var)) for var in M]

        # 6. Check Reachability using BDD (Membership Check)
        assignment = {var: val for var, val in zip(bdd_place_vars, m_cand)}

        # restrict returns 1 if path exists
        is_reachable = reachable_bdd.restrict(assignment).is_one()

        if is_reachable:
            print(f"  [Deadlock] FOUND Deadlock at iteration {iteration}!")
            print(f"  [Deadlock] Marking: {m_cand}")
            print(f"  [Deadlock] Time: {time.time() - start_time:.4f}s")
            return m_cand
        else:
            # 7. Spurious Solution -> Add Integer Cut (Canonical Cut)
            # Constraint: Sum(vars that are 1) - Sum(vars that are 0) <= (Num of 1s) - 1

            ones = [i for i, val in enumerate(m_cand) if val == 1]
            zeros = [i for i, val in enumerate(m_cand) if val == 0]

            cut_lhs = pulp.lpSum([M[i] for i in ones]) - pulp.lpSum(
                [M[i] for i in zeros]
            )
            cut_rhs = len(ones) - 1

            prob += (cut_lhs <= cut_rhs), f"Cut_Iter_{iteration}"

            # Optional: print(f"  [Deadlock] Iter {iteration}: Spurious {m_cand}. Cut added.")
