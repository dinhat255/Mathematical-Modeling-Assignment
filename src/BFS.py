from collections import deque
import numpy as np
from src.PetriNet import PetriNet
from typing import Set, Tuple


def bfs_reachable(pn: PetriNet) -> Set[Tuple[int, ...]]:
    # Initialize the queue with the initial marking
    queue = deque([pn.M0])

    # Set to store visited markings (as tuples for hashability)
    visited = set()
    visited.add(tuple(int(x) for x in pn.M0))

    # BFS loop
    while queue:
        current_marking = queue.popleft()

        # Try to fire each transition (use I.shape[0] to get number of transitions)
        for t_idx in range(pn.I.shape[0]):
            # Check if transition t_idx is enabled
            # A transition is enabled if current_marking >= I[t_idx, :]
            # I is a matrix where rows are transitions and columns are places
            if np.all(current_marking >= pn.I[t_idx, :]):
                # Fire the transition: new_marking = current_marking - I[t_idx, :] + O[t_idx, :]
                new_marking = current_marking - pn.I[t_idx, :] + pn.O[t_idx, :]

                # Check 1-safe constraint: each place can have at most 1 token
                if np.all(new_marking <= 1):
                    # Convert to tuple for hashing (convert np.int64 to int)
                    new_marking_tuple = tuple(int(x) for x in new_marking)

                    # If not visited, add to queue and visited set
                    if new_marking_tuple not in visited:
                        visited.add(new_marking_tuple)
                        queue.append(new_marking)

    return visited
