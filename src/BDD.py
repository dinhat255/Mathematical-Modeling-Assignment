from typing import Tuple
import numpy as np
from pyeda.boolalg.bdd import BDDONE, BDDZERO
from pyeda.inter import BinaryDecisionDiagram, bddvar
from src.PetriNet import PetriNet


def _initial_state_bdd(
    place_vars, initial_marking: np.ndarray
) -> BinaryDecisionDiagram:
    """Characteristic function of the initial marking."""
    state = BDDONE
    for var, value in zip(place_vars, initial_marking):
        state &= var if value else ~var
    return state


def _build_transition_relation(
    pn: PetriNet, place_vars, next_place_vars
) -> BinaryDecisionDiagram:
    """
    Build the global transition relation R(X, X')
    following Pastor-Cortadella symbolic construction.
    """
    transitions = []
    num_transitions, num_places = pn.I.shape

    for t_idx in range(num_transitions):
        relation = BDDONE

        # Enabling condition: all preset places marked (1-safe assumption)
        for p_idx in range(num_places):
            if pn.I[t_idx, p_idx] > 0:
                relation &= place_vars[p_idx]
            # 1-safe guard: cannot place a token where one already exists unless it is consumed
            if pn.O[t_idx, p_idx] > 0 and pn.I[t_idx, p_idx] == 0:
                relation &= ~place_vars[p_idx]

        # State update for every place
        for p_idx in range(num_places):
            consumes = pn.I[t_idx, p_idx] > 0
            produces = pn.O[t_idx, p_idx] > 0

            # Next value after firing: (keep token if not consumed) OR (produce token)
            next_val = (place_vars[p_idx] if not consumes else BDDZERO) | (
                BDDONE if produces else BDDZERO
            )

            # Constrain X' to match computed next value
            relation &= ~(next_place_vars[p_idx] ^ next_val)

        transitions.append(relation)

    # Global relation is the disjunction of every transition firing
    relation_all = BDDZERO
    for rel in transitions:
        relation_all |= rel
    return relation_all


def bdd_reachable(pn: PetriNet) -> Tuple[BinaryDecisionDiagram, int]:
    """
    Symbolic reachability analysis using the Pastor-Cortadella BDD algorithm
    (Symbolic Analysis of Bounded Petri Nets). Assumes 1-safe nets so each
    place maps to a boolean variable.
    """
    place_vars = [bddvar(pid) for pid in pn.place_ids]
    next_place_vars = [bddvar(f"{pid}_next") for pid in pn.place_ids]

    # R collects visited markings, F is the current frontier (both over X variables)
    reachable = _initial_state_bdd(place_vars, pn.M0)
    frontier = reachable

    # Transition relation R(X, X')
    relation = _build_transition_relation(pn, place_vars, next_place_vars)
    rename_next_to_curr = {nvar: var for var, nvar in zip(place_vars, next_place_vars)}

    while not frontier.is_zero():
        # Post-image: ∃X (F(X) ∧ R(X, X'))
        successors_prime = (frontier & relation).smoothing(place_vars)
        successors = successors_prime.compose(rename_next_to_curr)

        new_states = successors & ~reachable
        if new_states.is_zero():
            break

        reachable |= new_states
        frontier = new_states

    count = reachable.satisfy_count()
    return reachable, count
