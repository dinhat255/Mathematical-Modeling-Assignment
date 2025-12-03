# Mathematical Modeling Assignment
**CO2011 - Symbolic and Algebraic Reasoning in Petri Nets**

## Requirements

- Python ≥ 3.8
- NumPy ≥ 1.20.0
- PyEDA ≥ 0.28.0
- PuLP ≥ 2.7.0

## Installation

```bash
pip install numpy pyeda pulp
```

## Project Structure

```
├── src/
│   ├── PetriNet.py      # Task 1: PNML parser
│   ├── BFS.py           # Task 2: BFS reachability
│   ├── DFS.py           # Task 2: DFS reachability
│   ├── BDD.py           # Task 3: Symbolic BDD
│   ├── DeadLock.py      # Task 4: Deadlock detection
│   └── Optimization.py  # Task 5: Optimization
├── main.py              # Run all tasks
├── run_task.py          # Run individual task
└── TestModel.pnml       # Test file (13 places)
```

## Usage

### Run All Tasks

```bash
python main.py
```

### Run Individual Tasks

```bash
python run_task.py 1         # Task 1: Parse PNML
python run_task.py 2bfs      # Task 2: BFS
python run_task.py 2dfs      # Task 2: DFS
python run_task.py 3         # Task 3: BDD
python run_task.py 4         # Task 4: Deadlock
python run_task.py 5         # Task 5: Optimization
```

### Use Custom PNML File

```bash
python run_task.py 1 your_model.pnml
```

## Task Summary

| Task | Description | Output |
|------|-------------|--------|
| 1 | Parse PNML, check consistency | I/O matrices, M0 |
| 2 | BFS/DFS reachability | 6 markings (~0.001s) |
| 3 | Symbolic BDD reachability | 6 markings (~35s) |
| 4 | Deadlock detection | Found deadlock |
| 5 | Maximize c^T M | Max value 20.0 |

## Notes

- **1-safe nets only** (marking ≤ 1)
- BDD is slow for small models (symbolic overhead)
- TestModel.pnml has **deadlock** by design

## Authors

CO2011 - Semester 1, 2025-2026  
HCMUT
