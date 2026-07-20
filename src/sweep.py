"""
Simple hyperparameter grid sweep runner that launches `src/train.py` with different
parameter combinations. Each run creates a separate ClearML task (Task.init in train.py)
so results and artifacts appear in ClearML.

Usage (local):
    python src/sweep.py

You can change the `grid` below to run different hyperparameters.
"""
import os
import subprocess
import itertools

# grid to sweep
grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

# Build combinations
keys = list(grid.keys())
combinations = list(itertools.product(*(grid[k] for k in keys)))

print(f"Running {len(combinations)} runs...")

for i, combo in enumerate(combinations, 1):
    params = dict(zip(keys, combo))
    # build args
    args = ["python", "src/train.py"]
    for k, v in params.items():
        if v is None:
            continue
        args += [f"--{k}", str(v)]

    # provide a descriptive ClearML task name
    task_name = f"sweep_run_{i}_ne{params.get('n_estimators')}_md{params.get('max_depth')}_mss{params.get('min_samples_split')}"

    env = os.environ.copy()
    env['CLEARML_TASK_NAME'] = task_name

    print(f"Run {i}/{len(combinations)}: {task_name} args={args}")
    res = subprocess.run(args, env=env)
    if res.returncode != 0:
        print(f"Run {i} failed with exit code {res.returncode}, continuing...")

print('Sweep finished.')
