"""
Cluster job script generator for CAPTAIN experiments on UCL milk (SLURM).

Assumes the repo is cloned on the cluster and the environment is set up with:
    uv sync

Workflow:
    1. cp cluster/config.example.yaml cluster/config.yaml  # fill in your details
    2. Edit the `sweep` dict at the bottom to define your hyperparameter grid
    3. python cluster/job_script_generator.py              # generates scripts
    4. scp python_submit_job.sh milk:/path/to/repo/        # copy script to cluster
    5. On milk: sbatch python_submit_job.sh

Generated files (gitignored, at repo root):
    python_param_file.txt   - one CLI flag combo per line (one SLURM array task each)
    python_submit_job.sh    - SLURM job script to copy and submit on the cluster
"""

import itertools
from pathlib import Path

import yaml


def load_config(path: str = "cluster/config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def generate_sweep_params(sweep: dict, output_path: str) -> int:
    """Write one CLI flag combo per line; returns the number of combinations."""
    keys = list(sweep.keys())
    values = [v if isinstance(v, list) else [v] for v in sweep.values()]

    lines = []
    for combo in itertools.product(*values):
        tokens = []
        for k, v in zip(keys, combo):
            if v is None:
                continue
            elif isinstance(v, bool):
                if v:
                    tokens.append(f"--{k}")
            else:
                tokens.extend([f"--{k}", str(v)])
        lines.append(" ".join(tokens))

    Path(output_path).write_text("\n".join(lines) + "\n")
    print(f"Generated {output_path} ({len(lines)} combinations)")
    return len(lines)


def generate_submit_script(cfg: dict, n_tasks: int, param_file: str, output: str = "python_submit_job.sh"):
    job = cfg["job"]
    project = cfg["project"]

    header = [
        f"#SBATCH --time={job['time']}",
        f"#SBATCH --mem={job['mem']}",
        f"#SBATCH --job-name={job['name']}",
        f"#SBATCH --output=logs/{job['name']}/%x_%A_%a.out",
    ]
    if job.get("gpu"):
        header.append(f"#SBATCH --gres=gpu:{job.get('gpu_count', 1)}")
        if job.get("specific_gpu"):
            header.append(f"#SBATCH --constraint={job['specific_gpu']}")
    if n_tasks > 0:
        header.append(f"#SBATCH --array=1-{n_tasks}")

    gpu_block = ""
    if job.get("gpu"):
        gpu_block = """\
echo "================================================================"
echo "GPU status (nvidia-smi):"
nvidia-smi
echo "================================================================"
"""

    header_str = "\n".join(header)
    script = f"""\
#!/bin/bash -l

{header_str}

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$PROJECT_ROOT/logs/{job['name']}"

echo "Starting job on $(hostname) at $(date)"
echo "Project root: $PROJECT_ROOT"
echo "Module: {project['module']}"

{gpu_block}
source "$PROJECT_ROOT/.venv/bin/activate"

LOG_FILE="$PROJECT_ROOT/logs/{job['name']}/txt_{job['name']}_${{SLURM_JOB_ID}}"
PARAMS=""

if [ -n "${{SLURM_ARRAY_TASK_ID}}" ]; then
    LOG_FILE="${{LOG_FILE}}_${{SLURM_ARRAY_TASK_ID}}.txt"
    PARAM_FILE="$PROJECT_ROOT/{param_file}"
    if [ -f "$PARAM_FILE" ]; then
        PARAMS=$(sed "${{SLURM_ARRAY_TASK_ID}}q;d" "$PARAM_FILE")
        echo "Task $SLURM_ARRAY_TASK_ID params: $PARAMS"
    else
        echo "Error: param file $PARAM_FILE not found" && exit 1
    fi
else
    LOG_FILE="${{LOG_FILE}}.txt"
fi

python -m {project['module']} $PARAMS > "$LOG_FILE" 2>&1

echo "Job finished at $(date)"
"""

    with open(output, "w") as f:
        f.write(script)
    print(f"Generated {output}")


if __name__ == "__main__":
    cfg = load_config("cluster/config.yaml")
    project = cfg["project"]
    param_file = "python_param_file.txt"

    # --- Edit this sweep dict to define your experiment grid ---
    # Each key becomes a CLI flag; lists generate all combinations.
    sweep = {
        "seed": [42, 15, 37],
    }
    # -----------------------------------------------------------

    n_tasks = generate_sweep_params(sweep, param_file)
    generate_submit_script(cfg, n_tasks, param_file)
