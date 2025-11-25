## Activate virtual environment
source /home/xwz/projects/BenchMARL/.venv/bin/activate

## Run the experiment with CUDA settings
CUDA_VISIBLE_DEVICES=2 uv run examples/running/run_masac_pinn.py