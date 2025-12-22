# Potential Field Visualization for Safe PINN Models

This document describes the barrier potential field visualization feature for Safe PINN models in BenchMARL.

## Overview

When training with Safe PINN (MASAC) or Safe PINN PPO (MAPPO) algorithms on `navigation_obs` scenarios, the evaluation phase now automatically generates and uploads barrier potential field heatmaps to wandb's Viz module. This helps visualize how the learned barrier function guides agents away from obstacles and other agents.

## How It Works

1. **During Evaluation**: At each evaluation step, the system:
   - Computes the barrier potential field over a 2D grid
   - Extracts agent, obstacle, and goal positions from the environment
   - Renders a combined visualization showing both the environment state and potential field
   - Uploads the visualization to wandb under the `Viz/` prefix

2. **Potential Field Computation**: The potential field is computed based on the Safe PINN model's barrier parameters:
   - For **SafePinnPPO**: Uses log-barrier function with parameters like `r_collision`, `barrier_epsilon`, and `r_communication`
   - For **SafePinn**: Uses quadratic barrier function

3. **Visualization**:
   - **Left panel**: Original environment render showing agents, obstacles, and goals
   - **Right panel**: Heatmap of barrier potential field (red = high potential/danger, blue = low potential/safe)

## wandb Integration

The visualizations are logged to wandb under:
- `Viz/barrier_potential_field`: Static image of the final evaluation step
- `Viz/potential_field_video`: Video showing potential field evolution over the evaluation episode

## Usage

The feature is automatically enabled when:
1. Using Safe PINN models (`--use-safe-pinn` flag or default behavior)
2. Training on navigation-related scenarios (e.g., `navigation_obs`)
3. wandb logging is enabled (default, or explicitly with `--no-wandb` to disable)
4. Rendering is enabled (default, or explicitly with `--no-render` to disable)

### Example Commands

```bash
# MAPPO with Safe PINN PPO (visualizations enabled by default)
python main.py --algorithm mappo --scenario navigation_obs

# MASAC with Safe PINN (visualizations enabled by default)
python main.py --algorithm masac --scenario navigation_obs

# Disable potential field visualization (disable render)
python main.py --algorithm mappo --scenario navigation_obs --no-render
```

## Requirements

The visualization feature requires `matplotlib` (automatically handled as optional dependency):

```bash
pip install matplotlib>=3.5.0
# or with uv
uv add matplotlib
```

If matplotlib is not installed, the visualization will be silently skipped without affecting training.

## Interpreting the Visualization

- **High Potential (Red)**: Areas near obstacles/agents where the barrier function creates strong repulsive forces
- **Low Potential (Blue/Cool colors)**: Safe areas where agents can navigate freely
- **Gradient**: The steepness of potential change indicates how strongly the barrier will push agents away

### Key Parameters Affecting the Potential Field

| Parameter | Description | Effect on Visualization |
|-----------|-------------|------------------------|
| `r_collision` | Collision radius | Larger = wider high-potential zones |
| `r_communication` | Communication range | Limits which agents contribute to potential |
| `barrier_epsilon` | Barrier smoothness | Smaller = sharper potential gradients |
| `use_log_barrier` | Log vs quadratic barrier | Log produces smoother gradients |

## Technical Details

The implementation consists of:

1. **`PotentialFieldVisualizer`** (`gemsmarl/experiment/potential_visualizer.py`):
   - Computes potential fields over a configurable grid
   - Renders heatmaps with matplotlib
   - Creates combined environment + potential visualizations

2. **Logger Integration** (`gemsmarl/experiment/logger.py`):
   - `log_potential_field()` method uploads images/videos to wandb

3. **Experiment Integration** (`gemsmarl/experiment/experiment.py`):
   - `_evaluation_loop()` automatically detects Safe PINN models and triggers visualization

## Customization

You can customize the visualization by modifying `PotentialFieldVisualizer` parameters:

```python
visualizer = PotentialFieldVisualizer(
    world_bounds=(-1.5, 1.5, -1.5, 1.5),  # Grid extent
    grid_resolution=50,                     # Grid density (higher = more detail)
    device="cuda:0",                        # Computation device
)
```
