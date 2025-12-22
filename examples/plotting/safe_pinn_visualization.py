"""
Example Usage: Safe-PINN Publication Visualizations
====================================================

This script demonstrates how to use all the new visualization methods
for generating Nature Communications quality figures.

Author: BenchMARL Team
Date: 2025-12-22
"""

import numpy as np
import torch
import wandb
from gemsmarl.experiment.potential_visualizer import (
    PotentialFieldVisualizer,
    extract_positions_from_env,
    get_safe_pinn_model
)


def example_1_basic_usage(env, policy, current_state, step):
    """Example 1: Basic usage with automatic position extraction."""
    
    # Initialize visualizer
    visualizer = PotentialFieldVisualizer(
        world_bounds=(-1.5, 1.5, -1.5, 1.5),
        grid_resolution=50,
        device="cuda:0"
    )
    
    # Extract Safe-PINN model from policy
    safe_pinn_model = get_safe_pinn_model(policy)
    if safe_pinn_model is None:
        print("Warning: Not a Safe-PINN model")
        return
    
    # Extract positions from environment
    positions = extract_positions_from_env(env, env_index=0)
    agent_pos = positions['agents']
    obs_pos = positions['obstacles']
    goal_pos = positions['goals']
    
    # Convert to tensors
    agent_pos_t = torch.tensor(agent_pos, dtype=torch.float32, device="cuda:0")
    obs_pos_t = torch.tensor(obs_pos, dtype=torch.float32, device="cuda:0") if len(obs_pos) > 0 else None
    goal_pos_t = torch.tensor(goal_pos, dtype=torch.float32, device="cuda:0") if len(goal_pos) > 0 else None
    
    # Compute potential fields
    barrier_field = visualizer.compute_potential_field(
        safe_pinn_model, agent_pos_t, obs_pos_t
    )
    
    # Render basic visualization
    barrier_image = visualizer.render_potential_field(
        barrier_field,
        agent_positions=agent_pos,
        obstacle_positions=obs_pos,
        goal_positions=goal_pos,
        title="Barrier Potential Field"
    )
    
    # Log to wandb
    wandb.log({
        "Viz/barrier_potential": wandb.Image(barrier_image),
        "step": step
    })
    
    return barrier_image


def example_2_publication_figure(env, policy, current_state, step):
    """Example 2: Generate comprehensive publication figure."""
    
    # High-resolution visualizer for publication
    visualizer = PotentialFieldVisualizer(
        world_bounds=(-1.5, 1.5, -1.5, 1.5),
        grid_resolution=100,  # Higher resolution for publication
        device="cuda:0"
    )
    
    # Get model and positions
    safe_pinn_model = get_safe_pinn_model(policy)
    positions = extract_positions_from_env(env, env_index=0)
    
    # Generate comprehensive 6-panel figure
    pub_figure = visualizer.render_publication_figure(
        model=safe_pinn_model,
        sample_state=current_state,
        agent_positions=positions['agents'],
        obstacle_positions=positions['obstacles'],
        goal_positions=positions['goals'],
        step=step,
        figsize=(20, 12),
        dpi=150  # Publication quality
    )
    
    # Save to file
    from PIL import Image
    Image.fromarray(pub_figure).save(
        f'publication_figure_step_{step}.png',
        dpi=(300, 300)
    )
    
    # Log to wandb
    wandb.log({
        "Viz/publication_figure": wandb.Image(pub_figure),
        "step": step
    })
    
    return pub_figure


def example_3_individual_visualizations(env, policy, current_state, step):
    """Example 3: Generate all individual visualizations separately."""
    
    visualizer = PotentialFieldVisualizer(
        world_bounds=(-1.5, 1.5, -1.5, 1.5),
        grid_resolution=75,
        device="cuda:0"
    )
    
    safe_pinn_model = get_safe_pinn_model(policy)
    positions = extract_positions_from_env(env)
    
    agent_pos = positions['agents']
    obs_pos = positions['obstacles']
    goal_pos = positions['goals']
    
    # Convert to tensors
    agent_pos_t = torch.tensor(agent_pos, dtype=torch.float32, device="cuda:0")
    obs_pos_t = torch.tensor(obs_pos, dtype=torch.float32, device="cuda:0") if len(obs_pos) > 0 else None
    goal_pos_t = torch.tensor(goal_pos, dtype=torch.float32, device="cuda:0") if len(goal_pos) > 0 else None
    
    # 1. Compute all potential fields
    print("Computing potential fields...")
    barrier_field = visualizer.compute_potential_field(
        safe_pinn_model, agent_pos_t, obs_pos_t
    )
    
    if goal_pos_t is not None:
        task_field = visualizer.compute_task_potential_field(
            safe_pinn_model, agent_pos_t, goal_pos_t
        )
        total_field = visualizer.compute_total_potential_field(
            safe_pinn_model, agent_pos_t, obs_pos_t, goal_pos_t
        )
    else:
        task_field = np.zeros_like(barrier_field)
        total_field = barrier_field
    
    # 2. Generate 3D surface plot
    print("Rendering 3D surface plot...")
    surface_3d = visualizer.render_3d_surface_plot(
        potential_field=barrier_field,
        agent_positions=agent_pos,
        obstacle_positions=obs_pos,
        title="3D Barrier Potential Surface",
        elev=30,
        azim=45,
        cmap="viridis"
    )
    
    # 3. Generate energy flow diagram
    print("Rendering energy flow diagram...")
    flow_diagram = visualizer.render_energy_flow_diagram(
        potential_field=barrier_field,
        agent_positions=agent_pos,
        obstacle_positions=obs_pos,
        goal_positions=goal_pos,
        arrow_density=15
    )
    
    # 4. Generate safety margin contours
    print("Rendering safety contours...")
    safety_contours = visualizer.render_safety_margin_contours(
        potential_field=barrier_field,
        agent_positions=agent_pos,
        obstacle_positions=obs_pos
    )
    
    # 5. Generate energy decomposition
    print("Rendering energy decomposition...")
    energy_decomp = visualizer.render_energy_decomposition(
        model=safe_pinn_model,
        sample_state=current_state
    )
    
    # 6. Render task potential
    print("Rendering task potential...")
    task_image = visualizer.render_potential_field(
        task_field,
        agent_positions=agent_pos,
        obstacle_positions=obs_pos,
        goal_positions=goal_pos,
        title="Task Potential Field",
        cmap="Blues"
    )
    
    # 7. Render total potential
    print("Rendering total potential...")
    total_image = visualizer.render_potential_field(
        total_field,
        agent_positions=agent_pos,
        obstacle_positions=obs_pos,
        goal_positions=goal_pos,
        title="Total Potential Field",
        cmap="viridis"
    )
    
    # Log all to wandb
    wandb.log({
        "Viz/barrier_potential": wandb.Image(visualizer.render_potential_field(
            barrier_field, agent_pos, obs_pos, goal_pos, "Barrier Potential"
        )),
        "Viz/task_potential": wandb.Image(task_image),
        "Viz/total_potential": wandb.Image(total_image),
        "Viz/3d_surface": wandb.Image(surface_3d),
        "Viz/energy_flow": wandb.Image(flow_diagram),
        "Viz/safety_contours": wandb.Image(safety_contours),
        "Viz/energy_decomposition": wandb.Image(energy_decomp),
        "step": step
    })
    
    return {
        'barrier': barrier_field,
        'task': task_field,
        'total': total_field,
        'surface_3d': surface_3d,
        'flow': flow_diagram,
        'contours': safety_contours,
        'energy': energy_decomp
    }


def example_4_multiple_3d_views(env, policy, current_state, step):
    """Example 4: Generate multiple 3D views for supplementary materials."""
    
    visualizer = PotentialFieldVisualizer(
        world_bounds=(-1.5, 1.5, -1.5, 1.5),
        grid_resolution=100,
        device="cuda:0"
    )
    
    safe_pinn_model = get_safe_pinn_model(policy)
    positions = extract_positions_from_env(env)
    
    agent_pos_t = torch.tensor(positions['agents'], dtype=torch.float32, device="cuda:0")
    obs_pos_t = torch.tensor(positions['obstacles'], dtype=torch.float32, device="cuda:0") if len(positions['obstacles']) > 0 else None
    
    barrier_field = visualizer.compute_potential_field(
        safe_pinn_model, agent_pos_t, obs_pos_t
    )
    
    # Generate multiple viewing angles
    views = []
    angles = [
        (20, 30, "Low angle view"),
        (30, 45, "Standard view"),
        (40, 60, "High angle view"),
        (30, 135, "Opposite view"),
    ]
    
    for elev, azim, desc in angles:
        print(f"Rendering 3D view: {desc} (elev={elev}, azim={azim})")
        view_image = visualizer.render_3d_surface_plot(
            potential_field=barrier_field,
            agent_positions=positions['agents'],
            obstacle_positions=positions['obstacles'],
            title=f"3D Surface - {desc}",
            elev=elev,
            azim=azim,
            cmap="plasma"
        )
        views.append(view_image)
        
        # Log to wandb
        wandb.log({
            f"Viz/3d_surface_elev{elev}_azim{azim}": wandb.Image(view_image),
            "step": step
        })
    
    return views


def example_5_time_series_video(env, policy, rollout_data, max_steps=100):
    """Example 5: Generate time-series video of potential field evolution."""
    
    visualizer = PotentialFieldVisualizer(
        world_bounds=(-1.5, 1.5, -1.5, 1.5),
        grid_resolution=50,  # Lower resolution for video
        device="cuda:0"
    )
    
    safe_pinn_model = get_safe_pinn_model(policy)
    
    frames = []
    
    for step in range(min(max_steps, len(rollout_data))):
        print(f"Generating frame {step}/{max_steps}")
        
        # Get state at this step
        state = rollout_data[step]
        positions = extract_positions_from_env(env, env_index=0)
        
        # Generate combined visualization
        combined_image = visualizer.render_combined_visualization(
            env_frame=env.render()[0],  # Get environment render
            potential_field=visualizer.compute_potential_field(
                safe_pinn_model,
                torch.tensor(positions['agents'], dtype=torch.float32, device="cuda:0"),
                torch.tensor(positions['obstacles'], dtype=torch.float32, device="cuda:0") if len(positions['obstacles']) > 0 else None
            ),
            agent_positions=positions['agents'],
            obstacle_positions=positions['obstacles'],
            goal_positions=positions['goals'],
            step=step
        )
        
        frames.append(combined_image)
    
    # Log video to wandb
    wandb.log({
        "Viz/potential_field_evolution": wandb.Video(
            np.array(frames).transpose(0, 3, 1, 2),  # (T, H, W, C) -> (T, C, H, W)
            fps=10,
            format="mp4"
        )
    })
    
    return frames


def example_6_comparative_analysis(env, policy_list, model_names, current_state, step):
    """Example 6: Compare potential fields across different models."""
    
    visualizer = PotentialFieldVisualizer(
        world_bounds=(-1.5, 1.5, -1.5, 1.5),
        grid_resolution=75,
        device="cuda:0"
    )
    
    positions = extract_positions_from_env(env)
    agent_pos_t = torch.tensor(positions['agents'], dtype=torch.float32, device="cuda:0")
    obs_pos_t = torch.tensor(positions['obstacles'], dtype=torch.float32, device="cuda:0") if len(positions['obstacles']) > 0 else None
    
    # Compute potential fields for each model
    for policy, name in zip(policy_list, model_names):
        safe_pinn_model = get_safe_pinn_model(policy)
        if safe_pinn_model is None:
            continue
        
        barrier_field = visualizer.compute_potential_field(
            safe_pinn_model, agent_pos_t, obs_pos_t
        )
        
        # Render
        image = visualizer.render_potential_field(
            barrier_field,
            agent_positions=positions['agents'],
            obstacle_positions=positions['obstacles'],
            goal_positions=positions['goals'],
            title=f"Barrier Potential - {name}"
        )
        
        # Log
        wandb.log({
            f"Viz/comparison_{name}": wandb.Image(image),
            "step": step
        })


# Main execution example
if __name__ == "__main__":
    """
    This would be called from the evaluation loop in experiment.py
    """
    
    # Example integration in evaluation loop:
    """
    # In gemsmarl/experiment/experiment.py, _evaluation_loop method:
    
    from gemsmarl.experiment.potential_visualizer import PotentialFieldVisualizer, get_safe_pinn_model
    
    # Initialize visualizer once
    if self.cfg.render and self.logger is not None:
        safe_pinn_model = get_safe_pinn_model(self.policy)
        if safe_pinn_model is not None:
            visualizer = PotentialFieldVisualizer(
                world_bounds=(-1.5, 1.5, -1.5, 1.5),
                grid_resolution=100,  # High res for publication
                device=self.device
            )
    
    # During evaluation loop:
    for step in range(max_steps):
        # ... existing evaluation code ...
        
        # Generate visualizations every N steps
        if step % 10 == 0 and visualizer is not None:
            positions = extract_positions_from_env(self.eval_env, env_index=0)
            
            # Generate publication figure
            pub_figure = visualizer.render_publication_figure(
                model=safe_pinn_model,
                sample_state=current_state,
                agent_positions=positions['agents'],
                obstacle_positions=positions['obstacles'],
                goal_positions=positions['goals'],
                step=step,
                dpi=150
            )
            
            # Log to wandb
            self.logger.log_potential_field(pub_figure, step, "publication_figure")
    """
    
    print("See function docstrings for usage examples")
    print("Import this module in your evaluation script to use the examples")
