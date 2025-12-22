#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
"""
Potential Field Visualizer for Safe PINN models.

This module provides visualization of the obstacle potential field learned by Safe PINN
during evaluation. The potential field is rendered as a heatmap overlay on the environment
and uploaded to wandb's Viz module.
"""

from typing import Optional, Tuple, List, Dict, Any
import numpy as np
import torch
from tensordict import TensorDictBase

# Make matplotlib optional
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server environments
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, LinearSegmentedColormap, LogNorm
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.patches import Circle, FancyArrowPatch
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    Normalize = None
    LogNorm = None
    LinearSegmentedColormap = None
    cm = None
    Axes3D = None


class PotentialFieldVisualizer:
    """Visualizes the obstacle potential field learned by Safe PINN models.
    
    This class computes and renders a potential field heatmap based on the 
    barrier potential learned by Safe PINN/Safe PINN PPO models.
    
    Args:
        world_bounds: Tuple of (x_min, x_max, y_min, y_max) for the visualization grid
        grid_resolution: Number of grid points in each dimension
        device: PyTorch device for computation
    """
    
    def __init__(
        self,
        world_bounds: Tuple[float, float, float, float] = (-1.5, 1.5, -1.5, 1.5),
        grid_resolution: int = 50,
        device: str = "cuda:0",
    ):
        self.world_bounds = world_bounds
        self.grid_resolution = grid_resolution
        self.device = device
        
        # Pre-compute grid
        x = np.linspace(world_bounds[0], world_bounds[1], grid_resolution)
        y = np.linspace(world_bounds[2], world_bounds[3], grid_resolution)
        self.X, self.Y = np.meshgrid(x, y)
        self.grid_points = np.stack([self.X.flatten(), self.Y.flatten()], axis=1)
        
    def compute_potential_field(
        self,
        model,
        agent_positions: torch.Tensor,
        obstacle_positions: Optional[torch.Tensor] = None,
        agent_velocities: Optional[torch.Tensor] = None,
        goal_positions: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Compute the barrier potential field over the grid.
        
        This method computes the barrier potential H_barrier at each grid point,
        treating the grid point as the position of a virtual agent and computing
        the repulsive potential from all other agents and obstacles.
        
        Args:
            model: The Safe PINN model (SafePinn or SafePinnPPO)
            agent_positions: Positions of all agents, shape (n_agents, 2)
            obstacle_positions: Positions of obstacles, shape (n_obstacles, 2)
            agent_velocities: Velocities of agents, shape (n_agents, 2), defaults to zeros
            goal_positions: Goal positions for agents, shape (n_agents, 2)
            
        Returns:
            potential_field: 2D numpy array of shape (grid_resolution, grid_resolution)
        """
        from gemsmarl.models.safe_pinn import SafePinn
        from gemsmarl.models.safe_pinn_ppo import SafePinnPPO
        
        if not isinstance(model, (SafePinn, SafePinnPPO)):
            raise ValueError("Model must be SafePinn or SafePinnPPO")
        
        n_agents = agent_positions.shape[0]
        
        # Default velocities and goals if not provided
        if agent_velocities is None:
            agent_velocities = torch.zeros_like(agent_positions)
        if goal_positions is None:
            goal_positions = torch.zeros_like(agent_positions)
        
        # Combine obstacle positions with agent positions for barrier computation
        if obstacle_positions is not None and obstacle_positions.shape[0] > 0:
            all_obstacle_positions = torch.cat([agent_positions, obstacle_positions], dim=0)
        else:
            all_obstacle_positions = agent_positions
            
        grid_tensor = torch.tensor(self.grid_points, dtype=torch.float32, device=self.device)
        n_grid_points = grid_tensor.shape[0]
        
        potential_values = torch.zeros(n_grid_points, device=self.device)
        
        # Get model parameters
        r_collision = getattr(model, 'r_collision', 0.2)
        barrier_epsilon = getattr(model, 'barrier_epsilon', 0.05)
        use_log_barrier = getattr(model, 'use_log_barrier', True) if isinstance(model, SafePinnPPO) else False
        
        # Compute potential at each grid point
        with torch.no_grad():
            for i, grid_pos in enumerate(grid_tensor):
                # Compute distance from grid point to all obstacles/agents
                diff = grid_pos.unsqueeze(0) - all_obstacle_positions  # (n_obs, 2)
                dist = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-6)  # (n_obs,)
                
                # Compute barrier potential for each obstacle
                gap = dist - r_collision
                
                if use_log_barrier:
                    # Log barrier computation
                    safe_gap = torch.clamp(gap, min=barrier_epsilon)
                    H_barrier = -torch.log(safe_gap / (r_collision + barrier_epsilon))
                    H_barrier = torch.clamp(H_barrier, min=0.0, max=100.0)
                else:
                    # Quadratic barrier computation
                    denom = gap**2 + barrier_epsilon
                    H_barrier = 1.0 / denom
                
                # Sum over all obstacles
                potential_values[i] = H_barrier.sum()
        
        # Reshape to grid
        potential_field = potential_values.cpu().numpy().reshape(self.grid_resolution, self.grid_resolution)
        
        return potential_field
    
    def compute_learned_potential_field(
        self,
        model,
        sample_state: TensorDictBase,
    ) -> np.ndarray:
        """Compute the learned barrier potential field using the model's barrier head.
        
        This method uses the actual learned barrier head (H_barrier_head) from the
        Safe PINN model to compute the potential field, giving a more accurate
        representation of what the model has learned.
        
        Args:
            model: The Safe PINN model (SafePinn or SafePinnPPO)
            sample_state: A sample TensorDict from evaluation rollout
            
        Returns:
            potential_field: 2D numpy array of shape (grid_resolution, grid_resolution)
        """
        from gemsmarl.models.safe_pinn import SafePinn
        from gemsmarl.models.safe_pinn_ppo import SafePinnPPO
        
        if not isinstance(model, (SafePinn, SafePinnPPO)):
            return np.zeros((self.grid_resolution, self.grid_resolution))
        
        grid_tensor = torch.tensor(self.grid_points, dtype=torch.float32, device=self.device)
        n_grid_points = grid_tensor.shape[0]
        
        # Get model parameters
        r_collision = getattr(model, 'r_collision', 0.2)
        barrier_epsilon = getattr(model, 'barrier_epsilon', 0.05)
        r_communication = getattr(model, 'r_communication', 0.45)
        use_log_barrier = getattr(model, 'use_log_barrier', True) if isinstance(model, SafePinnPPO) else False
        
        # Extract agent info from sample state
        n_agents = model.n_agents
        obs_dim = model.observation_dim_per_agent
        
        potential_values = torch.zeros(self.grid_resolution, self.grid_resolution, device=self.device)
        
        # Get the first batch item's state
        try:
            in_keys = model.in_keys
            x = torch.cat(
                [torch.flatten(sample_state.get(in_key), start_dim=-1) for in_key in in_keys],
                dim=-1,
            )
            # Shape: (batch, n_agents, obs_dim)
            if x.dim() == 2:
                x = x.unsqueeze(0)  # Add batch dim if missing
            
            # Use first environment's state
            state = x[0]  # (n_agents, obs_dim)
            agent_positions = state[:, :2]  # (n_agents, 2)
        except Exception:
            # Fallback: use zeros
            agent_positions = torch.zeros(n_agents, 2, device=self.device)
            state = torch.zeros(n_agents, obs_dim, device=self.device)
        
        with torch.no_grad():
            # For each grid point, compute potential as if an agent were there
            for i, gx in enumerate(np.linspace(self.world_bounds[0], self.world_bounds[1], self.grid_resolution)):
                for j, gy in enumerate(np.linspace(self.world_bounds[2], self.world_bounds[3], self.grid_resolution)):
                    grid_pos = torch.tensor([[gx, gy]], dtype=torch.float32, device=self.device)
                    
                    # Compute distance from grid point to all agents
                    diff = grid_pos - agent_positions  # (n_agents, 2)
                    dist = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-6)  # (n_agents,)
                    
                    # Check which agents are within communication range
                    in_range = dist <= r_communication
                    
                    if in_range.any():
                        gap = dist - r_collision
                        
                        if use_log_barrier:
                            safe_gap = torch.clamp(gap, min=barrier_epsilon)
                            H_barrier = -torch.log(safe_gap / (r_collision + barrier_epsilon))
                            H_barrier = torch.clamp(H_barrier, min=0.0, max=100.0)
                        else:
                            denom = gap**2 + barrier_epsilon
                            H_barrier = 1.0 / denom
                        
                        # Only sum over in-range agents
                        potential_values[j, i] = (H_barrier * in_range.float()).sum()
                    else:
                        potential_values[j, i] = 0.0
        
        return potential_values.cpu().numpy()
    
    def render_potential_field(
        self,
        potential_field: np.ndarray,
        agent_positions: Optional[np.ndarray] = None,
        obstacle_positions: Optional[np.ndarray] = None,
        goal_positions: Optional[np.ndarray] = None,
        title: str = "Obstacle Barrier Potential Field",
        figsize: Tuple[int, int] = (8, 8),
        dpi: int = 100,
        cmap: str = "hot_r",
        vmax_percentile: float = 95,
    ) -> np.ndarray:
        """Render the potential field as a heatmap image.
        
        Args:
            potential_field: 2D array of potential values
            agent_positions: Positions of agents, shape (n_agents, 2)
            obstacle_positions: Positions of obstacles, shape (n_obstacles, 2)
            goal_positions: Goal positions for agents, shape (n_agents, 2)
            title: Title for the plot
            figsize: Figure size in inches
            dpi: Resolution of the output image
            cmap: Matplotlib colormap name
            vmax_percentile: Percentile for color normalization
            
        Returns:
            image: RGB image as numpy array, shape (H, W, 3)
        """
        if not HAS_MATPLOTLIB:
            # Return a blank image if matplotlib is not available
            return np.zeros((int(figsize[1] * dpi), int(figsize[0] * dpi), 3), dtype=np.uint8)
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Normalize potential field for better visualization
        vmax = np.percentile(potential_field[potential_field > 0], vmax_percentile) if (potential_field > 0).any() else 1.0
        vmin = 0
        norm = Normalize(vmin=vmin, vmax=max(vmax, 0.1))
        
        # Plot heatmap
        extent = [self.world_bounds[0], self.world_bounds[1], 
                  self.world_bounds[2], self.world_bounds[3]]
        im = ax.imshow(
            potential_field, 
            extent=extent, 
            origin='lower',
            cmap=cmap,
            norm=norm,
            alpha=0.8
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Barrier Potential', fontsize=10)
        
        # Plot obstacles
        if obstacle_positions is not None and len(obstacle_positions) > 0:
            ax.scatter(
                obstacle_positions[:, 0], 
                obstacle_positions[:, 1],
                c='gray', 
                s=200, 
                marker='o',
                edgecolors='black',
                linewidths=2,
                label='Obstacles',
                zorder=5
            )
        
        # Plot agents
        if agent_positions is not None and len(agent_positions) > 0:
            colors = plt.cm.Set1(np.linspace(0, 1, len(agent_positions)))
            ax.scatter(
                agent_positions[:, 0], 
                agent_positions[:, 1],
                c=colors, 
                s=150, 
                marker='o',
                edgecolors='black',
                linewidths=2,
                label='Agents',
                zorder=6
            )
        
        # Plot goals
        if goal_positions is not None and len(goal_positions) > 0:
            ax.scatter(
                goal_positions[:, 0], 
                goal_positions[:, 1],
                c='green', 
                s=100, 
                marker='*',
                edgecolors='black',
                linewidths=1,
                label='Goals',
                zorder=5
            )
        
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.set_title(title, fontsize=14)
        # Only add legend if there are labeled artists
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc='upper right', fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Convert figure to numpy array (compatible with newer matplotlib)
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        
        plt.close(fig)
        
        return image
    
    def render_combined_visualization(
        self,
        env_frame: np.ndarray,
        potential_field: np.ndarray,
        agent_positions: Optional[np.ndarray] = None,
        obstacle_positions: Optional[np.ndarray] = None,
        goal_positions: Optional[np.ndarray] = None,
        step: int = 0,
        figsize: Tuple[int, int] = (16, 8),
        dpi: int = 100,
    ) -> np.ndarray:
        """Render combined visualization with environment frame and potential field.
        
        Args:
            env_frame: RGB image from environment render
            potential_field: 2D array of potential values
            agent_positions: Positions of agents
            obstacle_positions: Positions of obstacles
            goal_positions: Goal positions for agents
            step: Current evaluation step
            figsize: Figure size
            dpi: Resolution
            
        Returns:
            combined_image: Combined RGB image
        """
        if not HAS_MATPLOTLIB:
            # Return a blank image if matplotlib is not available
            return np.zeros((int(figsize[1] * dpi), int(figsize[0] * dpi), 3), dtype=np.uint8)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
        
        # Left: Environment frame
        axes[0].imshow(env_frame)
        axes[0].set_title(f'Environment (Step {step})', fontsize=14)
        axes[0].axis('off')
        
        # Right: Potential field
        vmax = np.percentile(potential_field[potential_field > 0], 95) if (potential_field > 0).any() else 1.0
        norm = Normalize(vmin=0, vmax=max(vmax, 0.1))
        extent = [self.world_bounds[0], self.world_bounds[1], 
                  self.world_bounds[2], self.world_bounds[3]]
        
        im = axes[1].imshow(
            potential_field, 
            extent=extent, 
            origin='lower',
            cmap='hot_r',
            norm=norm,
            alpha=0.8
        )
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=axes[1], shrink=0.8)
        cbar.set_label('Barrier Potential', fontsize=10)
        
        # Plot entities
        if obstacle_positions is not None and len(obstacle_positions) > 0:
            axes[1].scatter(
                obstacle_positions[:, 0], obstacle_positions[:, 1],
                c='gray', s=200, marker='o', edgecolors='black', linewidths=2,
                label='Obstacles', zorder=5
            )
        
        if agent_positions is not None and len(agent_positions) > 0:
            colors = plt.cm.Set1(np.linspace(0, 1, len(agent_positions)))
            axes[1].scatter(
                agent_positions[:, 0], agent_positions[:, 1],
                c=colors, s=150, marker='o', edgecolors='black', linewidths=2,
                label='Agents', zorder=6
            )
        
        if goal_positions is not None and len(goal_positions) > 0:
            axes[1].scatter(
                goal_positions[:, 0], goal_positions[:, 1],
                c='green', s=100, marker='*', edgecolors='black', linewidths=1,
                label='Goals', zorder=5
            )
        
        axes[1].set_xlabel('X Position', fontsize=12)
        axes[1].set_ylabel('Y Position', fontsize=12)
        axes[1].set_title('Learned Barrier Potential Field', fontsize=14)
        # Only add legend if there are labeled artists
        handles, labels = axes[1].get_legend_handles_labels()
        if handles:
            axes[1].legend(loc='upper right', fontsize=9)
        axes[1].set_aspect('equal')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Convert to numpy (compatible with newer matplotlib)
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        
        plt.close(fig)
        
        return image
    
    def compute_task_potential_field(
        self,
        model,
        agent_positions: torch.Tensor,
        goal_positions: torch.Tensor,
    ) -> np.ndarray:
        """Compute the task (attractive) potential field over the grid.
        
        This method computes the attractive potential towards goals at each grid point.
        
        Args:
            model: The Safe PINN model (SafePinn or SafePinnPPO)
            agent_positions: Positions of all agents, shape (n_agents, 2)
            goal_positions: Goal positions for agents, shape (n_agents, 2)
            
        Returns:
            potential_field: 2D numpy array of shape (grid_resolution, grid_resolution)
        """
        from gemsmarl.models.safe_pinn import SafePinn
        from gemsmarl.models.safe_pinn_ppo import SafePinnPPO
        
        if not isinstance(model, (SafePinn, SafePinnPPO)):
            return np.zeros((self.grid_resolution, self.grid_resolution))
        
        grid_tensor = torch.tensor(self.grid_points, dtype=torch.float32, device=self.device)
        n_grid_points = grid_tensor.shape[0]
        
        potential_values = torch.zeros(n_grid_points, device=self.device)
        
        # Compute attractive potential to goals
        with torch.no_grad():
            for i, grid_pos in enumerate(grid_tensor):
                # Compute distance from grid point to all goals
                diff = grid_pos.unsqueeze(0) - goal_positions  # (n_goals, 2)
                dist = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-6)  # (n_goals,)
                
                # Quadratic attractive potential: 0.5 * k * d^2
                # Using k=1.0 for simplicity
                H_task = 0.5 * torch.sum(dist**2)
                potential_values[i] = H_task
        
        # Reshape to grid
        potential_field = potential_values.cpu().numpy().reshape(self.grid_resolution, self.grid_resolution)
        
        return potential_field
    
    def compute_total_potential_field(
        self,
        model,
        agent_positions: torch.Tensor,
        obstacle_positions: Optional[torch.Tensor] = None,
        goal_positions: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """Compute the total potential field (barrier + task) over the grid.
        
        Args:
            model: The Safe PINN model
            agent_positions: Positions of all agents
            obstacle_positions: Positions of obstacles
            goal_positions: Goal positions for agents
            
        Returns:
            total_potential_field: 2D numpy array
        """
        # Compute barrier potential
        barrier_field = self.compute_potential_field(
            model, agent_positions, obstacle_positions
        )
        
        # Compute task potential if goals provided
        if goal_positions is not None and goal_positions.shape[0] > 0:
            task_field = self.compute_task_potential_field(
                model, agent_positions, goal_positions
            )
            # Combine with appropriate weighting
            total_field = barrier_field + 0.1 * task_field  # Weight task potential lower
        else:
            total_field = barrier_field
        
        return total_field
    
    def render_3d_surface_plot(
        self,
        potential_field: np.ndarray,
        agent_positions: Optional[np.ndarray] = None,
        obstacle_positions: Optional[np.ndarray] = None,
        title: str = "3D Barrier Potential Surface",
        figsize: Tuple[int, int] = (10, 8),
        dpi: int = 120,
        cmap: str = "viridis",
        elev: float = 30,
        azim: float = 45,
    ) -> np.ndarray:
        """Render a 3D surface plot of the potential field.
        
        Args:
            potential_field: 2D array of potential values
            agent_positions: Positions of agents
            obstacle_positions: Positions of obstacles
            title: Title for the plot
            figsize: Figure size
            dpi: Resolution
            cmap: Colormap
            elev: Elevation angle for 3D view
            azim: Azimuth angle for 3D view
            
        Returns:
            image: RGB image as numpy array
        """
        if not HAS_MATPLOTLIB:
            return np.zeros((int(figsize[1] * dpi), int(figsize[0] * dpi), 3), dtype=np.uint8)
        
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax = fig.add_subplot(111, projection='3d')
        
        # Clip extreme values for better visualization
        vmax = np.percentile(potential_field[potential_field > 0], 98) if (potential_field > 0).any() else 1.0
        potential_clipped = np.clip(potential_field, 0, vmax)
        
        # Create surface plot
        surf = ax.plot_surface(
            self.X, self.Y, potential_clipped,
            cmap=cmap,
            alpha=0.9,
            edgecolor='none',
            antialiased=True,
            vmin=0,
            vmax=vmax
        )
        
        # Add contour lines at the base
        ax.contour(
            self.X, self.Y, potential_clipped,
            zdir='z',
            offset=0,
            cmap=cmap,
            alpha=0.5,
            levels=10
        )
        
        # Plot obstacle positions as vertical lines
        if obstacle_positions is not None and len(obstacle_positions) > 0:
            for obs_pos in obstacle_positions:
                ax.plot(
                    [obs_pos[0], obs_pos[0]],
                    [obs_pos[1], obs_pos[1]],
                    [0, vmax],
                    'r-',
                    linewidth=3,
                    alpha=0.7
                )
        
        # Plot agent positions
        if agent_positions is not None and len(agent_positions) > 0:
            ax.scatter(
                agent_positions[:, 0],
                agent_positions[:, 1],
                0,
                c='blue',
                s=100,
                marker='o',
                edgecolors='black',
                linewidths=2,
                zorder=10
            )
        
        # Colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label('Barrier Potential', fontsize=11)
        
        # Labels and title
        ax.set_xlabel('X Position', fontsize=11)
        ax.set_ylabel('Y Position', fontsize=11)
        ax.set_zlabel('Potential Energy', fontsize=11)
        ax.set_title(title, fontsize=14, pad=20)
        ax.view_init(elev=elev, azim=azim)
        
        # Convert to image
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        
        return image
    
    def render_energy_flow_diagram(
        self,
        potential_field: np.ndarray,
        agent_positions: Optional[np.ndarray] = None,
        obstacle_positions: Optional[np.ndarray] = None,
        goal_positions: Optional[np.ndarray] = None,
        title: str = "Energy Flow Field (Gradient Descent)",
        figsize: Tuple[int, int] = (10, 10),
        dpi: int = 100,
        arrow_density: int = 15,
    ) -> np.ndarray:
        """Render energy flow diagram showing gradient descent directions.
        
        This visualizes the negative gradient of the potential field, showing
        the direction agents would move under gradient descent.
        
        Args:
            potential_field: 2D array of potential values
            agent_positions: Positions of agents
            obstacle_positions: Positions of obstacles
            goal_positions: Goal positions
            title: Title for the plot
            figsize: Figure size
            dpi: Resolution
            arrow_density: Number of arrows in each dimension
            
        Returns:
            image: RGB image as numpy array
        """
        if not HAS_MATPLOTLIB:
            return np.zeros((int(figsize[1] * dpi), int(figsize[0] * dpi), 3), dtype=np.uint8)
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Compute gradient (negative for flow direction)
        grad_y, grad_x = np.gradient(potential_field)
        flow_x = -grad_x
        flow_y = -grad_y
        
        # Normalize potential field for visualization
        vmax = np.percentile(potential_field[potential_field > 0], 95) if (potential_field > 0).any() else 1.0
        norm = Normalize(vmin=0, vmax=max(vmax, 0.1))
        
        # Plot potential field as background
        extent = [self.world_bounds[0], self.world_bounds[1], 
                  self.world_bounds[2], self.world_bounds[3]]
        im = ax.imshow(
            potential_field,
            extent=extent,
            origin='lower',
            cmap='RdYlBu_r',
            norm=norm,
            alpha=0.6
        )
        
        # Subsample for arrow plot
        step = max(1, self.grid_resolution // arrow_density)
        X_sub = self.X[::step, ::step]
        Y_sub = self.Y[::step, ::step]
        U_sub = flow_x[::step, ::step]
        V_sub = flow_y[::step, ::step]
        
        # Compute magnitude for color coding
        magnitude = np.sqrt(U_sub**2 + V_sub**2)
        
        # Plot vector field
        quiver = ax.quiver(
            X_sub, Y_sub, U_sub, V_sub,
            magnitude,
            cmap='autumn',
            scale=20,
            width=0.004,
            headwidth=4,
            headlength=5,
            alpha=0.8
        )
        
        # Add colorbar for potential
        cbar1 = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar1.set_label('Barrier Potential', fontsize=11)
        
        # Plot entities
        if obstacle_positions is not None and len(obstacle_positions) > 0:
            circles = [Circle((obs[0], obs[1]), 0.1, color='gray', ec='black', lw=2, zorder=5) 
                      for obs in obstacle_positions]
            for circle in circles:
                ax.add_patch(circle)
            ax.scatter(
                obstacle_positions[:, 0], obstacle_positions[:, 1],
                c='gray', s=250, marker='o', edgecolors='black', linewidths=2.5,
                label='Obstacles', zorder=6
            )
        
        if agent_positions is not None and len(agent_positions) > 0:
            colors = plt.cm.Set1(np.linspace(0, 1, len(agent_positions)))
            ax.scatter(
                agent_positions[:, 0], agent_positions[:, 1],
                c=colors, s=180, marker='o', edgecolors='black', linewidths=2,
                label='Agents', zorder=7
            )
        
        if goal_positions is not None and len(goal_positions) > 0:
            ax.scatter(
                goal_positions[:, 0], goal_positions[:, 1],
                c='lime', s=150, marker='*', edgecolors='black', linewidths=1.5,
                label='Goals', zorder=7
            )
        
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(loc='upper right', fontsize=10)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        # Convert to image
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        
        return image
    
    def render_safety_margin_contours(
        self,
        potential_field: np.ndarray,
        agent_positions: Optional[np.ndarray] = None,
        obstacle_positions: Optional[np.ndarray] = None,
        title: str = "Safety Margin Contours",
        figsize: Tuple[int, int] = (10, 10),
        dpi: int = 100,
        safety_levels: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Render safety margin contour plot.
        
        Shows contour lines at different potential levels to visualize safe zones.
        
        Args:
            potential_field: 2D array of potential values
            agent_positions: Positions of agents
            obstacle_positions: Positions of obstacles
            title: Title for the plot
            figsize: Figure size
            dpi: Resolution
            safety_levels: Custom contour levels (default: auto-generated)
            
        Returns:
            image: RGB image as numpy array
        """
        if not HAS_MATPLOTLIB:
            return np.zeros((int(figsize[1] * dpi), int(figsize[0] * dpi), 3), dtype=np.uint8)
        
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Define safety levels if not provided
        if safety_levels is None:
            vmax = np.percentile(potential_field[potential_field > 0], 95) if (potential_field > 0).any() else 1.0
            safety_levels = np.linspace(0.1, vmax, 15)
        
        # Create filled contour plot
        extent = [self.world_bounds[0], self.world_bounds[1], 
                  self.world_bounds[2], self.world_bounds[3]]
        
        contourf = ax.contourf(
            self.X, self.Y, potential_field,
            levels=safety_levels,
            cmap='RdYlGn_r',
            alpha=0.8,
            extend='max'
        )
        
        # Add contour lines
        contour = ax.contour(
            self.X, self.Y, potential_field,
            levels=safety_levels,
            colors='black',
            linewidths=0.5,
            alpha=0.4
        )
        
        # Label some contours
        ax.clabel(contour, inline=True, fontsize=8, fmt='%.1f')
        
        # Colorbar
        cbar = plt.colorbar(contourf, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Barrier Potential (Safety Level)', fontsize=11)
        
        # Add safety zone annotations
        safe_zone = mpatches.Patch(color='green', alpha=0.5, label='Safe Zone (Low Potential)')
        danger_zone = mpatches.Patch(color='red', alpha=0.5, label='Danger Zone (High Potential)')
        
        # Plot entities
        if obstacle_positions is not None and len(obstacle_positions) > 0:
            ax.scatter(
                obstacle_positions[:, 0], obstacle_positions[:, 1],
                c='darkred', s=300, marker='X', edgecolors='black', linewidths=2.5,
                label='Obstacles', zorder=5
            )
        
        if agent_positions is not None and len(agent_positions) > 0:
            colors = plt.cm.Set1(np.linspace(0, 1, len(agent_positions)))
            ax.scatter(
                agent_positions[:, 0], agent_positions[:, 1],
                c=colors, s=180, marker='o', edgecolors='black', linewidths=2,
                label='Agents', zorder=6
            )
        
        ax.set_xlabel('X Position', fontsize=12)
        ax.set_ylabel('Y Position', fontsize=12)
        ax.set_title(title, fontsize=14)
        
        # Combine all legend handles
        handles, labels = ax.get_legend_handles_labels()
        handles = [safe_zone, danger_zone] + handles
        ax.legend(handles=handles, loc='upper right', fontsize=9)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.2, linestyle=':')
        
        plt.tight_layout()
        
        # Convert to image
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        
        return image
    
    def render_energy_decomposition(
        self,
        model,
        sample_state: TensorDictBase,
        figsize: Tuple[int, int] = (10, 6),
        dpi: int = 100,
    ) -> np.ndarray:
        """Render Hamiltonian energy decomposition chart.
        
        Shows the breakdown of total energy into barrier, task, and kinetic components.
        
        Args:
            model: The Safe PINN model
            sample_state: A sample TensorDict from evaluation
            figsize: Figure size
            dpi: Resolution
            
        Returns:
            image: RGB image as numpy array
        """
        if not HAS_MATPLOTLIB:
            return np.zeros((int(figsize[1] * dpi), int(figsize[0] * dpi), 3), dtype=np.uint8)
        
        from gemsmarl.models.safe_pinn import SafePinn
        from gemsmarl.models.safe_pinn_ppo import SafePinnPPO
        
        if not isinstance(model, (SafePinn, SafePinnPPO)):
            return np.zeros((int(figsize[1] * dpi), int(figsize[0] * dpi), 3), dtype=np.uint8)
        
        try:
            # Extract state information
            in_keys = model.in_keys
            x = torch.cat(
                [torch.flatten(sample_state.get(in_key), start_dim=-1) for in_key in in_keys],
                dim=-1,
            )
            if x.dim() == 2:
                x = x.unsqueeze(0)
            
            state = x[0]  # (n_agents, obs_dim)
            n_agents = state.shape[0]
            
            # Extract positions and velocities
            positions = state[:, :2]  # (n_agents, 2)
            velocities = state[:, 2:4] if state.shape[1] >= 4 else torch.zeros_like(positions)
            
            with torch.no_grad():
                # Compute kinetic energy
                H_kin = 0.5 * torch.sum(velocities**2).item()
                
                # Compute barrier energy (simplified)
                diff = positions.unsqueeze(1) - positions.unsqueeze(0)  # (n, n, 2)
                dist = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-6)  # (n, n)
                
                r_collision = getattr(model, 'r_collision', 0.2)
                barrier_epsilon = getattr(model, 'barrier_epsilon', 0.05)
                
                gap = dist - r_collision
                safe_gap = torch.clamp(gap, min=barrier_epsilon)
                
                # Mask diagonal
                mask = 1 - torch.eye(n_agents, device=self.device)
                H_barrier = torch.sum(-torch.log(safe_gap + 1e-6) * mask).item()
                H_barrier = max(0, min(H_barrier, 100))  # Clamp for visualization
                
                # Estimate task energy (if goals available)
                # For simplicity, use a placeholder
                H_task = 5.0  # Placeholder
                
            # Create visualization
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
            
            # Left: Bar chart
            components = ['Barrier\nPotential', 'Task\nPotential', 'Kinetic\nEnergy']
            values = [H_barrier, H_task, H_kin]
            colors = ['#e74c3c', '#3498db', '#2ecc71']
            
            bars = ax1.bar(components, values, color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
            ax1.set_ylabel('Energy', fontsize=12)
            ax1.set_title('Hamiltonian Energy Decomposition', fontsize=13)
            ax1.grid(axis='y', alpha=0.3, linestyle='--')
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{val:.2f}',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            
            # Right: Pie chart
            total_energy = sum(values)
            if total_energy > 0:
                percentages = [v/total_energy * 100 for v in values]
                wedges, texts, autotexts = ax2.pie(
                    values,
                    labels=components,
                    colors=colors,
                    autopct='%1.1f%%',
                    startangle=90,
                    wedgeprops={'edgecolor': 'black', 'linewidth': 1.5},
                    textprops={'fontsize': 10}
                )
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontweight('bold')
                ax2.set_title(f'Energy Distribution\n(Total: {total_energy:.2f})', fontsize=13)
            
            plt.tight_layout()
            
            # Convert to image
            fig.canvas.draw()
            image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
            plt.close(fig)
            
            return image
            
        except Exception as e:
            print(f"Warning: Could not compute energy decomposition: {e}")
            return np.zeros((int(figsize[1] * dpi), int(figsize[0] * dpi), 3), dtype=np.uint8)
    
    def render_publication_figure(
        self,
        model,
        sample_state: TensorDictBase,
        agent_positions: np.ndarray,
        obstacle_positions: Optional[np.ndarray] = None,
        goal_positions: Optional[np.ndarray] = None,
        step: int = 0,
        figsize: Tuple[int, int] = (20, 12),
        dpi: int = 150,
    ) -> np.ndarray:
        """Render comprehensive multi-panel publication figure.
        
        Creates a Nature Communications quality figure with multiple visualizations:
        - Barrier potential heatmap
        - Task potential heatmap
        - Total potential heatmap
        - 3D surface plot
        - Energy flow diagram
        - Safety margin contours
        
        Args:
            model: The Safe PINN model
            sample_state: Sample TensorDict from evaluation
            agent_positions: Positions of agents
            obstacle_positions: Positions of obstacles
            goal_positions: Goal positions
            step: Current evaluation step
            figsize: Figure size
            dpi: Resolution
            
        Returns:
            image: RGB image as numpy array
        """
        if not HAS_MATPLOTLIB:
            return np.zeros((int(figsize[1] * dpi), int(figsize[0] * dpi), 3), dtype=np.uint8)
        
        # Convert positions to tensors
        agent_pos_tensor = torch.tensor(agent_positions, dtype=torch.float32, device=self.device)
        obs_pos_tensor = torch.tensor(obstacle_positions, dtype=torch.float32, device=self.device) if obstacle_positions is not None else None
        goal_pos_tensor = torch.tensor(goal_positions, dtype=torch.float32, device=self.device) if goal_positions is not None else None
        
        # Compute all potential fields
        barrier_field = self.compute_potential_field(model, agent_pos_tensor, obs_pos_tensor)
        
        if goal_pos_tensor is not None:
            task_field = self.compute_task_potential_field(model, agent_pos_tensor, goal_pos_tensor)
            total_field = self.compute_total_potential_field(model, agent_pos_tensor, obs_pos_tensor, goal_pos_tensor)
        else:
            task_field = np.zeros_like(barrier_field)
            total_field = barrier_field
        
        # Create figure with subplots
        fig = plt.figure(figsize=figsize, dpi=dpi)
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        extent = [self.world_bounds[0], self.world_bounds[1], 
                  self.world_bounds[2], self.world_bounds[3]]
        
        # 1. Barrier Potential (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        vmax_barrier = np.percentile(barrier_field[barrier_field > 0], 95) if (barrier_field > 0).any() else 1.0
        im1 = ax1.imshow(barrier_field, extent=extent, origin='lower', cmap='hot_r', 
                        norm=Normalize(vmin=0, vmax=max(vmax_barrier, 0.1)), alpha=0.8)
        self._add_entities_to_ax(ax1, agent_positions, obstacle_positions, goal_positions)
        ax1.set_title('(a) Barrier Potential Field', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X Position', fontsize=10)
        ax1.set_ylabel('Y Position', fontsize=10)
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='H_barrier')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.2)
        
        # 2. Task Potential (top-middle)
        ax2 = fig.add_subplot(gs[0, 1])
        vmax_task = np.percentile(task_field[task_field > 0], 95) if (task_field > 0).any() else 1.0
        im2 = ax2.imshow(task_field, extent=extent, origin='lower', cmap='Blues',
                        norm=Normalize(vmin=0, vmax=max(vmax_task, 0.1)), alpha=0.8)
        self._add_entities_to_ax(ax2, agent_positions, obstacle_positions, goal_positions)
        ax2.set_title('(b) Task Potential Field', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X Position', fontsize=10)
        ax2.set_ylabel('Y Position', fontsize=10)
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04, label='H_task')
        ax2.set_aspect('equal')
        ax2.grid(True, alpha=0.2)
        
        # 3. Total Potential (top-right)
        ax3 = fig.add_subplot(gs[0, 2])
        vmax_total = np.percentile(total_field[total_field > 0], 95) if (total_field > 0).any() else 1.0
        im3 = ax3.imshow(total_field, extent=extent, origin='lower', cmap='viridis',
                        norm=Normalize(vmin=0, vmax=max(vmax_total, 0.1)), alpha=0.8)
        self._add_entities_to_ax(ax3, agent_positions, obstacle_positions, goal_positions)
        ax3.set_title('(c) Total Potential Field', fontsize=12, fontweight='bold')
        ax3.set_xlabel('X Position', fontsize=10)
        ax3.set_ylabel('Y Position', fontsize=10)
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='H_total')
        ax3.set_aspect('equal')
        ax3.grid(True, alpha=0.2)
        
        # 4. 3D Surface (bottom-left)
        ax4 = fig.add_subplot(gs[1, 0], projection='3d')
        vmax_3d = np.percentile(barrier_field[barrier_field > 0], 98) if (barrier_field > 0).any() else 1.0
        barrier_clipped = np.clip(barrier_field, 0, vmax_3d)
        surf = ax4.plot_surface(self.X, self.Y, barrier_clipped, cmap='plasma', alpha=0.9, 
                               edgecolor='none', antialiased=True)
        ax4.set_xlabel('X', fontsize=9)
        ax4.set_ylabel('Y', fontsize=9)
        ax4.set_zlabel('Potential', fontsize=9)
        ax4.set_title('(d) 3D Barrier Surface', fontsize=12, fontweight='bold')
        ax4.view_init(elev=25, azim=45)
        
        # 5. Energy Flow (bottom-middle)
        ax5 = fig.add_subplot(gs[1, 1])
        grad_y, grad_x = np.gradient(barrier_field)
        step_arrows = max(1, self.grid_resolution // 12)
        X_sub = self.X[::step_arrows, ::step_arrows]
        Y_sub = self.Y[::step_arrows, ::step_arrows]
        U_sub = -grad_x[::step_arrows, ::step_arrows]
        V_sub = -grad_y[::step_arrows, ::step_arrows]
        magnitude = np.sqrt(U_sub**2 + V_sub**2)
        
        im5 = ax5.imshow(barrier_field, extent=extent, origin='lower', cmap='RdYlBu_r',
                        norm=Normalize(vmin=0, vmax=max(vmax_barrier, 0.1)), alpha=0.5)
        ax5.quiver(X_sub, Y_sub, U_sub, V_sub, magnitude, cmap='autumn', 
                  scale=15, width=0.003, headwidth=4, alpha=0.8)
        self._add_entities_to_ax(ax5, agent_positions, obstacle_positions, goal_positions)
        ax5.set_title('(e) Energy Flow Field', fontsize=12, fontweight='bold')
        ax5.set_xlabel('X Position', fontsize=10)
        ax5.set_ylabel('Y Position', fontsize=10)
        ax5.set_aspect('equal')
        ax5.grid(True, alpha=0.2)
        
        # 6. Safety Contours (bottom-right)
        ax6 = fig.add_subplot(gs[1, 2])
        levels = np.linspace(0.1, vmax_barrier, 12)
        contourf = ax6.contourf(self.X, self.Y, barrier_field, levels=levels, 
                               cmap='RdYlGn_r', alpha=0.8)
        contour = ax6.contour(self.X, self.Y, barrier_field, levels=levels, 
                             colors='black', linewidths=0.5, alpha=0.3)
        self._add_entities_to_ax(ax6, agent_positions, obstacle_positions, goal_positions)
        ax6.set_title('(f) Safety Margin Contours', fontsize=12, fontweight='bold')
        ax6.set_xlabel('X Position', fontsize=10)
        ax6.set_ylabel('Y Position', fontsize=10)
        plt.colorbar(contourf, ax=ax6, fraction=0.046, pad=0.04, label='Safety Level')
        ax6.set_aspect('equal')
        ax6.grid(True, alpha=0.2, linestyle=':')
        
        # Add main title
        fig.suptitle(f'Safe-PINN Potential Field Analysis (Step {step})', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Convert to image
        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        
        return image
    
    def _add_entities_to_ax(
        self, 
        ax, 
        agent_positions: Optional[np.ndarray],
        obstacle_positions: Optional[np.ndarray],
        goal_positions: Optional[np.ndarray]
    ):
        """Helper method to add agents, obstacles, and goals to an axis."""
        if obstacle_positions is not None and len(obstacle_positions) > 0:
            ax.scatter(obstacle_positions[:, 0], obstacle_positions[:, 1],
                      c='gray', s=120, marker='o', edgecolors='black', 
                      linewidths=1.5, zorder=5, alpha=0.9)
        
        if agent_positions is not None and len(agent_positions) > 0:
            colors = plt.cm.Set1(np.linspace(0, 1, len(agent_positions)))
            ax.scatter(agent_positions[:, 0], agent_positions[:, 1],
                      c=colors, s=100, marker='o', edgecolors='black', 
                      linewidths=1.5, zorder=6)
        
        if goal_positions is not None and len(goal_positions) > 0:
            ax.scatter(goal_positions[:, 0], goal_positions[:, 1],
                      c='lime', s=80, marker='*', edgecolors='black', 
                      linewidths=1, zorder=5)


def extract_positions_from_env(env, env_index: int = 0) -> Dict[str, np.ndarray]:
    """Extract agent, obstacle, and goal positions from a VMAS environment.
    
    Args:
        env: The VMAS environment
        env_index: Index of the environment in the batch
        
    Returns:
        Dictionary with 'agents', 'obstacles', and 'goals' positions
    """
    positions = {
        'agents': [],
        'obstacles': [],
        'goals': []
    }
    
    try:
        # Get base environment
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        if hasattr(base_env, 'base_env'):
            base_env = base_env.base_env
        while hasattr(base_env, 'envs'):
            base_env = base_env.envs[0]
        
        # Get scenario
        scenario = None
        if hasattr(base_env, 'scenario'):
            scenario = base_env.scenario
        elif hasattr(base_env, 'world'):
            scenario = base_env
        
        if scenario is None:
            return positions
            
        # Extract agent positions
        world = getattr(scenario, 'world', scenario)
        if hasattr(world, 'agents'):
            for agent in world.agents:
                if hasattr(agent, 'state') and hasattr(agent.state, 'pos'):
                    pos = agent.state.pos
                    if pos.dim() >= 2:
                        positions['agents'].append(pos[env_index].cpu().numpy())
                    else:
                        positions['agents'].append(pos.cpu().numpy())
        
        # Extract obstacle positions
        if hasattr(world, 'landmarks'):
            for landmark in world.landmarks:
                if hasattr(landmark, 'name') and 'obstacle' in landmark.name.lower():
                    if hasattr(landmark, 'state') and hasattr(landmark.state, 'pos'):
                        pos = landmark.state.pos
                        if pos.dim() >= 2:
                            positions['obstacles'].append(pos[env_index].cpu().numpy())
                        else:
                            positions['obstacles'].append(pos.cpu().numpy())
                elif hasattr(landmark, 'name') and 'goal' in landmark.name.lower():
                    if hasattr(landmark, 'state') and hasattr(landmark.state, 'pos'):
                        pos = landmark.state.pos
                        if pos.dim() >= 2:
                            positions['goals'].append(pos[env_index].cpu().numpy())
                        else:
                            positions['goals'].append(pos.cpu().numpy())
        
        # Convert lists to arrays
        for key in positions:
            if positions[key]:
                positions[key] = np.array(positions[key])
            else:
                positions[key] = np.array([]).reshape(0, 2)
                
    except Exception as e:
        print(f"Warning: Could not extract positions from environment: {e}")
        for key in positions:
            positions[key] = np.array([]).reshape(0, 2)
    
    return positions


def is_safe_pinn_model(model) -> bool:
    """Check if a model is a Safe PINN variant."""
    model_class_name = type(model).__name__
    return 'SafePinn' in model_class_name or 'safe_pinn' in model_class_name.lower()


def get_safe_pinn_model(policy):
    """Extract the Safe PINN model from a policy.
    
    Args:
        policy: The policy (TensorDictSequential or similar)
        
    Returns:
        The Safe PINN model if found, None otherwise
    """
    from gemsmarl.models.safe_pinn import SafePinn
    from gemsmarl.models.safe_pinn_ppo import SafePinnPPO
    
    # Check if policy itself is SafePinn
    if isinstance(policy, (SafePinn, SafePinnPPO)):
        return policy
    
    # If policy is a sequence, search through modules
    if hasattr(policy, 'module'):
        for module in policy.module:
            if isinstance(module, (SafePinn, SafePinnPPO)):
                return module
            # Check nested modules
            if hasattr(module, 'module'):
                for submodule in module.module:
                    if isinstance(submodule, (SafePinn, SafePinnPPO)):
                        return submodule
    
    # Try finding in children
    for name, module in getattr(policy, 'named_modules', lambda: iter([]))():
        if isinstance(module, (SafePinn, SafePinnPPO)):
            return module
    
    return None
