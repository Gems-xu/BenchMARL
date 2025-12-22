#!/usr/bin/env python3
"""
Test script to verify all Safe-PINN visualizations render correctly.

This script:
1. Creates a PotentialFieldVisualizer
2. Creates mock data (agent positions, obstacles, goals)
3. Generates all visualization types
4. Verifies each visualization produces valid RGB images
5. (Optional) Logs to wandb if available

Run with: uv run python test_visualizations.py
"""

import numpy as np
import torch
import sys
from pathlib import Path

# Ensure we can import from the project
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from gemsmarl.experiment.potential_visualizer import (
    PotentialFieldVisualizer,
    HAS_MATPLOTLIB,
)


def create_mock_data(device="cpu"):
    """Create mock agent, obstacle, and goal positions for testing."""
    # Create 3 agents in different positions
    agent_positions = np.array([
        [-0.5, 0.0],
        [0.5, 0.3],
        [0.0, -0.5],
    ], dtype=np.float32)
    
    # Create 2 obstacles
    obstacle_positions = np.array([
        [0.0, 0.5],
        [-0.3, -0.3],
    ], dtype=np.float32)
    
    # Create 3 goals (one per agent)
    goal_positions = np.array([
        [0.8, 0.8],
        [-0.8, 0.8],
        [0.0, 0.8],
    ], dtype=np.float32)
    
    # Create tensors
    agent_pos_t = torch.tensor(agent_positions, dtype=torch.float32, device=device)
    obs_pos_t = torch.tensor(obstacle_positions, dtype=torch.float32, device=device)
    goal_pos_t = torch.tensor(goal_positions, dtype=torch.float32, device=device)
    
    return {
        'agents': agent_positions,
        'obstacles': obstacle_positions,
        'goals': goal_positions,
        'agents_t': agent_pos_t,
        'obstacles_t': obs_pos_t,
        'goals_t': goal_pos_t,
    }


class MockSafePinnModel:
    """Mock Safe PINN model for testing visualizations without actual model.
    
    We need to patch isinstance to work with this mock.
    """
    
    def __init__(self, n_agents=3, obs_dim=6, device="cpu"):
        self.n_agents = n_agents
        self.observation_dim_per_agent = obs_dim
        self.r_collision = 0.15
        self.barrier_epsilon = 0.05
        self.r_communication = 0.5
        self.use_log_barrier = True
        self.device = device
        self.in_keys = ["observation"]


# We need to create a proper mock that passes isinstance checks
# Let's patch the validation in the compute methods
def create_patched_visualizer(device="cpu"):
    """Create a visualizer with patched methods that skip model validation."""
    visualizer = PotentialFieldVisualizer(
        world_bounds=(-1.5, 1.5, -1.5, 1.5),
        grid_resolution=50,
        device=device
    )
    
    # Store original methods
    original_compute_potential = visualizer.compute_potential_field
    original_compute_task = visualizer.compute_task_potential_field
    original_compute_total = visualizer.compute_total_potential_field
    
    def patched_compute_potential_field(model, agent_positions, obstacle_positions=None, 
                                         agent_velocities=None, goal_positions=None):
        """Patched version that computes barrier potential without model validation."""
        n_agents = agent_positions.shape[0]
        
        if agent_velocities is None:
            agent_velocities = torch.zeros_like(agent_positions)
        if goal_positions is None:
            goal_positions = torch.zeros_like(agent_positions)
        
        if obstacle_positions is not None and obstacle_positions.shape[0] > 0:
            all_obstacle_positions = torch.cat([agent_positions, obstacle_positions], dim=0)
        else:
            all_obstacle_positions = agent_positions
            
        grid_tensor = torch.tensor(visualizer.grid_points, dtype=torch.float32, device=visualizer.device)
        n_grid_points = grid_tensor.shape[0]
        
        potential_values = torch.zeros(n_grid_points, device=visualizer.device)
        
        r_collision = getattr(model, 'r_collision', 0.2)
        barrier_epsilon = getattr(model, 'barrier_epsilon', 0.05)
        use_log_barrier = getattr(model, 'use_log_barrier', True)
        
        with torch.no_grad():
            for i, grid_pos in enumerate(grid_tensor):
                diff = grid_pos.unsqueeze(0) - all_obstacle_positions
                dist = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-6)
                
                gap = dist - r_collision
                
                if use_log_barrier:
                    safe_gap = torch.clamp(gap, min=barrier_epsilon)
                    H_barrier = -torch.log(safe_gap / (r_collision + barrier_epsilon))
                    H_barrier = torch.clamp(H_barrier, min=0.0, max=100.0)
                else:
                    denom = gap**2 + barrier_epsilon
                    H_barrier = 1.0 / denom
                
                potential_values[i] = H_barrier.sum()
        
        return potential_values.cpu().numpy().reshape(visualizer.grid_resolution, visualizer.grid_resolution)
    
    def patched_compute_task_field(model, agent_positions, goal_positions):
        """Patched version that computes task potential without model validation."""
        grid_tensor = torch.tensor(visualizer.grid_points, dtype=torch.float32, device=visualizer.device)
        n_grid_points = grid_tensor.shape[0]
        
        potential_values = torch.zeros(n_grid_points, device=visualizer.device)
        
        with torch.no_grad():
            for i, grid_pos in enumerate(grid_tensor):
                diff = grid_pos.unsqueeze(0) - goal_positions
                dist = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-6)
                H_task = 0.5 * torch.sum(dist**2)
                potential_values[i] = H_task
        
        return potential_values.cpu().numpy().reshape(visualizer.grid_resolution, visualizer.grid_resolution)
    
    def patched_compute_total_field(model, agent_positions, obstacle_positions=None, goal_positions=None):
        """Patched version that computes total potential without model validation."""
        barrier_field = patched_compute_potential_field(model, agent_positions, obstacle_positions)
        
        if goal_positions is not None and goal_positions.shape[0] > 0:
            task_field = patched_compute_task_field(model, agent_positions, goal_positions)
            total_field = barrier_field + 0.1 * task_field
        else:
            total_field = barrier_field
        
        return total_field
    
    # Patch the methods
    visualizer.compute_potential_field = patched_compute_potential_field
    visualizer.compute_task_potential_field = patched_compute_task_field
    visualizer.compute_total_potential_field = patched_compute_total_field
    
    return visualizer


def validate_image(image: np.ndarray, name: str) -> bool:
    """Validate that the image is a proper RGB numpy array."""
    if image is None:
        print(f"  ❌ {name}: Image is None")
        return False
    
    if not isinstance(image, np.ndarray):
        print(f"  ❌ {name}: Not a numpy array (got {type(image)})")
        return False
    
    if len(image.shape) != 3:
        print(f"  ❌ {name}: Wrong shape dimensions (expected 3, got {len(image.shape)})")
        return False
    
    if image.shape[2] != 3:
        print(f"  ❌ {name}: Wrong color channels (expected 3, got {image.shape[2]})")
        return False
    
    if image.shape[0] < 100 or image.shape[1] < 100:
        print(f"  ❌ {name}: Image too small ({image.shape[0]}x{image.shape[1]})")
        return False
    
    if image.dtype != np.uint8:
        print(f"  ⚠️  {name}: Non-standard dtype ({image.dtype}), converting...")
        # This is just a warning, not a failure
    
    print(f"  ✅ {name}: Valid ({image.shape[0]}x{image.shape[1]}x{image.shape[2]}, dtype={image.dtype})")
    return True


def test_barrier_potential_field(visualizer, data, model):
    """Test barrier potential field computation and rendering."""
    print("\n📊 Testing Barrier Potential Field...")
    
    # Compute
    barrier_field = visualizer.compute_potential_field(
        model, data['agents_t'], data['obstacles_t']
    )
    
    if not isinstance(barrier_field, np.ndarray):
        print(f"  ❌ Barrier field is not numpy array")
        return False
    
    print(f"  ✅ Barrier field computed: shape={barrier_field.shape}, "
          f"min={barrier_field.min():.4f}, max={barrier_field.max():.4f}")
    
    # Render
    image = visualizer.render_potential_field(
        barrier_field,
        agent_positions=data['agents'],
        obstacle_positions=data['obstacles'],
        goal_positions=data['goals'],
        title="Barrier Potential Field"
    )
    
    return validate_image(image, "Barrier Potential Render") 


def test_task_potential_field(visualizer, data, model):
    """Test task potential field computation and rendering."""
    print("\n📊 Testing Task Potential Field...")
    
    # Compute
    task_field = visualizer.compute_task_potential_field(
        model, data['agents_t'], data['goals_t']
    )
    
    if not isinstance(task_field, np.ndarray):
        print(f"  ❌ Task field is not numpy array")
        return False
    
    print(f"  ✅ Task field computed: shape={task_field.shape}, "
          f"min={task_field.min():.4f}, max={task_field.max():.4f}")
    
    # Render
    image = visualizer.render_potential_field(
        task_field,
        agent_positions=data['agents'],
        obstacle_positions=data['obstacles'],
        goal_positions=data['goals'],
        title="Task Potential Field",
        cmap="Blues"
    )
    
    return validate_image(image, "Task Potential Render")


def test_total_potential_field(visualizer, data, model):
    """Test total potential field computation."""
    print("\n📊 Testing Total Potential Field...")
    
    # Compute
    total_field = visualizer.compute_total_potential_field(
        model, data['agents_t'], data['obstacles_t'], data['goals_t']
    )
    
    if not isinstance(total_field, np.ndarray):
        print(f"  ❌ Total field is not numpy array")
        return False
    
    print(f"  ✅ Total field computed: shape={total_field.shape}, "
          f"min={total_field.min():.4f}, max={total_field.max():.4f}")
    
    # Render
    image = visualizer.render_potential_field(
        total_field,
        agent_positions=data['agents'],
        obstacle_positions=data['obstacles'],
        goal_positions=data['goals'],
        title="Total Potential Field",
        cmap="viridis"
    )
    
    return validate_image(image, "Total Potential Render")


def test_3d_surface_plot(visualizer, data, model):
    """Test 3D surface plot rendering."""
    print("\n📊 Testing 3D Surface Plot...")
    
    # First compute the barrier field
    barrier_field = visualizer.compute_potential_field(
        model, data['agents_t'], data['obstacles_t']
    )
    
    # Render 3D surface
    image = visualizer.render_3d_surface_plot(
        potential_field=barrier_field,
        agent_positions=data['agents'],
        obstacle_positions=data['obstacles'],
        title="3D Barrier Potential Surface",
        elev=30,
        azim=45,
        cmap="viridis"
    )
    
    return validate_image(image, "3D Surface Plot")


def test_energy_flow_diagram(visualizer, data, model):
    """Test energy flow diagram rendering."""
    print("\n📊 Testing Energy Flow Diagram...")
    
    # First compute the barrier field
    barrier_field = visualizer.compute_potential_field(
        model, data['agents_t'], data['obstacles_t']
    )
    
    # Render energy flow
    image = visualizer.render_energy_flow_diagram(
        potential_field=barrier_field,
        agent_positions=data['agents'],
        obstacle_positions=data['obstacles'],
        goal_positions=data['goals'],
        arrow_density=12
    )
    
    return validate_image(image, "Energy Flow Diagram")


def test_safety_margin_contours(visualizer, data, model):
    """Test safety margin contours rendering."""
    print("\n📊 Testing Safety Margin Contours...")
    
    # First compute the barrier field
    barrier_field = visualizer.compute_potential_field(
        model, data['agents_t'], data['obstacles_t']
    )
    
    # Render contours
    image = visualizer.render_safety_margin_contours(
        potential_field=barrier_field,
        agent_positions=data['agents'],
        obstacle_positions=data['obstacles']
    )
    
    return validate_image(image, "Safety Contours")


def test_energy_decomposition(visualizer, data, model):
    """Test energy decomposition rendering."""
    print("\n📊 Testing Energy Decomposition...")
    
    # Create mock sample state (TensorDict-like object)
    from tensordict import TensorDict
    
    # Create state with positions and velocities
    obs_dim = model.observation_dim_per_agent
    n_agents = model.n_agents
    
    # Create observation tensor: [pos_x, pos_y, vel_x, vel_y, goal_x, goal_y]
    obs = torch.zeros(1, n_agents, obs_dim, device=model.device)
    obs[0, :, :2] = data['agents_t']  # positions
    obs[0, :, 2:4] = torch.randn(n_agents, 2, device=model.device) * 0.1  # velocities
    
    sample_state = TensorDict({
        "observation": obs
    }, batch_size=[1, n_agents])
    
    # Render energy decomposition
    image = visualizer.render_energy_decomposition(
        model=model,
        sample_state=sample_state
    )
    
    return validate_image(image, "Energy Decomposition")


def test_publication_figure(visualizer, data, model):
    """Test publication figure rendering."""
    print("\n📊 Testing Publication Figure (6-panel)...")
    
    # Create mock sample state
    from tensordict import TensorDict
    
    obs_dim = model.observation_dim_per_agent
    n_agents = model.n_agents
    
    obs = torch.zeros(1, n_agents, obs_dim, device=model.device)
    obs[0, :, :2] = data['agents_t']
    obs[0, :, 2:4] = torch.randn(n_agents, 2, device=model.device) * 0.1
    
    sample_state = TensorDict({
        "observation": obs
    }, batch_size=[1, n_agents])
    
    # Render publication figure
    image = visualizer.render_publication_figure(
        model=model,
        sample_state=sample_state,
        agent_positions=data['agents'],
        obstacle_positions=data['obstacles'],
        goal_positions=data['goals'],
        step=100,
        figsize=(20, 12),
        dpi=100  # Lower for faster testing
    )
    
    return validate_image(image, "Publication Figure")


def test_combined_visualization(visualizer, data, model):
    """Test combined visualization rendering."""
    print("\n📊 Testing Combined Visualization...")
    
    # Create mock environment frame
    env_frame = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
    
    # Compute barrier field
    barrier_field = visualizer.compute_potential_field(
        model, data['agents_t'], data['obstacles_t']
    )
    
    # Render combined
    image = visualizer.render_combined_visualization(
        env_frame=env_frame,
        potential_field=barrier_field,
        agent_positions=data['agents'],
        obstacle_positions=data['obstacles'],
        goal_positions=data['goals'],
        step=100
    )
    
    return validate_image(image, "Combined Visualization")


def main():
    """Run all visualization tests."""
    print("=" * 60)
    print("🧪 Safe-PINN Visualization Test Suite")
    print("=" * 60)
    
    if not HAS_MATPLOTLIB:
        print("❌ matplotlib not installed - cannot run tests")
        return False
    
    print("\n✅ matplotlib is available")
    
    # Check CUDA availability
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device}")
    
    # Initialize visualizer with patched methods for mock model compatibility
    print("\n📦 Initializing PotentialFieldVisualizer (patched for mock model)...")
    visualizer = create_patched_visualizer(device=device)
    print(f"  ✅ Grid: {visualizer.grid_resolution}x{visualizer.grid_resolution}")
    print(f"  ✅ Bounds: {visualizer.world_bounds}")
    
    # Create mock data
    print("\n📦 Creating mock data...")
    data = create_mock_data(device=device)
    print(f"  ✅ Agents: {data['agents'].shape}")
    print(f"  ✅ Obstacles: {data['obstacles'].shape}")
    print(f"  ✅ Goals: {data['goals'].shape}")
    
    # Create mock model
    print("\n📦 Creating mock Safe PINN model...")
    model = MockSafePinnModel(n_agents=3, obs_dim=6, device=device)
    print(f"  ✅ n_agents: {model.n_agents}")
    print(f"  ✅ r_collision: {model.r_collision}")
    print(f"  ✅ use_log_barrier: {model.use_log_barrier}")
    
    # Run tests
    results = {}
    
    print("\n" + "=" * 60)
    print("🔬 Running Visualization Tests")
    print("=" * 60)
    
    # Test each visualization type
    try:
        results['barrier_potential'] = test_barrier_potential_field(visualizer, data, model)
    except Exception as e:
        print(f"  ❌ Barrier Potential Test Failed: {e}")
        results['barrier_potential'] = False
    
    try:
        results['task_potential'] = test_task_potential_field(visualizer, data, model)
    except Exception as e:
        print(f"  ❌ Task Potential Test Failed: {e}")
        results['task_potential'] = False
    
    try:
        results['total_potential'] = test_total_potential_field(visualizer, data, model)
    except Exception as e:
        print(f"  ❌ Total Potential Test Failed: {e}")
        results['total_potential'] = False
    
    try:
        results['3d_surface'] = test_3d_surface_plot(visualizer, data, model)
    except Exception as e:
        print(f"  ❌ 3D Surface Test Failed: {e}")
        results['3d_surface'] = False
    
    try:
        results['energy_flow'] = test_energy_flow_diagram(visualizer, data, model)
    except Exception as e:
        print(f"  ❌ Energy Flow Test Failed: {e}")
        results['energy_flow'] = False
    
    try:
        results['safety_contours'] = test_safety_margin_contours(visualizer, data, model)
    except Exception as e:
        print(f"  ❌ Safety Contours Test Failed: {e}")
        results['safety_contours'] = False
    
    try:
        results['energy_decomposition'] = test_energy_decomposition(visualizer, data, model)
    except Exception as e:
        print(f"  ❌ Energy Decomposition Test Failed: {e}")
        results['energy_decomposition'] = False
    
    try:
        results['publication_figure'] = test_publication_figure(visualizer, data, model)
    except Exception as e:
        print(f"  ❌ Publication Figure Test Failed: {e}")
        results['publication_figure'] = False
    
    try:
        results['combined_visualization'] = test_combined_visualization(visualizer, data, model)
    except Exception as e:
        print(f"  ❌ Combined Visualization Test Failed: {e}")
        results['combined_visualization'] = False
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results Summary")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print("\n" + "-" * 60)
    print(f"  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All visualization tests PASSED!")
        print("   You can now use these visualizations in your experiments.")
        print("   They will be logged to wandb's Viz module.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
    
    print("\n" + "=" * 60)
    print("📝 wandb Logging Keys (Viz/ prefix):")
    print("=" * 60)
    print("  • Viz/barrier_potential_field")
    print("  • Viz/barrier_potential_heatmap")
    print("  • Viz/task_potential_field")
    print("  • Viz/total_potential_field")
    print("  • Viz/3d_barrier_surface")
    print("  • Viz/energy_flow_diagram")
    print("  • Viz/safety_margin_contours")
    print("  • Viz/energy_decomposition")
    print("  • Viz/publication_figure")
    print("  • Viz/potential_field_video")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
