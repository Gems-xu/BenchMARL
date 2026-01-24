#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.

"""Callback for tracking and logging safety metrics during training and evaluation."""

from __future__ import annotations

from typing import List, Dict, Any, Optional
import warnings

import torch
from tensordict import TensorDictBase

from gemsmarl.experiment.callback import Callback


class SafetyMetricsCallback(Callback):
    """
    Callback for extracting and tracking safety metrics from multi-agent environments.
    
    Collects comprehensive safety metrics including:
    
    **Collision Metrics:**
    - agent_collision_rew: Collision penalty rewards for each agent (mean, min, max, std)
    - collision_rate: Percentage of timesteps where collisions occur
    - min_collision_distance: Minimum distance to obstacles/other agents (mean, min, max, std)
    
    **Constraint Metrics:**
    - constraint_value: Constraint satisfaction values (mean, min, max)
    - constraint_satisfaction_rate: Percentage of timesteps satisfying constraints
    
    **Goal Metrics:**
    - distance_to_goal: Distance to target goal (mean, min, max)
    - goal_reached_rate: Percentage of timesteps where agent is at goal
    
    These metrics are designed to work with environments like:
    - navigation_obs
    - navigation_obs_unicycle
    - navigation (if available)
    
    All collected metrics are organized under the "Safety/" prefix in wandb for better visualization.
    
    Example:
        >>> callback = SafetyMetricsCallback(evaluate_only=True)
        >>> # Use with Experiment: experiment.callbacks.append(callback)
        >>> # Metrics logged to wandb: Safe/safety/collision_rate, Safe/safety/constraint_satisfaction_rate, etc.
    """

    def __init__(self, evaluate_only: bool = False):
        """
        Initialize the SafetyMetricsCallback.
        
        Args:
            evaluate_only: If True, only collect metrics during evaluation.
                          If False, collect during both training and evaluation.
        """
        super().__init__()
        self.evaluate_only = evaluate_only
        self.last_evaluation_metrics: Dict[str, Any] = {}
        self.current_episode_metrics: Dict[str, List] = {}

    def on_batch_collected(self, batch: TensorDictBase):
        """
        Called at the end of every collection step.
        Extract safety metrics from the collected batch.
        
        Args:
            batch: TensorDict containing the collected data
        """
        if self.evaluate_only:
            return
            
        self._extract_metrics_from_batch(batch)

    def on_evaluation_end(self, rollouts: List[TensorDictBase]):
        """
        Called at the end of every evaluation.
        Extract safety metrics from evaluation rollouts.
        
        Args:
            rollouts: List of rollout TensorDicts from evaluation episodes
        """
        metrics = self._extract_metrics_from_rollouts(rollouts)
        self.last_evaluation_metrics = metrics
        if metrics:
            print(f"[SafetyMetrics] Collected {len(metrics)} safety metrics")

    def _extract_metrics_from_batch(self, batch: TensorDictBase) -> Dict[str, Any]:
        """
        Extract safety metrics from a collected batch.
        
        Args:
            batch: TensorDict from data collection
            
        Returns:
            Dictionary containing extracted metrics
        """
        metrics = {}
        
        try:
            # Try to extract agent collision rewards
            for group_key in batch.keys():
                if isinstance(group_key, str) and group_key not in ["info", "next"]:
                    # This might be a group name
                    if (group_key, "info") in batch.keys():
                        info_data = batch[(group_key, "info")]
                        
                        # Extract collision rewards if available
                        if "agent_collision_rew" in info_data.keys():
                            collision_rew = info_data["agent_collision_rew"]
                            if collision_rew is not None:
                                metrics[f"{group_key}/agent_collision_rew_mean"] = float(
                                    collision_rew.mean().item() if collision_rew.numel() > 0 else 0.0
                                )
                        
                        # Extract constraint values if available
                        if "constraints" in info_data.keys():
                            constraints = info_data["constraints"]
                            if constraints is not None:
                                metrics[f"{group_key}/constraint_value_mean"] = float(
                                    constraints.mean().item() if constraints.numel() > 0 else 0.0
                                )
        except Exception as e:
            warnings.warn(f"Error extracting metrics from batch: {e}")
        
        return metrics

    def _extract_metrics_from_rollouts(self, rollouts: List[TensorDictBase]) -> Dict[str, Any]:
        """
        Extract comprehensive safety metrics from evaluation rollouts.
        
        Args:
            rollouts: List of rollout TensorDicts
            
        Returns:
            Dictionary containing extracted metrics with stats (min, max, mean, std)
        """
        metrics = {}
        
        if not rollouts:
            return metrics
        
        try:
            # Collect metrics across all rollouts
            all_collision_rews = []
            all_constraint_values = []
            all_min_distances = []
            all_distances_to_goal = []
            all_on_goal = []
            collision_count = 0
            total_steps = 0
            
            for idx, rollout in enumerate(rollouts):
                try:
                    total_steps += rollout.batch_size[0]  # Count timesteps
                    
                    # Try to find agent info in the rollout
                    for key in rollout.keys():
                        if isinstance(key, str) and key not in ["info", "next", "done"]:
                            # Check next group info using get() to avoid nested key issues
                            try:
                                next_data = rollout.get("next", None)
                                if next_data is None:
                                    continue
                                
                                group_data = next_data.get(key, None)
                                if group_data is None:
                                    continue
                                
                                info_data = group_data.get("info", None)
                                if info_data is None:
                                    continue
                                
                                # Collision reward
                                if "agent_collision_rew" in info_data.keys():
                                    rew = info_data["agent_collision_rew"]
                                    if rew is not None and rew.numel() > 0:
                                        all_collision_rews.append(rew.flatten())
                                        # Count negative rewards (collisions)
                                        collision_count += (rew < -1e-6).sum().item()
                                
                                # Constraint values
                                if "constraints" in info_data.keys():
                                    constr = info_data["constraints"]
                                    if constr is not None and constr.numel() > 0:
                                        all_constraint_values.append(constr.flatten())
                                
                                # Minimum collision distance
                                if "min_collision_distance" in info_data.keys():
                                    dist = info_data["min_collision_distance"]
                                    if dist is not None and dist.numel() > 0:
                                        all_min_distances.append(dist.flatten())
                                
                                # Distance to goal
                                if "distance_to_goal" in info_data.keys():
                                    dist_goal = info_data["distance_to_goal"]
                                    if dist_goal is not None and dist_goal.numel() > 0:
                                        all_distances_to_goal.append(dist_goal.flatten())
                                
                                # On goal
                                if "on_goal" in info_data.keys():
                                    on_goal = info_data["on_goal"]
                                    if on_goal is not None and on_goal.numel() > 0:
                                        all_on_goal.append(on_goal.flatten())
                            
                            except Exception as nested_error:
                                # Skip this group if there's an error accessing nested data
                                continue
                except Exception as e:
                    warnings.warn(f"Error processing rollout: {e}")
                    continue
            
            # Aggregate collision reward metrics
            if all_collision_rews:
                collision_rews_tensor = torch.cat(all_collision_rews)
                metrics["safety/agent_collision_rew_mean"] = float(collision_rews_tensor.mean().item())
                metrics["safety/agent_collision_rew_min"] = float(collision_rews_tensor.min().item())
                metrics["safety/agent_collision_rew_max"] = float(collision_rews_tensor.max().item())
                metrics["safety/agent_collision_rew_std"] = float(collision_rews_tensor.std().item())
            
            # Aggregate constraint metrics
            if all_constraint_values:
                constraint_tensor = torch.cat(all_constraint_values)
                metrics["safety/constraint_value_mean"] = float(constraint_tensor.mean().item())
                metrics["safety/constraint_value_min"] = float(constraint_tensor.min().item())
                metrics["safety/constraint_value_max"] = float(constraint_tensor.max().item())
                metrics["safety/constraint_satisfaction_rate"] = float(
                    (constraint_tensor >= 0).float().mean().item()
                )
            
            # Aggregate minimum distance metrics
            if all_min_distances:
                min_dist_tensor = torch.cat(all_min_distances)
                # Filter out invalid distances (999)
                valid_distances = min_dist_tensor[min_dist_tensor < 100]
                if valid_distances.numel() > 0:
                    metrics["safety/min_collision_distance_mean"] = float(valid_distances.mean().item())
                    metrics["safety/min_collision_distance_min"] = float(valid_distances.min().item())
                    metrics["safety/min_collision_distance_max"] = float(valid_distances.max().item())
                    metrics["safety/min_collision_distance_std"] = float(valid_distances.std().item())
            
            # Aggregate distance to goal metrics
            if all_distances_to_goal:
                dist_goal_tensor = torch.cat(all_distances_to_goal)
                metrics["safety/distance_to_goal_mean"] = float(dist_goal_tensor.mean().item())
                metrics["safety/distance_to_goal_min"] = float(dist_goal_tensor.min().item())
                metrics["safety/distance_to_goal_max"] = float(dist_goal_tensor.max().item())
            
            # Aggregate on-goal metrics
            if all_on_goal:
                on_goal_tensor = torch.cat(all_on_goal)
                # Convert boolean to float for mean calculation
                if on_goal_tensor.dtype == torch.bool:
                    on_goal_tensor = on_goal_tensor.float()
                metrics["safety/goal_reached_rate"] = float(on_goal_tensor.mean().item())
            
            # Collision rate
            if total_steps > 0:
                metrics["safety/collision_rate"] = float(collision_count / total_steps)
            
        except Exception as e:
            warnings.warn(f"Error extracting metrics from rollouts: {e}")
        
        return metrics

    def get_last_evaluation_metrics(self) -> Dict[str, Any]:
        """
        Get the metrics from the last evaluation.
        
        Returns:
            Dictionary containing the last evaluation metrics
        """
        return self.last_evaluation_metrics.copy()

    def reset_metrics(self):
        """Reset tracked metrics."""
        self.last_evaluation_metrics = {}
        self.current_episode_metrics = {}
