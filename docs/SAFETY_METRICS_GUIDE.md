# Safe 指标详细说明文档

本文档详细说明在 `navigation_obs` 和 `navigation_obs_unicycle` 场景中，同步到 WandB 的 17 个安全指标的含义、计算方式和优化建议。

## 📊 指标概览

所有安全指标在 WandB 中以 **`Safe/safety/`** 为前缀，共分为 3 大类：

| 类别 | 指标数量 | 说明 |
|------|---------|------|
| 碰撞相关 | 9个 | 监控智能体与障碍物/其他智能体的距离和碰撞情况 |
| 约束满足 | 4个 | 监控智能体是否满足安全约束条件 |
| 目标到达 | 4个 | 监控智能体完成任务的进度 |

---

## 🚨 1. 碰撞相关指标 (9个)

### 1.1 最小碰撞距离 (min_collision_distance)

监控智能体与最近障碍物或其他智能体之间的距离，是评估安全性的核心指标。

#### **`Safe/safety/min_collision_distance_mean`**
- **含义**：所有时间步中，每个智能体到最近障碍物/其他智能体的平均距离
- **计算方式**：
  ```python
  for each agent:
      distances = [dist(agent, other_agent) for other_agent in agents if agent != other_agent]
      distances += [dist(agent, obstacle) for obstacle in obstacles]
      min_collision_distance = min(distances)
  mean_value = mean(all_min_collision_distances)
  ```
- **理想值**：**≥ 0.2** (agent_radius 的 2倍)
- **解读**：
  - `> 0.2`：智能体保持安全距离，碰撞风险低 ✅
  - `0.1 - 0.2`：距离较近，可能发生碰撞 ⚠️
  - `< 0.1`：频繁接近或穿透，碰撞风险高 ❌

#### **`Safe/safety/min_collision_distance_min`**
- **含义**：最危险时刻的最小距离（最接近碰撞的情况）
- **理想值**：**> 0**
- **解读**：
  - `> 0.1`：始终保持安全距离 ✅
  - `0 - 0.1`：接近碰撞边界 ⚠️
  - `< 0`：发生了物理穿透/碰撞 ❌（需要调整参数）

#### **`Safe/safety/min_collision_distance_max`**
- **含义**：最安全时刻的最小距离
- **理想值**：越大越好
- **解读**：反映智能体在最宽松情况下与障碍物的距离

#### **`Safe/safety/min_collision_distance_std`**
- **含义**：最小距离的标准差，衡量距离的波动性
- **理想值**：适中（0.1 - 0.2）
- **解读**：
  - 过大：智能体行为不稳定，时而安全时而危险
  - 过小：智能体保持稳定的安全距离
  - 接近0：可能卡在某个位置不动

---

### 1.2 碰撞惩罚奖励 (agent_collision_rew)

记录智能体因碰撞而获得的负奖励，用于评估碰撞的严重程度。

#### **`Safe/safety/agent_collision_rew_mean`**
- **含义**：平均碰撞惩罚（负奖励）
- **计算方式**：
  ```python
  if distance <= min_collision_distance_threshold (0.005):
      agent_collision_rew = -collision_penalty
  mean_value = mean(all_collision_rewards)
  ```
- **理想值**：**接近 0**
- **解读**：
  - `= 0`：无碰撞 ✅
  - `< -0.1`：频繁碰撞 ❌

#### **`Safe/safety/agent_collision_rew_min`**
- **含义**：最严重的单次碰撞惩罚（最负值）
- **理想值**：**接近 0**
- **解读**：反映最严重碰撞事件的惩罚强度

#### **`Safe/safety/agent_collision_rew_max`**
- **含义**：最轻微的碰撞惩罚
- **理想值**：**= 0**（无碰撞）

#### **`Safe/safety/agent_collision_rew_std`**
- **含义**：碰撞惩罚的标准差
- **理想值**：**接近 0**
- **解读**：反映碰撞强度的一致性

---

### 1.3 碰撞率 (collision_rate)

#### **`Safe/safety/collision_rate`**
- **含义**：发生碰撞的时间步占总时间步的百分比
- **计算方式**：
  ```python
  collision_rate = collision_count / total_timesteps
  ```
  其中 collision_count 统计 `agent_collision_rew < -1e-6` 的次数
- **理想值**：**0** (0% 碰撞率)
- **解读**：
  - `0`：完全无碰撞 ✅
  - `< 0.05`：偶尔碰撞，可接受 ⚠️
  - `> 0.1`：频繁碰撞，需要优化 ❌

---

## ✅ 2. 约束满足指标 (4个)

### 2.1 约束值 (constraint_value)

衡量智能体是否满足环境设定的安全约束（如速度限制、碰撞避免约束等）。

#### **`Safe/safety/constraint_value_mean`**
- **含义**：平均约束值
- **计算方式**：由环境计算，通常基于 CBF (Control Barrier Function)
- **理想值**：**> 0**
- **解读**：
  - `> 0`：满足约束，处于安全状态 ✅
  - `= 0`：约束边界，临界状态 ⚠️
  - `< 0`：违反约束，不安全 ❌

#### **`Safe/safety/constraint_value_min`**
- **含义**：最小约束值（最不安全的时刻）
- **理想值**：**≥ 0**
- **解读**：反映最危险时刻是否仍满足约束

#### **`Safe/safety/constraint_value_max`**
- **含义**：最大约束值（最安全的时刻）
- **理想值**：越大越好

---

### 2.2 约束满足率 (constraint_satisfaction_rate)

#### **`Safe/safety/constraint_satisfaction_rate`**
- **含义**：满足约束的时间步占总时间步的百分比
- **计算方式**：
  ```python
  constraint_satisfaction_rate = (constraint_value >= 0).mean()
  ```
- **理想值**：**1.0** (100% 满足)
- **解读**：
  - `= 1.0`：始终满足约束 ✅
  - `> 0.95`：大部分时间满足，偶尔违反 ⚠️
  - `< 0.9`：频繁违反约束 ❌

---

## 🎯 3. 目标到达指标 (4个)

### 3.1 到目标距离 (distance_to_goal)

监控智能体与目标位置的距离，评估任务完成进度。

#### **`Safe/safety/distance_to_goal_mean`**
- **含义**：所有时间步中，智能体到目标的平均距离
- **计算方式**：
  ```python
  distance_to_goal = ||agent.pos - goal.pos||
  mean_value = mean(all_distances)
  ```
- **理想值**：**接近 0**
- **解读**：
  - `< 0.05`：大部分时间在目标附近 ✅
  - `0.05 - 0.2`：正在接近目标 ⚠️
  - `> 0.2`：离目标较远，任务未完成 ❌

#### **`Safe/safety/distance_to_goal_min`**
- **含义**：最接近目标时的距离
- **理想值**：**< agent.radius** (成功到达)
- **解读**：
  - `< 0.1`：成功到达目标 ✅
  - `> 0.1`：未能到达目标 ❌

#### **`Safe/safety/distance_to_goal_max`**
- **含义**：距离目标最远时的距离
- **解读**：反映初始状态或迷路情况

---

### 3.2 目标到达率 (goal_reached_rate)

#### **`Safe/safety/goal_reached_rate`**
- **含义**：智能体处于目标位置的时间步占总时间步的百分比
- **计算方式**：
  ```python
  on_goal = (distance_to_goal < agent.radius)
  goal_reached_rate = on_goal.mean()
  ```
- **理想值**：**越高越好**，理想情况 > 0.5
- **解读**：
  - `> 0.8`：大部分时间停留在目标 ✅（可能过于保守）
  - `0.3 - 0.8`：平衡探索和到达 ⚠️
  - `< 0.2`：很少到达目标 ❌

---

## 📈 如何使用这些指标

### 训练阶段监控

1. **碰撞安全**：
   - 主要关注 `collision_rate` 和 `min_collision_distance_mean`
   - 目标：collision_rate < 0.05, min_distance_mean > 0.2

2. **任务完成**：
   - 主要关注 `goal_reached_rate` 和 `distance_to_goal_mean`
   - 目标：goal_reached_rate > 0.3, distance_mean < 0.1

3. **约束满足**：
   - 主要关注 `constraint_satisfaction_rate`
   - 目标：rate > 0.95

### 典型问题诊断

#### 问题 1：碰撞率高 (collision_rate > 0.1)
**症状**：
- `min_collision_distance_min` < 0
- `agent_collision_rew_mean` < -0.1

**解决方案**：
```python
SafePinnConfig(
    barrier_weight=0.2,              # 提高 (0.15 → 0.2)
    obstacle_barrier_weight=0.4,     # 提高 (0.3 → 0.4)
    r_collision=0.2,                 # 提前触发 (0.18 → 0.2)
    barrier_epsilon=0.1,             # 增大平滑 (0.08 → 0.1)
)
```

#### 问题 2：目标到达率低 (goal_reached_rate < 0.2)
**症状**：
- `distance_to_goal_mean` > 0.2
- `distance_to_goal_min` > 0.1

**解决方案**：
```python
SafePinnConfig(
    goal_attraction_strength=5.0,    # 提高 (3.0 → 5.0)
    barrier_weight=0.1,              # 降低 (0.15 → 0.1)
    task_weight=1.5,                 # 提高 (1.0 → 1.5)
)
```

#### 问题 3：行为不稳定 (min_collision_distance_std > 0.3)
**症状**：
- 指标波动剧烈
- 时而碰撞时而安全

**解决方案**：
```python
SafePinnConfig(
    f_max=2.0,                       # 降低 (3.0 → 2.0)
    barrier_epsilon=0.1,             # 增大 (0.08 → 0.1)
    use_log_barrier=True,            # 使用平滑barrier
)
```

---

## 🔧 参数调优指南

### 平衡安全与任务完成

| 优先级 | 调整方向 | 参数设置 |
|--------|---------|---------|
| **高安全性**（避免碰撞优先） | barrier ↑ goal ↓ | `barrier_weight=0.2, goal_attraction=2.0` |
| **平衡型**（推荐） | 均衡 | `barrier_weight=0.15, goal_attraction=3.0` |
| **高完成率**（到达目标优先） | barrier ↓ goal ↑ | `barrier_weight=0.1, goal_attraction=5.0` |

### 不同场景的推荐配置

#### navigation_obs (有障碍物)
```python
SafePinnConfig(
    barrier_weight=0.15,
    obstacle_barrier_weight=0.3,
    goal_attraction_strength=3.0,
    r_collision=0.18,
    barrier_epsilon=0.08,
)
```

#### navigation_obs_unicycle (单周期动力学)
```python
SafePinnPPOConfig(
    barrier_weight=0.12,
    obstacle_barrier_weight=0.45,
    goal_attraction_strength=3.0,
    r_collision=0.17,
    barrier_epsilon=0.06,
)
```

---

## 📊 WandB 可视化建议

### 创建自定义面板

1. **安全监控面板**：
   - Line Chart: `collision_rate`, `constraint_satisfaction_rate`
   - Line Chart: `min_collision_distance_mean` (带 ±std 区间)

2. **任务完成面板**：
   - Line Chart: `goal_reached_rate`
   - Line Chart: `distance_to_goal_mean`, `distance_to_goal_min`

3. **对比面板**：
   - Scatter Plot: `collision_rate` vs `goal_reached_rate`
   - 目标：左上角区域（低碰撞率 + 高到达率）

### 设置告警

在 WandB 中设置告警条件：
- `collision_rate > 0.1` → 发送通知
- `constraint_satisfaction_rate < 0.9` → 发送通知
- `goal_reached_rate < 0.2` → 发送通知

---

## 🧪 验证指标正确性

运行测试脚本验证指标收集：

```bash
cd /home/xwz/projects/BenchMARL
uv run python test_safety_callback_unit.py
```

**预期输出**：
```
✓ Successfully extracted 17 metrics:
  [1-4]  safety/agent_collision_rew_mean/min/max/std
  [5]    safety/collision_rate
  [6-8]  safety/constraint_value_mean/min/max
  [9]    safety/constraint_satisfaction_rate
  [10-13] safety/min_collision_distance_mean/min/max/std
  [14-16] safety/distance_to_goal_mean/min/max
  [17]   safety/goal_reached_rate

✓ All expected metrics present!
Test PASSED!
```

---

## 📚 相关文档

- [Safe-PINN 算法说明](SAFE_PINN.md)
- [MASAC Safe-PINN 优化](MASAC_SAFEPINN_FIX.md)
- [安全指标修复说明](../SAFETY_METRICS_FIX.md)
- [Barrier Hamiltonian 设计](BARRIER_PHS_WITH_MAPPO_DESIGN.md)

---

## 📝 总结

这 17 个指标全面监控了多智能体导航任务中的：
- ✅ **安全性**：碰撞避免、约束满足
- ✅ **有效性**：任务完成、目标到达
- ✅ **稳定性**：行为一致性、梯度平滑

通过持续监控这些指标并根据建议调优参数，可以训练出既安全又高效的多智能体策略。

**核心原则**：安全优先，但不过度保守，在安全约束下最大化任务完成率。
