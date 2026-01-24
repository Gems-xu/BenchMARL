# 安全指标修复说明

## 问题描述
在Wandb中看不到Safe模块的安全指标，并出现以下警告：
```
UserWarning: Error processing rollout: Nested membership checks with tuples of strings is only supported when setting `include_nested=True`.
```

## 根本原因
TensorDict在访问嵌套键时（如`rollout[("next", key, "info")]`）需要特殊处理，直接使用元组作为键会导致错误。

## 修复方案

### 1. 修改键访问方式
**文件:** `gemsmarl/experiment/safety_metrics_callback.py`

**改动:** 将直接访问嵌套键改为使用`.get()`方法逐层访问

**修改前:**
```python
if ("next", key, "info") in rollout.keys():
    info_data = rollout[("next", key, "info")]
```

**修改后:**
```python
next_data = rollout.get("next", None)
if next_data is None:
    continue

group_data = next_data.get(key, None)
if group_data is None:
    continue

info_data = group_data.get("info", None)
if info_data is None:
    continue
```

### 2. 修复布尔类型mean()错误
**问题:** `on_goal`字段是布尔类型，直接调用`.mean()`会报错

**修复:**
```python
if on_goal_tensor.dtype == torch.bool:
    on_goal_tensor = on_goal_tensor.float()
metrics["safety/goal_reached_rate"] = float(on_goal_tensor.mean().item())
```

### 3. 使用索引而非对象比较
**问题:** TensorDict对象不能直接比较（`if rollout == rollouts[0]`）

**修复:**
```python
for idx, rollout in enumerate(rollouts):
    if idx == 0:  # 使用索引而非对象比较
        # debug code
```

## 验证

运行单元测试验证修复：
```bash
cd /home/xwz/projects/BenchMARL
uv run python test_safety_callback_unit.py
```

**预期输出:**
```
✓ Successfully extracted 17 metrics:
  - safety/agent_collision_rew_mean
  - safety/agent_collision_rew_min/max/std
  - safety/collision_rate
  - safety/constraint_value_mean/min/max
  - safety/constraint_satisfaction_rate
  - safety/min_collision_distance_mean/min/max/std
  - safety/distance_to_goal_mean/min/max
  - safety/goal_reached_rate

✓ All expected metrics present!
Test PASSED!
```

## 现在可以正常工作

运行训练时，你应该能在Wandb中看到`Safe/safety/`前缀下的所有安全指标：

```bash
python main.py \
    --algorithm mappo \
    --scenario navigation_obs_unicycle \
    --env vmas \
    --device cuda \
    --seed 0
```

在评估阶段，控制台会输出：
```
[SafetyMetrics] Collected 17 safety metrics
```

然后在Wandb的Run页面中查找"Safe"部分，应该能看到所有17个安全指标。

## 收集的指标

### 碰撞相关 (9个指标)
- `Safe/safety/agent_collision_rew_mean/min/max/std` - 碰撞惩罚统计
- `Safe/safety/collision_rate` - 碰撞率
- `Safe/safety/min_collision_distance_mean/min/max/std` - 最小距离统计

### 约束相关 (4个指标)
- `Safe/safety/constraint_value_mean/min/max` - 约束值统计
- `Safe/safety/constraint_satisfaction_rate` - 约束满足率

### 目标相关 (4个指标)
- `Safe/safety/distance_to_goal_mean/min/max` - 目标距离统计
- `Safe/safety/goal_reached_rate` - 目标到达率

## 文件清单

**修改的文件:**
- `gemsmarl/experiment/safety_metrics_callback.py` - 修复TensorDict访问和类型错误
- `gemsmarl/experiment/experiment.py` - 简化日志输出

**新增测试文件:**
- `test_safety_callback_unit.py` - 单元测试验证功能
