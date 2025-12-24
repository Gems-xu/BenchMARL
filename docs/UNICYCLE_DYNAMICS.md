# Unicycle Dynamics for NavigationObsUnicycleScenario

## 概述

已成功将 `NavigationObsUnicycleScenario` 场景改造为支持真正的 **Unicycle 动力学模型**。

## Unicycle 动力学模型

Unicycle 是一种非完整约束系统，其运动受以下动力学方程描述：

```
dx/dt = v * cos(θ)
dy/dt = v * sin(θ)
dθ/dt = ω
```

其中：
- `(x, y)`: 机器人的2D位置
- `θ`: 机器人的朝向角（heading angle）
- `v`: 线性速度（forward/backward motion）
- `ω`: 角速度（rotation rate）

## 主要实现细节

### 1. 智能体状态表示

每个智能体现在有以下状态属性：
- `agent.state.pos`: 位置 `[x, y]`
- `agent.state.vel`: 速度 `[vx, vy]`（由v和θ计算得出）
- `agent.orientation`: 朝向角 `θ`（自定义属性，避免与Lidar传感器冲突）
- `agent.ang_vel_command`: 角速度命令 `ω`

### 2. 动作空间

动作空间是2D的连续空间：
- `action[0]`: 标准化的线性速度 `[-1, 1]` → 映射到 `[-0.5, 0.5] m/s`
- `action[1]`: 标准化的角速度 `[-1, 1]` → 映射到 `[-3.0, 3.0] rad/s`

映射在 `process_action()` 方法中完成。

### 3. 观测空间

每个智能体的观测包括：
- 位置 `[x, y]`: 2维
- 速度 `[vx, vy]`: 2维
- 朝向信息 `[cos(θ), sin(θ)]`: 2维（便于神经网络处理）
- 目标相对位置: 2维（如果不观察所有目标）
- Lidar距离读数: 12维（if collisions enabled）

**总观测维度**: 20维（position 2 + velocity 2 + orientation 2 + goal 2 + lidar 12）

### 4. Unicycle 动力学积分

在每个仿真步骤中：

1. **获取动作**: 从策略网络接收 `[v_normalized, ω_normalized]`
2. **映射到实际值**: `v = v_normalized * max_linear_velocity`, `ω = ω_normalized * max_angular_velocity`
3. **计算速度**: 
   ```
   vx = v * cos(θ)
   vy = v * sin(θ)
   ```
4. **更新位置**: VMAS物理引擎使用计算的 `[vx, vy]` 更新位置
5. **更新朝向**: 在 `post_step()` 中手动更新：
   ```
   θ_new = θ_old + ω * dt
   ```

### 5. 关键改变

#### 文件: `gemsmarl/environments/vmas/scenarios/navigation_obs_unicycle.py`

**新增方法**:
- `process_action()`: 将标准化的[-1,1]动作映射到实际的unicycle命令
- `post_step()`: 在物理步骤后更新朝向角

**修改的属性**:
- 用 `agent.orientation` 替代 `agent.state.rot`（避免Lidar维度冲突）
- 添加 `agent.ang_vel_command` 存储角速度命令

**修改的方法**:
- `make_world()`: 禁用默认drag，增加substeps用于稳定的数值积分
- `observation()`: 添加 `[cos(θ), sin(θ)]` 表示朝向
- `reset_world_at()`: 初始化随机朝向角

#### 文件: `gemsmarl/environments/vmas/common.py`

- 添加 `NAVIGATION_OBS_UNICYCLE` 到 `VmasTask` 枚举
- 注册 `navigation_obs_unicycle` 到 `CUSTOM_SCENARIOS` 映射

#### 文件: `gemsmarl/conf/task/vmas/navigation_obs_unicycle.yaml`

- 创建新的Hydra配置文件
- 包含unicycle特定的参数：`max_linear_velocity`, `max_angular_velocity`

## 使用方法

### 直接运行场景测试

```bash
python gemsmarl/environments/vmas/scenarios/navigation_obs_unicycle.py
```

### 使用main.py进行训练

#### MAPPO 算法（推荐用于on-policy）
```bash
python main.py \
  --env vmas \
  --scenario navigation_obs_unicycle \
  --algorithm mappo \
  --device cuda:0 \
  --seed 42
```

#### MASAC 算法（off-policy）
```bash
python main.py \
  --env vmas \
  --scenario navigation_obs_unicycle \
  --algorithm masac \
  --device cuda:0 \
  --seed 42
```

#### 使用Safe-PINN模型
```bash
python main.py \
  --env vmas \
  --scenario navigation_obs_unicycle \
  --algorithm mappo \
  --use-safe-pinn \
  --device cuda:0 \
  --seed 42
```

## 配置参数

在 `gemsmarl/conf/task/vmas/navigation_obs_unicycle.yaml` 中调整：

```yaml
max_steps: 100                    # 每个episode的最大步数
n_agents: 4                       # 智能体数量
collisions: True                  # 是否启用碰撞检测
n_obstacles: 3                    # 障碍物数量
obstacle_radius: 0.1              # 障碍物半径
max_linear_velocity: 0.5          # 最大线性速度 (m/s)
max_angular_velocity: 3.0         # 最大角速度 (rad/s)
lidar_range: 0.35                 # Lidar检测范围
agent_radius: 0.1                 # 智能体半径
```

## 性能特性

- ✅ 真正的Unicycle非完整约束运动学
- ✅ 与VMAS Lidar传感器兼容
- ✅ 支持批量环境（多进程训练）
- ✅ 与Safe-PINN和PINN模型兼容
- ✅ 支持MAPPO和MASAC算法

## 与原始navigation_obs场景的区别

| 特性 | 原始navigation_obs | 新的navigation_obs_unicycle |
|------|------------------|---------------------------|
| 动力学模型 | 直接速度控制（质点） | 真实Unicycle非完整系统 |
| 动作空间 | [vx, vy] 直接速度 | [v, ω] 线速度和角速度 |
| 朝向 | 无（不需要） | θ（必需的unicycle状态） |
| 观测维度 | 16 | 20 |
| 运动约束 | 无 | 非完整约束 |

## 测试结果

场景已在以下配置下成功测试：
- ✅ 4个环境，4个智能体，3个障碍物
- ✅ VMAS render视图正确显示朝向箭头
- ✅ Lidar传感器正常工作
- ✅ MAPPO训练收敛
- ✅ 约束值计算正确

## 数学模型细节

### Unicycle运动学

给定控制输入 `(v, ω)`，机器人状态 `x = [x, y, θ]` 的演化为：

$$
\begin{aligned}
\dot{x} &= v \cos(\theta) \\
\dot{y} &= v \sin(\theta) \\
\dot{\theta} &= \omega
\end{aligned}
$$


### 观测中的朝向编码

我们使用 $(cos(\theta), sin(\theta))$ 而不是直接的 $\theta$，因为：
1. 避免角度360°不连续性
2. 神经网络更容易学习三角函数编码
3. 与VMAS Lidar传感器兼容

## 未来改进

- [ ] 添加加速度约束
- [ ] 支持后轮驱动vs前轮驱动切换
- [ ] 路径跟踪损失函数
- [ ] 可视化轨迹历史
