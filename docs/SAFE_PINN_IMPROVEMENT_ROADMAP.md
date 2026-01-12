# Safe PINN 算法改进路线图

## 问题诊断总结

### 核心问题
**MAPPO + Safe PINN (遵循PHS框架) 表现差的根本原因**：

```
╔══════════════════════════════════════════════════════════════════╗
║  PHS方程: ẋ = (J - R) ∇H                                         ║
║                                                                  ║
║  问题: J矩阵随机初始化 → j12可正可负                             ║
║       如果j12为负 → (J-R)∇H方向错误 → agents远离目标             ║
║                                                                  ║
║  MASAC能修复: Q值提供"这个action不好"的信号 → actor调整J         ║
║  MAPPO不能修复: Policy Gradient仅有稀疏奖励 → J符号难以纠正       ║
╚══════════════════════════════════════════════════════════════════╝
```

---

## 改进方案优先级

### 第一优先级：紧急修复（建议立即实施）

#### 方案1：强制J矩阵的物理先验 ✅ **已实现**

**修改文件**: `gemsmarl/models/pinn.py`

**原理**: J矩阵的对角块$j_{12}$决定了$\dot{p}$对$\nabla_q H$的响应方向。标准Hamiltonian需要$j_{12} > 0$。

**实现方法**: 使用方法2（绝对值方法）

**修改代码**:

```python
# Att_R.forward() 和 Att_J.forward() 方法中
def forward(self, x, laplacian, scenario_name):
    # ... existing attention layers ...
    
    x = self.mlp_out(x.reshape(-1, self.hidden_dim))
    j12_raw = x.sum(1).sum(1).reshape(batch, self.na)
    
    # ========== 修改开始 ==========
    # 方法1: 使用softplus确保正值
    # j12 = torch.nn.functional.softplus(j12_raw) + 0.01

    # 方法2: 使用绝对值（更简单）
    # Fix: Use absolute value to ensure j12 is always positive (standard Hamiltonian structure)
    # This ensures correct force direction: dp/dt = -j12 * ∇H_q
    j12 = torch.abs(j12_raw) + 0.01  # Add small offset to prevent zero

    # 方法3: 使用sigmoid缩放到正区间
    # j12 = torch.sigmoid(j12_raw) * 2.0 + 0.1  # [0.1, 2.1]
    # ========== 修改结束 ==========

    j21 = -j12  # Keep antisymmetric structure
    
    # ... rest of code ...
```

**修改位置**:
- Line 207 (Att_R.forward)
- Line 293 (Att_J.forward)

**预期效果** ✅:
- ✅ J矩阵始终满足标准Hamiltonian结构
- ✅ $\dot{p} = -j_{12} \nabla_q H$ 始终产生正确的力方向
- ✅ 与MASAC和MAPPO都兼容
- ✅ 消除初始化随机性导致的方向错误

**测试验证**:
运行命令进行训练验证：
```bash
CUDA_VISIBLE_DEVICES=2 uv run main.py --algorithm mappo --env vmas --scenario navigation_obs_unicycle --device cuda:0
```

预期观察：
- Agents应该直奔目标，而不是相反方向
- Eval reward应该快速上升
- 不应该再出现初始方向错误的情况

---

#### 方案2：训练早期使用物理先验，后期切换到学习

**修改文件**: `gemsmarl/models/safe_pinn_ppo.py`

**原理**: 在agent还不知道正确方向时使用人工指导，学会基本行为后再切换到学习的(J-R)。

**新增方法**:

```python
def _compute_prior_dynamics(self, grad_H_goal, grad_H_barrier, batch_size):
    """使用物理先验计算动力学，确保方向正确"""
    # 标准Hamiltonian结构: dp/dt = -dH/dq
    # 对于goal: 力 = -∇H_goal = -(q - goal) = goal - q (指向goal)
    # 对于barrier: 力 = -∇H_barrier (远离障碍)
    
    # 分离位置和动量梯度
    grad_goal_q = grad_H_goal[:, :self.action_dim_per_agent]
    grad_barrier_q = grad_H_barrier[:, :self.action_dim_per_agent]
    
    # 物理正确的力
    force_goal = -self.task_weight * grad_goal_q  # 吸引向goal
    force_barrier = -self.barrier_weight * grad_barrier_q  # 排斥障碍
    
    # 组合力
    total_force = force_goal + force_barrier
    
    # 构造dx (简化版动力学)
    # dq/dt = p/m (由动能决定)
    # dp/dt = total_force
    dq = grad_H_goal[:, self.action_dim_per_agent:2*self.action_dim_per_agent]  # dH/dp
    dp = total_force
    
    dx_prior = torch.cat([dq, dp], dim=-1)
    return dx_prior

def _forward(self, tensordict):
    # ... 现有代码 ...
    
    # 计算学习的PHS动力学
    dx_learned = torch.bmm(J_mean - R_mean, grad_H_total.unsqueeze(2)).squeeze(2)
    
    # 计算物理先验动力学
    dx_prior = self._compute_prior_dynamics(grad_H_goal, grad_H_barrier, batch_size)
    
    # 混合比例 (0→1 随训练进行)
    blend_steps = 500  # 500步过渡
    alpha = min(1.0, self._training_steps.item() / blend_steps)
    
    # 早期使用prior，后期使用learned
    dx_final = (1 - alpha) * dx_prior + alpha * dx_learned
    
    # ... 后续代码 ...
```

**预期效果**:
- ✅ 训练早期有正确方向指导
- ✅ 后期平滑过渡到学习的系统
- ✅ 同时获得物理一致性和学习灵活性

---

### 第二优先级：性能优化（建议实施）

#### 方案3：Barrier势能设计优化

**问题**: 当前barrier在距离接近安全距离时可能梯度不够大

**修改文件**: `gemsmarl/models/safe_pinn_ppo.py`

```python
def _compute_improved_barrier(self, dist, r_collision, epsilon):
    """改进的barrier函数，在危险区域更陡峭"""
    gap = dist - r_collision
    
    # 三段式barrier
    # Zone 1: 安全区 (gap > epsilon) - 软约束
    # Zone 2: 警戒区 (0 < gap <= epsilon) - 硬约束
    # Zone 3: 碰撞区 (gap <= 0) - 超强约束
    
    # 基础barrier
    safe_gap = torch.clamp(gap, min=0.001)
    ratio = safe_gap / (r_collision + epsilon)
    base_barrier = torch.nn.functional.softplus(-torch.log(ratio + 1e-6), beta=2.0)
    
    # 危险区域增益
    danger_mask = (gap < epsilon).float()
    danger_gain = 1.0 + 3.0 * danger_mask  # 危险区域3倍增益
    
    # 碰撞区域惩罚
    collision_mask = (gap < 0).float()
    collision_penalty = 10.0 * collision_mask * torch.abs(gap)
    
    return base_barrier * danger_gain + collision_penalty
```

---

#### 方案4：添加速度方向考虑

**问题**: 当前barrier不考虑agents的运动方向

```python
def _compute_predictive_barrier(self, q_batch, v_batch, r_collision, dt=0.1):
    """预测性barrier，考虑速度方向"""
    n_agents = q_batch.shape[1]
    
    # 当前距离
    diff = q_batch.unsqueeze(2) - q_batch.unsqueeze(1)  # (b, n, n, 2)
    current_dist = torch.norm(diff, dim=-1)  # (b, n, n)
    
    # 预测未来位置
    q_future = q_batch + v_batch * dt
    diff_future = q_future.unsqueeze(2) - q_future.unsqueeze(1)
    future_dist = torch.norm(diff_future, dim=-1)
    
    # 如果正在接近（future_dist < current_dist），增加barrier权重
    approaching_mask = (future_dist < current_dist).float()
    approach_factor = 1.0 + approaching_mask  # 接近时2倍权重
    
    # 最终距离用于barrier计算
    effective_dist = torch.min(current_dist, future_dist)
    
    return effective_dist, approach_factor
```

---

### 第三优先级：架构改进（中期目标）

#### 方案5：分离物理层和学习层

```python
class SafePinnPPO_Hybrid(nn.Module):
    """混合架构: 固定物理先验 + 可学习修正"""
    
    def __init__(self, ...):
        # 物理先验层（固定）
        self.J_prior = self._create_standard_J(n_agents)  # 标准Hamiltonian J
        self.R_prior = self._create_damping_R(n_agents)   # 阻尼矩阵
        
        # 可学习修正层（小幅度调整）
        self.J_correction = Att_J(...)  # 输出受限的小修正
        self.R_correction = Att_R(...)
        
        # 修正强度（可调）
        self.correction_scale = 0.1  # 限制修正幅度
    
    def _create_standard_J(self, n):
        """标准Hamiltonian互联矩阵"""
        dim = n * 2 * 2  # n agents, (q,p), 2D
        J = torch.zeros(dim, dim)
        for i in range(n):
            # 对每个agent: J = [[0, I], [-I, 0]]
            idx_q = i * 2
            idx_p = (n + i) * 2
            J[idx_q:idx_q+2, idx_p:idx_p+2] = torch.eye(2)
            J[idx_p:idx_p+2, idx_q:idx_q+2] = -torch.eye(2)
        return J
    
    def forward(self, x):
        # 获取先验
        J = self.J_prior.expand(batch_size, -1, -1)
        R = self.R_prior.expand(batch_size, -1, -1)
        
        # 添加小幅修正
        J = J + self.correction_scale * self.J_correction(x)
        R = R + self.correction_scale * self.R_correction(x)
        
        # 计算动力学
        dx = torch.bmm(J - R, grad_H.unsqueeze(2)).squeeze(2)
        return dx
```

**优势**:
- ✅ 始终保持物理一致性
- ✅ 学习层只做微调
- ✅ 对初始化不敏感

---

#### 方案6：课程学习

```python
class CurriculumSafePinn:
    """课程学习: 逐步增加难度"""
    
    def __init__(self):
        self.phases = [
            {"obstacles": 0, "agents_interact": False, "steps": 200},
            {"obstacles": 1, "agents_interact": False, "steps": 300},
            {"obstacles": 3, "agents_interact": True, "steps": 500},
        ]
        self.current_phase = 0
    
    def get_hamiltonian(self, state, training_step):
        phase = self._get_phase(training_step)
        
        # Phase 1: 只有H_goal
        H = H_goal + H_kin
        
        # Phase 2: 添加静态障碍物
        if phase >= 1:
            H = H + H_barrier_obstacle
        
        # Phase 3: 添加agent交互
        if phase >= 2:
            H = H + H_barrier_agents + H_task
        
        return H
```

---

### 第四优先级：安全性增强（长期目标）

#### 方案7：Control Barrier Function (CBF) 集成

```python
class CBFSafetyLayer:
    """Control Barrier Function安全层"""
    
    def __init__(self, r_safe=0.2, gamma=1.0):
        self.r_safe = r_safe
        self.gamma = gamma
    
    def compute_cbf_constraint(self, q_i, q_j, v_i, v_j):
        """
        CBF约束: dh/dt >= -γ * h
        其中 h(x) = ||q_i - q_j||² - r_safe²
        """
        # 安全函数
        diff = q_i - q_j
        h = torch.sum(diff**2) - self.r_safe**2
        
        # 安全函数导数
        rel_vel = v_i - v_j
        dh_dt = 2 * torch.sum(diff * rel_vel)
        
        # CBF约束
        constraint = dh_dt + self.gamma * h
        
        return h, constraint
    
    def filter_action(self, u_nominal, state):
        """如果nominal action违反CBF约束，进行修正"""
        q_batch = state[:, :, 0:2]
        v_batch = state[:, :, 2:4]
        
        # 检查所有agent对
        for i in range(n_agents):
            for j in range(i+1, n_agents):
                h, constraint = self.compute_cbf_constraint(
                    q_batch[:, i], q_batch[:, j],
                    v_batch[:, i], v_batch[:, j]
                )
                
                # 如果约束将被违反，修正action
                if constraint < 0:
                    u_nominal = self._project_to_safe(u_nominal, i, j, constraint)
        
        return u_nominal
```

---

#### 方案8：运行时安全监控

```python
class RuntimeSafetyMonitor:
    """运行时安全监控，记录碰撞事件"""
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.collision_log = []
        self.near_miss_log = []
    
    def check_safety(self, state, step):
        q_batch = state[:, :, 0:2]
        
        for i in range(n_agents):
            for j in range(i+1, n_agents):
                dist = torch.norm(q_batch[:, i] - q_batch[:, j])
                
                if dist < collision_threshold:
                    self.collision_log.append({
                        "step": step,
                        "agents": (i, j),
                        "distance": dist.item(),
                        "severity": "COLLISION"
                    })
                elif dist < near_miss_threshold:
                    self.near_miss_log.append({
                        "step": step,
                        "agents": (i, j),
                        "distance": dist.item(),
                        "severity": "NEAR_MISS"
                    })
    
    def generate_report(self):
        """生成安全报告"""
        return {
            "total_collisions": len(self.collision_log),
            "total_near_misses": len(self.near_miss_log),
            "collision_rate": len(self.collision_log) / total_steps,
            "most_frequent_pairs": self._analyze_pairs(),
        }
```

---

## 实施计划

### Phase 1: 紧急修复 (1-2天)

1. **实施方案1** (约束J矩阵符号)
   - 修改 `pinn.py` 中的 `Att_J.forward()`
   - 验证MAPPO训练效果

2. **测试验证**
   - 运行 `navigation_obs_unicycle` 场景
   - 确认agents朝向目标运动
   - 比较修改前后的reward曲线

### Phase 2: 性能优化 (3-5天)

1. **实施方案2** (混合策略)
   - 修改 `safe_pinn_ppo.py`
   - 测试不同blend_steps的效果

2. **实施方案3** (改进barrier)
   - 修改barrier计算逻辑
   - 验证碰撞率下降

### Phase 3: 架构改进 (1-2周)

1. **实施方案5** (混合架构)
   - 重构模型架构
   - 全面测试

2. **实施方案6** (课程学习)
   - 设计课程阶段
   - 评估学习效率

### Phase 4: 安全增强 (持续)

1. **集成CBF** (方案7)
2. **添加监控** (方案8)
3. **持续优化**

---

## 评估指标

| 指标 | 当前水平 | 目标 |
|------|---------|------|
| 目标达到率 | ~60% | >95% |
| 碰撞率 (agent-agent) | ~15% | <3% |
| 碰撞率 (agent-obstacle) | ~20% | <5% |
| 运动流畅性 | 卡顿明显 | 流畅 |
| Eval reward (mean) | ~-5 | >+5 |
| 训练稳定性 | 梯度爆炸 | 稳定 |

---

## 参考资源

1. Port-Hamiltonian Systems: [van der Schaft, 2000]
2. Control Barrier Functions: [Ames et al., 2017]
3. Safe RL: [García & Fernández, 2015]
4. Physics-Informed Neural Networks: [Raissi et al., 2019]

---

*文档版本: v1.0*
*更新日期: 2026-01-11*
