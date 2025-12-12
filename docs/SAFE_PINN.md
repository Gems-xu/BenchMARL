# PINN 与 Safe PINN 模型对比指南

本文档旨在说明 Physics-Informed Neural Network (PINN) 及其衍生变体 Safe PINN 和 Safe PINN PPO 之间的关键区别。

## 1. PINN (Physics-Informed Neural Network)

**适用场景**: 通用多智能体控制，需要符合物理动力学约束。

PINN (基于 LEMURS 架构) 将物理约束直接嵌入到神经网络架构中，而不是作为损失函数的一部分。

- **工作原理**:
  - 使用一系列注意力机制 (`Att_R`, `Att_J`, `Att_H`) 来学习系统的哈密顿量 (Hamiltonian)。
  - **H (总能量)** = 动能 + 势能 (任务目标)。
  - 控制律 $u$ 根据哈密顿动力学方程直接计算：$\dot{x} = (J - R) \nabla H + g(x)u$。
- **优点**: 保证系统的耗散性和物理一致性，训练收敛更稳定。
- **局限**: 缺乏明确的安全/避障机制，智能体可能会发生碰撞。

## 2. Safe PINN (Standard)

**适用场景**: Off-policy 算法 (如 MASAC)，需要安全避障。

Safe PINN 在 PINN 的基础上引入了 **Barrier Potential (势垒势能)** 来实现安全性。

- **工作原理**:
  - **H (总能量)** = 动能 + 任务势能 + **Barrier 势能**。
  - Barrier 函数形式: $B(d) = \frac{k}{(d - r_{coll})^2 + \epsilon}$。
  - 当智能体接近障碍物或彼此时，Barrier 势能剧增，产生巨大的排斥力。
- **配置为 MASAC 优化**:
  - 依赖 Off-policy 算法的 Q 值函数来平滑高方差的 barrier 梯度。
  - 使用较大的力饱和 (`f_max=10.0`)。
  - Barrier 梯度权重隐含为 1.0 (与任务梯度相等)。

## 3. Safe PINN PPO (Optimized for On-policy)

**适用场景**: On-policy 算法 (如 MAPPO)，需要安全避障且训练稳定。

MAPPO 对梯度方差非常敏感，直接使用标准的 Safe PINN 会导致梯度爆炸和策略崩溃。Safe PINN PPO 专门针对此进行了优化。

- **核心改进**:
  1. **Log-Barrier 函数**: 使用 $B(d) = -k \log(\frac{d}{r_{coll}})$ 代替反比例函数。Log 函数的梯度增长更平滑，避免了硬接触时的梯度爆炸。
  2. **梯度权重平衡**: 显式分离任务梯度和 barrier 梯度。
     - **Adaptive Scaling**: 根据任务梯度的大小动态调整 barrier 权重。
     - 默认 barrier 权重较低 (0.05-0.1)，防止安全约束在训练初期完全主导策略。
  3. **渐进式 Schedule**:
     - **Warmup**: 训练初期 barrier 权重从 0 线性增加，让智能体先学会移动。
     - **Plateau**: 中期保持 barrier 权重，强化安全意识。
     - **Decay**: 后期衰减 barrier 权重，允许任务目标主导微调。
  4. **更严格的稳定性控制**:
     - 力饱和 (`f_max`) 从 10.0 降至 1.0。
     - 对 barrier 梯度进行归一化处理。
     - 增大了 `barrier_epsilon` (0.05) 以提高数值稳定性。

## 总结对比

| 特性 | PINN | Safe PINN | Safe PINN PPO |
|------|------|-----------|---------------|
| **核心机制** | 能量整形 (Energy Shaping) | 能量整形 + 势垒函数 (Barrier) | 能量整形 + Log 势垒 + 自适应调度 |
| **主要目标** | 任务完成 & 物理一致性 | 任务完成 + 强安全约束 | 任务完成 + 稳定安全约束 |
| **适用算法** | 所有 | Off-policy (MASAC, DDPG) | On-policy (MAPPO, PPO) |
| **Barrier 类型** | 无 | 硬势垒 ($1/x^2$) | 软势垒 ($\log x$) |
| **梯度稳定性** | 高 | 低 (易爆炸) | 中 (经优化) |
| **力饱和 (f_max)** | N/A | High (10.0) | Low (1.0) |

---

### 如何在 main.py 中使用

代码已自动根据选择的算法切换模型配置：

```bash
# 运行 MAPPO (自动使用 Safe PINN PPO)
uv run main.py --algorithm mappo --env vmas --scenario navigation_obs --use-safe-pinn
# 运行 MASAC (自动使用 Standard Safe PINN)
uv run main.py --algorithm masac --env vmas --scenario navigation_obs --use-safe-pinn
```
