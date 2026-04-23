---
title: '大模型强化学习方法详解：PPO、GRPO、DAPO、GSPO'
date: 2026-02-19
permalink: /posts/2026/02/reinforcement-learning-methods-for-llm/
tags:
  - 强化学习
  - 大语言模型
  - PPO
  - GRPO
  - DAPO
  - GSPO
  - RLHF
---

随着大语言模型（LLM）的快速发展，强化学习（Reinforcement Learning, RL）已成为提升模型性能、对齐人类偏好的关键技术。本文将详细介绍四种主流的强化学习方法：PPO、GRPO、DAPO和GSPO，并总结它们之间的区别。

# 1. PPO（Proximal Policy Optimization）

## 1.1 背景介绍

PPO（Proximal Policy Optimization，近端策略优化）是由OpenAI在2017年提出的一种策略梯度方法，是目前大语言模型RLHF（Reinforcement Learning from Human Feedback）中最经典、应用最广泛的算法。

## 1.2 核心思想

PPO的核心思想是**限制策略更新的幅度**，避免过大的策略变化导致训练不稳定。它通过引入裁剪（Clipping）机制来实现这一目标。

## 1.3 数学公式

PPO的目标函数为：

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是新旧策略的概率比
- $\hat{A}_t$ 是优势函数的估计
- $\epsilon$ 是裁剪系数（通常取0.1或0.2）

## 1.4 在LLM中的应用

在大语言模型的RLHF训练中，PPO的工作流程如下：
1. **奖励模型训练**：使用人类偏好数据训练一个奖励模型（Reward Model）
2. **策略优化**：使用PPO算法优化语言模型策略，最大化奖励模型给出的分数
3. **KL散度约束**：添加KL散度惩罚项，防止策略偏离原始SFT模型太远

完整的PPO-RLHF目标函数：

$$
\mathcal{L}_{PPO-RLHF} = \mathbb{E}_{x \sim D, y \sim \pi_\theta(y|x)} \left[ r(x, y) - \beta \cdot KL(\pi_\theta || \pi_{ref}) \right]
$$

## 1.5 优缺点

**优点：**
- 训练稳定，采样效率高
- 实现相对简单
- 经过大规模验证，效果可靠

**缺点：**
- 需要同时维护策略模型、参考模型、奖励模型和价值模型（4个模型）
- 计算资源消耗大
- 超参数敏感，调参复杂

---

# 2. GRPO（Group Relative Policy Optimization）

## 2.1 背景介绍

GRPO（Group Relative Policy Optimization，组相对策略优化）是由DeepSeek团队提出的一种改进的强化学习方法，首次应用于DeepSeek-Math和DeepSeek-R1模型的训练中。

## 2.2 核心思想

GRPO的核心创新在于：
1. **去除价值网络（Critic）**：不再需要单独训练一个价值网络
2. **组内相对奖励**：通过同一prompt下多个输出的相对奖励来计算优势函数
3. **简化训练流程**：减少需要维护的模型数量

## 2.3 数学公式

GRPO的优势估计方法：

对于每个prompt $x_i$，采样一组输出 $\{y_{i,1}, y_{i,2}, ..., y_{i,G}\}$，计算组内相对优势：

$$
\hat{A}_{i,j} = \frac{r(x_i, y_{i,j}) - \text{mean}(\{r(x_i, y_{i,k})\}_{k=1}^{G})}{\text{std}(\{r(x_i, y_{i,k})\}_{k=1}^{G})}
$$

GRPO的目标函数：

$$
\mathcal{L}_{GRPO} = \mathbb{E}_{x_i, \{y_{i,j}\}_{j=1}^{G}} \left[ \frac{1}{G} \sum_{j=1}^{G} \min \left( r_{i,j}(\theta) \hat{A}_{i,j}, \text{clip}(r_{i,j}(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_{i,j} \right) - \beta \cdot KL(\pi_\theta || \pi_{ref}) \right]
$$

## 2.4 技术特点

1. **组采样策略**：每个prompt采样多个（如8-16个）响应
2. **标准化优势**：使用组内标准化避免奖励尺度问题
3. **无需价值网络**：显著降低内存和计算需求

## 2.5 优缺点

**优点：**
- 只需要3个模型（策略模型、参考模型、奖励模型），节省内存
- 训练更加稳定
- 在数学推理等任务上表现优异

**缺点：**
- 需要每个prompt采样多个输出，增加推理开销
- 组内样本数量影响优势估计的准确性

---

# 3. DAPO（Decoupled Clip and Dynamic Sampling Policy Optimization）

## 3.1 背景介绍

DAPO（Decoupled Clip and Dynamic Sampling Policy Optimization，解耦裁剪和动态采样策略优化）是字节跳动团队针对长链推理（Chain-of-Thought）任务提出的改进算法，旨在解决传统PPO/GRPO在复杂推理任务中的局限性。

## 3.2 核心思想

DAPO针对GRPO的几个关键问题进行了改进：
1. **解耦裁剪（Decoupled Clipping）**：分别处理正负优势的裁剪
2. **动态采样（Dynamic Sampling）**：根据样本难度动态调整采样策略
3. **Token级别的KL惩罚**：更精细的KL散度控制

## 3.3 数学公式

### 解耦裁剪机制

传统PPO/GRPO使用相同的裁剪范围处理正负优势，DAPO将其解耦：

$$
L^{DAPO}(\theta) = \mathbb{E}_t \left[ 
\begin{cases}
\min(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon_{low}, 1+\epsilon_{high}) \hat{A}_t) & \text{if } \hat{A}_t > 0 \\
\max(r_t \hat{A}_t, \text{clip}(r_t, 1-\epsilon_{low}, 1+\epsilon_{high}) \hat{A}_t) & \text{if } \hat{A}_t < 0
\end{cases}
\right]
$$

其中 $\epsilon_{low}$ 和 $\epsilon_{high}$ 可以独立设置。

### 动态采样策略

DAPO引入过滤机制，跳过过于简单或过于困难的样本：

$$
\mathcal{D}_{filtered} = \{(x_i, \{y_{i,j}\}) | \text{acc}(x_i) \in (\tau_{low}, \tau_{high})\}
$$

其中 $\text{acc}(x_i)$ 是prompt $x_i$ 的组内正确率。

### Token级别KL惩罚

$$
\mathcal{L}_{KL}^{token} = \sum_{t=1}^{T} KL(\pi_\theta(\cdot|s_t) || \pi_{ref}(\cdot|s_t))
$$

## 3.4 技术特点

1. **非对称裁剪**：允许正优势样本有更大的更新空间
2. **难度自适应**：过滤掉太简单（全对）或太难（全错）的样本
3. **精细化KL控制**：在token级别进行KL约束
4. **无需奖励模型**：可以直接使用规则奖励（如数学题正确性）

## 3.5 优缺点

**优点：**
- 在长链推理任务（数学、代码）上效果显著
- 样本利用效率更高
- 避免了简单样本和困难样本的训练浪费

**缺点：**
- 超参数更多，需要仔细调整
- 动态采样可能导致训练样本分布偏移
- 实现相对复杂

---

# 4. GSPO（Group Sampling Policy Optimization）

## 4.1 背景介绍

GSPO（Group Sampling Policy Optimization，组采样策略优化）是一种结合了组采样思想和策略优化的方法，在GRPO的基础上进行了进一步的改进和优化。

## 4.2 核心思想

GSPO的核心改进包括：
1. **改进的组采样策略**：更高效的组内样本生成方式
2. **优化的优势估计**：引入基线减少方差
3. **自适应KL系数**：根据训练进度动态调整KL惩罚强度

## 4.3 数学公式

### 改进的优势估计

GSPO使用带基线的优势估计：

$$
\hat{A}_{i,j} = r(x_i, y_{i,j}) - b(x_i)
$$

其中基线 $b(x_i)$ 可以是组内奖励的均值或其他估计。

### 自适应KL系数

$$
\beta_{t+1} = \beta_t \cdot \begin{cases}
\alpha & \text{if } KL_t > KL_{target} \\
1/\alpha & \text{if } KL_t < KL_{target}
\end{cases}
$$

### GSPO目标函数

$$
\mathcal{L}_{GSPO} = \mathbb{E} \left[ \sum_{j=1}^{G} w_j \cdot L^{clip}(r_j, \hat{A}_j) - \beta_t \cdot KL(\pi_\theta || \pi_{ref}) \right]
$$

其中 $w_j$ 是样本权重，可以根据奖励或优势进行加权。

## 4.4 技术特点

1. **重要性加权**：对高质量样本赋予更高权重
2. **自适应正则化**：KL惩罚系数随训练动态调整
3. **高效采样**：通过批量采样减少推理开销
4. **稳定训练**：引入多种稳定化技术

## 4.5 优缺点

**优点：**
- 样本利用效率高
- 训练过程稳定
- 支持灵活的奖励函数设计

**缺点：**
- 需要合理设计样本加权策略
- 自适应KL系数可能导致训练不稳定
- 对组大小敏感

---

# 5. 方法对比总结

## 5.1 核心特点对比表

| 特性 | PPO | GRPO | DAPO | GSPO |
|------|-----|------|------|------|
| **价值网络** | 需要 | 不需要 | 不需要 | 不需要 |
| **模型数量** | 4个 | 3个 | 3个 | 3个 |
| **优势估计** | GAE | 组内标准化 | 组内+解耦 | 组内+基线 |
| **裁剪方式** | 对称 | 对称 | 非对称解耦 | 对称 |
| **采样策略** | 单样本 | 组采样 | 动态组采样 | 加权组采样 |
| **KL控制** | 序列级 | 序列级 | Token级 | 自适应 |
| **计算开销** | 高 | 中 | 中 | 中 |
| **内存需求** | 高 | 较低 | 较低 | 较低 |

## 5.2 适用场景对比

| 方法 | 适用场景 |
|------|----------|
| **PPO** | 通用RLHF训练、对话系统、内容生成，需要稳定可靠的训练效果 |
| **GRPO** | 资源受限场景、数学推理、代码生成，需要高效训练 |
| **DAPO** | 长链推理任务（数学证明、复杂代码）、需要处理难度差异大的数据集 |
| **GSPO** | 需要灵活奖励设计、自适应训练强度控制的场景 |

## 5.3 技术演进路线

```
PPO（经典方法）
  │
  ├──→ GRPO（去除价值网络，组相对优势）
  │      │
  │      ├──→ DAPO（解耦裁剪，动态采样，Token级KL）
  │      │
  │      └──→ GSPO（自适应KL，重要性加权）
```

## 5.4 主要区别总结

1. **价值网络的去留**
   - PPO需要价值网络估计状态价值
   - GRPO/DAPO/GSPO通过组采样替代价值网络

2. **优势函数估计方式**
   - PPO使用GAE（Generalized Advantage Estimation）
   - GRPO使用组内奖励的标准化差异
   - DAPO在GRPO基础上引入解耦裁剪
   - GSPO引入带基线的优势估计和重要性加权

3. **裁剪策略**
   - PPO/GRPO/GSPO使用对称裁剪
   - DAPO使用非对称解耦裁剪，允许正优势有更大更新空间

4. **采样策略**
   - PPO每个prompt单次采样
   - GRPO固定组采样
   - DAPO动态过滤采样
   - GSPO加权组采样

5. **KL散度控制**
   - PPO/GRPO使用序列级KL惩罚
   - DAPO使用更精细的Token级KL惩罚
   - GSPO使用自适应KL系数

## 5.5 选择建议

- **资源充足、追求稳定**：选择PPO
- **资源有限、数学/代码任务**：选择GRPO
- **长链推理、难度差异大**：选择DAPO
- **需要灵活控制、自适应训练**：选择GSPO

---

# 参考资料

1. Schulman, J., et al. "Proximal Policy Optimization Algorithms." (2017)
2. DeepSeek-AI. "DeepSeek-Math: Pushing the Limits of Mathematical Reasoning." (2024)
3. DeepSeek-AI. "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs." (2025)
4. ByteDance. "DAPO: Decoupled Clip and Dynamic Sampling Policy Optimization." (2025)
