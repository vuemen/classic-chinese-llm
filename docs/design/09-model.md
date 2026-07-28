# 模型层设计文档

**所属阶段:** Phase 4 — 模型实现与预训练（核心阶段）
**涉及模块:** `src/classic_chinese_llm/model/`
**日期:** 2026-07-28

---

## 1. 需求概述

### 1.1 功能需求

| 编号 | 需求 | 说明 |
|------|------|------|
| F1 | RMSNorm 归一化层 | 实现 Root Mean Square Layer Normalization，仅保留缩放参数，去除平移参数和均值中心化 |
| F2 | RoPE 旋转位置编码 | 实现 Rotary Position Embedding，直接作用于 Q/K 向量，支持训练时未见过的序列长度 |
| F3 | MultiHeadAttention | 多头因果自注意力，通过 `F.scaled_dot_product_attention` 后端自动启用 FlashAttention，causal mask 保证自回归 |
| F4 | SwiGLU FFN | 门控前馈网络：gate + up 投影 → SiLU 激活 → element-wise 乘法 → down 投影 |
| F5 | TransformerBlock | 单个 Decoder 层：Pre-norm 残差结构（RMSNorm → Attention → Residual → RMSNorm → FFN → Residual） |
| F6 | TransformerLM | 完整模型：Token Embedding → TransformerBlock × N → Final RMSNorm → LM Head，权重共享 |
| F7 | Generator | 自回归生成器：Greedy / Temperature / Top-K / Top-P / Repetition Penalty / Beam Search + 流式输出 |

### 1.2 非功能需求

- **纯 `torch.nn` 实现**: 零 HuggingFace 模型代码依赖（`transformers` 仅用于 tokenizer 互操作）
- **~157M 参数**: 必须在 12GB VRAM 内完成训练（BF16 混合精度），batch_size=8、seq_len=1024 时显存占用 < 10GB
- **模块可组合性**: 每个组件独立可测，可替换（如 RoPE → AliBi、SwiGLU → ReLU）
- **数值稳定性**: BF16 下 softmax、LayerNorm/RMSNorm 不溢出
- **训练/推理一致性**: 同一模型代码支持训练和推理（通过 `model.train()` / `model.eval()` 切换）
- **与 checkpoint 系统兼容**: `model.state_dict()` 可直接传入 `CheckpointState.model_state_dict`

### 1.3 模型规格（来自 `ModelConfig`）

| 参数 | 数值 | 说明 |
|------|------|------|
| `vocab_size` | 32,000 | 与 SentencePiece Unigram tokenizer 的 vocab_size 一致 |
| `d_model` | 768 | 隐藏维度 / embedding 维度 |
| `n_layers` | 14 | Transformer Block 层数 |
| `n_heads` | 12 | 注意力头数（d_model / n_heads = 64 每头） |
| `d_ff` | 3,072 | SwiGLU FFN 中间层维度（d_model × 4） |
| `max_seq_len` | 2,048 | 最大序列长度（训练 + 推理） |
| `dropout` | 0.0 | 预训练阶段不使用 dropout（减少正则化，加速收敛） |
| `head_dim` | 64 | 每头维度 = d_model / n_heads |

### 1.4 模型架构全景图

```
                           ┌──────────────────────────────────────┐
                           │            INPUT (token IDs)          │
                           │       shape: (batch, seq_len)         │
                           │       例: [32, 2048] 训练时            │
                           └──────────────────┬───────────────────┘
                                              │
                           ┌──────────────────▼───────────────────┐
                           │          TOKEN EMBEDDING              │
                           │    nn.Embedding(32000, 768)           │
                           │    参数: 32,000 × 768 = 24,576,000    │
                           │    shape: (batch, seq_len, 768)       │
                           └──────────────────┬───────────────────┘
                                              │
              ┌───────────────────────────────┼───────────────────────────────┐
              │                               │                               │
              ▼                               ▼                               ▼
    ┌─────────────────────┐     ┌─────────────────────┐     ┌─────────────────────┐
    │  TRANSFORMER BLOCK  │     │  TRANSFORMER BLOCK  │     │  TRANSFORMER BLOCK  │
    │       #1 / 14       │     │       #2 / 14       │     │      #14 / 14       │
    │                     │     │                     │     │                     │
    │ ┌─────────────────┐ │     │ ┌─────────────────┐ │     │ ┌─────────────────┐ │
    │ │    RMSNorm      │ │     │ │    RMSNorm      │ │     │ │    RMSNorm      │ │
    │ │  (Attn Norm)    │ │     │ │  (Attn Norm)    │ │     │ │  (Attn Norm)    │ │
    │ └───────┬─────────┘ │     │ └───────┬─────────┘ │     │ └───────┬─────────┘ │
    │         ▼           │     │         ▼           │     │         ▼           │
    │ ┌─────────────────┐ │     │ ┌─────────────────┐ │     │ ┌─────────────────┐ │
    │ │ MultiHeadAttn   │ │     │ │ MultiHeadAttn   │ │     │ │ MultiHeadAttn   │ │
    │ │ 12 heads × 64d  │ │     │ │ 12 heads × 64d  │ │     │ │ 12 heads × 64d  │ │
    │ │ + RoPE on Q,K   │ │     │ │ + RoPE on Q,K   │ │     │ │ + RoPE on Q,K   │ │
    │ │ + Causal Mask    │ │     │ │ + Causal Mask    │ │     │ │ + Causal Mask    │ │
    │ └───────┬─────────┘ │     │ └───────┬─────────┘ │     │ └───────┬─────────┘ │
    │         │  + (残差)  │     │         │  + (残差)  │     │         │  + (残差)  │
    │         ▼           │     │         ▼           │     │         ▼           │
    │ ┌─────────────────┐ │     │ ┌─────────────────┐ │     │ ┌─────────────────┐ │
    │ │    RMSNorm      │ │     │ │    RMSNorm      │ │     │ │    RMSNorm      │ │
    │ │  (FFN Norm)     │ │     │ │  (FFN Norm)     │ │     │ │  (FFN Norm)     │ │
    │ └───────┬─────────┘ │     │ └───────┬─────────┘ │     │ └───────┬─────────┘ │
    │         ▼           │     │         ▼           │     │         ▼           │
    │ ┌─────────────────┐ │     │ ┌─────────────────┐ │     │ ┌─────────────────┐ │
    │ │   SwiGLU FFN    │ │     │ │   SwiGLU FFN    │ │     │ │   SwiGLU FFN    │ │
    │ │ gate+up+down    │ │     │ │ gate+up+down    │ │     │ │ gate+up+down    │ │
    │ │ d_ff = 3072     │ │     │ │ d_ff = 3072     │ │     │ │ d_ff = 3072     │ │
    │ └───────┬─────────┘ │     │ └───────┬─────────┘ │     │ └───────┬─────────┘ │
    │         │  + (残差)  │     │         │  + (残差)  │     │         │  + (残差)  │
    │         ▼           │     │         ▼           │     │         ▼           │
    │   (输出: 768 维)     │     │   (输出: 768 维)     │     │   (输出: 768 维)     │
    └─────────┬───────────┘     └─────────┬───────────┘     └─────────┬───────────┘
              │                           │                           │
              └───────────────────────────┼───────────────────────────┘
                                          │
                           ┌──────────────▼──────────────┐
                           │       FINAL RMSNorm          │
                           │     参数: 768 (γ only)       │
                           │   shape: (batch, seq_len,    │
                           │           768)               │
                           └──────────────┬──────────────┘
                                          │
                           ┌──────────────▼──────────────┐
                           │          LM HEAD             │
                           │  nn.Linear(768, 32000)       │
                           │  权重共享 with Token Embed    │
                           │  shape: (batch, seq_len,     │
                           │          32000)              │
                           └──────────────┬──────────────┘
                                          │
                           ┌──────────────▼──────────────┐
                           │       OUTPUT (logits)        │
                           │   shape: (batch, seq_len,    │
                           │          32000)              │
                           │  → CrossEntropyLoss 计算     │
                           └─────────────────────────────┘


                  ╔═══════════════════════════════════════════════╗
                  ║        模型规模速览                              ║
                  ╠═══════════════════╦═══════════════════════════╣
                  ║ 总参数量           ║ ~157M (156,718,848)       ║
                  ║ Transformer 层数   ║ 14                        ║
                  ║ 隐藏维度 d_model   ║ 768                       ║
                  ║ 注意力头数 n_heads ║ 12 (每头 64 维)           ║
                  ║ FFN 中间维度 d_ff  ║ 3072 (d_model × 4)       ║
                  ║ 词汇量 vocab_size  ║ 32,000                    ║
                  ║ 最大序列长度       ║ 2,048                     ║
                  ║ Embedding/LM Head  ║ 权重共享 (节省 ~24.6M)    ║
                  ║ 位置编码           ║ RoPE (零额外参数)          ║
                  ║ 激活函数           ║ SwiGLU (门控 FFN)         ║
                  ║ 归一化             ║ RMSNorm (Pre-norm)        ║
                  ║ 注意力             ║ 稠密 Causal + FlashAttn   ║
                  ╚═══════════════════╩═══════════════════════════╝


              单个 TransformerBlock 内部结构（Pre-norm 残差）:

                ┌─────────────────────────────────────────┐
                │           INPUT: x (768 维)              │
                └──────────────────┬──────────────────────┘
                                   │
                  ┌────────────────┴────────────────┐
                  │                                 │
                  ▼                                 │
        ┌─────────────────┐                         │
        │   RMSNorm       │  ← 先归一化              │
        │   (γ, 768 维)   │                         │
        └────────┬────────┘                         │
                 │                                  │
                 ▼                                  │
        ┌─────────────────┐                         │
        │ MultiHeadAttn   │  ← 再计算                │
        │ · Q/K/V 投影    │                         │
        │ · RoPE on Q,K   │                         │
        │ · FlashAttn     │                         │
        │ · Causal Mask   │                         │
        │ · O 投影        │                         │
        └────────┬────────┘                         │
                 │                                  │
                 │  x = x + Attn(RMSNorm(x))        │
                 │  ┌─────────────────────────────┐ │
                 │  │       残差连接 (加法)         │◄┘
                 │  └─────────────┬───────────────┘
                 │                │
                 ▼                ▼
        ┌──────────────────────────────────────────┐
        │        中间状态: h (768 维)               │
        └──────────────────┬───────────────────────┘
                           │
          ┌────────────────┴────────────────┐
          │                                 │
          ▼                                 │
┌─────────────────┐                         │
│   RMSNorm       │  ← 先归一化              │
│   (γ, 768 维)   │                         │
└────────┬────────┘                         │
         │                                  │
         ▼                                  │
┌─────────────────┐                         │
│  SwiGLU FFN     │  ← 再计算                │
│  · gate 投影    │                         │
│  · up 投影      │                         │
│  · SiLU ⊙ up   │                         │
│  · down 投影    │                         │
└────────┬────────┘                         │
         │                                  │
         │  h = h + FFN(RMSNorm(h))         │
         │  ┌─────────────────────────────┐ │
         │  │       残差连接 (加法)         │◄┘
         │  └─────────────┬───────────────┘
         │                │
         ▼                ▼
┌──────────────────────────────────────────┐
│           OUTPUT: (768 维)                │
│        传入下一个 TransformerBlock         │
└──────────────────────────────────────────┘
```

---

## 2. 方案选型与对比

### 2.1 注意力机制：稠密 vs 稀疏

这是模型架构层面最根本的选择。

| 方案 | 原理 | 复杂度 | 长序列 | 实现复杂度 | 文言文适配 | 结论 |
|------|------|--------|--------|-----------|-----------|------|
| **Dense (Full Causal)** | 每个 token 关注所有历史 token | O(n²) | 受限于 max_seq_len | ⭐ 简单 | ⭐⭐⭐ | ✅ 选用 |
| Sliding Window | 每个 token 仅关注前 W 个 token | O(nW) | 理论上无限 | ⭐⭐ | ⭐⭐ | ❌ |
| Sparse (Sparse Transformer) | 固定稀疏模式（如跨步 + 局部） | O(n√n) | 较好 | ⭐⭐⭐ | ⭐ | ❌ |
| Mamba / SSM | 状态空间模型，线性复杂度 | O(n) | 天然长序列 | ⭐⭐⭐⭐ | ⭐ | ❌ |

**详细分析**:

**Dense Attention（选用）**：

文言文场景下稠密注意力的充分理由：

1. **序列长度短**: 文言文 max_seq_len=2,048，在 FlashAttention 下 O(n²) 的 2,048² ≈ 4M 次交互在 GPU 上完全可承受。稠密注意力的主要瓶颈（长序列）对文言文不存在。

2. **文言文需要全上下文**: 文言文的指代省略（主语省略、宾语省略、代词省略）极为常见，理解一句话往往需要回溯很远的上文。稠密注意力让每个 token 能关注到 2,048 范围内的所有历史信息。

3. **FlashAttention 消除了稠密的内存瓶颈**: `F.scaled_dot_product_attention` 在 Ampere+ GPU 上自动使用 FlashAttention-2 内核，将 O(n²) 的内存降到 O(n)，使稠密注意力在 2K 序列长度下的内存和计算都不是瓶颈。

4. **实现简单 = 正确性高**: 不需要实现复杂的稀疏 mask 生成、块划分逻辑，降低了 bug 风险。

```python
# 稠密 Causal Attention 的核心就是一行 mask
# causal_mask: (T, T) 的上三角矩阵，-inf 屏蔽未来 token
causal_mask = torch.triu(
    torch.full((seq_len, seq_len), float("-inf"), device=device),
    diagonal=1,
)
```

**Sliding Window + Global**:

```python
# Mistral 使用的滑动窗口 + 全局注意力的混合方案
# 问题: 文言文中重要的全局上下文 token 难以预先确定
# 增加了一个超参数（窗口大小 W），引入不必要的复杂度
```

**Sparse / Mamba**:
- 稀疏 Transformer 的固定模式（如跨步）对文言文的灵活指代关系不友好
- Mamba/SSM 是新兴架构，在 157M 小模型规模上的表现未经充分验证；本项目的学习目标包括经典 Transformer，SSM 偏离了学习路径

**最终选择: 稠密 Causal Attention + FlashAttention 后端**。在 2,048 序列长度约束下，稠密注意力是最可靠、最可解释、最易实现的选择。

---

### 2.2 归一化位置：Pre-norm vs Post-norm

| 方案 | 公式 | 训练稳定性 | 梯度流动 | 最终效果 | 主流采用 | 结论 |
|------|------|-----------|---------|---------|---------|------|
| **Pre-norm** | x + F(Norm(x)) | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | GPT-3, LLaMA, Mistral, Qwen | ✅ 选用 |
| Post-norm | Norm(x + F(x)) | ⭐ | ⭐ | ⭐⭐ | 原始 Transformer, GPT-2 | ❌ |

**详细分析**:

原始的 Transformer（Vaswani et al. 2017）和 GPT-2 使用 Post-norm：

```python
# Post-norm (GPT-2 风格)
def forward(self, x):
    # 残差分支的输出先与输入相加，再做归一化
    attn_out = self.attn(x)
    x = self.ln_1(x + attn_out)     # Norm 在残差之后
    ffn_out = self.ffn(x)
    x = self.ln_2(x + ffn_out)      # Norm 在残差之后
    return x
```

Post-norm 的问题：
- 残差分支的输出逐层累积，不经归一化直接相加，导致深层梯度爆炸风险增大
- 学习率 warmup 几乎是必需的，否则训练早期容易发散
- 在 14 层规模下，Post-norm 的梯度不稳定性比 Pre-norm 明显

**Pre-norm（选用）**：

```python
# Pre-norm (LLaMA/Mistral 风格)
def forward(self, x):
    # 先对输入做归一化，残差分支的输出直接相加
    attn_out = self.attn(self.norm_1(x))
    x = x + attn_out                  # 残差路径未经归一化，直接传播
    ffn_out = self.ffn(self.norm_2(x))
    x = x + ffn_out                   # 残差路径未经归一化，直接传播
    return x
```

Pre-norm 的优势：
1. **训练稳定**: 归一化在残差分支的输入端，而非输出端。这使得残差路径（skip connection）的梯度可以"无障碍"地从顶层传播到底层
2. **warmup 宽容**: 即使没有学习率 warmup，Pre-norm 也通常不会发散（本项目仍保留 warmup 作为最佳实践）
3. **业界共识**: GPT-3 开始，几乎所有现代开源 LLM（LLaMA、Mistral、Qwen、DeepSeek）都使用 Pre-norm

```python
# Pre-norm 梯度流动示意
# 残差路径: ∂L/∂x_l = ∂L/∂x_L + ... (直接传播，无衰减)
# 分支路径: ∂L/∂x_l += 经 F(·) 的梯度 (可能衰减)
# → 总有一路梯度能"直达"浅层，避免了梯度消失
```

**最终选择: Pre-norm**。

---

### 2.3 激活函数：SwiGLU vs ReLU vs GELU

| 方案 | 公式 | 参数量（同等 d_ff） | 效果（同参数预算） | 计算开销 | 主流采用 | 结论 |
|------|------|-------------------|-------------------|---------|---------|------|
| **SwiGLU** | (x·W_g ⊙ SiLU(x·W_u))·W_d | 3×d_model×d_ff | ⭐⭐⭐ 最优 | ⭐⭐ | LLaMA, PaLM | ✅ 选用 |
| ReLU | max(0, x·W_1)·W_2 | 2×d_model×d_ff | ⭐ 基线 | ⭐⭐⭐ | GPT-2 | ❌ |
| GELU | x·Φ(x)·W_1·W_2 | 2×d_model×d_ff | ⭐⭐ | ⭐⭐ | BERT, GPT-3 | ❌ |
| GEGLU | (x·W_g ⊙ GELU(x·W_u))·W_d | 3×d_model×d_ff | ⭐⭐ | ⭐ | — | ❌ |

**详细分析**:

SwiGLU 来自 Google 的 PaLM 论文（Shazeer, 2020），核心创新是用门控机制替代简单的非线性激活：

```python
# ReLU FFN (GPT-2, d_ff=3072)
# 参数量: d_model × d_ff + d_ff × d_model = 2 × 768 × 3072 = 4,718,592
def relu_ffn(x):
    return F.relu(x @ W1 + b1) @ W2 + b2

# SwiGLU FFN (LLaMA, d_ff=3072)
# 参数量: 3 × d_model × d_ff = 3 × 768 × 3072 = 7,077,888
def swiglu_ffn(x):
    gate = x @ W_gate          # 门控投影
    up = x @ W_up               # 上投影
    return (F.silu(gate) * up) @ W_down  # 门控 + 下投影
```

**关键洞察——"等参数量比较"**：

SwiGLU 有三个权重矩阵（gate, up, down），而 ReLU 只有两个（W1, W2）。在 d_ff 相同的情况下，SwiGLU FFN 的参数量是 ReLU 的 1.5 倍（3 vs 2 个投影矩阵）。

为保证 ~157M 的总参数量不变，我们采用以下策略：
- 将 d_ff 从传统 GPT-2 的 4×d_model 保持为 3,072（仍是 4×768）
- SwiGLU 多用了一个投影矩阵，但 transformer 块总数（14 层）保持不变
- 最终总参数量约 157M，SwiGLU 的额外参数通过 embedding 共享（节省 ~24M）来抵消

**为什么 SwiGLU 比 ReLU 更好**：

1. **门控机制**: SwiGLU 的 gate 分支学习"哪些信息应该通过"，相当于一个可学习的过滤器。对于文言文中需要精细处理的虚词（"之"、"乎"、"者"、"也"），门控可以学会选择性保留或抑制信息
2. **更平滑的梯度**: SiLU（Sigmoid Linear Unit）= x·σ(x)，比 ReLU 的硬截断（x<0 时梯度为零）有更平滑的梯度流
3. **实证优越**: PaLM、LLaMA、Mistral 等模型的一致选择，在同等训练预算下 SwiGLU 的困惑度（PPL）比 ReLU/GELU 低 2-5%

**最终选择: SwiGLU**。

---

### 2.4 位置编码：RoPE vs Learned vs AliBi vs Sinusoidal

| 方案 | 原理 | 外推能力 | 额外参数 | 相对位置 | 实现复杂度 | 结论 |
|------|------|---------|---------|---------|-----------|------|
| **RoPE** | Q/K 向量乘以旋转矩阵，位置信息编码为旋转角度 | ⭐⭐⭐ 优秀 | 0 | ✅ 天然支持 | ⭐⭐ | ✅ 选用 |
| Learned | 可学习的绝对位置 embedding | ⭐ 差 | max_seq_len × d_model | ❌ | ⭐⭐⭐ | ❌ |
| AliBi | 在 attention score 上加线性偏置 | ⭐⭐⭐ 优秀 | 0 | ⚠️ 线性偏置 | ⭐⭐⭐ | ❌ |
| Sinusoidal | 固定正弦位置编码 | ⭐ 差 | 0 | ❌ | ⭐⭐⭐ | ❌ |

**详细分析**:

**RoPE（选用）**:

Rotary Position Embedding（Su et al., 2021）是目前 Decoder-only Transformer 的事实标准。其核心思想：

```python
# RoPE 数学原理
# 对 Q/K 向量的每对相邻维度 (2i, 2i+1) 施加二维旋转
# 旋转角度 θ_i 随维度递减: θ_i = base^(-2i/d), base=10000

# Q 的第 i 对维度旋转:
# [q_{2i}]'   = cos(θ_i · pos) · q_{2i} - sin(θ_i · pos) · q_{2i+1}
# [q_{2i+1}]' = sin(θ_i · pos) · q_{2i} + cos(θ_i · pos) · q_{2i+1}

# 关键性质: Q_m · K_n 的内积仅依赖于位置差 (m - n)
# 这意味着 RoPE 天然编码了相对位置关系
```

RoPE 的核心优势：

1. **天然相对位置**: 两个 token 的 attention score 仅取决于它们的相对距离 (m-n)，而非绝对位置 m 和 n。这比绝对位置编码更能捕捉文言文中长距离的句法依赖
2. **外推能力**: RoPE 可以外推到训练时未见过的序列长度（虽然需要调整 base frequency 或使用 NTK-aware scaling），at 12GB VRAM 约束下 2,048 已足够，但为未来扩展留有余地
3. **零额外参数**: 不增加可训练参数，全部计算是确定性的旋转操作
4. **逐层应用**: 每一层都独立应用 RoPE，使不同层可以关注不同粒度的位置信息

**Learned Absolute Position（GPT-2 原版）**:

```python
# Learned Position Embedding
self.wpe = nn.Embedding(max_seq_len, d_model)  # 2048 × 768 = 1.57M 参数
x = token_embed + position_embed  # 直接相加
```

❌ 缺点：
- 无法外推到训练时未见过的位置（位置 2049+ 无 embedding）
- 位置 embedding 与 token embedding 直接相加，将语义和位置信息混淆在一起
- 无相对位置信息——无法直接建模"token A 在 token B 前面 N 个位置"这样的关系

**AliBi（Attention with Linear Biases）**:

```python
# AliBi: 在 attention score 上加一个与距离成正比的负偏置
# score = Q·K^T / √d - m * |i - j|   (m 是每头不同的斜率)
```

AliBi 的优势是实现更简单（不需要旋转计算），外推能力极强。但 RoPE 在 LLaMA 系模型的广泛验证下已成为社区标准，对于本项目（学习目标包括理解主流 Transformer），实现 RoPE 更有学习价值。

**Sinusoidal（原始 Transformer）**:

GPT-2 不使用 Sinusoidal，而使用 Learned。Sinusoidal 的固定编码无外推能力，且绝对位置信息不如 Learned 灵活，不如 RoPE 的相对位置建模优雅。

**最终选择: RoPE**。与 LLaMA/Mistral/Qwen 等主流模型保持一致，零额外参数，天然相对位置，优秀外推能力。

---

### 2.5 归一化：RMSNorm vs LayerNorm

| 方案 | 公式 | 计算量 | 参数量 | 效果 | 主流采用 | 结论 |
|------|------|--------|--------|------|---------|------|
| **RMSNorm** | x · γ / RMS(x) | ⭐⭐ 少 | d_model | ⭐⭐⭐ | LLaMA, Mistral | ✅ 选用 |
| LayerNorm | (x - μ)/σ · γ + β | ⭐ 多 | 2×d_model | ⭐⭐⭐ | GPT-2, BERT | ❌ |

**详细分析**:

```python
# LayerNorm: 减均值 + 除标准差 + 缩放 + 平移
def layer_norm(x):
    mean = x.mean(dim=-1, keepdim=True)     # O(d)
    std = x.std(dim=-1, keepdim=True)        # O(d)
    return (x - mean) / std * γ + β          # 2d 个参数

# RMSNorm: 仅除 RMS（均方根），不减去均值，不添加平移参数
def rms_norm(x):
    rms = torch.sqrt((x ** 2).mean(dim=-1, keepdim=True))  # O(d)
    return x / rms * γ                       # d 个参数（无 β）
```

RMSNorm 相比 LayerNorm 的优势：

1. **更快**: 省略了均值计算和减法操作，约减少 15-20% 的归一化计算时间。在大规模训练中（~100K steps），累计节省可观
2. **更少参数**: 每层少 d_model（768）个 β 参数。14 层共节省 14 × 2 × 768 = 21,504 个参数，虽然总占比很小（~0.01%），但"去掉不必要的参数"符合设计原则
3. **效果不减**: LLaMA 论文的消融实验表明 RMSNorm 与 LayerNorm 在训练 loss 上无显著差异

**为什么去均值不影响效果**：

LayerNorm 的均值中心化（re-centering）被证明不是归一化效果的核心因素。真正关键的是方差缩放（re-scaling）——将激活值的尺度控制在合理范围内，避免梯度爆炸/消失。RMSNorm 保留了方差缩放，去掉的仅仅是均值中心化，而后者在 Pre-norm 架构下由残差连接的恒等映射路径部分弥补。

**最终选择: RMSNorm**。

---

### 2.6 Embedding 权重共享：Tied vs Untied

| 方案 | 参数量 | 效果 | 梯度 | 主流采用 | 结论 |
|------|--------|------|------|---------|------|
| **Tied** | vocab_size × d_model（一份） | 略低于 Untied（PPL 差 < 0.5） | 共享梯度，隐式正则化 | GPT-2, LLaMA (部分) | ✅ 选用 |
| Untied | 2 × vocab_size × d_model（两份独立） | 理论上限更高 | 独立更新 | 大模型常见 | ❌ 参数过多 |

```python
# Tied: Embedding 和 LM Head 共享同一权重矩阵
self.token_embedding = nn.Embedding(vocab_size, d_model)   # 32K × 768 = 24.6M
self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
self.lm_head.weight = self.token_embedding.weight          # ← 共享！

# Untied: 两份独立权重
self.token_embedding = nn.Embedding(vocab_size, d_model)   # 24.6M
self.lm_head = nn.Linear(d_model, vocab_size, bias=False)  # 24.6M（额外）
# Untied 总计: 24.6 + 24.6 = 49.2M（仅 embedding/lm_head 就占 31%）
```

Tied embeddings 节省 ~24.6M 参数。在 ~157M 的总预算下，这 24.6M 可以"重新分配"给更多的 transformer 层或更宽的 FFN，对模型效果的提升远大于 untied embeddings 带来的微弱 PPL 改善。

对于文言文小模型（157M），Tied embeddings 额外提供了一种隐式正则化：输入 embedding 和输出投影共享参数，迫使模型在输入和输出空间使用一致的语义表示，减少了过拟合风险（因为文言文训练数据有限）。

**最终选择: Tied embeddings**。

---

### 2.7 注意力实现：FlashAttention 后端

| 方案 | 原理 | 速度 | 内存 | 精度 | 结论 |
|------|------|------|------|------|------|
| **F.scaled_dot_product_attention** | PyTorch 2.0+ 统一后端，自动调度最优实现 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ✅ 选用 |
| 手动实现 | 自己写 Q·K^T / softmax / ×V | ⭐ | ⭐ | ⭐⭐⭐ | ❌ |
| xformers | Meta 的 FlashAttention 库 | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ❌ 额外依赖 |

**最终选择: `torch.nn.functional.scaled_dot_product_attention`**。

这是 PyTorch 2.0+ 引入的统一注意力后端，自动根据硬件和输入特性选择最优实现：
- Ampere+ GPU（A100, RTX 3090/4090）：自动使用 FlashAttention-2 内核
- 旧 GPU：使用 Memory Efficient Attention（xformers 风格）
- CPU：使用标准实现

无需额外依赖，API 简洁：

```python
import torch.nn.functional as F

attn_output = F.scaled_dot_product_attention(
    query, key, value,
    attn_mask=causal_mask,         # causal attention
    dropout_p=0.0,                 # 预训练不使用 attention dropout
    is_causal=True,                # 等价于传入上三角 mask
)
```

---

## 3. 参数量详细计算

### 3.1 逐项计算

所有计算使用 `d_model=768`, `n_layers=14`, `n_heads=12`, `d_ff=3072`, `vocab_size=32000`, `head_dim=64`, `max_seq_len=2048`。

```
═══════════════════════════════════════════════════════════════════
组件                          计算公式                 参数量         占比
═══════════════════════════════════════════════════════════════════

【Embedding 层】
Token Embedding               vocab_size × d_model
                              32,000 × 768              24,576,000   15.7%

【每层 TransformerBlock — 以单层计】
    
  MultiHeadAttention:
    Q 投影                    d_model × d_model
                              768 × 768                    589,824
    K 投影                    d_model × d_model
                              768 × 768                    589,824
    V 投影                    d_model × d_model
                              768 × 768                    589,824
    O 投影                    d_model × d_model
                              768 × 768                    589,824
    小计                                                2,359,296

  SwiGLU FFN:
    gate 投影                 d_model × d_ff
                              768 × 3,072                2,359,296
    up 投影                   d_model × d_ff
                              768 × 3,072                2,359,296
    down 投影                 d_ff × d_model
                              3,072 × 768                2,359,296
    小计                                                7,077,888

  RMSNorm × 2:
    Attention Norm            1 × d_model = 768                 768
    FFN Norm                  1 × d_model = 768                 768
    小计                                                    1,536

  单层 Block 合计                                       9,438,720

【全部 14 层 TransformerBlock】
14 × 9,438,720                                          132,142,080   84.3%

【Final RMSNorm】
输出归一化                   1 × d_model = 768                 768    0.0%

【LM Head】
与 Token Embedding 共享（Tied）                                 0    0.0%

═══════════════════════════════════════════════════════════════════
总计                                                    156,718,848  100.0%
═══════════════════════════════════════════════════════════════════
```

**总计: ~156.7M ≈ 157M 参数**。

### 3.2 参数分布可视化

```
                        ~157M 总参数量分布
  ┌─────────────────────────────────────────────────────────────────────────┐
  │                                                                         │
  │  Token Embedding  ██████████████▉                 15.7%    24,576,000   │
  │                                                                         │
  │  Attention ×14    ████████████████████▊           21.1%    33,030,144   │
  │  (Q/K/V/O 投影)                                                        │
  │                                                                         │
  │  SwiGLU FFN ×14   ████████████████████████████████████████████████████▊ │
  │  (gate/up/down)                          63.2%             99,090,432   │
  │                                                                         │
  │  RMSNorm ×29      ▏                               ~0.01%       22,272   │
  │  (14×2 + 1 final)                                                       │
  │                                                                         │
  │  LM Head           (与 Embedding 共享)              0%              0   │
  │                                                                         │
  ├─────────────────────────────────────────────────────────────────────────┤
  │  TOTAL                                             100%    156,718,848   │
  └─────────────────────────────────────────────────────────────────────────┘


              每层 TransformerBlock 参数分解 (共 ~9.44M / 层):

    ┌───────────────────────────────────────────────────────┐
    │                                                       │
    │     MultiHeadAttention:  2.36M  (25.0%)               │
    │     ┌─────────────────────────────────────────────┐   │
    │     │  Q proj:  768×768 =  589,824                 │   │
    │     │  K proj:  768×768 =  589,824                 │   │
    │     │  V proj:  768×768 =  589,824                 │   │
    │     │  O proj:  768×768 =  589,824                 │   │
    │     │  Total:                 2,359,296 params      │   │
    │     └─────────────────────────────────────────────┘   │
    │                                                       │
    │     SwiGLU FFN:         7.08M  (75.0%)               │
    │     ┌─────────────────────────────────────────────┐   │
    │     │  gate proj:  768×3072 = 2,359,296            │   │
    │     │  up proj:    768×3072 = 2,359,296            │   │
    │     │  down proj: 3072×768 = 2,359,296             │   │
    │     │  Total:                 7,077,888 params      │   │
    │     └─────────────────────────────────────────────┘   │
    │                                                       │
    │     RMSNorm ×2:            1,536  (0.02%)             │
    │     ┌─────────────────────────────────────────────┐   │
    │     │  Attn Norm γ:  768                            │   │
    │     │  FFN Norm γ:   768                            │   │
    │     │  Total:              1,536 params              │   │
    │     └─────────────────────────────────────────────┘   │
    │                                                       │
    │     ─────────────────────────────────────────────     │
    │     单层合计:            9,438,720 params              │
    └───────────────────────────────────────────────────────┘


              参数占比饼图（文字模拟）:

                     ┌──────────────────────┐
                     │     ╱‾‾‾‾‾‾‾‾╲       │
                     │   ╱             ╲     │
                     │  │   FFN  63.2%  │    │  ← SwiGLU 的 3 个投影矩阵
                     │  │               │    │     是参数的绝对主体
                     │  │               │    │
                     │   ╲    Attn     ╱     │  ← Q/K/V/O 投影
                     │    ╲   21.1%  ╱      │
                     │     ╲        ╱       │
                     │      ╲      ╱  Emb   │  ← Token Embedding
                     │       ╲    ╱  15.7%  │
                     │        ‾‾‾‾‾‾        │
                     └──────────────────────┘
                     RMSNorm ≈ 0.01%（不可见）
```

关键观察：

1. **FFN 层是参数主体**: SwiGLU 的 3 个投影矩阵占据了模型参数的 63.2%，这是 Transformer 架构的普遍特征——FFN 负责存储知识，Attention 负责信息路由
2. **Embedding 占 15.7%**: 在 32K vocab、768 维下合理。如果 vocab 扩大到 64K，embedding 将占 ~27%，挤压 Transformer 层参数
3. **Tied embeddings 节省 24.6M**: 若 untied，总参数将达 ~181M，超出 12GB VRAM 舒适范围

### 3.3 显存占用估算（BF16 混合精度训练）

以 AdamW 优化器为例，混合精度训练中每个参数的存储组成：

```
BF16 模型权重:    157M × 2 bytes =   314 MB     (forward pass)
BF16 梯度:       157M × 2 bytes =   314 MB     (backward pass)
FP32 Master 权重: 157M × 4 bytes =   628 MB     (optimizer step)
FP32 Adam m:      157M × 4 bytes =   628 MB     (1st moment)
FP32 Adam v:      157M × 4 bytes =   628 MB     (2nd moment)
─────────────────────────────────────────
模型状态总计:                      2,512 MB = ~2.5 GB
```

激活值（batch_size=8, seq_len=1024, d_model=768, 14 层）：

```
Attention 激活:   8 × 1024 × 768 × 14 × 4 bytes ≈ 352 MB (BF16)
FFN 激活:        8 × 1024 × 3072 × 14 × 4 bytes ≈ 1,408 MB (BF16，SwiGLU中间值)
残差 + Norm:     8 × 1024 × 768 × 14 × 2 bytes ≈ 176 MB
─────────────────────────────────────────
激活值总计:                                  约 1.9 GB
```

```
模型状态:      ~2.5 GB
激活值:        ~1.9 GB
CUDA Context:  ~1.0 GB
PyTorch 开销:  ~0.5 GB
───────────────────────
总计:           ~5.9 GB (训练峰值)
```

在 batch_size=8、seq_len=1024 和 gradient_accumulation_steps=4 的配置下，训练峰值显存约 **5.9 GB**。即使扩展到 seq_len=2048，激活值加倍到 ~3.8GB，总峰值约 **7.8 GB**，在 12GB VRAM 内安全运行，留有充足余量。

> 注：上述为估算值，实际值可能因 PyTorch 版本、CUDA 版本和驱动不同而波动 ±15%。建议通过 `torch.cuda.memory_stats()` 运行时监控。

---

## 4. 组件详细设计

### 4.1 RMSNorm

**位置**: `src/classic_chinese_llm/model/layers.py`

```python
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    与 LayerNorm 的区别:
    - 不减去均值（去中心化不是归一化的核心，可以省略）
    - 不添加平移参数 β（每层节省 d_model 个参数）
    - 仅保留缩放参数 γ（可学习的 gain）

    公式:
        RMSNorm(x) = x / RMS(x) · γ
        RMS(x) = sqrt(mean(x²) + ε)

    其中 γ 是可学习的缩放参数，ε 是数值稳定性常数。

    Args:
        d_model: 归一化维度
        eps: 防止除零的小常数（默认 1e-6）
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))  # γ：可学习缩放
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model) or (batch, d_model)
        # 计算 RMS: sqrt(mean(x²))
        # 保持 float32 计算以保证数值精度（BF16 下尤为重要）
        x_float = x.float()
        rms = torch.sqrt((x_float ** 2).mean(dim=-1, keepdim=True) + self.eps)
        # 归一化 + 缩放，转回原始 dtype
        # 在 float32 下完成乘法再 cast，保留最高精度
        return (x_float / rms * self.weight).to(x.dtype)
```

**数值稳定性说明**：

在 BF16 下，`x²` 可能因精度不足导致 RMS 计算误差。因此 `forward` 中先将输入转为 float32 计算 RMS 和除法，再转回原始 dtype。这个开销很小（仅归一化层），但能避免训练中后期的不稳定。

---

### 4.2 RoPE（Rotary Position Embedding）

**位置**: `src/classic_chinese_llm/model/layers.py`

```
              RoPE 旋转位置编码原理:

  核心思想: 将 Q/K 向量的相邻维度配对，看作 d/2 个二维向量，
           对每对向量施加与位置相关的旋转，使 attention score
           自然编码相对位置关系。

  ┌─────────────────────────────────────────────────────────────────┐
  │                  二维旋转示意（第 i 对维度）                       │
  │                                                                 │
  │              q[2i+1] (y轴)                                      │
  │                  │                                              │
  │                  │        ★ q'  (旋转后)                         │
  │                  │      ╱                                       │
  │                  │    ╱   θ = pos × θ_i                         │
  │                  │  ╱     (旋转角度 = 位置 × 频率)                 │
  │                  │╱                                               │
  │                  ├────────────★ q (旋转前)                        │
  │                  │           q[2i] (x轴)                         │
  │                  │                                              │
  │   旋转矩阵:                                                      │
  │   ┌                     ┐ ┌         ┐   ┌                      ┐│
  │   │ cos(θ)  -sin(θ)    │ │ q[2i]   │ = │ q'[2i]               ││
  │   │ sin(θ)   cos(θ)    │ │ q[2i+1] │   │ q'[2i+1]             ││
  │   └                     ┘ └         ┘   └                      ┘│
  └─────────────────────────────────────────────────────────────────┘


              RoPE 频率分配（head_dim=64 → 32 对旋转）:

  维度索引 i:    0          8          16         24         31
                 │          │          │          │          │
  θ_i 值:      1.00      0.56       0.24       0.06       0.01
  (base=10000)
                 │          │          │          │          │
  含义:       最快旋转   较快旋转    中速旋转    慢速旋转    最慢旋转
              (近处注意)                                   (远处注意)
                 │          │          │          │          │
  波长 ≈       2π×1      2π/.56     2π/.24     2π/.06     2π/.01
              ≈6 tokens ≈11 tokens ≈26 tokens ≈105 tokens ≈628 tokens
                 └──────────────────────────────────────────────┘
                  低维捕捉短距离依赖        高维捕捉长距离依赖

  在 max_seq_len=2048 时，最慢频率（θ_31≈0.01）旋转角度:
  pos=2048 × 0.01 = 20.5 rad ≈ 3.3 周期 → 仍在合理范围内，不会混淆位置
```

RoPE 的实现分为两部分：
1. **频率预计算**（`precompute_freqs_cis`）：在模型初始化时计算所有位置的 cos/sin 值
2. **旋转应用**（`apply_rotary_emb`）：在每次 forward 中将旋转应用到 Q/K 向量

```python
def precompute_freqs_cis(
    d_model: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
) -> torch.Tensor:
    """预计算 RoPE 频率的 cos/sin 值（复数形式）。

    RoPE 的核心思想:
    - 将 d_model 维的 Q/K 向量按相邻维度配对，看作 d_model/2 个二维向量
    - 每个二维向量对乘以一个旋转矩阵，旋转角度随维度递减
    - θ_i = theta^(-2i/d_model), i = 0, 1, ..., d_model/2 - 1
    - 第 i 对的旋转角度 = position × θ_i

    返回 freq_cis，形状为 (max_seq_len, d_model//2, 2):
    freq_cis[pos, i] = (cos(pos·θ_i), sin(pos·θ_i))

    使用时通过复数乘法一次性完成所有旋转:
    x_rotated = x_complex * freq_cis
    """
    # 计算频率: θ_i = 10000^(-2i/d), i = 0, 2, 4, ..., d-2
    freq = 1.0 / (
        theta ** (torch.arange(0, d_model, 2, dtype=torch.float32, device=device) / d_model)
    )
    # 所有位置的旋转角度: position × θ_i
    positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    angles = torch.outer(positions, freq)  # (max_seq_len, d_model//2)
    # 返回 (cos, sin) 对
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=-1)


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> torch.Tensor:
    """将预计算的 RoPE 频率应用到输入张量。

    Args:
        x: (batch, n_heads, seq_len, head_dim) 的 Q 或 K
        freqs_cis: (seq_len, head_dim//2, 2) 的预计算 (cos, sin)

    Returns:
        应用 RoPE 后的张量，形状与输入相同

    实现技巧:
    1. 将 x 重塑为 (..., head_dim//2, 2)，每对相邻维度看作 (real, imag)
    2. 复数乘法: (a+bi) * (cos+sin i) = (a·cos - b·sin) + (a·sin + b·cos)i
    3. 实部: x[..., 0] * cos - x[..., 1] * sin
    4. 虚部: x[..., 0] * sin + x[..., 1] * cos
    """
    seq_len = x.shape[2]
    freqs = freqs_cis[:seq_len]  # 截取到当前序列长度（推理时可能短于 max_seq_len）

    # 将 head_dim 拆为 (head_dim//2, 2) 以应用旋转
    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)

    cos = freqs[..., 0].unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim//2)
    sin = freqs[..., 1].unsqueeze(0).unsqueeze(0)

    # 复数旋转公式
    x_out = torch.empty_like(x_reshaped)
    x_out[..., 0] = x_reshaped[..., 0] * cos - x_reshaped[..., 1] * sin
    x_out[..., 1] = x_reshaped[..., 0] * sin + x_reshaped[..., 1] * cos

    return x_out.reshape_as(x).to(x.dtype)
```

**RoPE 频率的 base frequency（theta）选择**：

LLaMA 使用 theta=10000（原始 Transformer 的默认值）。在 seq_len=2048 的约束下，10000 的 base frequency 完全足够——最长的位置 2048，最慢的频率分量为 θ_d/2-1 = 10000^(-766/768) ≈ 0.00016，旋转角度 2048 × 0.00016 ≈ 0.33 弧度，仍在一个周期内，不会出现位置混淆。

如果未来需要扩展到 4096+ 序列长度，可以通过 NTK-aware scaling 调整 theta（如 LLaMA-2-7B 将 theta 调整为 500000），但现阶段保持 10000 即可。

---

### 4.3 MultiHeadAttention

**位置**: `src/classic_chinese_llm/model/layers.py`

```
               MultiHeadAttention 内部数据流（单头示意，实际 12 头并行）
              
  INPUT x: (batch, seq_len, 768)
       │
       ├──────────┬──────────┬──────────┐
       ▼          ▼          ▼          ▼
  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐
  │ Q Proj │ │ K Proj │ │ V Proj │ │ O Proj │    ← 四个独立的 Linear(768,768)
  │  768→  │ │  768→  │ │  768→  │ │  768←  │       bias=False
  │  768   │ │  768   │ │  768   │ │  768   │
  └───┬────┘ └───┬────┘ └───┬────┘ └───┬────┘
      │          │          │          │
      ▼          ▼          ▼          │
  Reshape: (B, 12, S, 64) 每个 head 独立   │
      │          │          │          │
      ▼          ▼          │          │
  ┌──────────────┐          │          │
  │  Apply RoPE  │  ← 旋转  │          │    ← Q 和 K 分别应用 RoPE
  │  对 Q 和 K    │   位置编码│          │       V 不应用（V 存"内容"非"位置"）
  └──────┬───────┘          │          │
         │                  │          │
         ▼                  ▼          ▼
  ┌───────────────────────────────────────────┐
  │         Scaled Dot-Product Attention       │
  │                                            │
  │   scores = Q @ K^T / √64  (√head_dim)     │
  │      ┌──────────────────────────┐          │
  │      │  Causal Mask 应用:        │          │    ← 自回归约束
  │      │  上三角区域 → -inf         │          │       token_i 只能看到
  │      │  ┌─────────────────┐     │          │       token_j (j ≤ i)
  │      │  │ Q₁·K₁  -∞   -∞ │     │          │
  │      │  │ Q₂·K₁ Q₂·K₂ -∞ │     │          │
  │      │  │ Q₃·K₁ Q₃·K₂ ...│     │          │
  │      │  └─────────────────┘     │          │
  │      └──────────────────────────┘          │
  │                                            │
  │   weights = softmax(scores)                │
  │   output  = weights @ V                    │
  │                                            │
  │  后端: F.scaled_dot_product_attention()    │    ← 自动选择 FlashAttention
  │        (is_causal=True)                     │       或 Memory-Efficient Attention
  └──────────────────────┬────────────────────┘
                         │
                         ▼
  Shape: (B, 12, S, 64) ──→ Merge heads → (B, S, 768)
                         │
                         ▼
                  ┌──────────────┐
                  │    O Proj    │  ← 多头输出拼接后投影回 d_model
                  └──────┬───────┘
                         │
                         ▼
  OUTPUT: (batch, seq_len, 768)


              注意力头的角色分工（概念示意）:

      Head 1     Head 2     Head 3    ...    Head 12
      ┌────┐    ┌────┐    ┌────┐           ┌────┐
      │ 主谓 │    │ 动宾 │    │ 修饰 │         │ 跨句 │
      │ 关系 │    │ 搭配 │    │ 关系 │   ...   │ 指代 │
      └────┘    └────┘    └────┘           └────┘
        └──────────┬──────────┘               │
                   │                          │
                   ▼                          ▼
            拼接为 (768,) 向量 → O 投影融合 → 下一层
```

```python
class MultiHeadAttention(nn.Module):
    """多头因果自注意力。

    实现要点:
    1. Q/K/V 投影: 分别通过线性层将 d_model 映射到 n_heads × head_dim
    2. RoPE: 仅对 Q 和 K 应用旋转位置编码（V 不使用位置信息）
    3. Scaled Dot-Product Attention: 通过 F.scaled_dot_product_attention（FlashAttention 后端）
    4. O 投影: 将多头输出拼接后映射回 d_model
    5. Causal Mask: 通过 is_causal=True 保证自回归（每个 token 只能看到自身及之前的 token）

    Args:
        d_model: 隐藏维度 (768)
        n_heads: 注意力头数 (12)
        head_dim: 每头维度 (64 = 768 / 12)
        dropout: attention dropout 概率（预训练阶段为 0.0）
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model 必须能被 n_heads 整除"

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Q/K/V 投影：将 d_model 映射到 n_heads × head_dim = d_model
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) 的输入
            freqs_cis: RoPE 预计算频率

        Returns:
            (batch, seq_len, d_model) 的输出
        """
        batch, seq_len, _ = x.shape

        # 1. Q/K/V 线性投影 + 重塑为多头
        # (batch, seq_len, d_model) → (batch, n_heads, seq_len, head_dim)
        q = self.q_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        # 2. 对 Q 和 K 应用 RoPE（V 不应用）
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        # 3. Scaled Dot-Product Attention (FlashAttention 后端)
        # is_causal=True 自动生成上三角 mask，屏蔽未来 token
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # 4. 合并多头 + O 投影
        # (batch, n_heads, seq_len, head_dim) → (batch, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, seq_len, self.d_model)
        return self.o_proj(attn_output)
```

**设计要点**：

1. **Q/K 投影无 bias**: 与 LLaMA/Mistral 一致，去掉 bias 减少参数量且对注意力机制无影响（LayerNorm/RMSNorm 已做了归一化）

2. **RoPE 仅应用于 Q 和 K**: V（Value）不需要位置信息——V 存储的是"内容"，应该与位置无关。Q 和 K 用于计算 attention score（"谁关注谁"），需要知道相对位置

3. **is_causal=True 代替手动 mask**: PyTorch 的 `is_causal=True` 等效于传入上三角的 `-inf` mask，且 FlashAttention 内核对此有专门优化，比手动构建 mask 更快

4. **训练时关闭 attention dropout**: `dropout=0.0`。在预训练阶段，attention dropout 不是必要的正则化手段——SwiGLU 的门控机制和 weight decay 已提供足够的正则化

---

### 4.4 SwiGLU FFN

**位置**: `src/classic_chinese_llm/model/layers.py`

```
              标准 FFN (ReLU) vs SwiGLU FFN 对比:

  ┌─────────────────────────────────────────────────────────────────────┐
  │                    标准 FFN (GPT-2 风格)                             │
  │                                                                     │
  │    x (768) ──→ W_up (768×3072) ──→ ReLU ──→ W_down (3072×768) ──→  │
  │                                     │                               │
  │                              硬截断: x<0 → 0                        │
  │                              梯度:  x<0 → 0 (dead ReLU 风险)         │
  │                                                                     │
  │    参数量: 2 × 768 × 3072 = 4,718,592                              │
  └─────────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────────┐
  │                    SwiGLU FFN (本项目选用)                            │
  │                                                                     │
  │                         ┌─→ gate_proj (768×3072) ──→ SiLU ──┐      │
  │                         │        "门控信号"              σ(x)·x │      │
  │    x (768) ──┬──────────┤                                      ├─→ ⊙ ──→ down_proj (3072×768) ──→
  │              │          │                                      │    ↑                               │
  │              │          └─→ up_proj (768×3072) ───────────────┘  逐元素                              │
  │              │                     "内容信号"                     乘法                                │
  │              │                                                                                       │
  │              │   门控机制:                                                                             │
  │              │   · gate 分支决定"哪些信息可以通过"（输出值范围 ≈ (0, 1)）                                │
  │              │   · up 分支提供"信息内容"                                                               │
  │              │   · gate ⊙ up 实现选择性信息过滤                                                        │
  │              │                                                                                       │
  │              │   SiLU(x) = x · σ(x)                                                                  │
  │              │   ┌────────────────────────────┐                                                      │
  │              │   │  x > 0:  σ(x) → 1          │   ≈ 接近线性（保留正信号）                              │
  │              │   │  x < 0:  σ(x) → 0          │   ≈ 接近截断（抑制负信号）                              │
  │              │   │  x ≈ 0:  平滑过渡           │   无 dead neuron 问题                                 │
  │              │   └────────────────────────────┘                                                      │
  │              │                                                                                       │
  │    参数量: 3 × 768 × 3072 = 7,077,888  (是标准 FFN 的 1.5 倍)                                        │
  └─────────────────────────────────────────────────────────────────────┘


              SwiGLU 门控示意（单个 token、单维度）:

                 输入: x[0] (768 维向量中的 1 个维度)
                   │
         ┌─────────┴─────────┐
         │                   │
         ▼                   ▼
    gate[0]              up[0]
    = W_gate[0]·x       = W_up[0]·x
    = 0.8               = 2.5
         │                   │
         ▼                   │
    SiLU(0.8)               │
    = 0.8 × σ(0.8)          │
    = 0.8 × 0.69            │
    = 0.55  ← "通过率 55%"   │
         │                   │
         └───────┬───────────┘
                 ▼
        gate ⊙ up = 0.55 × 2.5 = 1.375  ← 信息被门控选择性通过
                 │
                 ▼
        down_proj 将 3072 维映射回 768 维
```

```python
class SwiGLUFFN(nn.Module):
    """SwiGLU 门控前馈网络。

    标准 FFN (ReLU):  x → W_up → ReLU → W_down
    SwiGLU FFN:       x → W_gate → SiLU ─┐
                      x → W_up ──────────→ ⊙ → W_down

    门控机制: gate 分支学习"哪些信息可以通过"，与 up 分支的投影逐元素相乘
    这使得 FFN 可以有选择地激活或抑制不同的特征维度。

    SiLU (Sigmoid Linear Unit):  SiLU(x) = x · σ(x)
    - σ(x) 是 sigmoid 函数，输出 (0, 1) 之间的值
    - x > 0 时，σ(x) → 1，SiLU ≈ x（接近 linear）
    - x < 0 时，σ(x) → 0，SiLU → 0（接近 cutoff）
    - 不像 ReLU 硬截断，SiLU 在负值区域有平滑的过渡

    Args:
        d_model: 输入/输出维度 (768)
        d_ff: 中间层维度 (3,072 = 4 × d_model)
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        # gate 分支: 控制信号，SiLU 激活后取 (0, 1) 范围
        gate = F.silu(self.gate_proj(x))
        # up 分支: 内容信号，线性投影
        up = self.up_proj(x)
        # 门控: gate ⊙ up → 下投影
        return self.down_proj(gate * up)
```

**为什么 d_ff = 3072 (4 × d_model)**：

SwiGLU 有三个权重矩阵（gate, up, down），而 ReLU FFN 只有两个。如果使用传统的 d_ff = 4 × d_model：
- ReLU FFN: 2 × 768 × 3072 = 4.7M 参数/层
- SwiGLU FFN: 3 × 768 × 3072 = 7.1M 参数/层

SwiGLU 每层多 ~2.4M 参数，14 层多 ~33M 参数。这些额外参数的代价通过 embedding 共享（节省 ~24.6M）来部分抵消，总参数仍在 ~157M 范围内。关键点是：在相同参数预算下，SwiGLU 的效果优于 ReLU——这是 PaLM 论文的核心发现。

---

### 4.5 TransformerBlock

**位置**: `src/classic_chinese_llm/model/transformer.py`

```python
class TransformerBlock(nn.Module):
    """单个 Decoder-only Transformer 层。

    Pre-norm 残差结构:
        x = x + Attention(RMSNorm(x))
        x = x + SwiGLUFFN(RMSNorm(x))

    梯度流动分析:
    - 残差路径: x → + → x' → + → x''
        梯度可以通过残差路径"无障碍"传播
    - 注意力分支: x → RMSNorm → Attention → + → ...
        梯度经过 Attention 的反向传播，可能因 softmax 饱和而衰减
    - FFN 分支: 同注意力分支

    Pre-norm 的优势：即使注意力/FFN 分支的梯度衰减，
    残差路径仍能将梯度直接从深层传到浅层。

    Args:
        d_model: 隐藏维度
        n_heads: 注意力头数
        d_ff: FFN 中间维度
        dropout: 预训练阶段为 0.0
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLUFFN(d_model, d_ff)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            freqs_cis: RoPE 预计算频率（传递给 Attention）

        Returns:
            (batch, seq_len, d_model)
        """
        # Pre-norm: 先归一化再做注意力
        x = x + self.attn(self.attn_norm(x), freqs_cis)
        # Pre-norm: 先归一化再做 FFN
        x = x + self.ffn(self.ffn_norm(x))
        return x
```

---

### 4.6 TransformerLM（完整模型）

**位置**: `src/classic_chinese_llm/model/transformer.py`

```python
class TransformerLM(nn.Module):
    """完整的 Decoder-only Transformer 语言模型。

    架构:
        Input (token IDs)
          │
          ▼
        Token Embedding ─────────────── Tied Weights ──────────────┐
          │                                                         │
          ▼                                                         │
        ┌──────────────────────┐  × N (n_layers)                   │
        │  TransformerBlock    │                                    │
        │  ├─ RMSNorm          │                                    │
        │  ├─ MultiHeadAttn    │  ← RoPE (每层独立应用)              │
        │  ├─ RMSNorm          │                                    │
        │  └─ SwiGLUFFN        │                                    │
        └──────────────────────┘                                    │
          │                                                         │
          ▼                                                         │
        Final RMSNorm                                                │
          │                                                         │
          ▼                                                         │
        LM Head ─────────────── 权重共享 ───────────────────────────┘
          │
          ▼
        Logits (vocab_size)

    Args:
        config: ModelConfig 配置对象
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        # Token Embedding（权重与 LM Head 共享）
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer Blocks
        self.layers = nn.ModuleList([
            TransformerBlock(
                d_model=config.d_model,
                n_heads=config.n_heads,
                d_ff=config.d_ff,
                dropout=config.dropout,
            )
            for _ in range(config.n_layers)
        ])

        # Final RMSNorm（在 LM Head 之前）
        self.final_norm = RMSNorm(config.d_model)

        # LM Head（权重与 token_embedding 共享）
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight  # Tied weights

        # RoPE 频率预计算（所有层共享同一组频率）
        freqs_cis = precompute_freqs_cis(
            d_model=config.d_model // config.n_heads,  # head_dim, 非全局 d_model
            max_seq_len=config.max_seq_len,
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        # 初始化权重
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """LLaMA 风格的权重初始化。

        - nn.Linear: 使用小的正态分布初始化（std = 0.02 / sqrt(2 * n_layers)）
          参考 DeepNet 的思想：初始化时让残差分支的输出接近零，
          使得初始状态接近恒等映射，有利于训练早期的稳定性
        - nn.Embedding: 正态分布，std = d_model^(-0.5)
        """
        if isinstance(module, nn.Linear):
            std = 0.02 / math.sqrt(2 * self.config.n_layers)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.d_model ** -0.5)

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) 的 token ID 序列

        Returns:
            (batch, seq_len, vocab_size) 的 logits
        """
        # Token Embedding
        x = self.token_embedding(input_ids)  # (batch, seq_len, d_model)

        # 逐层 TransformerBlock
        for layer in self.layers:
            x = layer(x, self.freqs_cis)

        # Final RMSNorm → LM Head
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        return logits

    def get_num_params(self) -> int:
        """返回可训练参数总数。"""
        return sum(p.numel() for p in self.parameters())

    def get_device(self) -> torch.device:
        """返回模型所在设备。"""
        return next(self.parameters()).device
```

**关键设计点**：

1. **`register_buffer("freqs_cis", ..., persistent=False)`**: RoPE 频率不是可训练参数，而是固定缓冲区。`persistent=False` 表示它不参与 `state_dict`，避免在 checkpoint 中保存重复的确定性数据

2. **权重初始化（参考 DeepNet）**: 传统初始化（如 Xavier/Kaiming）在 14 层的 Pre-norm Transformer 上工作正常，但 LLaMA 风格的初始化（`std = 0.02 / sqrt(2 * n_layers)`）使残差分支在训练早期的输出接近零，模型初始状态接近恒等映射，更有利于深层网络的训练稳定性

3. **`lm_head.weight = token_embedding.weight`**: Python 赋值将两者的 `weight` 属性绑定到同一个 `Parameter` 对象。反向传播时梯度自动累加到这个共享参数上

4. **接收 `ModelConfig` 而非独立参数**: 遵循项目的依赖注入模式，模型所有超参数来自配置系统

---

### 4.7 Generator（生成器）

**位置**: `src/classic_chinese_llm/model/generation.py`

```
              自回归生成流程:

  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │   输入 Prompt: "子曰："                                          │
  │         │                                                       │
  │         ▼                                                       │
  │   ┌──────────┐     ┌──────────┐     ┌──────────┐               │
  │   │ Tokenize │ →   │  Model   │ →   │  Sample  │               │
  │   │ "子曰："  │     │ Forward  │     │ 1 Token  │               │
  │   │ [2, 子,  │     │ Pass     │     │  (eg."學")│              │
  │   │  曰, :]  │     │          │     │          │               │
  │   └──────────┘     └──────────┘     └────┬─────┘               │
  │                                          │                      │
  │         ┌────────────────────────────────┘                      │
  │         │  新 token 追加到序列末尾                               │
  │         ▼                                                       │
  │   序列: [2, 子, 曰, :, 學]                                      │
  │         │                                                       │
  │         │ 重复直到: 生成 EOS 或 达到 max_new_tokens              │
  │         ▼                                                       │
  │   输出: "學而時習之，不亦說乎？"                                   │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘


              四种采样策略对比（logits → token 的过程）:

  原始 logits (vocab_size=32000):
  ┌───────────────┬───────┬───────┬───────┬───────┬─────┬─────┬─────┐
  │ token:        │ "學"  │ "子"  │ "曰"  │ "而"  │ ... │ ... │ ... │
  │ logit:        │  5.2  │  2.1  │ -0.3  │  4.8  │     │     │     │
  │ softmax prob: │ 0.31  │ 0.05  │ 0.01  │ 0.23  │ ... │ ... │ ... │
  └───────────────┴───────┴───────┴───────┴───────┴─────┴─────┴─────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  Greedy (deterministic):                                        │
  │    argmax(logits) → "學" (概率最高的 token)                       │
  │    特点: 确定性、可复现、缺乏多样性                                │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  Temperature Sampling:                                          │
  │    logits' = logits / T                                         │
  │    T=0.5: 差异放大 → 更确定性                                     │
  │    T=1.0: 无变化                                                 │
  │    T=2.0: 差异缩小 → 更随机                                       │
  │    然后 softmax → multinomial 采样                                │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  Top-K Sampling (K=50):                                          │
  │    保留概率最高的 50 个 token，其余 → -inf                        │
  │    → 避免极低概率的"垃圾 token"被采样                             │
  └─────────────────────────────────────────────────────────────────┘

  ┌─────────────────────────────────────────────────────────────────┐
  │  Top-P (Nucleus) Sampling (P=0.9):                               │
  │    对 softmax 后的概率降序排列:                                    │
  │    "學"(0.31) + "而"(0.23) + "習"(0.12) + ... = 累积概率          │
  │    保留累积概率 ≤ 0.9 的最小 token 集合                           │
  │    → 动态调整候选数（高置信度时少选，低置信度时多选）              │
  └─────────────────────────────────────────────────────────────────┘


              Beam Search (num_beams=3) 示意图:

  Step 1:                  Step 2:                  Step 3:
  ┌────┬──────┐           ┌────┬──────┐           ┌────┬──────┐
  │ B1 │ "學" │─0.31     │ B1 │ "學而"│─0.31×0.45 │ B1 │"學而時"│─0.14×0.52
  ├────┼──────┤           ├────┼──────┤           ├────┼──────┤
  │ B2 │ "而" │─0.23     │ B2 │ "而學"│─0.23×0.30 │ B2 │"學而習"│─0.14×0.28
  ├────┼──────┤           ├────┼──────┤           ├────┼──────┤
  │ B3 │ "習" │─0.12     │ B3 │ "學  │─0.31×0.22 │ B3 │"而時習"│─0.07×0.38
  └────┴──────┘           │    │  習" │           └────┴──────┘
  保留 Top-3 序列          └────┴──────┘           保留 Top-3 序列
                          保留 Top-3 序列
```

```python
@dataclass
class GenerationConfig:
    """生成参数配置。"""

    max_new_tokens: int = 256          # 最大生成 token 数
    temperature: float = 1.0           # 温度（> 0，1.0 表示无缩放）
    top_k: int = 0                     # Top-K 采样（0 表示不使用）
    top_p: float = 1.0                 # Top-P (nucleus) 采样（1.0 表示不使用）
    repetition_penalty: float = 1.0    # 重复惩罚（1.0 表示不惩罚）
    num_beams: int = 1                 # Beam Search 的 beam 数（1 表示贪心/采样）
    do_sample: bool = True             # 是否使用采样（False = 贪心解码）
    eos_token_id: int = 3              # EOS token ID
    pad_token_id: int = 0              # PAD token ID


class Generator:
    """自回归文本生成器。

    支持的解码策略:
    - Greedy (temperature=0 或 do_sample=False): 每步选最高概率 token
    - Temperature Sampling: 在 softmax 之前将 logits 除以 temperature
    - Top-K Sampling: 仅保留概率最高的 K 个 token，其余设为零
    - Top-P (Nucleus) Sampling: 保留累积概率 ≥ p 的最小 token 集合
    - Repetition Penalty: 对已生成的 token 施加惩罚，降低重复概率
    - Beam Search: 维护 K 条候选序列，每次扩展时保留最优的 K 条

    支持 KV Cache 加速推理（逐 token 生成时可复用之前的 K/V）。

    设计上 Generator 只处理 tensor（token IDs），不绑定 tokenizer。
    文本 ↔ token 的转换由调用方负责，保持接口简洁和可组合性。

    Args:
        model: TransformerLM 模型实例（必须在目标设备上）。
    """

    def __init__(self, model: TransformerLM) -> None:
        self.model = model
        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig | None = None,
    ) -> torch.Tensor:
        """从 token IDs 生成文本（非流式）。

        Args:
            input_ids: (1, prompt_len) 或 (prompt_len,) 的 prompt token IDs。
            config: 生成配置。

        Returns:
            (1, prompt_len + new_tokens) 的完整序列（含 prompt）。
            (token_ids, decoded_text) 的元组
        """
        ...

    @torch.no_grad()
    def generate_stream(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig | None = None,
    ) -> Generator[int, None, None]:
        """从 token IDs 逐 token 流式生成。

        每次 yield 一个新生成的 token ID (int)。
        调用方负责将 token ID 解码为文本。
        """
        ...
```

**生成策略详解**:

采样策略拆分为独立辅助函数（见 `src/classic_chinese_llm/model/generation.py`）:

```python
# 1. Repetition Penalty: 降低已出现 token 的概率
def _apply_repetition_penalty(logits, generated, penalty) -> Tensor: ...
# 2. Temperature: 控制随机性
#    logits = logits / temperature
# 3. Top-K: 仅保留概率最高的 K 个 token
def _top_k_filter(logits, k) -> Tensor: ...
# 4. Top-P (Nucleus): 保留累积概率 ≥ p 的最小 token 集合
def _top_p_filter(logits, p) -> Tensor: ...
# 采样: softmax → multinomial 或 argmax
            if logits[token_id] > 0:
                logits[token_id] /= config.repetition_penalty  # 惩罚正概率
            else:
                logits[token_id] *= config.repetition_penalty  # 惩罚负 logit

    # 2. Temperature: 控制随机性
    if config.temperature > 0:
        logits = logits / config.temperature

    # 3. Top-K: 仅保留概率最高的 K 个 token
    if config.top_k > 0:
        top_k_values, _ = torch.topk(logits, config.top_k)
        logits[logits < top_k_values[-1]] = float("-inf")

    # 4. Top-P (Nucleus): 保留累积概率 ≥ p 的最小 token 集合
    if config.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # 找到累积概率超过 top_p 的位置
        sorted_mask = cumulative_probs > config.top_p
        # 至少保留一个 token
        sorted_mask[1:] = sorted_mask[:-1].clone()
        sorted_mask[0] = False
        logits[sorted_indices[sorted_mask]] = float("-inf")

    # 5. 采样
    if config.do_sample:
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1)
    else:
        return torch.argmax(logits, dim=-1, keepdim=True)
```

**Beam Search 策略**:

```python
# Beam Search 核心思想:
# 维护一个大小为 beam_size 的优先队列，保存当前最优的 K 条候选序列
# 每步:
#   1. 对 K 条序列分别计算下一步的所有 token 概率
#   2. 取所有 K × vocab_size 个候选中的 top-K 条序列
#   3. 如果某条序列生成了 EOS，将其标记为"完成"并从活跃队列中移除
#   4. 当所有序列都完成或达到 max_new_tokens 时停止
#
# 与采样的区别: Beam Search 是确定性的最大化策略（目标是找概率最高的序列）
# 采样是随机性的（目标是生成多样化的文本）
```

**KV Cache 设计**（推理加速）:

```python
# 无 KV Cache: 每次生成新 token 都要重新计算整个序列的 Attention
# 有 KV Cache: 缓存之前计算过的 K/V，新 token 只需计算增量

# 对于自回归生成（逐 token），KV Cache 将每步的 Attention 复杂度:
#   无 Cache: O(n²)  →  有 Cache: O(n)
# 对于 max_new_tokens=256，加速约 128 倍（2,304² vs 平均 1,152 步的 O(n))

class KVCache:
    """简单的 KV Cache 实现。
    
    存储每层的 K 和 V，每次 forward 只计算新 token 并追加。
    """
    def __init__(self, n_layers: int):
        self.keys: list[torch.Tensor | None] = [None] * n_layers
        self.values: list[torch.Tensor | None] = [None] * n_layers
    
    def update(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor):
        """追加新的 K/V 到缓存。"""
        if self.keys[layer_idx] is None:
            self.keys[layer_idx] = k
            self.values[layer_idx] = v
        else:
            self.keys[layer_idx] = torch.cat([self.keys[layer_idx], k], dim=2)
            self.values[layer_idx] = torch.cat([self.values[layer_idx], v], dim=2)
```

---

## 5. 模块结构

```
src/classic_chinese_llm/model/
├── __init__.py          # 导出: TransformerLM, Generator, GenerationConfig
├── layers.py            # RMSNorm, precompute_freqs_cis, apply_rotary_emb,
│                        #   MultiHeadAttention, SwiGLUFFN
├── transformer.py       # TransformerBlock, TransformerLM
└── generation.py        # GenerationConfig, Generator, KVCache
```

**模块依赖**:

```
layers.py  ←── 无内部依赖（仅依赖 torch.nn）
    │
transformer.py  ←── 依赖 layers.py, ModelConfig (config 模块)
    │
generation.py  ←── 依赖 transformer.py, tokenizer (PreTrainedTokenizerFast)
```

---

## 6. 接口定义汇总

### 6.1 层组件（layers.py）

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...

def precompute_freqs_cis(
    d_model: int, max_seq_len: int, theta: float = 10000.0, device=None
) -> torch.Tensor: ...

def apply_rotary_emb(
    x: torch.Tensor, freqs_cis: torch.Tensor
) -> torch.Tensor: ...

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0) -> None: ...
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor: ...

class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None: ...
    def forward(self, x: torch.Tensor) -> torch.Tensor: ...
```

### 6.2 完整模型（transformer.py）

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0) -> None: ...
    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor: ...

class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig) -> None: ...
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor: ...
    def get_num_params(self) -> int: ...
    def get_device(self) -> torch.device: ...
```

### 6.3 生成器（generation.py）

```python
@dataclass
class GenerationConfig:
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    num_beams: int = 1
    do_sample: bool = True
    eos_token_id: int = 3
    pad_token_id: int = 0

class Generator:
    def __init__(self, model: TransformerLM, tokenizer: PreTrainedTokenizerFast) -> None: ...
    def generate(self, prompt: str | list[int], config: GenerationConfig | None) -> tuple[list[int], str]: ...
    def generate_stream(self, prompt: str | list[int], config: GenerationConfig | None) -> Generator[str, None, None]: ...
```

---

## 7. 与其他模块的关系

```
                         Phase 3: Tokenizer
                 ┌──────────────────────────────┐
                 │  PreTrainedTokenizerFast      │
                 │  vocab_size = 32,000          │──┐
                 │  BOS=2, EOS=3, PAD=0         │  │
                 └──────────────────────────────┘  │
                                                   │ vocab_size 决定
                                                   │ Embedding 维度
                                                   ▼
┌──────────────────────────────────────────────────────┐
│                  Phase 4: Model                       │
│                                                       │
│  ModelConfig ──→ TransformerLM ──→ Logits             │
│  (config 模块)    │                                    │
│                   ├─→ state_dict() ──→ CheckpointState │
│                   │                   (utils 模块)     │
│                   └─→ Generator ──→ 文本               │
│                        │                               │
│                        └──→ tokenizer.decode()         │
└──────────────────────────────────────────────────────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
    Phase 5: SFT   Phase 6: Chat   Phase 7: Eval
    (训练层消费     (对话层调用      (评测层调用
     model 做       Generator       model 计算
     forward)       生成回复)        perplexity)
```

**上游依赖**:
- **Config 模块**: `ModelConfig` 提供所有模型超参数
- **Tokenizer 模块**: `vocab_size=32000` 约束 embedding 维度；`PreTrainedTokenizerFast` 为 Generator 提供编码/解码

**下游依赖**:
- **Training 模块**: 调用 `TransformerLM.forward()` 计算 logits → loss
- **Checkpoint 系统**: `model.state_dict()` → `CheckpointState.model_state_dict`
- **Inference / Chat**: 使用 `Generator` 进行自回归生成

---

## 8. 验证清单

### 形状正确性
- [ ] RMSNorm: 输入 (2, 128, 768) → 输出 (2, 128, 768)，std≈1
- [ ] RoPE: Q/K 旋转后形状不变，无 NaN/Inf
- [ ] MultiHeadAttention: (2, 128, 768) → (2, 128, 768)，因果 mask 正确
- [ ] SwiGLUFFN: (2, 128, 768) → (2, 128, 768)
- [ ] TransformerBlock: (2, 128, 768) → (2, 128, 768)
- [ ] TransformerLM: input_ids (2, 128) → logits (2, 128, 32000)

### 参数计算验证
- [ ] `model.get_num_params()` ≈ 156,718,848
- [ ] `token_embedding.weight is lm_head.weight`（Tied weights 验证）
- [ ] `lm_head.weight` 有梯度流动（`weight.grad is not None` after backward）

### 数值正确性
- [ ] 对长度为 4 的随机序列做 forward + backward，loss 下降
- [ ] 与相同配置的 HF GPT-2（d_model=768, n_layer=14, n_head=12）在同一输入下的 logits 在合理误差范围内（允许因 RoPE vs Learned Position 差异）
- [ ] BF16 forward 不产生 NaN/Inf

### 生成功能
- [ ] Greedy 解码：相同 prompt 产生确定性相同输出
- [ ] Temperature=0.0 等价于 Greedy
- [ ] Top-K 采样：vocab 中被保留的候选数 ≤ K
- [ ] Top-P 采样：保留候选的累积概率 ≥ p
- [ ] Repetition Penalty > 1.0 减少重复
- [ ] Beam Search (num_beams=3)：beam_size=3 维护正确
- [ ] Generator 不修改底层模型的参数（`torch.no_grad()`）

### 边界情况
- [ ] seq_len=1（单个 token）forward 正常
- [ ] seq_len=2048（最大长度）forward 不 OOM
- [ ] batch_size=1 和 batch_size=8 输出一致（无 batch 依赖）
- [ ] 空 prompt → 正确从 BOS 开始生成
