# 训练层设计文档

**所属阶段:** Phase 4 — 模型实现与预训练（核心阶段）
**涉及模块:** `src/classic_chinese_llm/training/` + `scripts/pretrain.py` + `scripts/finetune.py`
**日期:** 2026-07-28

---

## 1. 需求概述

### 1.1 功能需求

| 编号 | 需求 | 说明 |
|------|------|------|
| F1 | 通用 Trainer | 可复用的训练循环：梯度累积、混合精度（BF16）、学习率调度、梯度裁剪、checkpoint 断点续训 |
| F2 | 预训练（Causal LM） | 对全序列计算 next-token cross-entropy loss，数据来自清洗后的文言文 JSONL |
| F3 | 指令微调（SFT） | Chat template 格式化对话数据，loss 仅计算在 assistant 回复 token 上（label masking） |
| F4 | 回调系统 | 插件式回调钩子：`on_train_begin`、`on_step_end`、`on_eval_end`、`on_epoch_end`、`on_train_end` |
| F5 | Data Collator | 动态 batch 内 padding、attention mask 构建、SFT label masking（非 assistant 位置设为 -100） |
| F6 | 学习率调度 | Cosine warmup decay 调度器，支持 warmup 阶段线性增长 + 主体阶段余弦衰减 |
| F7 | 显存约束 | BF16 混合精度训练，batch_size=8、seq_len=1024-2048 时显存峰值 < 10GB |
| F8 | CLI 训练脚本 | `scripts/pretrain.py` 和 `scripts/finetune.py` 作为入口，接收 YAML 配置文件 |

### 1.2 非功能需求

- **中断续训**: 训练可在任意 step 中断（Ctrl+C / OOM / 硬件故障），restart 后从最近的 checkpoint 无缝恢复（loss 曲线连续）
- **显存安全**: 不出现 CUDA OOM（12GB 内），留有充足余量应对 PyTorch 版本差异
- **日志可追溯**: 每个 step 的 loss、LR、tokens/sec、GPU 显存记录到日志文件和终端
- **混合精度兼容**: 自动检测 BF16 支持（Ampere+ GPU），不支持时降级为 FP16，再不支持降级为 FP32
- **确定性**: 固定 seed 下，同一 checkpoint 恢复后的训练结果可复现（相同 step 产生相同的 loss）

### 1.3 训练配置（来自 `TrainingConfig` / `PretrainConfig` / `SFTConfig`）

| 配置项 | 预训练值 | SFT 值 | 说明 |
|--------|---------|--------|------|
| `batch_size` | 8 | 4 | 单步 batch size（受显存限制） |
| `gradient_accumulation_steps` | 4 | 8 | 梯度累积步数 |
| 有效 batch size | 32 | 32 | batch_size × grad_accum |
| `learning_rate` | 3e-4 | 1e-4 | 峰值学习率 |
| `weight_decay` | 0.1 | 0.01 | AdamW 权重衰减 |
| `warmup_steps` | 1,000 | 100 | 学习率 warmup 步数 |
| `max_steps` | 100,000 | — | 预训练总步数（与 max_epochs 互斥） |
| `max_epochs` | — | 3 | SFT 总 epoch 数（与 max_steps 互斥） |
| `eval_every` | 500 | 200 | 评估间隔（步数） |
| `save_every` | 2,000 | 500 | checkpoint 保存间隔（步数） |
| `max_checkpoints` | 5 | 3 | 保留的 checkpoint 数量 |
| `optimizer.name` | adamw | adamw | 优化器 |
| `optimizer.betas` | (0.9, 0.95) | (0.9, 0.95) | Adam betas |
| `scheduler.name` | cosine | cosine | LR 调度器 |
| `scheduler.min_lr` | 3e-5 | 3e-5 | 最小学习率（最大 LR 的 10%） |

---

## 2. 方案选型与对比

### 2.1 训练框架：自建 Trainer vs HF Trainer

| 方案 | 灵活性 | 学习价值 | 代码量 | 文言文定制 | 依赖 | 结论 |
|------|--------|---------|--------|-----------|------|------|
| **自建 Trainer** | ⭐⭐⭐ | ⭐⭐⭐ 核心学习目标 | ⭐⭐ ~500 行 | ✅ 完全可控 | 仅 torch + accelerate | ✅ 选用 |
| HF Trainer | ⭐⭐ | ⭐ | ⭐⭐ 最少 | ⚠️ 需要适配 | transformers | ❌ |
| PyTorch Lightning | ⭐⭐ | ⭐⭐ | ⭐⭐ | ⚠️ 需要适配 | lightning | ❌ |

**详细分析**:

**自建 Trainer（选用）**：

本项目的第一性原理是从零理解 Transformer 的完整训练流程。自建 Trainer 的学习价值体现在：

1. **理解训练循环的每一个细节**: 梯度累积如何影响 loss 的 scale、autocast 的 scope 如何划定、LR scheduler 的 step 时机（per-step vs per-epoch）、`optimizer.zero_grad()` 与梯度累积的交互
2. **精确控制**: 文言文 SFT 的 label masking 需要与 ChatML 模板深度绑定，自建 Data Collator 比覆写 HF 的 collator 更直接
3. **零额外依赖**: 项目已引入 `accelerate`（混合精度 + 设备管理），不需要 `transformers` 的 Trainer

```python
# 自建 Trainer 的核心循环（精简示意）
def train(self):
    self.model.train()
    for step in range(self.start_step, self.max_steps):
        # 1. 梯度累积循环
        self.optimizer.zero_grad()
        accum_loss = 0.0
        for micro_step in range(self.grad_accum_steps):
            batch = next(self.train_iter)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = self.model(batch["input_ids"])
                loss = F.cross_entropy(logits, batch["labels"])
                loss = loss / self.grad_accum_steps  # ← 关键！归一化
            loss.backward()
            accum_loss += loss.item()
        
        # 2. 梯度裁剪 + 优化器步进
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        # 3. 定期评估 + 保存
        if step % self.eval_every == 0:
            val_loss = self._evaluate()
        if step % self.save_every == 0:
            self._save_checkpoint(step)
```

**HF Trainer**：

```python
# HF Trainer 封装了训练循环，但难以深度定制
from transformers import Trainer, TrainingArguments
trainer = Trainer(
    model=model, args=TrainingArguments(...),
    train_dataset=train_dataset, data_collator=collator,
)
trainer.train()
# 问题1: model 不是 HF PreTrainedModel，需要实现特定接口
# 问题2: ChatML label masking 需要复杂的 data collator 覆写
# 问题3: 学习目标——理解训练循环——被屏蔽了
```

**PyTorch Lightning**：

类似 HF Trainer，引入了 `LightningModule` 的概念和大量的回调/日志系统，学习曲线高，且与本项目的学习目标（从零理解训练流程）冲突。

**最终选择: 自建 Trainer**。

---

### 2.2 混合精度：BF16 vs FP16 vs FP32

| 方案 | 动态范围 | 精度（尾数） | Loss Scaling | 硬件要求 | 训练稳定性 | 结论 |
|------|---------|-------------|-------------|---------|-----------|------|
| **BF16** | 大（8-bit 指数，同 FP32） | 7-bit | 不需要 | Ampere+ (RTX 30/40 系列, A100) | ⭐⭐⭐ 极佳 | ✅ 首选 |
| FP16 | 小（5-bit 指数） | 10-bit | 需要 GradScaler | Volta+ (几乎所有 CUDA GPU) | ⭐⭐ 需要调参 | ⚠️ 降级方案 |
| FP32 | 最大 | 23-bit | 不需要 | 所有设备 | ⭐⭐⭐ 最佳但不现实 | ❌ 显存不足 |

**详细分析**:

BF16（bfloat16）是 Google TPU 和 NVIDIA Ampere 架构引入的半精度格式。与 FP16 的关键区别在于指数位数：

```
FP32: 1 符号位 + 8 指数位 + 23 尾数位  →  动态范围 ~10^-38 到 10^38
BF16: 1 符号位 + 8 指数位 + 7 尾数位   →  动态范围同 FP32，但精度降低
FP16: 1 符号位 + 5 指数位 + 10 尾数位 →  动态范围 ~10^-8 到 65504
```

BF16 的核心优势：

1. **不需要 Loss Scaling**: BF16 的 8-bit 指数与 FP32 相同，动态范围覆盖了深度学习中绝大多数梯度和激活值的范围。FP16 的 5-bit 指数很容易溢出（gradient > 65504 → Inf），因此需要 `GradScaler` 动态调整 loss 的 scale
2. **训练更稳定**: 在 SwiGLU 的门控值（可能接近 0 或 1）、RMSNorm 的 RMS 计算（可能极小的方差）等场景中，BF16 的动态范围确保不出现 Inf/NaN
3. **硬件普适性**: 项目目标硬件（12GB+ NVIDIA GPU）几乎都是 RTX 30 系列以上，全部支持 BF16

降级策略：

```python
# 自动选择最优 dtype
from classic_chinese_llm.utils.device import detect_device, get_dtype

device_info = detect_device()
dtype = get_dtype(device_info, preference="bf16")
# 优先级: BF16 > FP16 > FP32
```

**最终选择: BF16 首选，自动降级 FP16/FP32**。

---

### 2.3 学习率调度：Cosine vs Linear vs Constant

| 方案 | 收敛速度 | 最终 Loss | 超参数 | 调优难度 | 主流采用 | 结论 |
|------|---------|----------|--------|---------|---------|------|
| **Cosine Warmup Decay** | ⭐⭐⭐ | ⭐⭐⭐ | min_lr, warmup_steps | ⭐⭐ | Chinchilla, LLaMA, GPT-3 | ✅ 选用 |
| Linear Warmup Decay | ⭐⭐ | ⭐⭐ | min_lr, warmup_steps | ⭐⭐ | — | ❌ |
| Constant with Warmup | ⭐⭐ | ⭐ | warmup_steps | ⭐⭐⭐ | — | ❌ |
| Cosine (no warmup) | ⭐ | ⭐ | min_lr | ⭐⭐⭐ | — | ❌ 不稳定 |

**详细分析**:

```python
# Cosine Warmup Decay 的完整公式:
# Phase 1 (Warmup): t ∈ [0, warmup_steps)
#   lr(t) = peak_lr × t / warmup_steps
# Phase 2 (Decay): t ∈ [warmup_steps, max_steps)
#   progress = (t - warmup_steps) / (max_steps - warmup_steps)
#   lr(t) = min_lr + 0.5 × (peak_lr - min_lr) × (1 + cos(π × progress))
```

**为什么需要 Warmup**:

在训练的初始阶段，模型权重是随机初始化的，梯度方向不稳定。如果直接用峰值学习率：
- 梯度的协方差矩阵尚未稳定，大步长更新可能导致模型进入不理想的参数空间
- Pre-norm 架构虽然比 Post-norm 对 warmup 更宽容，但 1,000 步 warmup 仍是最佳实践

在预热阶段，学习率从 0（或接近 0）线性增长到峰值。这给了 Adam 优化器的时间积累梯度二阶矩估计（v），让后续的大步长更新更稳定。

**为什么 Cosine Decay 优于 Linear**:

Cosine 衰减在训练初期缓慢下降（保持较高的学习率探索参数空间），在训练末期快速下降（精细收敛到局部最优）。Linear 衰减的恒定下降速度在训练初期可能过快地降低了探索能力，在训练末期又不够精细。

**最终选择: Cosine Warmup Decay**。

---

### 2.4 SFT Loss 策略：仅 Assistant vs 全序列

| 方案 | Loss 计算位置 | 训练效率 | 回答质量 | 实现复杂度 | 结论 |
|------|-------------|---------|---------|-----------|------|
| **仅 Assistant** | 只在 `<\|assistant\|>...<\|end\|>` 段计算 loss | ⭐⭐ | ⭐⭐⭐ | ⭐⭐ | ✅ 选用 |
| 全序列 | 所有 token（含 system/user/特殊 token） | ⭐⭐⭐ | ⭐ | ⭐⭐⭐ | ❌ |
| 仅 Completion | 只在最后一个 assistant 段计算 | ⭐⭐ | ⭐⭐ | ⭐⭐ | ❌ |

**详细分析**:

SFT 的 ChatML 格式化后序列示意：

```
<|bos|> <|system|> 你是一个文言文专家... <|end|>
        <|user|> 请用文言文写一首诗 <|end|>
        <|assistant|> 春江潮水连海平... <|end|> <|eos|>
```

**仅 Assistant（选用）**:

```
Token:  <bos> <sys> 你 是 ... <end> <user> 请 用 ... <end> <asst> 春 江 ... <end> <eos>
Label:   -100  -100 -100 ... -100  -100  -100 -100 ... -100  -100  春 江 ... <end> <eos>
                                                                         ↑
                                                       仅此段参与 loss 计算
```

- 模型只学习"给定 system prompt + user 问题 → 生成 assistant 回答"
- -100 是 PyTorch `CrossEntropyLoss` 的默认 `ignore_index`，这些位置的 loss 梯度为零
- 避免模型去学习预测 system prompt 和 user 问题的格式（这些是固定的模板，不应被"生成"）

**全序列（不选用）**：

```python
# Label = input_ids 整体右移一位
# 问题: 模型被训练为"续写模板格式"，而非"根据问题生成回答"
# 在推理时，system/user 是预设的，不需要模型生成
```

**仅 Completion（不选用）**：

仅最后一个 assistant 段的变体，丢失了多轮对话中早期 assistant 回复的学习信号。

**最终选择: 仅 Assistant token 计算 loss**。

---

### 2.5 回调设计：类钩子 vs 事件总线 vs 函数注册

| 方案 | 类型安全 | 钩子顺序 | IDE 支持 | 实现复杂度 | 结论 |
|------|---------|---------|---------|-----------|------|
| **类继承 + 方法覆写** | ✅ | ✅ 显式调用顺序 | ✅ 自动补全 | ⭐ | ✅ 选用 |
| 事件总线 | ❌ | ⚠️ 依赖注册顺序 | ❌ | ⭐⭐⭐ | ❌ |
| 函数注册 | ⚠️ | ⚠️ | ⚠️ | ⭐⭐ | ❌ |

**最终选择: 类继承 + 方法覆写**（类似 Keras/Lightning）。基类提供空的钩子方法，子类选择性覆写。Trainer 在特定时机显式调用钩子，调用顺序由 Trainer 代码控制。

```python
class Callback(ABC):
    """回调基类。所有钩子默认为 no-op。"""
    def on_train_begin(self, trainer: "Trainer") -> None: ...
    def on_step_end(self, trainer: "Trainer", loss: float, lr: float) -> None: ...
    def on_eval_end(self, trainer: "Trainer", metrics: dict[str, float]) -> None: ...
    def on_epoch_end(self, trainer: "Trainer") -> None: ...
    def on_train_end(self, trainer: "Trainer") -> None: ...
```

---

## 3. 显存预算详细分析

### 3.1 训练峰值显存组成（12GB VRAM 内）

以 AdamW 优化器、BF16 混合精度、batch_size=8、seq_len=1024、d_model=768、n_layers=14 为例：

```
═══════════════════════════════════════════════════════════════
组件                          计算方式                    显存
═══════════════════════════════════════════════════════════════

【模型状态】
BF16 模型权重                 157M × 2 bytes           314 MB
BF16 梯度（反向传播后）       157M × 2 bytes           314 MB
FP32 Master 权重（优化器用）  157M × 4 bytes           628 MB
FP32 Adam m（一阶动量）       157M × 4 bytes           628 MB
FP32 Adam v（二阶动量）       157M × 4 bytes           628 MB
─────────────────────────────────────────────────────────
模型状态小计                                         2,512 MB

【激活值】(batch=8, seq_len=1024)
Attention 中间值              8×1024×768×14×2B×2     352 MB
  (Q/K/V 投影输出 × 层数 × BF16)
SwiGLU 中间值                 8×1024×3072×14×2B      704 MB
  (gate/up 投影 + element-wise 乘积, BF16)
残差连接 + RMSNorm 输出       8×1024×768×14×2B×2     352 MB
  (每层 2 个残差路径, BF16)
Softmax 输出 (attn weights)   8×12×1024×1024×14×2B   176 MB (估算)
─────────────────────────────────────────────────────────
激活值小计                                         ~1,584 MB

【其他】
CUDA Context / cuBLAS 工作区                           ~800 MB
PyTorch 分配器缓存                                      ~500 MB
─────────────────────────────────────────────────────────
其他小计                                             ~1,300 MB

═══════════════════════════════════════════════════════════════
训练峰值总计 (seq_len=1024)                         ~5,400 MB
训练峰值总计 (seq_len=2048, 激活值×2)               ~7,000 MB
═══════════════════════════════════════════════════════════════
```

**安全余量**: 12GB - 7.0GB = **5.0GB** 余量。

即使在 worst-case（seq_len=2048 + PyTorch 内存碎片 + 驱动开销），12GB VRAM 仍足够。如果实际运行中出现 OOM，优先调整策略：

1. 降低 `batch_size`: 8 → 4（激活值减半）
2. 降低 `seq_len`: 2048 → 1024（激活值减半）
3. 启用 `torch.cuda.empty_cache()` 定期清理碎片
4. 使用 `gradient_checkpointing`（用计算换显存，但本项目 ~157M 小模型通常不需要）

### 3.2 混合精度训练的内存节省

```
FP32 全精度训练:
  模型权重 + 梯度: 157M × 4B × 2 = 1,256 MB
  优化器状态:      157M × 4B × 2 = 1,256 MB
  总计模型状态:                     2,512 MB  (vs BF16 的 2,512 MB??)

等等，这个计算有问题。让我重新算。

FP32 Training:
  模型权重:        157M × 4 bytes = 628 MB
  梯度:            157M × 4 bytes = 628 MB
  Adam m:          157M × 4 bytes = 628 MB
  Adam v:          157M × 4 bytes = 628 MB
  模型状态总计:                     2,512 MB  (相同！)

关键区别在于激活值:
  FP32 激活值 ≈ BF16 激活值 × 2（所有中间张量翻倍）：
  激活值:          ~1,584 × 2 ≈ 3,168 MB
  FP32 总计:       ~2,512 + 3,168 + 1,300 ≈ 7,000 MB (seq_len=1024)
  BF16 总计:       ~2,512 + 1,584 + 1,300 ≈ 5,400 MB (seq_len=1024)

BF16 相比 FP32 节省约 23%（主要是激活值减半）
BF16 相比 FP16 在模型状态上相同，但训练更稳定（不需要 loss scaling）
```

> 注：BF16 的模型状态与 FP32 相同，因为 optimizer 始终在 FP32 下维护 master weights。混合精度的核心节省来自**激活值减半**——前向传播的中间结果以 BF16 存储而非 FP32。

---

## 4. 组件详细设计

### 4.1 Trainer（通用训练循环）

**位置**: `src/classic_chinese_llm/training/trainer.py`

```python
class Trainer:
    """通用训练循环。

    职责:
    1. 管理训练状态（global_step, epoch, best_loss）
    2. 驱动训练循环：梯度累积 → 混合精度 → 梯度裁剪 → 优化器步进 → LR 调度
    3. 定期评估 + checkpoint 保存
    4. 调度回调钩子
    5. 处理中断信号（Ctrl+C → 优雅保存并退出）

    设计原则:
    - 与具体任务（Pretrain/SFT）解耦——loss 计算由调用方注入
    - 与模型架构解耦——仅依赖 nn.Module 的 forward + parameters 接口
    - 与 checkpoint 系统对接——使用 utils/checkpoint.py 的 CheckpointState

    Args:
        model: TransformerLM 模型实例
        config: PretrainConfig 或 SFTConfig
        train_dataloader: 训练数据加载器
        val_dataloader: 验证数据加载器（可为 None）
        device_info: 设备信息（来自 utils/device.py）
        checkpoint_dir: checkpoint 保存目录
        callbacks: 回调列表
        resume: 是否自动从最新 checkpoint 恢复
    """

    def __init__(
        self,
        model: nn.Module,
        config: Settings,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None,
        device_info: DeviceInfo,
        checkpoint_dir: Path,
        callbacks: list[Callback] | None = None,
        resume: bool = True,
    ) -> None:
        self.model = model
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device_info = device_info
        self.checkpoint_dir = checkpoint_dir
        self.callbacks = callbacks or []

        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self._should_stop = False

        # 优化器
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()

        # 混合精度
        self.dtype = get_dtype(device_info, preference=config.dtype)
        self.scaler = GradScaler() if self.dtype == torch.float16 else None

        # Loss 函数
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

        # 恢复
        if resume:
            self._try_resume()

    def _create_optimizer(self) -> torch.optim.AdamW:
        """创建 AdamW 优化器。

        - 不衰减 bias 和 RMSNorm 的 weight（它们是 1D 参数）
        - 仅衰减 2D 参数（Linear 层的 weight）
        - betas=(0.9, 0.95) 参考 LLaMA 的配置
        """
        opt_cfg = self.config.optimizer
        train_cfg = self.config.training

        # 分组：decay / no_decay
        decay_params = []
        no_decay_params = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # bias 和 1D 参数（RMSNorm weight）不衰减
            if param.ndim <= 1 or "bias" in name or "norm" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": train_cfg.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        return torch.optim.AdamW(
            param_groups,
            lr=train_cfg.learning_rate,
            betas=opt_cfg.betas,
            eps=opt_cfg.eps,
        )

    def _create_scheduler(self) -> torch.optim.lr_scheduler.LambdaLR:
        """创建 Cosine Warmup 学习率调度器。"""
        train_cfg = self.config.training
        sch_cfg = self.config.scheduler

        warmup = train_cfg.warmup_steps
        total = (
            train_cfg.max_steps
            if train_cfg.max_steps is not None
            else train_cfg.max_epochs * len(self.train_dataloader)  # type: ignore[operator]
        )

        def lr_lambda(step: int) -> float:
            """Cosine warmup decay 的核心公式。"""
            if step < warmup:
                # Phase 1: 线性 warmup
                return step / max(1, warmup)
            # Phase 2: Cosine decay
            progress = (step - warmup) / max(1, total - warmup)
            if progress >= 1.0:
                return sch_cfg.min_lr / train_cfg.learning_rate
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            ratio = sch_cfg.min_lr / train_cfg.learning_rate
            return ratio + (1.0 - ratio) * cosine_decay

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(self, loss_fn: Callable[[nn.Module, dict], torch.Tensor]) -> None:
        """主训练循环。

        Args:
            loss_fn: 损失计算函数，签名为 (model, batch) -> scalar loss
                     由 Pretrain/SFT 各自提供，实现任务特定的 loss 计算
        """
        self._notify_callbacks("on_train_begin")

        train_cfg = self.config.training
        accum_steps = train_cfg.gradient_accumulation_steps

        while not self._should_stop:
            self.epoch += 1

            for batch in self.train_dataloader:
                self.global_step += 1

                # ── 梯度累积循环 ──
                self.optimizer.zero_grad()
                step_loss = 0.0

                for micro_step in range(accum_steps):
                    # 如果 DataLoader 耗尽，跳出
                    # (实际实现中可能需要更复杂的 micro-batch 切分逻辑)

                    # 混合精度前向传播
                    with torch.cuda.amp.autocast(
                        device_type=self.device_info.device.type,
                        dtype=self.dtype,
                    ):
                        loss = loss_fn(self.model, batch)
                        loss = loss / accum_steps  # ← 归一化

                    # 反向传播（autocast 外）
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    step_loss += loss.item()

                # ── 梯度裁剪 ──
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )

                # ── 优化器步进 ──
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.scheduler.step()

                current_lr = self.scheduler.get_last_lr()[0]

                # ── 回调: step 结束 ──
                self._notify_callbacks(
                    "on_step_end", loss=step_loss, lr=current_lr
                )

                # ── 定期评估 ──
                if self.global_step % train_cfg.eval_every == 0:
                    metrics = self._evaluate(loss_fn)
                    self._notify_callbacks("on_eval_end", metrics=metrics)

                # ── 定期保存 ──
                if self.global_step % train_cfg.save_every == 0:
                    self._save(tag=f"step_{self.global_step}")

                # ── 终止条件 ──
                if train_cfg.max_steps and self.global_step >= train_cfg.max_steps:
                    self._should_stop = True
                    break

            if train_cfg.max_epochs and self.epoch >= train_cfg.max_epochs:
                break

        self._notify_callbacks("on_train_end")

    @torch.no_grad()
    def _evaluate(
        self, loss_fn: Callable[[nn.Module, dict], torch.Tensor]
    ) -> dict[str, float]:
        """在验证集上计算 loss。"""
        if self.val_dataloader is None:
            return {"val_loss": float("nan")}

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0

        for batch in self.val_dataloader:
            logits = self.model(batch["input_ids"])
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                batch["labels"].view(-1),
                ignore_index=-100,
            )
            total_loss += loss.item() * batch["labels"].numel()
            total_tokens += (batch["labels"] != -100).sum().item()

        self.model.train()

        avg_loss = total_loss / max(total_tokens, 1)
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self._save(tag="best")

        return {"val_loss": avg_loss, "best_loss": self.best_loss}

    def _save(self, tag: str) -> Path:
        """保存 checkpoint。"""
        state = CheckpointState(
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            global_step=self.global_step,
            epoch=self.epoch,
            best_loss=self.best_loss,
            rng_state={
                "torch": torch.random.get_rng_state(),
                "cuda": torch.cuda.random.get_rng_state_all()
                if torch.cuda.is_available()
                else None,
            },
            metadata={
                "config": self.config.model_dump(),
                "dtype": str(self.dtype),
            },
        )
        return save_checkpoint(
            state,
            self.checkpoint_dir,
            tag=tag,
            max_checkpoints=self.config.training.max_checkpoints,
        )

    def _try_resume(self) -> None:
        """尝试从最新 checkpoint 恢复训练。"""
        latest = find_latest_checkpoint(self.checkpoint_dir)
        if latest is None:
            logger.info("未找到 checkpoint，从头开始训练")
            return

        logger.info("正在恢复训练: %s", latest)
        state = load_checkpoint(latest, map_location=self.device_info.device)

        self.model.load_state_dict(state.model_state_dict)
        self.optimizer.load_state_dict(state.optimizer_state_dict)  # type: ignore[arg-type]
        self.global_step = state.global_step
        self.epoch = state.epoch
        self.best_loss = state.best_loss

        # 恢复 RNG 状态
        if state.rng_state:
            torch.random.set_rng_state(state.rng_state["torch"])
            cuda_state = state.rng_state.get("cuda")
            if cuda_state and torch.cuda.is_available():
                torch.cuda.random.set_rng_state_all(cuda_state)

        logger.info(
            "训练恢复完成: step=%d, epoch=%d, best_loss=%.4f",
            self.global_step, self.epoch, self.best_loss,
        )

    def _notify_callbacks(self, hook: str, **kwargs) -> None:
        """通知所有回调执行指定钩子。"""
        for cb in self.callbacks:
            getattr(cb, hook)(self, **kwargs)  # type: ignore[misc]
```

---

### 4.2 预训练（Causal LM Pretraining）

**位置**: `src/classic_chinese_llm/training/pretrain.py`

```python
def pretrain_loss_fn(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Causal Language Modeling 的 loss 计算。

    标准 next-token prediction:
    - 输入: tokens[0:-1]（模型内部通过 causal attention 自动处理）
    - 目标: tokens[1:]（由 Data Collator 构建的 labels）

    labels 中 PAD 位置已设为 -100（CrossEntropyLoss 的 ignore_index），
    不参与 loss 和梯度计算。

    Args:
        model: TransformerLM 模型
        batch: {"input_ids": (B, S), "labels": (B, S), "attention_mask": (B, S)}

    Returns:
        标量 loss（未除以 grad_accum_steps）
    """
    logits = model(input_ids=batch["input_ids"])  # (B, S, vocab_size)

    # 展平为 (B*S, vocab_size) vs (B*S,)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        batch["labels"].view(-1),
        ignore_index=-100,
    )
    return loss


class PretrainRunner:
    """预训练流程编排。

    职责:
    1. 构建预训练数据集（从 deduplicated.jsonl 加载原始文本）
    2. 创建 DataLoader
    3. 实例化 Trainer
    4. 启动训练

    数据格式:
    - 输入: data/processed/deduplicated.jsonl（SourceDocument.text 字段）
    - 每行为一段文言文文本
    - Tokenizer 编码后得到 input_ids
    - labels = input_ids（右移一位由 CrossEntropyLoss 的 shift 逻辑隐式处理）
    """

    def __init__(
        self,
        config: PretrainConfig,
        data_path: str | Path,
        tokenizer: PreTrainedTokenizerFast,
    ) -> None:
        self.config = config
        self.data_path = Path(data_path)
        self.tokenizer = tokenizer
        self.device_info = detect_device()

    def run(self) -> None:
        """执行完整的预训练流程。"""
        # 1. 加载数据
        train_dataset = PretrainDataset(
            self.data_path,
            self.tokenizer,
            max_seq_len=self.config.model.max_seq_len,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            collate_fn=DataCollator(
                pad_token_id=self.tokenizer.pad_token_id,
                max_length=self.config.model.max_seq_len,
                is_sft=False,
            ),
            num_workers=4,
            pin_memory=True,
        )

        # 2. 创建模型
        model = TransformerLM(self.config.model)
        model.to(self.device_info.device)

        # 3. 创建 Trainer
        paths = PathConfig.get()
        trainer = Trainer(
            model=model,
            config=self.config,
            train_dataloader=train_loader,
            val_dataloader=None,  # 预训练阶段可选
            device_info=self.device_info,
            checkpoint_dir=paths.checkpoint_dir,
            callbacks=[
                LoggingCallback(log_dir=paths.logs_dir),
                CheckpointCallback(),
            ],
            resume=True,
        )

        # 4. 开始训练
        trainer.train(loss_fn=pretrain_loss_fn)
```

---

### 4.3 指令微调（SFT）

**位置**: `src/classic_chinese_llm/training/sft.py`

```python
def _build_sft_labels(
    input_ids: torch.Tensor,
    assistant_token_id: int,
    end_token_id: int,
) -> torch.Tensor:
    """为 SFT 构建 label mask。

    算法:
    1. labels 初始设为 input_ids（右移一位的逻辑由 CrossEntropyLoss 隐式处理）
    2. 扫描序列找到所有 <|assistant|> token 的位置
    3. 从每个 <|assistant|> 到下一个 <|end|> 之间的 token 保留 label
    4. 其余位置设为 -100（ignore_index）

    ChatML 格式示例:
    <|bos|> <|system|> ... <|end|> <|user|> ... <|end|> <|assistant|> A1 A2 ... <|end|> <|eos|>
    位置:    0         1  ...   m       m+1     ...   n       n+1        n+2 n+3 ...   k       k+1
    Label:  -100      -100 ... -100    -100    ...  -100       A1        A2  A3  ... <|end|> <|eos|>
                                                                       ↑
                                                    从 <|assistant|>+1 到 <|end|> 保留
                                                    注意: 由于 CrossEntropy 内部做 predict[t]→target[t]
                                                    实际 labels 是 input_ids[1:]，所以 assistant 位置
                                                    在 label 中的偏移是 assistant_pos - 1

    多轮对话示例:
    <|user|> Q1 <|end|> <|assistant|> A1 <|end|> <|user|> Q2 <|end|> <|assistant|> A2 <|end|>
    每段 A1、A2 分别保留，Q1、Q2、特殊 token 设为 -100

    Args:
        input_ids: 单条样本的 token ID 序列 (S,)
        assistant_token_id: <|assistant|> 的 token ID
        end_token_id: <|end|> 的 token ID

    Returns:
        labels 张量 (S,)，非 assistant 位置为 -100
    """
    labels = input_ids.clone()
    seq_len = len(input_ids)

    # 找到所有 <|assistant|> 的位置
    assistant_positions = (input_ids == assistant_token_id).nonzero(as_tuple=True)[0]

    # 默认将所有位置设为忽略
    labels[:] = -100

    for asst_pos in assistant_positions:
        asst_pos = asst_pos.item()
        # <|assistant|> 本身也应被忽略（它是特殊 token，不需要预测）
        # 需要预测的是 <|assistant|> 之后、<|end|> 之前的内容

        # 找到对应的 <|end|>
        end_positions = (input_ids[asst_pos + 1 :] == end_token_id).nonzero(as_tuple=True)[0]
        if len(end_positions) == 0:
            # 没有 <|end|>：截断的序列，assistant 之后全部保留（不含 assistant 本身）
            labels[asst_pos + 1 :] = input_ids[asst_pos + 1 :]
        else:
            end_pos = asst_pos + 1 + end_positions[0].item()
            # 保留 asst_pos+1 到 end_pos（含 <|end|>）的内容，
            # 因为 <|end|> 也是模型需要学习生成的结束信号
            labels[asst_pos + 1 : end_pos + 1] = input_ids[asst_pos + 1 : end_pos + 1]

    return labels


def sft_loss_fn(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """SFT 的 loss 计算。

    与 pretrain_loss_fn 相同——labels 已经由 DataCollator 预处理完成，
    非 assistant 位置已设为 -100。

    Args:
        model: TransformerLM 模型
        batch: {"input_ids": (B, S), "labels": (B, S), "attention_mask": (B, S)}
    """
    logits = model(input_ids=batch["input_ids"])
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        batch["labels"].view(-1),
        ignore_index=-100,
    )
    return loss


class SFTRunner:
    """指令微调流程编排。

    职责:
    1. 加载 ChatML 格式的指令数据（FormattedSample.messages）
    2. 通过 tokenizer.apply_chat_template() 将 messages 转换为 input_ids
    3. 构建 SFT 专用的 labels
    4. 创建 DataLoader + Trainer
    5. 启动训练

    数据格式:
    - 输入: data/processed/instructions/train.jsonl（ChatML 格式）
    - 每行: {"messages": [{"role": "system", ...}, {"role": "user", ...},
              {"role": "assistant", ...}], "task_type": "..."}
    """

    def __init__(
        self,
        config: SFTConfig,
        train_data_path: str | Path,
        val_data_path: str | Path | None,
        pretrained_checkpoint: str | Path,
        tokenizer: PreTrainedTokenizerFast,
    ) -> None:
        self.config = config
        self.train_data_path = Path(train_data_path)
        self.val_data_path = Path(val_data_path) if val_data_path else None
        self.pretrained_checkpoint = Path(pretrained_checkpoint)
        self.tokenizer = tokenizer
        self.device_info = detect_device()

    def run(self) -> None:
        """执行完整的 SFT 流程。"""
        # 1. 加载预训练权重
        model = self._load_pretrained_model()

        # 2. 构建数据集
        train_dataset = SFTDataset(
            self.train_data_path,
            self.tokenizer,
            max_seq_len=self.config.model.max_seq_len,
            chat_template=self.config.chat_template,
        )
        val_dataset = (
            SFTDataset(
                self.val_data_path,
                self.tokenizer,
                max_seq_len=self.config.model.max_seq_len,
                chat_template=self.config.chat_template,
            )
            if self.val_data_path
            else None
        )

        # 3. DataLoaders
        collator = DataCollator(
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=self.config.model.max_seq_len,
            is_sft=True,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            collate_fn=collator,
        )
        val_loader = (
            DataLoader(val_dataset, batch_size=self.config.training.batch_size, collate_fn=collator)
            if val_dataset
            else None
        )

        # 4. Trainer
        paths = PathConfig.get()
        trainer = Trainer(
            model=model,
            config=self.config,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            device_info=self.device_info,
            checkpoint_dir=paths.checkpoint_dir / "sft",
            callbacks=[
                LoggingCallback(log_dir=paths.logs_dir),
                CheckpointCallback(),
                EarlyStoppingCallback(patience=5),
            ],
            resume=False,  # SFT 通常从预训练 checkpoint 开始，不自动 resume
        )

        trainer.train(loss_fn=sft_loss_fn)

    def _load_pretrained_model(self) -> TransformerLM:
        """从预训练 checkpoint 加载模型权重。

        加载策略:
        - 加载预训练模型的权重
        - 如果预训练模型的 vocab_size 与当前不同（tokenizer 添加了特殊 token），
          需要调整 embedding 层的大小
        - 不加载 optimizer state（SFT 使用新的 optimizer）
        """
        state = load_checkpoint(self.pretrained_checkpoint)
        model = TransformerLM(self.config.model)

        # 处理 vocab 大小不匹配（SFT 可能添加了 ChatML 特殊 token）
        pretrained_weights = state.model_state_dict
        if pretrained_weights["token_embedding.weight"].shape[0] != self.config.model.vocab_size:
            logger.warning("vocab_size 不匹配，将 resize embedding 并部分随机初始化")
            # resize + 从预训练权重复制可用的部分
            pretrained_weights = self._resize_embeddings(pretrained_weights, model)

        model.load_state_dict(pretrained_weights, strict=False)
        model.to(self.device_info.device)
        return model
```

---

### 4.4 回调系统（Callbacks）

**位置**: `src/classic_chinese_llm/training/callbacks.py`

```python
class Callback(ABC):
    """回调基类。

    所有钩子方法默认是 no-op。子类选择性覆写需要的方法。
    钩子的调用顺序由 Trainer 保证（按 callbacks 列表顺序）。
    """

    def on_train_begin(self, trainer: "Trainer") -> None:
        """训练开始前调用（创建 optimizer/scheduler 之后，第一个 step 之前）。"""

    def on_step_end(self, trainer: "Trainer", loss: float, lr: float) -> None:
        """每个 optimizer step 之后调用。"""

    def on_eval_end(self, trainer: "Trainer", metrics: dict[str, float]) -> None:
        """每次评估结束后调用。"""

    def on_epoch_end(self, trainer: "Trainer") -> None:
        """每个 epoch 结束时调用。"""

    def on_train_end(self, trainer: "Trainer") -> None:
        """训练完成（正常终止或 early stop）时调用。"""


class LoggingCallback(Callback):
    """训练日志回调。

    记录以下指标:
    - step: 当前步数
    - loss: 最近 step 的平均 loss
    - lr: 当前学习率
    - tokens/sec: token 处理速度
    - gpu_memory: GPU 显存使用量
    - val_loss: 验证 loss（仅在有评估时）
    - best_loss: 历史最佳验证 loss

    输出目标:
    - 终端: Rich 格式的彩色进度条 + 指标表格
    - 文件: logs/training.log（纯文本格式，便于事后分析）
    """

    def __init__(self, log_dir: Path, log_every: int = 10) -> None:
        self.log_dir = Path(log_dir)
        self.log_every = log_every
        self._step_times: list[float] = []  # 最近 N 步的耗时（用于计算 tokens/sec）
        self._losses: list[float] = []      # 最近 N 步的 loss
        self._start_time: float = 0.0

    def on_train_begin(self, trainer: "Trainer") -> None:
        self._start_time = time.time()
        # 初始化 tqdm 进度条
        total = trainer.config.training.max_steps or (
            trainer.config.training.max_epochs * len(trainer.train_dataloader)
            if trainer.config.training.max_epochs
            else None
        )
        self._pbar = tqdm(total=total, initial=trainer.global_step)

    def on_step_end(self, trainer: "Trainer", loss: float, lr: float) -> None:
        self._losses.append(loss)

        if trainer.global_step % self.log_every == 0:
            # 计算 tokens/sec
            elapsed = time.time() - self._start_time
            tokens = (
                trainer.config.training.batch_size
                * trainer.config.model.max_seq_len
                * trainer.config.training.gradient_accumulation_steps
                * self.log_every
            )
            tps = tokens / max(elapsed, 0.001)

            # GPU 显存
            gpu_mem = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0

            avg_loss = sum(self._losses) / max(len(self._losses), 1)

            # 更新进度条
            self._pbar.set_postfix({
                "loss": f"{avg_loss:.4f}",
                "lr": f"{lr:.2e}",
                "t/s": f"{tps:.0f}",
                "mem": f"{gpu_mem:.1f}G",
                "best": f"{trainer.best_loss:.4f}",
            })
            self._pbar.update(self.log_every)

            self._losses = []
            self._start_time = time.time()


class CheckpointCallback(Callback):
    """Checkpoint 保存回调。

    两种保存模式:
    1. 周期性保存: 每 save_every 步保存（Trainer 中控制）
    2. 最佳模型保存: 当 eval loss 创新低时保存（Trainer 中控制）

    此回调负责在 Trainer 调用 _save() 时触发，
    Trainer 中已集成 save 逻辑，此回调主要用于记录日志。
    """


class EarlyStoppingCallback(Callback):
    """早停回调。

    如果连续 patience 次评估后 val_loss 没有改善，则停止训练。

    Args:
        patience: 容忍的评估次数（在 eval_every 步后判断）
        min_delta: 视为"改善"的最小 loss 下降量
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._counter = 0
        self._best_loss = float("inf")

    def on_eval_end(self, trainer: "Trainer", metrics: dict[str, float]) -> None:
        val_loss = metrics.get("val_loss", float("inf"))
        if val_loss < self._best_loss - self.min_delta:
            self._best_loss = val_loss
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                logger.info(
                    "早停触发: val_loss 连续 %d 次未改善 (best=%.4f, current=%.4f)",
                    self.patience, self._best_loss, val_loss,
                )
                trainer._should_stop = True
```

---

### 4.5 Data Collator

**位置**: `src/classic_chinese_llm/training/data_collator.py`

```python
class DataCollator:
    """动态 padding 的批次整理器。

    功能:
    1. 将 batch 中的样本 padding 到 batch 内最大长度（非全局 max_seq_len）
    2. 构建 attention_mask（1 = 真实 token，0 = padding）
    3. 构建 labels（pretrain: labels = input_ids 右移；SFT: 非 assistant 位置 = -100）

    Args:
        pad_token_id: PAD token ID
        max_length: 强制截断的最大长度（防止单样本超长）
        is_sft: 是否 SFT 模式（SFT 样本的 labels 由 SFTDataset 预处理完成）
    """

    def __init__(
        self,
        pad_token_id: int,
        max_length: int = 2048,
        is_sft: bool = False,
    ) -> None:
        self.pad_token_id = pad_token_id
        self.max_length = max_length
        self.is_sft = is_sft

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """
        输入: [{"input_ids": (S1,), "labels": (S1,)}, ...]
        输出: {"input_ids": (B, max_S), "attention_mask": (B, max_S), "labels": (B, max_S)}
        """
        # 强制截断到 max_length
        for item in batch:
            item["input_ids"] = item["input_ids"][:self.max_length]
            if "labels" in item:
                item["labels"] = item["labels"][:self.max_length]

        # 动态 padding：取 batch 内最大长度
        batch_max_len = max(item["input_ids"].size(0) for item in batch)

        input_ids_list = []
        attention_mask_list = []
        labels_list = []

        for item in batch:
            seq_len = item["input_ids"].size(0)
            pad_len = batch_max_len - seq_len

            # input_ids: 右侧 padding
            input_ids = F.pad(item["input_ids"], (0, pad_len), value=self.pad_token_id)
            input_ids_list.append(input_ids)

            # attention_mask: 1 = 真实 token，0 = padding
            attn_mask = torch.cat([
                torch.ones(seq_len, dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ])
            attention_mask_list.append(attn_mask)

            # labels: 同 input_ids，但 padding 位置设为 -100
            if "labels" in item:
                labels = F.pad(item["labels"], (0, pad_len), value=-100)
            else:
                # PyTorch CrossEntropyLoss 内部做 predict[t]→target[t] 的 shift
                # 所以 labels = input_ids（右移由 loss 函数处理）
                labels = input_ids.clone()
                labels[seq_len:] = -100  # padding 忽略

            labels_list.append(labels)

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list),
        }
```

**动态 padding vs 固定长度 padding**:

```python
# 固定 padding（效率低）:
# batch 中所有样本都 pad 到 max_seq_len=2048
# 问题: 如果 batch 内最大长度仅 512，浪费 75% 的计算

# 动态 padding（选用）:
# batch 内 pad 到 max_len_in_batch
# 优势: 平均节省 30-50% 的计算量（取决于数据分布）
# 代价: 每个 batch 的形状不同（对 FlashAttention 无影响）

# 文言文数据长度分布（估测）:
# 中位数: ~300 字 → ~180 tokens
# P90: ~1200 字 → ~700 tokens
# P99: ~2500 字 → ~1500 tokens
# 动态 padding 在 90% 的 batch 中将有效长度控制在 800 tokens 以下
```

---

### 4.6 数据集类

**位置**: `src/classic_chinese_llm/training/datasets.py`（可与 data_collator.py 合并或独立）

```python
class PretrainDataset(Dataset):
    """预训练数据集。

    从 cleaned JSONL 加载原始文言文文本，逐行 tokenize 后返回。

    数据来源: data/processed/deduplicated.jsonl
    """

    def __init__(
        self,
        data_path: Path,
        tokenizer: PreTrainedTokenizerFast,
        max_seq_len: int = 2048,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # 加载所有行
        self._samples: list[str] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                text = record.get("text", "").strip()
                if text:
                    self._samples.append(text)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        text = self._samples[idx]
        # Tokenize：返回 {"input_ids": [...], "attention_mask": [...]}
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors=None,  # 返回 list 而非 tensor
        )
        input_ids = torch.tensor(encoded["input_ids"], dtype=torch.long)
        # labels = input_ids（Causal LM 的 shift 逻辑由 loss 函数处理）
        return {"input_ids": input_ids, "labels": input_ids.clone()}


class SFTDataset(Dataset):
    """SFT 指令微调数据集。

    从 ChatML 格式的 JSONL 加载数据，使用 tokenizer.apply_chat_template()
    将 messages 转换为 input_ids，并构建仅 assistant 位置的 labels。

    数据来源: data/processed/instructions/train.jsonl
    """

    def __init__(
        self,
        data_path: Path,
        tokenizer: PreTrainedTokenizerFast,
        max_seq_len: int = 2048,
        chat_template: str = "classical_chinese_v1",
    ) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # 获取 ChatML 特殊 token 的 ID
        self.assistant_token_id = tokenizer.convert_tokens_to_ids("<|assistant|>")
        self.end_token_id = tokenizer.convert_tokens_to_ids("<|end|>")

        # 加载所有 ChatML 样本
        self._samples: list[list[dict[str, str]]] = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                messages = record.get("messages", [])
                if messages:
                    self._samples.append(messages)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        messages = self._samples[idx]

        # 使用 Chat Template 格式化 + Tokenize
        # add_generation_prompt=False: 训练时不需要追加 <|assistant|>
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            truncation=True,
            max_length=self.max_seq_len,
        )

        input_ids = torch.tensor(input_ids, dtype=torch.long)

        # 构建仅 assistant 位置的 labels
        labels = _build_sft_labels(
            input_ids,
            assistant_token_id=self.assistant_token_id,
            end_token_id=self.end_token_id,
        )

        return {"input_ids": input_ids, "labels": labels}
```

---

## 5. 模块结构

```
src/classic_chinese_llm/training/
├── __init__.py          # 导出: Trainer, PretrainRunner, SFTRunner, Callback, ...
├── trainer.py           # Trainer 通用训练循环
├── pretrain.py          # pretrain_loss_fn, PretrainRunner
├── sft.py               # sft_loss_fn, _build_sft_labels, SFTRunner
├── callbacks.py         # Callback 基类, LoggingCallback, CheckpointCallback,
│                        #   EarlyStoppingCallback
├── data_collator.py     # DataCollator (动态 padding + label mask)
└── datasets.py          # PretrainDataset, SFTDataset

scripts/
├── pretrain.py          # CLI: python scripts/pretrain.py --config configs/pretrain.yaml
└── finetune.py          # CLI: python scripts/finetune.py --config configs/sft.yaml
                         #      --pretrained-checkpoint models/checkpoints/checkpoint_best.pt
```

---

## 6. 接口定义汇总

### 6.1 Trainer

```python
class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: Settings,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader | None,
        device_info: DeviceInfo,
        checkpoint_dir: Path,
        callbacks: list[Callback] | None = None,
        resume: bool = True,
    ) -> None: ...

    def train(
        self,
        loss_fn: Callable[[nn.Module, dict[str, torch.Tensor]], torch.Tensor],
    ) -> None: ...
```

### 6.2 任务函数

```python
# pretrain.py
def pretrain_loss_fn(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor: ...

class PretrainRunner:
    def __init__(self, config: PretrainConfig, data_path: Path, tokenizer: PreTrainedTokenizerFast) -> None: ...
    def run(self) -> None: ...

# sft.py
def _build_sft_labels(input_ids: Tensor, assistant_token_id: int, end_token_id: int) -> Tensor: ...
def sft_loss_fn(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor: ...

class SFTRunner:
    def __init__(
        self, config: SFTConfig, train_data_path: Path, val_data_path: Path | None,
        pretrained_checkpoint: Path, tokenizer: PreTrainedTokenizerFast,
    ) -> None: ...
    def run(self) -> None: ...
```

### 6.3 Callbacks

```python
class Callback(ABC):
    def on_train_begin(self, trainer: Trainer) -> None: ...
    def on_step_end(self, trainer: Trainer, loss: float, lr: float) -> None: ...
    def on_eval_end(self, trainer: Trainer, metrics: dict[str, float]) -> None: ...
    def on_epoch_end(self, trainer: Trainer) -> None: ...
    def on_train_end(self, trainer: Trainer) -> None: ...

class LoggingCallback(Callback): ...
class CheckpointCallback(Callback): ...
class EarlyStoppingCallback(Callback): ...
```

### 6.4 Data

```python
class DataCollator:
    def __init__(self, pad_token_id: int, max_length: int = 2048, is_sft: bool = False) -> None: ...
    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]: ...

class PretrainDataset(Dataset): ...
class SFTDataset(Dataset): ...
```

---

## 7. 与其他模块的关系

```
                       Phase 2: Data                    Phase 3: Tokenizer
              ┌──────────────────────┐         ┌──────────────────────────┐
              │ deduplicated.jsonl   │         │ PreTrainedTokenizerFast  │
              │ instructions/*.jsonl │         │ .encode() .decode()      │
              └─────────┬────────────┘         │ .apply_chat_template()   │
                        │                      └────────────┬─────────────┘
                        │                                   │
         ┌──────────────┼───────────────────────────────────┼──────────────┐
         │              │          Phase 4: Training        │              │
         │              ▼                                   ▼              │
         │  ┌─────────────────┐              ┌──────────────────────┐     │
         │  │ PretrainDataset │──────────────│ Tokenizer encode     │     │
         │  │ SFTDataset      │──┐           │ Chat template format │     │
         │  └─────────────────┘  │           └──────────────────────┘     │
         │                       ▼                                       │
         │  ┌──────────────────────────────────────────────────────┐     │
         │  │ DataCollator                                          │     │
         │  │  ├─ Dynamic padding                                   │     │
         │  │  ├─ Attention mask                                    │     │
         │  │  └─ SFT label masking (non-assistant → -100)          │     │
         │  └──────────────────────────┬───────────────────────────┘     │
         │                             ▼                                 │
         │  ┌──────────────────────────────────────────────────────┐     │
         │  │ Trainer                                                │     │
         │  │  ├─ Gradient accumulation                             │     │
         │  │  ├─ BF16 autocast + (optional) GradScaler              │     │
         │  │  ├─ Gradient clipping                                 │     │
         │  │  ├─ Cosine warmup LR scheduler                        │     │
         │  │  ├─ Eval loop                                         │     │
         │  │  └─ Checkpoint save/resume                            │     │
         │  └──────────┬──────────────────┬────────────────────────┘     │
         │             │                  │                              │
         └─────────────┼──────────────────┼──────────────────────────────┘
                       │                  │
              ┌────────▼─────┐   ┌───────▼──────────┐
              │ TransformerLM │   │ CheckpointState   │
              │ (model 模块)   │   │ (utils 模块)      │
              └──────────────┘   └──────────────────┘
```

**上游依赖**:
- **Config 模块**: `PretrainConfig` / `SFTConfig` 提供所有训练超参数
- **Model 模块**: `TransformerLM` 的 forward 接口和 state_dict
- **Tokenizer 模块**: `PreTrainedTokenizerFast` 的 encode、apply_chat_template
- **Data 模块**: 清洗后的 JSONL 数据文件
- **Utils 模块**: DeviceInfo（设备检测）、CheckpointState（保存/恢复）、日志

**下游依赖**:
- **Phase 5 (SFT)**: 预训练产出的 checkpoint 作为 SFT 的初始化权重
- **Phase 6 (Inference/Chat)**: 训练产出的 checkpoint 被 Generator 加载
- **Phase 7 (Evaluation)**: 训练产出的 checkpoint 被评测模块加载

---

## 8. CLI 脚本设计

### 8.1 `scripts/pretrain.py`

```python
#!/usr/bin/env python3
"""预训练 CLI 入口。

用法:
    python scripts/pretrain.py --config configs/pretrain.yaml

    从 checkpoint 恢复:
    python scripts/pretrain.py --config configs/pretrain.yaml --resume

    使用环境变量覆盖:
    CCLLM_TRAINING__BATCH_SIZE=16 python scripts/pretrain.py --config configs/pretrain.yaml
"""

import argparse
import sys
from pathlib import Path

from classic_chinese_llm.config import load_config, PretrainConfig
from classic_chinese_llm.config.paths import PathConfig
from classic_chinese_llm.tokenizer.wrapper import build_tokenizer
from classic_chinese_llm.training.pretrain import PretrainRunner
from classic_chinese_llm.utils.logging_config import setup_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="文言文 LLM 预训练")
    parser.add_argument("--config", default="configs/pretrain.yaml", help="配置文件路径")
    parser.add_argument("--resume", action="store_true", help="从最新 checkpoint 恢复")
    args = parser.parse_args(argv)

    # 初始化项目路径
    project_root = Path(__file__).resolve().parent.parent
    PathConfig.initialize(project_root)
    paths = PathConfig.get()

    # 加载配置
    config = load_config(args.config, PretrainConfig)

    # 初始化日志
    setup_logging(
        level=config.logging.level,
        log_file=paths.logs_dir / "pretrain.log",
    )

    # 加载 tokenizer
    tokenizer_path = paths.tokenizer_dir / "classical_chinese.model"
    tokenizer = build_tokenizer(tokenizer_path)

    # 启动预训练
    data_path = paths.processed_data_dir / "deduplicated.jsonl"
    runner = PretrainRunner(config, data_path, tokenizer)
    runner.run()


if __name__ == "__main__":
    main()
```

### 8.2 `scripts/finetune.py`

```python
#!/usr/bin/env python3
"""指令微调 CLI 入口。

用法:
    python scripts/finetune.py \
        --config configs/sft.yaml \
        --pretrained-checkpoint models/checkpoints/checkpoint_best.pt
"""

import argparse
import sys
from pathlib import Path

from classic_chinese_llm.config import load_config, SFTConfig
from classic_chinese_llm.config.paths import PathConfig
from classic_chinese_llm.tokenizer.wrapper import build_tokenizer
from classic_chinese_llm.training.sft import SFTRunner
from classic_chinese_llm.utils.logging_config import setup_logging


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="文言文 LLM 指令微调")
    parser.add_argument("--config", default="configs/sft.yaml", help="配置文件路径")
    parser.add_argument(
        "--pretrained-checkpoint",
        required=True,
        help="预训练 checkpoint 路径",
    )
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parent.parent
    PathConfig.initialize(project_root)
    paths = PathConfig.get()

    config = load_config(args.config, SFTConfig)

    setup_logging(
        level=config.logging.level,
        log_file=paths.logs_dir / "sft.log",
    )

    tokenizer = build_tokenizer(paths.tokenizer_dir / "classical_chinese.model")

    runner = SFTRunner(
        config=config,
        train_data_path=paths.processed_data_dir / "instructions" / "train.jsonl",
        val_data_path=paths.processed_data_dir / "instructions" / "val.jsonl",
        pretrained_checkpoint=args.pretrained_checkpoint,
        tokenizer=tokenizer,
    )
    runner.run()


if __name__ == "__main__":
    main()
```

---

## 9. 验证清单

### 训练循环
- [ ] Trainer 在 CPU 上运行 10 步（小模型 d_model=64, n_layers=2）不报错
- [ ] 梯度累积正确：accum=4, batch=8 的 loss ≈ batch=32 的 loss（数值误差 < 1%）
- [ ] 梯度裁剪生效：post-clip gradient norm ≤ max_norm=1.0
- [ ] Cosine LR：第一步 lr ≈ min_lr，warmup_steps 步时 lr = peak_lr，最后一步 lr = min_lr
- [ ] Warmup 阶段 lr 线性增长（每步增量 ≈ peak_lr / warmup_steps）
- [ ] optimizer.zero_grad() 在每次累积循环前调用

### 混合精度
- [ ] BF16 autocast 下 forward + backward 不产生 NaN/Inf
- [ ] BF16 不支持时自动降级为 FP16（带 GradScaler）
- [ ] FP16 模式 GradScaler 正常工作（scale loss → backward → unscale → clip → step → update）
- [ ] RMSNorm 内部计算在 float32 下执行（精度保证）

### Checkpoint 恢复
- [ ] 训练中断后 resume，global_step 从断点继续
- [ ] resume 后 loss 连续（相对断点前无跳变）
- [ ] optimizer state、scheduler state、RNG state 全部恢复
- [ ] checkpoint 轮转：超过 max_checkpoints 时旧文件被删除
- [ ] best checkpoint 独立保留（不被轮转删除）
- [ ] Ctrl+C → 保存 checkpoint 后退出

### 预训练
- [ ] 小数据集（100 样本）过拟合：loss 在 50 步内降至 0.1 以下
- [ ] PAD token 位置的 loss 不计入（ignore_index=-100 生效）
- [ ] `labels = input_ids`（Causal LM shift 逻辑正确）

### 指令微调
- [ ] Chat template 格式化：messages → input_ids 包含正确的 `<|system|>`、`<|user|>`、`<|assistant|>` 分隔
- [ ] SFT labels: 非 assistant 位置的 label 全部为 -100
- [ ] SFT labels: assistant 位置的 label = 正确的 token ID
- [ ] SFT labels: `<|end|>` token 在 assistant 段内被保留（模型需要学习生成结束信号）
- [ ] 多轮对话：每个 assistant 段独立保留 label
- [ ] 多轮对话：中间轮次的 user 问题 label = -100

### Data Collator
- [ ] 动态 padding：batch 内 pad 到 max_len_in_batch（非全局 2048）
- [ ] attention_mask：真实 token = 1，padding token = 0
- [ ] 超长样本自动截断到 max_length

### 显存
- [ ] batch=8, seq=1024, BF16 训练 10 步无 OOM
- [ ] batch=8, seq=2048, BF16 训练 10 步无 OOM（预期峰值 < 8GB）
- [ ] 显存使用量在日志中可见

### 代码质量
- [ ] 所有函数 ≤ 50 行（CLAUDE.md 规范）
- [ ] 所有函数签名含完整类型注解
- [ ] 使用 `get_logger(__name__)`，无 `print()`
- [ ] mypy strict mode: `mypy src/classic_chinese_llm/training/` 零错误
- [ ] black + ruff 格式化通过
