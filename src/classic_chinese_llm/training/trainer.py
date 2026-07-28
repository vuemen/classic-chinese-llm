"""通用 Trainer —— 梯度累积 + 混合精度 + 学习率调度 + Checkpoint 管理。

与具体任务 (Pretrain/SFT) 解耦, loss 计算由调用方通过 loss_fn 注入。
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from classic_chinese_llm.config.settings import Settings
from classic_chinese_llm.training.callbacks import Callback
from classic_chinese_llm.utils.checkpoint import (
    CheckpointState,
    find_latest_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from classic_chinese_llm.utils.device import DeviceInfo, get_dtype
from classic_chinese_llm.utils.logging_config import get_logger

logger = get_logger(__name__)

# loss_fn 类型: 接收模型和 batch, 返回标量 loss
LossFn = Callable[[nn.Module, dict[str, torch.Tensor]], torch.Tensor]


class Trainer:
    """通用训练循环。

    职责:
    1. 管理训练状态 (global_step, epoch, best_loss)
    2. 驱动训练循环: 梯度累积 → 混合精度 → 梯度裁剪 → 优化器步进 → LR 调度
    3. 定期评估 + checkpoint 保存
    4. 调度回调钩子
    5. 支持中断续训 (Ctrl+C → 优雅保存)

    Args:
        model: TransformerLM 模型实例。
        config: PretrainConfig 或 SFTConfig。
        train_dataloader: 训练数据加载器。
        val_dataloader: 验证数据加载器 (可为 None)。
        device_info: 设备信息 (来自 utils/device.py)。
        checkpoint_dir: checkpoint 保存目录。
        callbacks: 回调列表。
        resume: 是否自动从最新 checkpoint 恢复。
    """

    def __init__(
        self,
        model: nn.Module,
        config: Settings,
        train_dataloader: DataLoader,  # type: ignore[type-arg]
        val_dataloader: DataLoader | None,  # type: ignore[type-arg]
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
        self.checkpoint_dir = Path(checkpoint_dir)
        self.callbacks = callbacks or []

        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float("inf")
        self._should_stop = False

        # 优化器
        self.optimizer = self._create_optimizer()

        # 计算总步数（必须在 scheduler 之前，scheduler 依赖 total_steps）
        self.total_steps = self._compute_total_steps()
        self.scheduler = self._create_scheduler()

        # 混合精度
        self.dtype = get_dtype(device_info, preference=config.dtype)
        self.scaler = torch.cuda.amp.GradScaler() if self.dtype == torch.float16 else None

        # 尝试恢复
        if resume:
            self._try_resume()

    def _compute_total_steps(self) -> int:
        """计算训练总步数。"""
        train_cfg = self.config.training
        if train_cfg.max_steps is not None:
            return train_cfg.max_steps
        if train_cfg.max_epochs is not None:
            return train_cfg.max_epochs * len(self.train_dataloader)
        raise ValueError("max_steps 或 max_epochs 必须提供一个")

    def _create_optimizer(self) -> torch.optim.AdamW:
        """创建带分组 weight decay 的 AdamW 优化器。

        - bias 和 1D 参数 (RMSNorm weight) 不衰减
        - 2D 参数 (Linear weight) 正常衰减
        """
        opt_cfg = self.config.optimizer
        train_cfg = self.config.training

        decay_params: list[nn.Parameter] = []
        no_decay_params: list[nn.Parameter] = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
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
        total = self.total_steps
        peak_lr = train_cfg.learning_rate
        min_lr = sch_cfg.min_lr

        def lr_lambda(step: int) -> float:
            if step < warmup:
                return step / max(1, warmup)
            progress = min((step - warmup) / max(1, total - warmup), 1.0)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (min_lr + (peak_lr - min_lr) * cosine_decay) / peak_lr

        return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

    def train(self, loss_fn: LossFn) -> None:
        """主训练循环。

        Args:
            loss_fn: 损失计算函数, 签名为 (model, batch) -> scalar loss。
        """
        self._notify_callbacks("on_train_begin")

        train_cfg = self.config.training
        accum_steps = train_cfg.gradient_accumulation_steps
        device = self.device_info.device

        # 确保模型在正确设备上
        self.model.to(device)
        self.model.train()

        while not self._should_stop:
            self.epoch += 1

            for batch in self.train_dataloader:
                self.global_step += 1

                # 将 batch 移到设备
                batch = {k: v.to(device) for k, v in batch.items()}

                # ── 梯度累积循环 ──
                # 在累积的第一步 zero_grad, 后续梯度直接累加
                if (self.global_step - 1) % accum_steps == 0:
                    self.optimizer.zero_grad()

                # 混合精度前向传播
                with torch.autocast(
                    device_type=device.type if device.type != "cpu" else "cuda",
                    dtype=self.dtype,
                ):
                    loss = loss_fn(self.model, batch)
                    loss = loss / accum_steps  # 归一化到有效 batch

                # 反向传播 (autocast 外)
                if self.scaler is not None:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()  # type: ignore[no-untyped-call]

                step_loss = loss.item() * accum_steps

                # 仅在累积足够步数后更新参数
                if self.global_step % accum_steps == 0:
                    # 梯度裁剪
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                    # 优化器步进
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.scheduler.step()

                current_lr = self.scheduler.get_last_lr()[0]

                # ── 回调: step 结束 ──
                self._notify_callbacks("on_step_end", loss=step_loss, lr=current_lr)

                # ── 定期评估 ──
                if self.global_step % train_cfg.eval_every == 0:
                    metrics = self._evaluate(loss_fn)
                    self._notify_callbacks("on_eval_end", metrics=metrics)

                # ── 定期保存 ──
                if self.global_step % train_cfg.save_every == 0:
                    self._save(tag=f"step_{self.global_step}")

                # ── 终止条件 ──
                if self.global_step >= self.total_steps:
                    self._should_stop = True
                    break

            self._notify_callbacks("on_epoch_end")

            if train_cfg.max_epochs and self.epoch >= train_cfg.max_epochs:
                break

        # 最终保存
        self._save(tag="latest")
        self._notify_callbacks("on_train_end")

    @torch.no_grad()
    def _evaluate(self, loss_fn: LossFn) -> dict[str, float]:
        """在验证集上计算平均 loss。

        Returns:
            {"val_loss": float, "perplexity": float}
        """
        if self.val_dataloader is None:
            return {"val_loss": float("nan")}

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        device = self.device_info.device

        for batch in self.val_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast(
                device_type=device.type if device.type != "cpu" else "cuda",
                dtype=self.dtype,
            ):
                logits = self.model(batch["input_ids"])
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    batch["labels"].view(-1),
                    ignore_index=-100,
                )
            total_loss += loss.item() * (batch["labels"] != -100).sum().item()
            total_tokens += (batch["labels"] != -100).sum().item()

        self.model.train()

        avg_loss = total_loss / max(total_tokens, 1)
        ppl = math.exp(min(avg_loss, 20))  # 防止溢出

        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            self._save(tag="best")

        return {"val_loss": avg_loss, "perplexity": ppl}

    def _save(self, tag: str) -> Path:
        """保存 checkpoint。

        Args:
            tag: checkpoint 标签 ("step_1000", "best", "latest" 等)。

        Returns:
            保存的 .pt 文件路径。
        """
        state = CheckpointState(
            model_state_dict=self.model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            global_step=self.global_step,
            epoch=self.epoch,
            best_loss=self.best_loss,
            rng_state={
                "torch": torch.random.get_rng_state(),
                "cuda": (
                    torch.cuda.random.get_rng_state_all() if torch.cuda.is_available() else None
                ),
            },
            metadata={
                "dtype": str(self.dtype),
                "config": self.config.model_dump(),
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
            logger.info("未找到 checkpoint, 从头开始训练")
            return

        logger.info("正在恢复训练: %s", latest)
        state = load_checkpoint(latest, map_location=str(self.device_info.device))

        self.model.load_state_dict(state.model_state_dict)
        if state.optimizer_state_dict is not None:
            self.optimizer.load_state_dict(state.optimizer_state_dict)
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
            self.global_step,
            self.epoch,
            self.best_loss,
        )

    def _notify_callbacks(self, hook: str, **kwargs: object) -> None:
        """通知所有回调执行指定钩子。"""
        for cb in self.callbacks:
            getattr(cb, hook)(self, **kwargs)  # dynamic dispatch
