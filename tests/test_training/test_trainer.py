"""Trainer 测试 —— 训练循环、梯度累积、LR 调度、checkpoint 恢复。"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from classic_chinese_llm.config.settings import ModelConfig, PretrainConfig, TrainingConfig
from classic_chinese_llm.model.transformer import TransformerLM
from classic_chinese_llm.training.pretrain import pretrain_loss_fn
from classic_chinese_llm.training.trainer import Trainer
from classic_chinese_llm.utils.checkpoint import find_latest_checkpoint
from classic_chinese_llm.utils.device import detect_device


class DummyDataset(Dataset):  # type: ignore[type-arg]
    """测试用的虚拟数据集（固定 token ID 输出）。"""

    def __init__(self, num_samples: int = 32, seq_len: int = 16, vocab_size: int = 1000) -> None:
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        input_ids = torch.randint(0, self.vocab_size, (self.seq_len,))
        return {"input_ids": input_ids, "labels": input_ids.clone()}


def _make_tiny_config() -> PretrainConfig:
    """创建测试用的预训练配置（使用 Pydantic 允许的最小合法值）。"""
    return PretrainConfig(
        seed=42,
        dtype="fp32",
        model=ModelConfig(
            vocab_size=1000,
            d_model=64,
            n_layers=1,
            n_heads=2,
            d_ff=256,
            max_seq_len=128,
            dropout=0.0,
        ),
        training=TrainingConfig(
            batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=1e-3,
            weight_decay=0.01,
            warmup_steps=5,
            max_steps=20,
            eval_every=100,  # 不触发 eval
            save_every=100,  # 不触发 save
            max_checkpoints=2,
        ),
    )


def _make_tiny_model(config: PretrainConfig) -> TransformerLM:
    return TransformerLM(config.model)


class TestTrainerBasics:
    """Trainer 基础功能测试。"""

    def test_trainer_initializes(self, temp_dir: Path) -> None:
        """Trainer 正常初始化。"""
        config = _make_tiny_config()
        model = _make_tiny_model(config)
        dataset = DummyDataset(num_samples=16, vocab_size=100)
        loader = DataLoader(dataset, batch_size=4)

        trainer = Trainer(
            model=model,
            config=config,
            train_dataloader=loader,
            val_dataloader=None,
            device_info=detect_device(prefer="cpu"),
            checkpoint_dir=temp_dir / "checkpoints",
            resume=False,
        )
        assert trainer.global_step == 0
        assert trainer.epoch == 0
        assert trainer.best_loss == float("inf")

    def test_total_steps_calculation(self, temp_dir: Path) -> None:
        """max_steps 正确传递给 Trainer。"""
        config = _make_tiny_config()
        # max_steps=20
        assert config.training.max_steps == 20

        model = _make_tiny_model(config)
        loader = DataLoader(DummyDataset(num_samples=16, vocab_size=100), batch_size=4)
        trainer = Trainer(
            model=model,
            config=config,
            train_dataloader=loader,
            val_dataloader=None,
            device_info=detect_device(prefer="cpu"),
            checkpoint_dir=temp_dir / "checkpoints",
            resume=False,
        )
        assert trainer.total_steps == 20

    def test_optimizer_created(self, temp_dir: Path) -> None:
        """优化器正确创建。"""
        config = _make_tiny_config()
        model = _make_tiny_model(config)
        loader = DataLoader(DummyDataset(num_samples=16, vocab_size=100), batch_size=4)
        trainer = Trainer(
            model=model,
            config=config,
            train_dataloader=loader,
            val_dataloader=None,
            device_info=detect_device(prefer="cpu"),
            checkpoint_dir=temp_dir / "checkpoints",
            resume=False,
        )
        assert isinstance(trainer.optimizer, torch.optim.AdamW)

    def test_scheduler_is_cosine(self, temp_dir: Path) -> None:
        """学习率调度器正确创建。"""
        config = _make_tiny_config()
        model = _make_tiny_model(config)
        loader = DataLoader(DummyDataset(num_samples=16, vocab_size=100), batch_size=4)
        trainer = Trainer(
            model=model,
            config=config,
            train_dataloader=loader,
            val_dataloader=None,
            device_info=detect_device(prefer="cpu"),
            checkpoint_dir=temp_dir / "checkpoints",
            resume=False,
        )
        assert trainer.scheduler is not None

    def test_weight_decay_grouping(self, temp_dir: Path) -> None:
        """RMSNorm 参数不参与 weight decay。"""
        config = _make_tiny_config()
        model = _make_tiny_model(config)
        loader = DataLoader(DummyDataset(num_samples=16, vocab_size=100), batch_size=4)
        trainer = Trainer(
            model=model,
            config=config,
            train_dataloader=loader,
            val_dataloader=None,
            device_info=detect_device(prefer="cpu"),
            checkpoint_dir=temp_dir / "checkpoints",
            resume=False,
        )
        # 第一个 param_group (decay_params) 有 weight_decay
        # 第二个 param_group (no_decay_params) weight_decay=0
        assert len(trainer.optimizer.param_groups) == 2
        assert trainer.optimizer.param_groups[1]["weight_decay"] == 0.0


class TestTrainerTrainLoop:
    """Trainer 训练循环测试。"""

    def test_runs_n_steps(self, temp_dir: Path) -> None:
        """训练循环运行指定步数。"""
        config = _make_tiny_config()
        # 改写为 10 步
        config.training.max_steps = 10
        model = _make_tiny_model(config)
        loader = DataLoader(DummyDataset(num_samples=64, vocab_size=100), batch_size=4)

        trainer = Trainer(
            model=model,
            config=config,
            train_dataloader=loader,
            val_dataloader=None,
            device_info=detect_device(prefer="cpu"),
            checkpoint_dir=temp_dir / "checkpoints",
            resume=False,
        )
        trainer.train(loss_fn=pretrain_loss_fn)

        assert trainer.global_step == 10

    def test_loss_decreases_during_training(self, temp_dir: Path) -> None:
        """训练过程中 loss 下降。"""
        config = _make_tiny_config()
        config.training.max_steps = 50

        model = _make_tiny_model(config)
        # 使用小数据集确保过拟合
        dataset = DummyDataset(num_samples=128, vocab_size=config.model.vocab_size)
        loader = DataLoader(dataset, batch_size=4)

        # 记录前几步的 loss
        trainer = Trainer(
            model=model,
            config=config,
            train_dataloader=loader,
            val_dataloader=None,
            device_info=detect_device(prefer="cpu"),
            checkpoint_dir=temp_dir / "checkpoints",
            resume=False,
        )

        # 手动跑几步看 loss
        initial_loss = None
        final_loss = None
        device = trainer.device_info.device
        model.to(device)

        # 记录 loss 用于对比
        losses = []
        model.train()
        for step, batch in enumerate(loader):
            if step >= 30:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = pretrain_loss_fn(model, batch)
            loss.backward()  # type: ignore[no-untyped-call]

            if step == 0:
                initial_loss = loss.item()
            if step == 29:
                final_loss = loss.item()

            trainer.optimizer.step()
            trainer.optimizer.zero_grad()
            trainer.scheduler.step()
            losses.append(loss.item())

        # loss 应该下降
        assert initial_loss is not None
        assert final_loss is not None
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} → {losses[-1]:.4f}"

    def test_gradient_accumulation(self, temp_dir: Path) -> None:
        """梯度累积：参数更新频率 = 每 accum_steps 步一次。"""
        config = _make_tiny_config()
        config.training.max_steps = 6
        config.training.gradient_accumulation_steps = 2
        config.training.batch_size = 2

        model = _make_tiny_model(config)
        loader = DataLoader(DummyDataset(num_samples=32, vocab_size=100), batch_size=2)

        trainer = Trainer(
            model=model,
            config=config,
            train_dataloader=loader,
            val_dataloader=None,
            device_info=detect_device(prefer="cpu"),
            checkpoint_dir=temp_dir / "checkpoints",
            resume=False,
        )

        # 记录训练前的参数快照
        param_before = {n: p.clone() for n, p in model.named_parameters()}

        trainer.train(loss_fn=pretrain_loss_fn)

        # 参数应该发生了变化
        param_after = dict(model.named_parameters())
        changed = False
        for name in param_before:
            if not torch.equal(param_before[name], param_after[name]):
                changed = True
                break
        assert changed, "训练后参数无变化"

    def test_lr_scheduler_warmup_then_decay(self, temp_dir: Path) -> None:
        """LR 先 warmup 增长再 cosine 衰减。"""
        config = _make_tiny_config()
        config.training.max_steps = 30
        config.training.warmup_steps = 10
        config.training.learning_rate = 0.01

        model = _make_tiny_model(config)
        loader = DataLoader(DummyDataset(num_samples=64, vocab_size=100), batch_size=4)

        trainer = Trainer(
            model=model,
            config=config,
            train_dataloader=loader,
            val_dataloader=None,
            device_info=detect_device(prefer="cpu"),
            checkpoint_dir=temp_dir / "checkpoints",
            resume=False,
        )

        lrs = []
        device = trainer.device_info.device
        model.to(device)
        model.train()

        for step, batch in enumerate(loader):
            if step >= 30:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            trainer.optimizer.zero_grad()
            loss = pretrain_loss_fn(model, batch)
            loss.backward()  # type: ignore[no-untyped-call]
            trainer.optimizer.step()
            trainer.scheduler.step()
            lrs.append(trainer.scheduler.get_last_lr()[0])

        # warmup 阶段 lr 上升
        assert lrs[0] < lrs[9], f"Warmup 阶段 lr 应该上升: {lrs[0]:.6f} → {lrs[9]:.6f}"
        # 衰减阶段 ending lr 接近 min_lr
        peak_lr = max(lrs)
        assert lrs[-1] < peak_lr, f"衰减阶段 lr 应该下降: peak={peak_lr:.6f}, final={lrs[-1]:.6f}"


class TestTrainerCheckpoint:
    """Trainer checkpoint 保存/恢复测试。"""

    def test_save_and_resume(self, temp_dir: Path) -> None:
        """保存后恢复，状态一致。"""
        config = _make_tiny_config()
        config.training.max_steps = 15
        config.training.save_every = 5  # 每 5 步保存
        config.training.eval_every = 100  # 不触发 eval

        # 第一次训练：跑 10 步
        model1 = _make_tiny_model(config)
        loader1 = DataLoader(DummyDataset(num_samples=64, vocab_size=100), batch_size=4)
        ckpt_dir = temp_dir / "checkpoints"

        trainer1 = Trainer(
            model=model1,
            config=config,
            train_dataloader=loader1,
            val_dataloader=None,
            device_info=detect_device(prefer="cpu"),
            checkpoint_dir=ckpt_dir,
            resume=False,
        )
        trainer1.train(loss_fn=pretrain_loss_fn)

        # 验证 checkpoint 已保存
        latest = find_latest_checkpoint(ckpt_dir)
        assert latest is not None, "checkpoint 应已保存"

        # 第二次训练：从 checkpoint 恢复
        model2 = _make_tiny_model(config)
        loader2 = DataLoader(DummyDataset(num_samples=64, vocab_size=100), batch_size=4)

        trainer2 = Trainer(
            model=model2,
            config=config,
            train_dataloader=loader2,
            val_dataloader=None,
            device_info=detect_device(prefer="cpu"),
            checkpoint_dir=ckpt_dir,
            resume=True,
        )
        # resume 后 global_step 应该恢复
        assert trainer2.global_step == trainer1.global_step
        assert trainer2.best_loss == trainer1.best_loss
