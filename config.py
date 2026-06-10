"""
Prompt Guard v1 — Configuration
All hyperparameters and paths in one place.
"""

from dataclasses import dataclass, field
from typing import Optional
import os


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_name: str = "microsoft/mdeberta-v3-base"
    num_labels: int = 3  # BENIGN=0, INJECTION=1, JAILBREAK=2
    max_seq_length: int = 512
    classifier_dropout: float = 0.1


@dataclass
class DataConfig:
    """Dataset configuration."""
    # Output paths
    output_dir: str = "./data/processed"
    cache_dir: str = "./data/cache"

    # Processing
    max_samples_per_class: Optional[int] = None  # None = use all
    min_text_length: int = 10  # Skip very short texts
    max_text_length: int = 10000  # Skip extremely long texts
    test_size: float = 0.1
    val_size: float = 0.1
    seed: int = 42

    # Balancing strategy: "downsample", "upsample", or None
    balance_strategy: str = "downsample"

    # Label mapping
    label2id: dict = field(default_factory=lambda: {
        "BENIGN": 0,
        "INJECTION": 1,
        "JAILBREAK": 2,
    })
    id2label: dict = field(default_factory=lambda: {
        0: "BENIGN",
        1: "INJECTION",
        2: "JAILBREAK",
    })


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    output_dir: str = "./checkpoints"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4  # T4 16GB: DeBERTa disentangled attn is VRAM-heavy
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 8  # effective batch = 4 * 8 = 32
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "cosine"
    fp16: bool = False  # DeBERTa v3 has known NaN issues with FP16 (relative position embeds overflow)
    bf16: bool = False   # T4 doesn't support BF16 either — must train in FP32
    gradient_checkpointing: bool = True  # Critical for T4: trades compute for ~40% VRAM savings

    # Evaluation & saving
    eval_strategy: str = "steps"
    eval_steps: int = 500
    save_strategy: str = "steps"
    save_steps: int = 500
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "f1_macro"
    greater_is_better: bool = True

    # Regularization
    label_smoothing_factor: float = 0.1

    # Misc
    dataloader_num_workers: int = 0  # Colab has limited /dev/shm; >0 can crash DataLoader
    logging_steps: int = 100
    report_to: str = "none"  # Set to "wandb" to enable W&B
    seed: int = 42

    # Early stopping
    early_stopping_patience: int = 5


@dataclass
class InferenceConfig:
    """Inference configuration."""
    model_path: str = "./checkpoints/best"
    device: str = "auto"  # auto, cpu, cuda
    batch_size: int = 32
    max_seq_length: int = 512


@dataclass
class Config:
    """Master configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    # Project info
    project_name: str = "prompt-guard-v1"
    version: str = "1.0.0"


# Global config instance
config = Config()
