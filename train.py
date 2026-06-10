"""
Prompt Guard v1 — Training Script

Fine-tunes microsoft/mdeberta-v3-base for prompt injection / jailbreak classification
using HuggingFace Trainer API. Optimized for T4 GPU (16 GB VRAM).

Usage:
    python train.py [--config-overrides ...]
    python train.py --max-steps 10 --smoke-test  # Quick sanity check
"""

import argparse
import logging
import os
import sys

import numpy as np
from datasets import load_from_disk
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    """Compute classification metrics for the Trainer."""
    logits = eval_pred.predictions
    labels = eval_pred.label_ids
    predictions = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, predictions, average="weighted", zero_division=0),
        "precision_macro": precision_score(labels, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, predictions, average="macro", zero_division=0),
    }


def tokenize_dataset(dataset, tokenizer, max_length):
    """Tokenize the dataset."""
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,  # Dynamic padding via DataCollator
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing",
        num_proc=1,  # Colab has limited RAM; multi-proc duplicates memory
    )
    return tokenized


def get_class_weights(dataset):
    """Compute inverse-frequency class weights for imbalanced data."""
    import torch
    from collections import Counter

    label_counts = Counter(dataset["label"])
    total = sum(label_counts.values())
    num_classes = len(label_counts)

    weights = []
    for i in range(num_classes):
        count = label_counts.get(i, 1)
        weights.append(total / (num_classes * count))

    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    logger.info(f"Class weights: {weights}")
    return weights_tensor


class WeightedTrainer(Trainer):
    """Trainer with class-weighted cross-entropy loss."""

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        import torch.nn as nn
        import torch

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
        else:
            weight = None

        loss_fn = nn.CrossEntropyLoss(
            weight=weight,
            label_smoothing=self.args.label_smoothing_factor,
        )
        loss = loss_fn(logits.view(-1, self.model.config.num_labels), labels.view(-1))

        return (loss, outputs) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser(description="Train Prompt Guard v1")
    parser.add_argument("--data-dir", type=str, default=config.data.output_dir)
    parser.add_argument("--output-dir", type=str, default=config.training.output_dir)
    parser.add_argument("--model-name", type=str, default=config.model.model_name)
    parser.add_argument("--epochs", type=int, default=config.training.num_train_epochs)
    parser.add_argument("--batch-size", type=int, default=config.training.per_device_train_batch_size)
    parser.add_argument("--lr", type=float, default=config.training.learning_rate)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--smoke-test", action="store_true", help="Quick sanity check")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    if args.smoke_test:
        args.max_steps = 10
        config.training.eval_steps = 5
        config.training.save_steps = 5
        config.training.load_best_model_at_end = False  # No best checkpoint in 10 steps
        config.training.early_stopping_patience = 999  # Disable early stopping
        logger.info("🔥 SMOKE TEST MODE — running 10 steps only")

    # ── Load dataset ──
    logger.info("Loading processed dataset...")
    dataset = load_from_disk(args.data_dir)
    logger.info(f"  Train: {len(dataset['train'])} samples")
    logger.info(f"  Val:   {len(dataset['validation'])} samples")
    logger.info(f"  Test:  {len(dataset['test'])} samples")

    # ── Load tokenizer ──
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # ── Tokenize ──
    logger.info("Tokenizing dataset...")
    tokenized_dataset = {}
    for split in ["train", "validation", "test"]:
        tokenized_dataset[split] = tokenize_dataset(
            dataset[split], tokenizer, config.model.max_seq_length
        )

    # ── Load model ──
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=config.model.num_labels,
        id2label=config.data.id2label,
        label2id=config.data.label2id,
        classifier_dropout=config.model.classifier_dropout,
    )
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total params: {total_params:,}")
    logger.info(f"  Trainable params: {trainable_params:,}")

    # ── Class weights ──
    class_weights = None
    if not args.no_class_weights:
        class_weights = get_class_weights(dataset["train"])

    # ── Data collator ──
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # ── Training arguments ──
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=config.training.per_device_eval_batch_size,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        learning_rate=args.lr,
        weight_decay=config.training.weight_decay,
        warmup_ratio=config.training.warmup_ratio,
        lr_scheduler_type=config.training.lr_scheduler_type,
        fp16=config.training.fp16,
        bf16=config.training.bf16,
        gradient_checkpointing=config.training.gradient_checkpointing,
        eval_strategy=config.training.eval_strategy,
        eval_steps=config.training.eval_steps,
        save_strategy=config.training.save_strategy,
        save_steps=config.training.save_steps,
        save_total_limit=config.training.save_total_limit,
        load_best_model_at_end=config.training.load_best_model_at_end,
        metric_for_best_model=config.training.metric_for_best_model,
        greater_is_better=config.training.greater_is_better,
        label_smoothing_factor=config.training.label_smoothing_factor,
        dataloader_num_workers=config.training.dataloader_num_workers,
        logging_steps=config.training.logging_steps,
        report_to=config.training.report_to,
        seed=config.training.seed,
        max_steps=args.max_steps,
        dataloader_pin_memory=True,
        remove_unused_columns=True,
    )

    # ── Trainer ──
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.training.early_stopping_patience
            ),
        ],
    )

    # ── Train ──
    logger.info("=" * 60)
    logger.info("Starting training...")
    logger.info("=" * 60)

    train_result = trainer.train(resume_from_checkpoint=args.resume_from)

    # ── Save best model ──
    best_dir = os.path.join(args.output_dir, "best")
    trainer.save_model(best_dir)
    tokenizer.save_pretrained(best_dir)
    logger.info(f"Best model saved to: {best_dir}")

    # ── Log training metrics ──
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    # ── Evaluate on test set ──
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating on test set...")
    logger.info("=" * 60)

    test_metrics = trainer.evaluate(tokenized_dataset["test"], metric_key_prefix="test")
    trainer.log_metrics("test", test_metrics)
    trainer.save_metrics("test", test_metrics)

    logger.info("\n" + "=" * 60)
    logger.info("✓ Training complete!")
    logger.info("=" * 60)
    logger.info(f"  Test Accuracy:  {test_metrics.get('test_accuracy', 'N/A'):.4f}")
    logger.info(f"  Test F1 (macro): {test_metrics.get('test_f1_macro', 'N/A'):.4f}")
    logger.info(f"  Best model: {best_dir}")


if __name__ == "__main__":
    main()
