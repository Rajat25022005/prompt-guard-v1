"""
Prompt Guard v1 — Dataset Preparation (Tier 3: All 16 Datasets)

Downloads, merges, cleans, balances, and splits all available
prompt injection / jailbreak / benign datasets into a unified format.

MEMORY-SAFE: Uses HuggingFace Dataset (Arrow-backed, memory-mapped) throughout.
Never loads full datasets into Python lists — safe for Colab's ~12.7 GB RAM.

Usage:
    python data/prepare_dataset.py [--max-samples-per-class N] [--output-dir DIR]
    python data/prepare_dataset.py --skip-large   # Skip Necent (1.18M) and lmsys-1m
"""

import argparse
import hashlib
import logging
import os
import sys
from collections import Counter
from typing import Optional

from datasets import (
    Dataset,
    DatasetDict,
    concatenate_datasets,
    load_dataset,
)
from sklearn.model_selection import train_test_split

# Add parent dir to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Label constants
BENIGN = 0
INJECTION = 1
JAILBREAK = 2


# ─────────────────────────────────────────────────────────────────────────────
# Individual dataset loaders
# Each returns a HuggingFace Dataset with columns: {"text": str, "label": int}
# Arrow-backed → memory-mapped, NOT loaded into Python RAM.
# ─────────────────────────────────────────────────────────────────────────────


def _to_standard_dataset(records: list[dict]) -> Dataset:
    """Convert a small list of dicts to a HF Dataset. Only for small datasets."""
    if not records:
        return Dataset.from_dict({"text": [], "label": []})
    texts = [r["text"] for r in records]
    labels = [r["label"] for r in records]
    return Dataset.from_dict({"text": texts, "label": labels})


def load_necent() -> Dataset:
    """#1: Necent aggregated dataset (~1.18M rows). Processed via .map()."""
    logger.info("Loading Necent/llm-jailbreak-prompt-injection-dataset...")
    try:
        ds = load_dataset(
            "Necent/llm-jailbreak-prompt-injection-dataset",
            split="train",
            cache_dir=config.data.cache_dir,
        )

        def _map_necent(row):
            text = row.get("prompt", "") or ""
            prompt_type = str(row.get("prompt_type", "")).lower()
            is_harmful = row.get("prompt_harmful", 0)
            is_adversarial = row.get("prompt_adversarial", 0)

            if prompt_type == "jailbreak" or (is_harmful and not is_adversarial):
                label = JAILBREAK
            elif prompt_type == "prompt_injection" or is_adversarial:
                label = INJECTION
            elif not is_harmful and not is_adversarial:
                label = BENIGN
            else:
                label = INJECTION
            return {"text": text, "label": label}

        ds = ds.map(_map_necent, remove_columns=ds.column_names, desc="Mapping Necent")
        ds = ds.filter(lambda x: x["text"] and len(x["text"]) > 0, desc="Filtering empty")
        logger.info(f"  → {len(ds)} rows from Necent")
        return ds
    except Exception as e:
        logger.warning(f"  ✗ Failed to load Necent: {e}")
        return Dataset.from_dict({"text": [], "label": []})


def load_tensortrust() -> Dataset:
    """#2: TensorTrust prompt injection game (~170K)."""
    logger.info("Loading TensorTrust (ethz-spylab)...")
    try:
        try:
            ds = load_dataset("ethz-spylab/ttp_and_ttd", split="train", cache_dir=config.data.cache_dir)
        except Exception:
            ds = load_dataset("ethz-spylab/tensor-trust", split="train", cache_dir=config.data.cache_dir)

        def _map_tt(row):
            text = row.get("prompt", row.get("attack", row.get("text", ""))) or ""
            return {"text": text, "label": INJECTION}

        ds = ds.map(_map_tt, remove_columns=ds.column_names, desc="Mapping TensorTrust")
        ds = ds.filter(lambda x: len(x["text"]) > 0)
        logger.info(f"  → {len(ds)} rows from TensorTrust")
        return ds
    except Exception as e:
        logger.warning(f"  ✗ Failed to load TensorTrust: {e}")
        return Dataset.from_dict({"text": [], "label": []})


def load_wildguardmix() -> Dataset:
    """#3: Allen AI WildGuardMix (~92K)."""
    logger.info("Loading allenai/wildguardmix...")
    try:
        ds = load_dataset("allenai/wildguardmix", split="train", cache_dir=config.data.cache_dir)

        def _map_wg(row):
            text = row.get("prompt", row.get("instruction", "")) or ""
            is_harmful = str(row.get("prompt_harm_label", row.get("is_harmful", ""))).lower()
            is_adversarial = row.get("is_adversarial", False)

            if is_harmful in ("yes", "1", "true", "harmful"):
                label = JAILBREAK if is_adversarial else INJECTION
            else:
                label = BENIGN
            return {"text": text, "label": label}

        ds = ds.map(_map_wg, remove_columns=ds.column_names, desc="Mapping WildGuardMix")
        ds = ds.filter(lambda x: len(x["text"]) > 0)
        logger.info(f"  → {len(ds)} rows from WildGuardMix")
        return ds
    except Exception as e:
        logger.warning(f"  ✗ Failed to load WildGuardMix: {e}")
        return Dataset.from_dict({"text": [], "label": []})


def load_trustailab_wild() -> Dataset:
    """#4: TrustAIRLab in-the-wild jailbreak prompts (~21.5K)."""
    logger.info("Loading TrustAIRLab/in-the-wild-jailbreak-prompts...")
    try:
        ds = load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", split="train", cache_dir=config.data.cache_dir)

        def _map(row):
            text = row.get("prompt", row.get("text", "")) or ""
            return {"text": text, "label": JAILBREAK}

        ds = ds.map(_map, remove_columns=ds.column_names)
        ds = ds.filter(lambda x: len(x["text"]) > 0)
        logger.info(f"  → {len(ds)} rows from TrustAIRLab/wild")
        return ds
    except Exception as e:
        logger.warning(f"  ✗ Failed to load TrustAIRLab/wild: {e}")
        return Dataset.from_dict({"text": [], "label": []})


def load_neuralchemy() -> Dataset:
    """#5: neuralchemy prompt injection dataset (~22K)."""
    logger.info("Loading neuralchemy/Prompt-injection-dataset...")
    try:
        ds = load_dataset("neuralchemy/Prompt-injection-dataset", split="train", cache_dir=config.data.cache_dir)

        def _map(row):
            text = row.get("text", row.get("prompt", "")) or ""
            label_str = str(row.get("label", row.get("category", ""))).lower()
            if "jailbreak" in label_str:
                label = JAILBREAK
            elif "injection" in label_str or label_str == "1":
                label = INJECTION
            elif "benign" in label_str or label_str == "0":
                label = BENIGN
            else:
                label = INJECTION
            return {"text": text, "label": label}

        ds = ds.map(_map, remove_columns=ds.column_names)
        ds = ds.filter(lambda x: len(x["text"]) > 0)
        logger.info(f"  → {len(ds)} rows from neuralchemy")
        return ds
    except Exception as e:
        logger.warning(f"  ✗ Failed to load neuralchemy: {e}")
        return Dataset.from_dict({"text": [], "label": []})


def load_trustailab_jailbreakllms() -> Dataset:
    """#6: TrustAIRLab/JailbreakLLMs (~6.4K)."""
    logger.info("Loading TrustAIRLab/JailbreakLLMs...")
    try:
        ds = load_dataset("TrustAIRLab/JailbreakLLMs", split="train", cache_dir=config.data.cache_dir)

        def _map(row):
            text = row.get("prompt", row.get("text", "")) or ""
            is_jailbreak = row.get("is_jailbreak", row.get("label", 1))
            label = JAILBREAK if is_jailbreak else BENIGN
            return {"text": text, "label": label}

        ds = ds.map(_map, remove_columns=ds.column_names)
        ds = ds.filter(lambda x: len(x["text"]) > 0)
        logger.info(f"  → {len(ds)} rows from TrustAIRLab/JailbreakLLMs")
        return ds
    except Exception as e:
        logger.warning(f"  ✗ Failed to load TrustAIRLab/JailbreakLLMs: {e}")
        return Dataset.from_dict({"text": [], "label": []})


def load_slabs() -> Dataset:
    """#7: S-Labs prompt injection dataset (~5K)."""
    logger.info("Loading S-Labs/prompt-injection-dataset...")
    try:
        ds = load_dataset("S-Labs/prompt-injection-dataset", split="train", cache_dir=config.data.cache_dir)

        def _map(row):
            text = row.get("text", row.get("prompt", "")) or ""
            label_val = str(row.get("label", row.get("is_injection", 0))).lower()
            label = INJECTION if label_val in ("1", "true", "injection", "attack") else BENIGN
            return {"text": text, "label": label}

        ds = ds.map(_map, remove_columns=ds.column_names)
        ds = ds.filter(lambda x: len(x["text"]) > 0)
        logger.info(f"  → {len(ds)} rows from S-Labs")
        return ds
    except Exception as e:
        logger.warning(f"  ✗ Failed to load S-Labs: {e}")
        return Dataset.from_dict({"text": [], "label": []})


def load_centre_securite() -> Dataset:
    """#8: CentrePourLaSecuriteIA jailbreak dataset (~3K)."""
    logger.info("Loading CentrePourLaSecuriteIA/jailbreak-dataset...")
    try:
        ds = load_dataset("CentrePourLaSecuriteIA/jailbreak-dataset", split="train", cache_dir=config.data.cache_dir)

        def _map(row):
            text = row.get("prompt", row.get("text", "")) or ""
            return {"text": text, "label": JAILBREAK}

        ds = ds.map(_map, remove_columns=ds.column_names)
        ds = ds.filter(lambda x: len(x["text"]) > 0)
        logger.info(f"  → {len(ds)} rows from CentrePourLaSecuriteIA")
        return ds
    except Exception as e:
        logger.warning(f"  ✗ Failed to load CentrePourLaSecuriteIA: {e}")
        return Dataset.from_dict({"text": [], "label": []})


def load_casey27() -> Dataset:
    """#9: Casey27 JailbreakPrompts (~1K)."""
    logger.info("Loading Casey27/JailbreakPrompts...")
    try:
        ds = load_dataset("Casey27/JailbreakPrompts", split="train", cache_dir=config.data.cache_dir)

        def _map(row):
            text = row.get("prompt", row.get("text", "")) or ""
            return {"text": text, "label": JAILBREAK}

        ds = ds.map(_map, remove_columns=ds.column_names)
        ds = ds.filter(lambda x: len(x["text"]) > 0)
        logger.info(f"  → {len(ds)} rows from Casey27")
        return ds
    except Exception as e:
        logger.warning(f"  ✗ Failed to load Casey27: {e}")
        return Dataset.from_dict({"text": [], "label": []})


def load_deepset() -> Dataset:
    """#10: deepset/prompt-injections (662 rows)."""
    logger.info("Loading deepset/prompt-injections...")
    try:
        ds = load_dataset("deepset/prompt-injections", split="train", cache_dir=config.data.cache_dir)

        def _map(row):
            text = row.get("text", row.get("prompt", "")) or ""
            label = INJECTION if row.get("label", 0) == 1 else BENIGN
            return {"text": text, "label": label}

        ds = ds.map(_map, remove_columns=ds.column_names)
        ds = ds.filter(lambda x: len(x["text"]) > 0)
        logger.info(f"  → {len(ds)} rows from deepset")
        return ds
    except Exception as e:
        logger.warning(f"  ✗ Failed to load deepset: {e}")
        return Dataset.from_dict({"text": [], "label": []})


def load_advbench() -> Dataset:
    """#11: AdvBench harmful behaviors (~500)."""
    logger.info("Loading AdvBench...")
    try:
        ds = load_dataset("walledai/AdvBench", split="train", cache_dir=config.data.cache_dir)

        def _map(row):
            text = row.get("prompt", row.get("goal", row.get("text", ""))) or ""
            return {"text": text, "label": JAILBREAK}

        ds = ds.map(_map, remove_columns=ds.column_names)
        ds = ds.filter(lambda x: len(x["text"]) > 0)
        logger.info(f"  → {len(ds)} rows from AdvBench")
        return ds
    except Exception as e:
        logger.warning(f"  ✗ Failed to load AdvBench: {e}")
        return Dataset.from_dict({"text": [], "label": []})


def load_harmbench() -> Dataset:
    """#12: HarmBench (~500)."""
    logger.info("Loading HarmBench...")
    try:
        ds = load_dataset("harmbench/HarmBench", split="train", cache_dir=config.data.cache_dir)

        def _map(row):
            text = row.get("prompt", row.get("behavior", row.get("text", ""))) or ""
            return {"text": text, "label": JAILBREAK}

        ds = ds.map(_map, remove_columns=ds.column_names)
        ds = ds.filter(lambda x: len(x["text"]) > 0)
        logger.info(f"  → {len(ds)} rows from HarmBench")
        return ds
    except Exception as e:
        logger.warning(f"  ✗ Failed to load HarmBench: {e}")
        return Dataset.from_dict({"text": [], "label": []})


def load_prodnull() -> Dataset:
    """#13: prodnull/prompt-injection-repo-dataset (~2K)."""
    logger.info("Loading prodnull/prompt-injection-repo-dataset...")
    try:
        ds = load_dataset("prodnull/prompt-injection-repo-dataset", split="train", cache_dir=config.data.cache_dir)

        def _map(row):
            text = row.get("text", row.get("prompt", "")) or ""
            label_val = str(row.get("label", row.get("is_injection", 1))).lower()
            label = INJECTION if label_val in ("1", "true", "injection") else BENIGN
            return {"text": text, "label": label}

        ds = ds.map(_map, remove_columns=ds.column_names)
        ds = ds.filter(lambda x: len(x["text"]) > 0)
        logger.info(f"  → {len(ds)} rows from prodnull")
        return ds
    except Exception as e:
        logger.warning(f"  ✗ Failed to load prodnull: {e}")
        return Dataset.from_dict({"text": [], "label": []})


def load_lmsys_1m() -> Dataset:
    """#14: lmsys/lmsys-chat-1m — benign user prompts. Uses .map() to stay on disk."""
    logger.info("Loading lmsys/lmsys-chat-1m (benign prompts)...")
    try:
        ds = load_dataset("lmsys/lmsys-chat-1m", split="train", cache_dir=config.data.cache_dir)

        def _map(row):
            text = ""
            conversation = row.get("conversation", [])
            if conversation and len(conversation) > 0:
                first_msg = conversation[0]
                if isinstance(first_msg, dict):
                    text = first_msg.get("content", "") or ""
                elif isinstance(first_msg, str):
                    text = first_msg

            # Skip if flagged by OpenAI moderation
            is_flagged = False
            moderation = row.get("openai_moderation", {})
            if isinstance(moderation, dict):
                is_flagged = moderation.get("flagged", False)

            if is_flagged or not text:
                return {"text": "", "label": BENIGN}  # will be filtered out
            return {"text": text, "label": BENIGN}

        ds = ds.map(_map, remove_columns=ds.column_names, desc="Mapping lmsys-1m")
        ds = ds.filter(lambda x: len(x["text"]) > 0, desc="Filtering empty")
        logger.info(f"  → {len(ds)} benign rows from lmsys-chat-1m")
        return ds
    except Exception as e:
        logger.warning(f"  ✗ Failed to load lmsys-chat-1m: {e}")
        return Dataset.from_dict({"text": [], "label": []})


def load_lmsys_arena() -> Dataset:
    """#15: lmsys/chatbot_arena_conversations (~33K benign)."""
    logger.info("Loading lmsys/chatbot_arena_conversations...")
    try:
        ds = load_dataset("lmsys/chatbot_arena_conversations", split="train", cache_dir=config.data.cache_dir)

        def _map(row):
            text = ""
            conversation = row.get("conversation_a", row.get("conversation", []))
            if conversation and len(conversation) > 0:
                first_msg = conversation[0]
                if isinstance(first_msg, dict):
                    text = first_msg.get("content", "") or ""
                elif isinstance(first_msg, str):
                    text = first_msg

            toxic = row.get("toxic_chat_tag", False)
            if toxic or not text:
                return {"text": "", "label": BENIGN}
            return {"text": text, "label": BENIGN}

        ds = ds.map(_map, remove_columns=ds.column_names, desc="Mapping arena")
        ds = ds.filter(lambda x: len(x["text"]) > 0)
        logger.info(f"  → {len(ds)} benign rows from arena")
        return ds
    except Exception as e:
        logger.warning(f"  ✗ Failed to load arena: {e}")
        return Dataset.from_dict({"text": [], "label": []})


def load_oasst() -> Dataset:
    """#16: OpenAssistant/oasst1 (~66K benign)."""
    logger.info("Loading OpenAssistant/oasst1...")
    try:
        ds = load_dataset("OpenAssistant/oasst1", split="train", cache_dir=config.data.cache_dir)

        # Filter to prompter messages only
        ds = ds.filter(lambda x: x.get("role", "") == "prompter", desc="Filtering prompters")

        def _map(row):
            text = row.get("text", "") or ""
            # Basic spam check
            labels = row.get("labels", {})
            if isinstance(labels, dict):
                spam = labels.get("spam", 0)
                if isinstance(spam, (int, float)) and spam > 0.5:
                    return {"text": "", "label": BENIGN}
            return {"text": text, "label": BENIGN}

        ds = ds.map(_map, remove_columns=ds.column_names, desc="Mapping oasst1")
        ds = ds.filter(lambda x: len(x["text"]) > 0)
        logger.info(f"  → {len(ds)} benign rows from oasst1")
        return ds
    except Exception as e:
        logger.warning(f"  ✗ Failed to load oasst1: {e}")
        return Dataset.from_dict({"text": [], "label": []})


# ─────────────────────────────────────────────────────────────────────────────
# Processing pipeline — all Arrow-backed, memory-safe
# ─────────────────────────────────────────────────────────────────────────────


ALL_LOADERS = [
    ("Necent", load_necent, True),       # (name, fn, is_large)
    ("TensorTrust", load_tensortrust, True),
    ("WildGuardMix", load_wildguardmix, False),
    ("TrustAIRLab/wild", load_trustailab_wild, False),
    ("neuralchemy", load_neuralchemy, False),
    ("TrustAIRLab/JailbreakLLMs", load_trustailab_jailbreakllms, False),
    ("S-Labs", load_slabs, False),
    ("CentrePourLaSecuriteIA", load_centre_securite, False),
    ("Casey27", load_casey27, False),
    ("deepset", load_deepset, False),
    ("AdvBench", load_advbench, False),
    ("HarmBench", load_harmbench, False),
    ("prodnull", load_prodnull, False),
    ("lmsys-chat-1m", load_lmsys_1m, True),  # Large
    ("lmsys-arena", load_lmsys_arena, False),
    ("oasst1", load_oasst, False),
]


def deduplicate_dataset(ds: Dataset) -> Dataset:
    """Remove duplicates using text hash column — Arrow-backed, memory-safe."""
    logger.info(f"Deduplication: starting with {len(ds)} rows...")

    def _add_hash(example):
        h = hashlib.md5(example["text"].strip().lower().encode()).hexdigest()
        return {"_hash": h}

    ds = ds.map(_add_hash, desc="Computing hashes")

    # Use pandas for efficient dedup (still chunked via Arrow)
    df = ds.to_pandas()
    before = len(df)
    df = df.drop_duplicates(subset=["_hash"], keep="first")
    after = len(df)

    ds = Dataset.from_pandas(df[["text", "label"]], preserve_index=False)
    logger.info(f"Deduplication: {before} → {after} (removed {before - after})")
    return ds


def filter_quality_dataset(ds: Dataset) -> Dataset:
    """Apply quality filters — stays on Arrow."""
    before = len(ds)

    def _quality_check(example):
        text = example["text"].strip()
        if len(text) < config.data.min_text_length:
            return False
        if len(text) > config.data.max_text_length:
            return False
        printable_ratio = sum(c.isprintable() or c.isspace() for c in text) / max(len(text), 1)
        return printable_ratio >= 0.7

    ds = ds.filter(_quality_check, desc="Quality filtering")
    logger.info(f"Quality filter: {before} → {len(ds)} (removed {before - len(ds)})")
    return ds


def balance_classes_dataset(ds: Dataset, strategy: str = "downsample") -> Dataset:
    """Balance class distribution — uses .select() to stay memory-safe."""
    import random
    random.seed(config.data.seed)

    label_col = ds["label"]
    counter = Counter(label_col)
    logger.info(f"Class distribution before balancing: {dict(counter)}")

    if strategy == "downsample":
        min_count = min(counter.values())
        indices_by_label = {}
        for i, label in enumerate(label_col):
            indices_by_label.setdefault(label, []).append(i)

        selected_indices = []
        for label, indices in indices_by_label.items():
            if len(indices) > min_count:
                indices = random.sample(indices, min_count)
            selected_indices.extend(indices)

        random.shuffle(selected_indices)
        ds = ds.select(selected_indices)

    elif strategy == "upsample":
        max_count = max(counter.values())
        indices_by_label = {}
        for i, label in enumerate(label_col):
            indices_by_label.setdefault(label, []).append(i)

        selected_indices = []
        for label, indices in indices_by_label.items():
            if len(indices) < max_count:
                extra = random.choices(indices, k=max_count - len(indices))
                indices = indices + extra
            selected_indices.extend(indices)

        random.shuffle(selected_indices)
        ds = ds.select(selected_indices)

    counter_after = Counter(ds["label"])
    logger.info(f"Class distribution after {strategy}: {dict(counter_after)}")
    return ds


def create_splits(ds: Dataset) -> DatasetDict:
    """Create stratified train/val/test splits."""
    labels = ds["label"]

    all_indices = list(range(len(ds)))

    # First split: train+val vs test
    train_val_idx, test_idx = train_test_split(
        all_indices,
        test_size=config.data.test_size,
        random_state=config.data.seed,
        stratify=labels,
    )

    # Second split: train vs val
    train_val_labels = [labels[i] for i in train_val_idx]
    val_fraction = config.data.val_size / (1 - config.data.test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_fraction,
        random_state=config.data.seed,
        stratify=train_val_labels,
    )

    logger.info(f"Splits: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    return DatasetDict({
        "train": ds.select(train_idx),
        "validation": ds.select(val_idx),
        "test": ds.select(test_idx),
    })


def main():
    parser = argparse.ArgumentParser(description="Prepare Prompt Guard training data")
    parser.add_argument("--output-dir", type=str, default=config.data.output_dir)
    parser.add_argument("--max-samples-per-class", type=int, default=None)
    parser.add_argument("--balance", type=str, default=config.data.balance_strategy,
                        choices=["downsample", "upsample", "none"])
    parser.add_argument("--skip-large", action="store_true",
                        help="Skip Necent (1.18M), TensorTrust (170K), lmsys-1m — for faster runs or low-RAM envs")
    args = parser.parse_args()

    if args.max_samples_per_class:
        config.data.max_samples_per_class = args.max_samples_per_class

    # ── Step 1: Load all datasets as Arrow-backed HF Datasets ──
    logger.info("=" * 60)
    logger.info("Step 1: Loading all datasets (Arrow-backed)...")
    logger.info("=" * 60)

    loaded_datasets = []
    source_counts = {}

    for name, loader_fn, is_large in ALL_LOADERS:
        if args.skip_large and is_large:
            logger.info(f"  Skipping {name} (--skip-large)")
            continue
        try:
            ds = loader_fn()
            if len(ds) > 0:
                loaded_datasets.append(ds)
                source_counts[name] = len(ds)
            else:
                source_counts[name] = 0
        except Exception as e:
            logger.error(f"  Failed to load {name}: {e}")
            source_counts[name] = 0

    logger.info(f"\nSource breakdown:")
    for name, count in source_counts.items():
        logger.info(f"  {name}: {count:,}")

    if not loaded_datasets:
        logger.error("No data loaded! Check your internet connection and dataset access.")
        sys.exit(1)

    # ── Step 2: Concatenate into a single Arrow Dataset ──
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Concatenating datasets...")
    logger.info("=" * 60)
    combined = concatenate_datasets(loaded_datasets)
    logger.info(f"  Combined: {len(combined):,} rows")

    # Free references to individual datasets
    del loaded_datasets

    # ── Step 3: Deduplicate ──
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Deduplication...")
    logger.info("=" * 60)
    combined = deduplicate_dataset(combined)

    # ── Step 4: Quality filter ──
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Quality filtering...")
    logger.info("=" * 60)
    combined = filter_quality_dataset(combined)

    # ── Step 5: Balance classes ──
    logger.info("\n" + "=" * 60)
    logger.info("Step 5: Balancing classes...")
    logger.info("=" * 60)
    balance = args.balance if args.balance != "none" else None
    if balance:
        combined = balance_classes_dataset(combined, strategy=balance)

    # ── Step 6: Cap per class if requested ──
    if config.data.max_samples_per_class:
        import random
        random.seed(config.data.seed)
        logger.info(f"\nCapping to {config.data.max_samples_per_class} per class...")

        label_col = combined["label"]
        indices_by_label = {}
        for i, label in enumerate(label_col):
            indices_by_label.setdefault(label, []).append(i)

        selected = []
        for label, indices in indices_by_label.items():
            if len(indices) > config.data.max_samples_per_class:
                indices = random.sample(indices, config.data.max_samples_per_class)
            selected.extend(indices)
        random.shuffle(selected)
        combined = combined.select(selected)

    # ── Step 7: Create splits ──
    logger.info("\n" + "=" * 60)
    logger.info("Step 6: Creating train/val/test splits...")
    logger.info("=" * 60)
    dataset_dict = create_splits(combined)

    # ── Step 8: Save ──
    logger.info("\n" + "=" * 60)
    logger.info("Step 7: Saving processed dataset...")
    logger.info("=" * 60)

    os.makedirs(args.output_dir, exist_ok=True)
    dataset_dict.save_to_disk(args.output_dir)

    # Print final summary
    logger.info("\n" + "=" * 60)
    logger.info("✓ Dataset preparation complete!")
    logger.info("=" * 60)
    for split_name, split_ds in dataset_dict.items():
        label_counts = Counter(split_ds["label"])
        logger.info(f"  {split_name}: {len(split_ds):,} samples — {dict(label_counts)}")
    logger.info(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
