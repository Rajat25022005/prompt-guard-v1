"""
Prompt Guard v1 — One-Click Colab Training Script
Run this SINGLE cell in Google Colab to do everything:
Setup → Data Prep → Train (all epochs) → Evaluate → Download

Before running: Runtime → Change runtime type → T4 GPU
"""

# ============================================================
# STEP 1: Mount Google Drive (checkpoint safety net)
# ============================================================
from google.colab import drive
drive.mount('/content/drive')

import os
DRIVE_DIR = "/content/drive/MyDrive/prompt-guard-v1"
os.makedirs(f"{DRIVE_DIR}/checkpoints", exist_ok=True)
os.makedirs(f"{DRIVE_DIR}/results", exist_ok=True)
print("✅ Step 1/7: Drive mounted")

# ============================================================
# STEP 2: Clone repo & install dependencies
# ============================================================
os.chdir("/content")
if not os.path.exists("/content/prompt-guard-v1"):
    os.system("git clone https://github.com/Rajat25022005/prompt-guard-v1.git")
os.chdir("/content/prompt-guard-v1")
os.system("pip install -q -r requirements.txt")
print("✅ Step 2/7: Repo cloned & dependencies installed")

# ============================================================
# STEP 3: Verify GPU
# ============================================================
import torch
assert torch.cuda.is_available(), "❌ No GPU! Go to Runtime → Change runtime type → T4 GPU"
gpu_name = torch.cuda.get_device_name(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"✅ Step 3/7: GPU ready — {gpu_name} ({vram:.1f} GB)")

# ============================================================
# STEP 4: Prepare dataset
# ============================================================
print("\n" + "=" * 60)
print("Step 4/7: Preparing dataset (this takes 30-60 min)...")
print("=" * 60)

DATA_DIR = "./data/processed"
DRIVE_DATA = f"{DRIVE_DIR}/data_processed"

# Check if we have a Drive backup from a previous run
if os.path.exists(DRIVE_DATA) and os.path.exists(f"{DRIVE_DATA}/dataset_info.json"):
    print("Found dataset backup on Drive — restoring...")
    os.system(f"cp -r {DRIVE_DATA} {DATA_DIR}")
    print("✅ Dataset restored from Drive!")
else:
    # Use --skip-large for safer RAM usage on free Colab
    # Remove --skip-large if you want ALL 16 datasets (~1M+ samples)
    exit_code = os.system("python data/prepare_dataset.py --skip-large --output-dir ./data/processed")
    if exit_code != 0:
        print("⚠️ Full dataset failed, trying with max cap...")
        os.system("python data/prepare_dataset.py --skip-large --max-samples-per-class 50000 --output-dir ./data/processed")
    # Backup to Drive
    os.system(f"cp -r {DATA_DIR} {DRIVE_DATA}")
    print("✅ Dataset saved + backed up to Drive!")

print("✅ Step 4/7: Dataset ready")

# ============================================================
# STEP 5: Train all epochs
# ============================================================
print("\n" + "=" * 60)
print("Step 5/7: Training (all 3 epochs, ~4-5 hours)...")
print("=" * 60)

CKPT_DIR = f"{DRIVE_DIR}/checkpoints"

# Check for existing checkpoint to resume from
import glob
existing = sorted(glob.glob(f"{CKPT_DIR}/checkpoint-*"))
resume_flag = f"--resume-from {existing[-1]}" if existing else ""
if resume_flag:
    print(f"🔄 Resuming from: {existing[-1]}")

exit_code = os.system(
    f"python train.py "
    f"--epochs 3 "
    f"--output-dir {CKPT_DIR} "
    f"--data-dir {DATA_DIR} "
    f"{resume_flag}"
)

if exit_code != 0:
    print("⚠️ Training may have errored — check logs above")
else:
    print("✅ Step 5/7: Training complete!")

# ============================================================
# STEP 6: Evaluate
# ============================================================
print("\n" + "=" * 60)
print("Step 6/7: Evaluating model...")
print("=" * 60)

RESULTS_DIR = f"{DRIVE_DIR}/results"
os.system(
    f"python evaluate.py "
    f"--checkpoint {CKPT_DIR}/best "
    f"--data-dir {DATA_DIR} "
    f"--output-dir {RESULTS_DIR}"
)

# Show confusion matrix
try:
    from IPython.display import Image, display
    cm_path = f"{RESULTS_DIR}/confusion_matrix.png"
    if os.path.exists(cm_path):
        display(Image(cm_path))
except Exception:
    pass

# Show metrics
import json
report_path = f"{RESULTS_DIR}/classification_report.json"
if os.path.exists(report_path):
    with open(report_path) as f:
        report = json.load(f)
    print(f"\n📊 Test Results:")
    print(f"   Accuracy:     {report.get('accuracy', 'N/A'):.4f}")
    print(f"   F1 (macro):   {report.get('macro avg', {}).get('f1-score', 'N/A'):.4f}")
    print(f"   F1 (BENIGN):  {report.get('BENIGN', {}).get('f1-score', 'N/A'):.4f}")
    print(f"   F1 (INJECT):  {report.get('INJECTION', {}).get('f1-score', 'N/A'):.4f}")
    print(f"   F1 (JAILBK):  {report.get('JAILBREAK', {}).get('f1-score', 'N/A'):.4f}")

print("✅ Step 6/7: Evaluation complete!")

# ============================================================
# STEP 7: Zip & download model
# ============================================================
print("\n" + "=" * 60)
print("Step 7/7: Packaging model for download...")
print("=" * 60)

import shutil
from google.colab import files

model_dir = f"{CKPT_DIR}/best"
zip_path = "/content/prompt-guard-v1-model"

if os.path.exists(model_dir):
    shutil.make_archive(zip_path, "zip", model_dir)
    size_mb = os.path.getsize(f"{zip_path}.zip") / (1024 * 1024)
    print(f"📦 Model zipped: {size_mb:.1f} MB")
    print("📥 Download starting...")
    files.download(f"{zip_path}.zip")
else:
    print(f"⚠️ Best model not found at {model_dir}")
    print("Check your checkpoints on Drive")

# ============================================================
# DONE!
# ============================================================
print("\n" + "=" * 60)
print("🎉 ALL DONE! Your Prompt Guard model is trained!")
print("=" * 60)
print(f"\n📁 Everything is saved on Google Drive at:")
print(f"   {DRIVE_DIR}/checkpoints/best  — trained model")
print(f"   {DRIVE_DIR}/results/          — evaluation metrics")
print(f"\nTo use locally:")
print(f"   from inference import PromptGuard")
print(f'   guard = PromptGuard("./path-to-unzipped-model")')
print(f'   guard.classify("your prompt here")')
