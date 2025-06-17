import os
import shutil
import random
from collections import defaultdict
from pathlib import Path

# === 1. Define paths ===
original_dset = Path("/home/nvidia/git/tsfl/dset")
output_root = Path("/home/nvidia/spd")

# === 2. Clear output directory ===
if output_root.exists():
    shutil.rmtree(output_root)
output_root.mkdir(parents=True, exist_ok=True)

# === 3. Define target subsets ===
subsets = ["pre", "js11", "js12", "js13", "rp11", "rp12", "rp13", "rp21", "rp22", "rp23"]

# === 4. Load all class names ===
all_classes = [folder.name for folder in original_dset.iterdir() if folder.is_dir()]
random.seed(42)

# === 5. Create subset directories with 3–8 random class directories ===
subset_class_map = defaultdict(list)

for subset in subsets:
    subset_dir = output_root / subset
    subset_dir.mkdir(parents=True, exist_ok=True)

    selected_classes = random.sample(all_classes, random.randint(3, 8))
    for cls in selected_classes:
        class_dir = subset_dir / cls
        class_dir.mkdir(parents=True, exist_ok=True)
        subset_class_map[subset].append(cls)

# === 6. Count how many times each class appears across subsets ===
class_occurrence_count = defaultdict(int)
for subset_classes in subset_class_map.values():
    for cls in subset_classes:
        class_occurrence_count[cls] += 1

# === 7. Load image paths from original dataset per class ===
class_image_pool = dict()
class_image_quota = dict()

for cls in all_classes:
    class_folder = original_dset / cls
    images = sorted([img for img in class_folder.iterdir() if img.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]])
    class_image_pool[cls] = images
    occur = class_occurrence_count.get(cls, 0)
    class_image_quota[cls] = (len(images) // occur) if occur > 0 else 0

# === 8. Distribute images to each subset/class directory ===
class_distribution_tracker = defaultdict(int)

for subset in subsets:
    for cls in subset_class_map[subset]:
        quota = class_image_quota[cls]
        start_idx = class_distribution_tracker[cls]
        end_idx = start_idx + quota
        assigned_images = class_image_pool[cls][start_idx:end_idx]
        class_distribution_tracker[cls] += quota

        for img in assigned_images:
            dst = output_root / subset / cls / img.name
            shutil.copy(img, dst)

print("✅ Image distribution completed successfully.")
