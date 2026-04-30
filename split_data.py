import os
import shutil
import random

source_folder = "garbage_classification/garbage_classification"

dest_folder = "dataset"

for split in ['train', 'test']:
    for category in ['paper', 'glass', 'organic']:
        os.makedirs(os.path.join(dest_folder, split, category), exist_ok=True)

mapping = {
    'paper': ['paper', 'cardboard'],
    'glass': ['green-glass', 'brown-glass'],
    'organic': ['biological']
}

for category, source_folders in mapping.items():
    all_images = []

    for src in source_folders:
        src_path = os.path.join(source_folder, src)
        if os.path.exists(src_path):
            images = os.listdir(src_path)
            for img in images:
                all_images.append(os.path.join(src_path, img))

    random.shuffle(all_images)

    split_index = int(0.8 * len(all_images))
    train_images = all_images[:split_index]
    test_images = all_images[split_index:]

    for img_path in train_images:
        shutil.copy2(img_path, os.path.join(dest_folder, 'train', category))

    for img_path in test_images:
        shutil.copy2(img_path, os.path.join(dest_folder, 'test', category))

    print(f"{category}: {len(train_images)} train, {len(test_images)} test")

print("\nDataset split complete!")