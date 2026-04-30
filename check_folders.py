import os

source_folder = "garbage_classification/garbage_classification"

items = os.listdir(source_folder)
print("Folders found inside garbage_classification:")
for item in items:
    item_path = os.path.join(source_folder, item)
    if os.path.isdir(item_path):
        count = len(os.listdir(item_path))
        print(f"  '{item}' → {count} images")