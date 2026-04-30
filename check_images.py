import os

for category in ['glass', 'paper', 'organic']:
    path = f"dataset/test/{category}"
    images = os.listdir(path)
    print(f"\n{category.upper()} - first 5 images:")
    for img in images[:5]:
        print(f"  {img}")