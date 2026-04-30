import os
import random

path = "dataset/train/organic"
images = os.listdir(path)
print(f"Total organic training images: {len(images)}")
print(f"\nRandom 20 filenames:")
for img in random.sample(images, min(20, len(images))):
    print(f"  {img}")