import os
import random
import shutil

all_images_dir = "simple_images/all_images"
train_dir = "/images/trainimages"
test_dir = "/images/testimages"
split_ratio = 0.75

# Get all file names in the 'all_images' directory
all_files = os.listdir(all_images_dir)

# Group files with similar names
file_groups = {}
for filename in all_files:
    name, ext = os.path.splitext(filename)
    if name not in file_groups:
        file_groups[name] = {'name': name, 'exts': []}
    file_groups[name]['exts'].append(ext)
# Move %75 of the dataset to the 'train' directory and %25 to the 'test' directory
num_train_samples = int(len(file_groups) * split_ratio)
train_samples = {}
random_indices = random.sample(range(len(file_groups)), num_train_samples)

for i, sample in enumerate(file_groups.values()):
    dest_dir = train_dir if i in random_indices else test_dir
    for ext in sample['exts']:
        source_path = os.path.join(all_images_dir, f"{sample['name']}{ext}")
        dest_path = os.path.join(dest_dir, f"{sample['name']}{ext}")
        shutil.move(source_path, dest_path)
