import os
import random
import csv

true_test_path = 'true_test.csv'
preprocessed_img_dir = '../../imagery/preprocessed_segmented'
output_dir = './Test Batches'

os.makedirs(output_dir, exist_ok=True)

dish_ids = []
with open(true_test_path, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        dish_id = row['dish_id']
        dish_ids.append(dish_id)

valid_ids = []
for dish_id in dish_ids:
    img_path = os.path.join(preprocessed_img_dir, f'pp_{dish_id}.png')
    if os.path.exists(img_path):
        valid_ids.append(dish_id)

print(f"Total valid dish IDs with images: {len(valid_ids)}")

random.shuffle(valid_ids)
batch_size = 10
num_batches = len(valid_ids) // batch_size
remainder = len(valid_ids) % batch_size

for i in range(num_batches):
    batch_ids = valid_ids[i * batch_size: (i + 1) * batch_size]
    with open(os.path.join(output_dir, f'test_batch_{i + 1}.txt'), 'w') as f:
        f.write('\n'.join(batch_ids))

if remainder > 0:
    last_batch_ids = valid_ids[-remainder:]
    with open(os.path.join(output_dir, 'last_batch.txt'), 'w') as f:
        f.write('\n'.join(last_batch_ids))

print(f"Created {num_batches} full batches and 1 last batch with {remainder} IDs.")
