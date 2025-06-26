import os
import pickle
import numpy as np
import torch
import clip
from PIL import Image
import pandas as pd
# DEBUGGING
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches

embedding_file = "/Volumes/T9/nutrition5k_dataset/REMOVED_OUTLIERS/Splits/train_embeddings.pkl"
metadata_file = "/Volumes/T9/nutrition5k_dataset/REMOVED_OUTLIERS/true_metadata_cafe1_cleaned.csv"
preprocessed_dir = "/Volumes/T9/nutrition5k_dataset/imagery/preprocessed_segmented"
output_dir = "/Volumes/T9/nutrition5k_dataset/REMOVED_OUTLIERS/Batch Results"
os.makedirs(output_dir, exist_ok=True)

with open(embedding_file, "rb") as f:
    db = pickle.load(f)

metadata = pd.read_csv(metadata_file)
metadata = metadata.set_index("dish_id")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

batch_folder = input("Enter path to batch folder (with 10 images): ").strip()
batch_images = [f for f in os.listdir(batch_folder) if f.endswith(('.jpg', '.png'))]
batch_ids = [i.replace("dish_", "").replace(".jpg", "").replace(".png", "") for i in batch_images]

embeddings = {}
for img_id in batch_ids:
    img_path = os.path.join(preprocessed_dir, f"pp_dish_{img_id}.png")
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        continue
    img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)[0].cpu().numpy()
    embeddings[img_id] = emb

data = []
for img_id, query_emb in embeddings.items():
    sims = {}
    for db_id, entry in db.items():
        db_emb = entry["embedding"]
        sim = np.dot(query_emb, db_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(db_emb) + 1e-8)
        sims[db_id] = sim

    sorted_sims = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:5]

    for db_id, score in sorted_sims:
        row = metadata.loc[int(db_id)]
        data.append({
            "query_image": img_id,
            "matched_dish": db_id,
            "similarity": round(score, 4),
            "mass": row.mass,
            "calories": row.calories,
            "protein": row.protein,
            "fat": row.fat,
            "carbs": row.carbs
        })

result = pd.DataFrame(data)
out_file = os.path.join(output_dir, f"batch_results.csv")
result.to_csv(out_file, index=False)
print(f"Saved results to {out_file}")

# DEBUGGING
# for img_id in embeddings.keys():
#     query_img_path = os.path.join(batch_folder, f"dish_{img_id}.jpg")
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     axes = axes.flatten()
#
#     query_img = Image.open(query_img_path).resize((256, 256))
#     axes[0].imshow(query_img)
#     axes[0].set_title(f"Query: {img_id}")
#     axes[0].axis('off')
#
#     matches = result[result.query_image == img_id].reset_index()
#     for i in range(5):
#         img_name = matches.loc[i, 'matched_dish']
#         matched_img_path = os.path.join(preprocessed_dir, f"pp_dish_{img_name}.png")
#         matched_img = Image.open(matched_img_path).resize((256, 256))
#         axes[i+1].imshow(matched_img)
#         axes[i+1].set_title(f"Sim: {matches.loc[i, 'similarity']}")
#         axes[i+1].axis('off')
#
#     plt.tight_layout()
#     plt.show()
