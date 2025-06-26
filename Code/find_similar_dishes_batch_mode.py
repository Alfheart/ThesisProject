import os
import pickle
import numpy as np
import torch
import clip
from PIL import Image

embedding_file = "Splits/train_embeddings.pkl"
image_dir = "/Volumes/T9/nutrition5k_dataset/imagery/extracted_frames"
preprocessed_dir = "/Volumes/T9/nutrition5k_dataset/imagery/preprocessed_segmented"
save_dir = "/Volumes/T9/nutrition5k_dataset/REMOVED_OUTLIERS/Similar Dishes"
os.makedirs(save_dir, exist_ok=True)
threshold = 0.825
top_n = 0

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

with open(embedding_file, "rb") as f:
    db = pickle.load(f)

dish_ids = list(db.keys())
embeddings = np.stack([db[d]["embedding"] for d in dish_ids])

input_batch_dir = input("Enter the path to a test batch folder (with .jpg images): ").strip()
if not os.path.isdir(input_batch_dir):
    print("Batch folder not found.")
    exit(1)

for input_image_name in os.listdir(input_batch_dir):
    if not input_image_name.endswith(".jpg") or input_image_name.startswith("._"):
        continue

    input_image_path = os.path.join(input_batch_dir, input_image_name)
    print(f"\nQuery image file path: {input_image_path}")

    query_image_name = os.path.basename(input_image_path)
    query_dish_id = query_image_name.replace("dish_", "").replace(".jpg", "")
    preprocessed_query_path = os.path.join(preprocessed_dir, f"pp_dish_{query_dish_id}.png")

    if not os.path.exists(preprocessed_query_path):
        print("Background-removed version of this image not found.")
        continue

    img = preprocess(Image.open(preprocessed_query_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        query_emb = model.encode_image(img)[0].cpu().numpy()

    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
    sims = np.dot(embeddings, query_emb) / (norms + 1e-8)

    above_threshold_indices = np.where(sims >= threshold)[0]
    sorted_indices = above_threshold_indices[np.argsort(sims[above_threshold_indices])[::-1]]

    if len(sorted_indices) == 0:
        print(f"No similar dishes found for {query_image_name}")
        continue

    print(f"\n{len(sorted_indices)} similar images above threshold ({threshold}):")
    for idx in sorted_indices[:top_n]:
        did = dish_ids[idx]
        row = db[did]
        sim_score = sims[idx]
        full_img_path = os.path.join(image_dir, f"{did}.jpg")

        print(f"\nImage: {full_img_path}")
        print(f"Similarity Score: {sim_score:.4f}")
        print(f"   Mass (g):    {row['mass']}")
        print(f"   Calories:    {row['calories']}")
        print(f"   Protein (g): {row['protein']}")
        print(f"   Fat (g):     {row['fat']}")
        print(f"   Carbs (g):   {row['carbs']}")

# DEBUGGING
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
#
#     fig, axes = plt.subplots(2, 3, figsize=(15, 10))
#     axes = axes.flatten()
#
#     query_img = Image.open(input_image_path).resize((256, 256))
#     axes[0].imshow(query_img)
#     axes[0].set_title(f"Query: {query_dish_id}")
#     axes[0].axis('off')
#
#     for i, idx in enumerate(sorted_indices[:5]):
#         did = dish_ids[idx]
#         matched_img_path = os.path.join(preprocessed_dir, f"pp_dish_{did}.png")
#         matched_img = Image.open(matched_img_path).resize((256, 256))
#         axes[i + 1].imshow(matched_img)
#         axes[i + 1].set_title(f"Sim: {sims[idx]:.4f}")
#         axes[i + 1].axis('off')
#
#     plt.tight_layout()
#     plt.show()
