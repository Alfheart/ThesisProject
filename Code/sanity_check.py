import os
import pickle
import numpy as np
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

embedding_file = "../Splits/train_embeddings.pkl"
image_dir = "/Volumes/T9/nutrition5k_dataset/imagery/preprocessed_segmented"
save_dir = "Results"
os.makedirs(save_dir, exist_ok=True)
threshold = 0.825

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

with open(embedding_file, "rb") as f:
    db = pickle.load(f)

dish_ids = list(db.keys())
embeddings = np.stack([db[d]["embedding"] for d in dish_ids])

input_image_path = input("Enter the path to your query image (.png or .jpg): ").strip()
if not os.path.isfile(input_image_path):
    print("Image not found.")
    exit(1)

query_image_name = os.path.basename(input_image_path)
if query_image_name.startswith("pp_dish_"):
    preview_name = f"sanity_{query_image_name.split('.')[0]}.png"
else:
    preview_name = f"sanity_{os.path.splitext(query_image_name)[0]}.png"

out_path = os.path.join(save_dir, preview_name)

img = preprocess(Image.open(input_image_path).convert("RGB")).unsqueeze(0).to(device)
with torch.no_grad():
    query_emb = model.encode_image(img)[0].cpu().numpy()

norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
sims = np.dot(embeddings, query_emb) / (norms + 1e-8)

above_threshold_indices = np.where(sims >= threshold)[0]
sorted_indices = above_threshold_indices[np.argsort(sims[above_threshold_indices])][:9]

result_paths = []
result_scores = []
result_meta = []

for idx in sorted_indices:
    did = dish_ids[idx]
    row = db[did]
    img_path = os.path.join(image_dir, f"{did.replace('dish_', 'pp_dish_')}.png")
    sim_score = sims[idx]
    result_paths.append(img_path)
    result_scores.append(sim_score)
    result_meta.append({
        "mass": row["mass"],
        "calories": row["calories"],
        "protein": row["protein"],
        "fat": row["fat"],
        "carbs": row["carbs"]
    })

def plot_image_grid(query_img_path, query_name, img_paths, sim_scores, meta_list, out_path):
    fig, axes = plt.subplots(6, 4, figsize=(16, 20), gridspec_kw={'height_ratios': [0.5, 1, 1, 1, 1, 1]})
    axes = axes.flatten()

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    headers = ["Dish", "Nutritional Values", "Dish", "Nutritional Values"]
    for i in range(4):
        axes[i].text(0.5, 0.5, headers[i], ha='center', va='center', fontsize=24, fontweight="bold")

    query_img = Image.open(query_img_path).resize((256, 256))
    axes[4].imshow(query_img)
    axes[5].text(
        0.5, 0.5,
        "Mass (g):    —\n"
        "Calories:    —\n"
        "Protein (g): —\n"
        "Fat (g):     —\n"
        "Carbs (g):   —",
        fontsize=24, ha='center', va='center'
    )
    for ax in [axes[4], axes[5]]:
        ax.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=1))

    max_pairs = (len(axes) - 6) // 2
    for i in range(min(len(img_paths), max_pairs)):
        img = Image.open(img_paths[i]).resize((256, 256))
        score = sim_scores[i]
        info = meta_list[i]
        img_ax = 6 + i * 2
        txt_ax = 7 + i * 2

        axes[img_ax].imshow(img)
        axes[txt_ax].text(
            0.5, 0.5,
            f"Similarity:  {score:.4f}\n"
            f"Mass (g):    {info['mass']:.1f}\n"
            f"Calories:    {info['calories']:.1f}\n"
            f"Protein (g): {info['protein']:.1f}\n"
            f"Fat (g):     {info['fat']:.1f}\n"
            f"Carbs (g):   {info['carbs']:.1f}",
            fontsize=24,
            ha='center',
            va='center'
        )
        for ax in [axes[img_ax], axes[txt_ax]]:
            ax.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black', linewidth=1))

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"\nSanity preview saved to {out_path}")

