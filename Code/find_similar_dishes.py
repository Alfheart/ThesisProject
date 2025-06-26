import os
import pickle
import numpy as np
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

embedding_file = "Splits/train_embeddings.pkl"
image_dir = "/Volumes/T9/nutrition5k_dataset/imagery/preprocessed_segmented"
save_dir = "/Volumes/T9/nutrition5k_dataset/REMOVED_OUTLIERS/Similar Dishes"
os.makedirs(save_dir, exist_ok=True)
threshold = 0.825

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

with open(embedding_file, "rb") as f:
    db = pickle.load(f)

dish_ids = list(db)
embeddings = np.stack([db[d]["embedding"] for d in dish_ids])

input_image_path = input("Enter path to query image (.png or .jpg): ").strip()
if not os.path.isfile(input_image_path):
    print("Image not found.")
    exit()

query_name = os.path.basename(input_image_path).removesuffix(".png").removesuffix(".jpg")
preview_name = f"similar_to_{query_name}.png"
out_path = f"{save_dir}/{preview_name}"

img = preprocess(Image.open(input_image_path).convert("RGB")).unsqueeze(0).to(device)
with torch.no_grad():
    query_emb = model.encode_image(img).cpu().numpy()[0]

norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
sims = np.dot(embeddings, query_emb) / (norms + 1e-8)

idx = np.where(sims >= threshold)[0]
sorted_idx = idx[np.argsort(sims[idx])[::-1]]

if not len(sorted_idx):
    print(f"No similar dishes above threshold {threshold}.")
    exit()

result_paths = []
result_scores = []
result_meta = []
for i in sorted_idx[:9]:
    did = dish_ids[i]
    row = db[did]
    path = f"{image_dir}/{did.replace('dish_', 'pp_dish_')}.png"
    result_paths.append(path)
    result_scores.append(sims[i])
    result_meta.append({
        "mass": row["mass"],
        "calories": row["calories"],
        "protein": row["protein"],
        "fat": row["fat"],
        "carbs": row["carbs"]
    })
    print(f"\nImage: {path}")
    print(f"Similarity: {sims[i]:.4f}")
    print(f"Mass: {row['mass']}, Calories: {row['calories']}, Protein: {row['protein']}, Fat: {row['fat']}, Carbs: {row['carbs']}")

def plot_grid(query_img, img_paths, scores, meta, save_path):
    fig, axes = plt.subplots(6, 4, figsize=(16, 20), gridspec_kw={'height_ratios': [0.5, 1, 1, 1, 1, 1]})
    axes = axes.flatten()

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis("off")

    headers = ["Dish", "Nutritional Values", "Dish", "Nutritional Values"]
    for i in range(4):
        axes[i].text(0.5, 0.5, headers[i], ha='center', va='center', fontsize=24, fontweight="bold")

    img = Image.open(query_img).resize((256, 256))
    axes[4].imshow(img)
    axes[5].text(0.5, 0.5, "Mass: —\nCalories: —\nProtein: —\nFat: —\nCarbs: —", fontsize=24, ha='center', va='center')
    for ax in [axes[4], axes[5]]:
        ax.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black'))

    for i, (path, score, info) in enumerate(zip(img_paths, scores, meta)):
        img_ax = 6 + i * 2
        txt_ax = 7 + i * 2

        img = Image.open(path).resize((256, 256))
        axes[img_ax].imshow(img)
        axes[txt_ax].text(
            0.5, 0.5,
            f"Sim: {score:.4f}\nMass: {info['mass']:.1f}\nCalories: {info['calories']:.1f}\nProtein: {info['protein']:.1f}\nFat: {info['fat']:.1f}\nCarbs: {info['carbs']:.1f}",
            fontsize=24, ha='center', va='center'
        )
        for ax in [axes[img_ax], axes[txt_ax]]:
            ax.add_patch(patches.Rectangle((0, 0), 1, 1, fill=False, edgecolor='black'))

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"\nPreview saved to {save_path}")

if all(os.path.exists(p) for p in result_paths):
    plot_grid(input_image_path, result_paths, result_scores, result_meta, out_path)
else:
    print("Some images missing, preview not generated.")
