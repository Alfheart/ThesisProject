import os
import torch
import clip
from PIL import Image
import pandas as pd
import pickle
from tqdm import tqdm

image_dir = "/Volumes/T9/nutrition5k_dataset/imagery/preprocessed_segmented"
metadata_csv = "/Volumes/T9/nutrition5k_dataset/REMOVED_OUTLIERS/true_metadata_cafe1_cleaned.csv"
output_file = "/Volumes/T9/nutrition5k_dataset/REMOVED_OUTLIERS/clip_embeddings_segmented_cleaned.pkl"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

df = pd.read_csv(metadata_csv)
df["dish_id"] = df["dish_id"].astype(str)
df["mass"] = pd.to_numeric(df["mass"], errors="coerce").fillna(0.0)

results = {}

print("Embedding segmented images...")

for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding"):
    dish_id = row["dish_id"].removeprefix("dish_")
    img_path = f"{image_dir}/pp_dish_{dish_id}.png"
    if not os.path.exists(img_path):
        continue

    try:
        img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            embedding = model.encode_image(img).cpu().numpy()[0]
        results[f"dish_{dish_id}"] = {
            "embedding": embedding,
            "calories": row["calories"],
            "fat": row["fat"],
            "carbs": row["carbs"],
            "protein": row["protein"],
            "mass": row["mass"]
        }
    except Exception as e:
        print(f"Error on dish_{dish_id}: {e}")

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "wb") as f:
    pickle.dump(results, f)

print(f"Saved {len(results)} embeddings to {output_file}")
