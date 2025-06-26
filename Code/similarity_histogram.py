import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

embedding_path = "/Volumes/T9/nutrition5k_dataset/REMOVED_OUTLIERS/clip_embeddings_segmented_cleaned.pkl"
output_path = "/Volumes/T9/nutrition5k_dataset/REMOVED_OUTLIERS/similarity_histogram_all_images.png"

with open(embedding_path, "rb") as f:
    data = pickle.load(f)

valid_items = []
for dish_id, info in data.items():
    if "embedding" in info:
        vec = np.array(info["embedding"])
        if vec.ndim == 1 and vec.shape[0] == 512:
            valid_items.append((dish_id, vec))

image_paths = [item[0] for item in valid_items]
embeddings = np.stack([item[1] for item in valid_items])

all_similarities = []
print("Computing similarities across all images...")

for i in tqdm(range(len(embeddings))):
    sims = cosine_similarity(embeddings[i].reshape(1, -1), embeddings)[0]
    filtered = sims[sims < 0.9999]
    all_similarities.extend(filtered)

bins = np.linspace(0.5, 1.0, 51)
hist, bin_edges = np.histogram(all_similarities, bins=bins)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
derivative = np.diff(hist)
elbow_index = np.argmin(derivative)
elbow_similarity = bin_centers[elbow_index + 1]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

ax1.bar(bin_centers, hist, width=0.009, color='skyblue', edgecolor='black')
ax1.set_title("Cosine Similarities Histogram (Excluding Self-Matches)")
ax1.set_ylabel("Frequency")
ax1.grid(True)
ax1.axvline(x=elbow_similarity, color='red', linestyle='--', label=f'Elbow ~ {elbow_similarity:.3f}')
ax1.legend()

ax2.plot(bin_centers[1:], derivative, marker='o', color='gray')
ax2.axvline(x=elbow_similarity, color='red', linestyle='--')
ax2.set_title("Rate of Change in Similarity Histogram (1st Derivative)")
ax2.set_xlabel("Cosine Similarity")
ax2.set_ylabel(r"$\\Delta$ Frequency")
ax2.grid(True)

plt.tight_layout()
plt.savefig(output_path)
print(f"Histogram saved: {output_path}")
print(f"Elbow similarity threshold estimated: {elbow_similarity:.3f}")
