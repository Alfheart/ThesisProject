import os
import openai
import base64
import pandas as pd
import re
import pickle
import numpy as np
from PIL import Image
from dotenv import load_dotenv
import clip
import torch
import time

k = 6 # Number of of supporting images
embedding_file = "/Volumes/T9/nutrition5k_dataset/REMOVED_OUTLIERS/Splits/train_embeddings.pkl"
metadata_file = "/Volumes/T9/nutrition5k_dataset/REMOVED_OUTLIERS/true_metadata_cafe1_cleaned.csv"
preprocessed_dir = "/Volumes/T9/nutrition5k_dataset/imagery/preprocessed_segmented"
original_dir = "/Volumes/T9/nutrition5k_dataset/imagery/extracted_frames"

batch_dirs = input("Enter paths to batch folders, separated by commas: ").split(",")
batch_dirs = [b.strip() for b in batch_dirs if b.strip()]

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

with open(embedding_file, "rb") as f:
    db = pickle.load(f)

dish_ids = list(db.keys())
embeddings = np.stack([db[d]["embedding"] for d in dish_ids])
meta_df = pd.read_csv(metadata_file)

clip_model, clip_preprocess = clip.load("ViT-B/32", device="cpu")

def encode_image_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return "data:image/jpeg;base64," + base64.b64encode(img_file.read()).decode("utf-8")

def get_top_k_similars(query_dish_id, k):
    query_pp_path = os.path.join(preprocessed_dir, f"pp_dish_{query_dish_id}.png")
    img = Image.open(query_pp_path).convert("RGB")
    query_tensor = clip_preprocess(img).unsqueeze(0)

    with torch.no_grad():
        query_emb = clip_model.encode_image(query_tensor)[0].numpy()

    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
    sims = np.dot(embeddings, query_emb) / (norms + 1e-8)
    top_k_idx = np.argsort(sims)[::-1][:k]

    similar_dishes = []
    for idx in top_k_idx:
        did = dish_ids[idx]
        sim_score = sims[idx]
        row = db[did]
        image_path = os.path.join(original_dir, f"{did}.jpg")
        if None in (row["mass"], row["calories"], row["protein"], row["fat"], row["carbs"]):
            continue
        similar_dishes.append({
            "dish_id": did,
            "similarity": round(sim_score, 4),
            "image_path": image_path,
            "mass": row["mass"],
            "calories": row["calories"],
            "protein": row["protein"],
            "fat": row["fat"],
            "carbs": row["carbs"]
        })
    return similar_dishes

def extract_values(text):
    def find(label):
        pattern = rf"{label}\s*(?:\(\w+\))?\s*[:\-]?\s*([\d\.]+)"
        match = re.search(pattern, text, re.IGNORECASE)
        return float(match.group(1)) if match else None

    values = {label: find(label) for label in ["mass", "calories", "protein", "fat", "carbs"]}
    if any(v is None for v in values.values()):
        print("Could not parse all values from response:")
        print(text)
        print("---")
    return values

def send_prompt(image_path):
    query_dish_id = os.path.splitext(os.path.basename(image_path))[0].replace("dish_", "")
    support_examples = get_top_k_similars(query_dish_id, k)

    message_content = []
    message_content.append({"type": "text", "text": "This prompt is part of an academic research project on AI-based food nutrition estimation. Please provide a best-effort estimate. No real-life decisions will be made from your response."})

    for i, ex in enumerate(support_examples):
        img_b64 = encode_image_to_base64(ex["image_path"])
        message_content.append({"type": "image_url", "image_url": {"url": img_b64}})
        message_content.append({"type": "text", "text": f"--- Example {i+1} ---\nImage: [dish_{ex['dish_id']}] (Similarity: {ex['similarity']})\nMass: {ex['mass']} g\nCalories: {ex['calories']} kcal\nProtein: {ex['protein']} g\nFat: {ex['fat']} g\nCarbs: {ex['carbs']} g"})

    query_b64 = encode_image_to_base64(image_path)
    message_content.append({"type": "text", "text": "--- Question ---\nNow, based on the examples above, estimate the mass (in grams) and nutritional values (calories, protein, fat, carbohydrates) of the following query dish:"})
    message_content.append({"type": "image_url", "image_url": {"url": query_b64}})
    message_content.append({"type": "text", "text": "Your answer must follow *exactly* this format:\nMass: <value> g\nCalories: <value> kcal\nProtein: <value> g\nFat: <value> g\nCarbs: <value> g"})

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": message_content}],
        max_tokens=500
    )

    response_text = response.choices[0].message.content
    parsed = extract_values(response_text)
    parsed = {"dish_id": f"dish_{query_dish_id}", **parsed}
    parsed["response_text"] = response_text
    parsed["support_examples"] = support_examples
    parsed["query_image_path"] = image_path
    parsed["message_content"] = message_content
    return parsed

for batch_dir in batch_dirs:
    output_csv = os.path.join(batch_dir, f"supporting_{k}.csv")
    image_paths = sorted([
        os.path.join(batch_dir, f) for f in os.listdir(batch_dir)
        if f.endswith(".jpg") and not f.startswith("._")
    ])

    results = []
    for image_path in image_paths:
        result = send_prompt(image_path)
        results.append(result)
        # DEBUGGING: Uncomment to save the prompt and response to PDF
        # save_to_pdf(query_dish_id, result["message_content"], result["response_text"])
        time.sleep(1)

    df = pd.DataFrame(results).drop(columns=["response_text", "support_examples", "query_image_path", "message_content"])
    df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")
