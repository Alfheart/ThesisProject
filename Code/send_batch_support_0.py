import os
import openai
import base64
import pandas as pd
import re
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

batch_dir = input("Enter path to batch folder: ").strip()
output_csv = os.path.join(batch_dir, "supporting_0.csv")

def build_prompt(dish_id):
    return f"""You are a nutrition expert. For the image below labeled {dish_id}, estimate its total mass and nutritional content.

Do not describe ingredients or try to guess what the dish is. Only estimate values based on visual appearance.

Example of a proper answer:
Mass (g): <value>  
Calories (kcal): <value>  
Protein (g): <value>  
Fat (g): <value>  
Carbs (g): <value>

--- Question ---
What is the mass and nutritional content of the following dish?
Image: [{dish_id}]
"""

def encode_image_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        b64_img = base64.b64encode(img_file.read()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64_img}"

def extract_values(text):
    def find(pattern):
        match = re.search(pattern, text, re.IGNORECASE)
        return float(match.group(1)) if match else None

    return {
        "mass": find(r"Mass\s*\(g\):\s*([\d.]+)"),
        "calories": find(r"Calories\s*\(kcal\):\s*([\d.]+)"),
        "protein": find(r"Protein\s*\(g\):\s*([\d.]+)"),
        "fat": find(r"Fat\s*\(g\):\s*([\d.]+)"),
        "carbs": find(r"Carbs\s*\(g\):\s*([\d.]+)")
    }

def send_prompt(image_path):
    dish_id = os.path.splitext(os.path.basename(image_path))[0]
    prompt = build_prompt(dish_id)
    base64_image = encode_image_to_base64(image_path)

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "user", "content": prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": base64_image, "detail": "auto"}
                    }
                ]
            }
        ],
        max_tokens=500
    )

    response_text = response.choices[0].message.content
    extracted = extract_values(response_text)
    return {"dish_id": dish_id, **extracted}

image_paths = sorted([
    os.path.join(batch_dir, f)
    for f in os.listdir(batch_dir)
    if f.endswith(".jpg") and not f.startswith("._")
])[:10]

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(send_prompt, image_paths))

df = pd.DataFrame(results)
df.to_csv(output_csv, index=False)
print(f"Saved to {output_csv}")
