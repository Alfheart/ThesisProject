import pandas as pd
from pathlib import Path

cafe1_file = Path("dish_metadata_cafe1.csv")
output_file = Path("true_metadata_cafe1.csv")

num_fixed_columns = 6
ingredient_block_size = 7

results = []

with open(cafe1_file, "r") as f:
    for line in f:
        parts = line.strip().split(",")
        if len(parts) < num_fixed_columns + ingredient_block_size:
            continue

        dish_id = parts[0]
        pred_kcal = float(parts[1])
        pred_mass = float(parts[2])
        pred_fat = float(parts[3])
        pred_carb = float(parts[4])
        pred_protein = float(parts[5])

        ingredients = parts[num_fixed_columns:]

        ingredient_data = []
        use_predicted = False

        for i in range(0, len(ingredients), ingredient_block_size):
            try:
                ingr_name = ingredients[i + 1].lower()
                if "deprecated" in ingr_name:
                    use_predicted = True
                    break
            except IndexError:
                continue

        if use_predicted:
            results.append({
                "dish_id": dish_id,
                "calories": pred_kcal,
                "mass": pred_mass,
                "fat": pred_fat,
                "carbs": pred_carb,
                "protein": pred_protein
            })
        else:
            for i in range(0, len(ingredients), ingredient_block_size):
                try:
                    grams = float(ingredients[i + 2])
                    kcal = float(ingredients[i + 3])
                    fat = float(ingredients[i + 4])
                    carb = float(ingredients[i + 5])
                    protein = float(ingredients[i + 6])
                    ingredient_data.append((grams, kcal, fat, carb, protein))
                except (ValueError, IndexError):
                    continue

            total_mass = sum(x[0] for x in ingredient_data)
            total_kcal = sum(x[1] for x in ingredient_data)
            total_fat = sum(x[2] for x in ingredient_data)
            total_carb = sum(x[3] for x in ingredient_data)
            total_protein = sum(x[4] for x in ingredient_data)

            results.append({
                "dish_id": dish_id,
                "calories": total_kcal,
                "mass": total_mass,
                "fat": total_fat,
                "carbs": total_carb,
                "protein": total_protein
            })

df = pd.DataFrame(results)
df.to_csv(output_file, index=False)
print(f"Saved {len(df)} dishes to {output_file}")
