import os
import pandas as pd
import numpy as np

test_batches_dir = "/Volumes/T9/nutrition5k_dataset/imagery/test_batches"
true_metadata_file = "/Volumes/T9/nutrition5k_dataset/REMOVED_OUTLIERS/true_metadata_cafe1_cleaned.csv"
output_excel = "/Volumes/T9/nutrition5k_dataset/REMOVED_OUTLIERS/Results/MAE_supporting_6.xlsx"

true_df = pd.read_csv(true_metadata_file)
true_df["dish_id"] = true_df["dish_id"].astype(str)

outlier_dish_ids = set()
for folder in os.listdir(test_batches_dir):
    folder_path = os.path.join(test_batches_dir, folder)
    file_path = os.path.join(folder_path, "supporting_6.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df["dish_id"] = df["dish_id"].astype(str)
        joined = df.merge(true_df, on="dish_id", suffixes=("_pred", "_true"))
        joined["carbs_mape"] = np.abs((joined["carbs_pred"] - joined["carbs_true"]) / (joined["carbs_true"] + 1e-8)) * 100
        outlier_ids = joined.loc[joined["carbs_mape"] > 500, "dish_id"]
        outlier_dish_ids.update(outlier_ids)

print("Excluded extreme outliers:")
for dish_id in sorted(outlier_dish_ids):
    print(f"- {dish_id}")
print(f"Total excluded: {len(outlier_dish_ids)}\n")

def calc_mae_and_mape(pred_df, true_df):
    joined = pred_df.merge(true_df, on="dish_id", suffixes=("_pred", "_true"))
    mae = {}
    mape = {}
    for col in ["mass", "calories", "protein", "fat", "carbs"]:
        true_col = joined[f"{col}_true"]
        pred_col = joined[f"{col}_pred"]
        mae[col] = np.mean(np.abs(pred_col - true_col))
        mape[col] = np.mean(np.abs((pred_col - true_col) / (true_col + 1e-8))) * 100
    return mae, mape

all_preds = []
for folder in os.listdir(test_batches_dir):
    folder_path = os.path.join(test_batches_dir, folder)
    if not os.path.isdir(folder_path):
        continue
    file_path = os.path.join(folder_path, "supporting_6.csv")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df["dish_id"] = df["dish_id"].astype(str)
        df = df[~df["dish_id"].isin(outlier_dish_ids)]
        all_preds.append(df)

if not all_preds:
    raise ValueError("No supporting_6.csv files found!")

pred_df = pd.concat(all_preds, ignore_index=True)
mae, mape = calc_mae_and_mape(pred_df, true_df)

row = {}
for key in ["calories", "mass", "fat", "carbs", "protein"]:
    formatted = f"{round(mae[key], 2)} / {round(mape[key], 1)}%"
    row[f"{key.title()} MAE"] = formatted

macro_mae_percent = np.mean([mape["carbs"], mape["protein"], mape["fat"]])
row["Macronutrient MAE"] = f"{round(macro_mae_percent, 1)}%"

df = pd.DataFrame([row])
df.to_excel(output_excel, index=False)

print(f"MAE and MAPE table saved to {output_excel}")
