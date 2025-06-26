import os
import pandas as pd
import numpy as np

test_batches_dir = "/Volumes/T9/nutrition5k_dataset/imagery/test_batches"
true_metadata_file = "/Volumes/T9/nutrition5k_dataset/REMOVED_OUTLIERS/true_metadata_cafe1_cleaned.csv"
output_file = "/Volumes/T9/nutrition5k_dataset/REMOVED_OUTLIERS/Results/support_choice.csv"

true_df = pd.read_csv(true_metadata_file)
true_df["dish_id"] = true_df["dish_id"].astype(str)

outlier_dish_ids = set()
for support_count in range(0, 8):
    for folder in os.listdir(test_batches_dir):
        folder_path = os.path.join(test_batches_dir, folder)
        file_path = os.path.join(folder_path, f"supporting_{support_count}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df["dish_id"] = df["dish_id"].astype(str)
            joined = df.merge(true_df, on="dish_id", suffixes=("_pred", "_true"))
            joined["carbs_mape"] = np.abs((joined["carbs_pred"] - joined["carbs_true"]) / (joined["carbs_true"] + 1e-8)) * 100
            new_outliers = joined.loc[joined["carbs_mape"] > 1000, "dish_id"]
            outlier_dish_ids.update(new_outliers)

print("Excluded extreme outliers (carbs MAPE > 1000%):")
for dish_id in sorted(outlier_dish_ids):
    print(f"- {dish_id}")
print(f"Total excluded: {len(outlier_dish_ids)}\n")

def calc_metrics(pred_df, true_df):
    joined = pred_df.merge(true_df, on="dish_id", suffixes=("_pred", "_true"))
    mae = {}
    mape = {}
    for col in ["mass", "calories", "protein", "fat", "carbs"]:
        true_col = joined[f"{col}_true"]
        pred_col = joined[f"{col}_pred"]
        mae[col] = np.mean(np.abs(pred_col - true_col))
        mape[col] = np.mean(np.abs((pred_col - true_col) / (true_col + 1e-8))) * 100
    return mae, mape

mae_table = []
mape_table = []

for support_count in range(0, 8):
    all_preds = []
    for folder in os.listdir(test_batches_dir):
        folder_path = os.path.join(test_batches_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        file_path = os.path.join(folder_path, f"supporting_{support_count}.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df["dish_id"] = df["dish_id"].astype(str)
            df = df[~df["dish_id"].isin(outlier_dish_ids)]
            all_preds.append(df)
    if not all_preds:
        continue
    pred_df = pd.concat(all_preds, ignore_index=True)
    mae, mape = calc_metrics(pred_df, true_df)

    mae_row = {"# supporting images": support_count}
    mape_row = {"# supporting images": support_count}
    mae_row.update({f"{k} (g/kcal)": round(v, 2) for k, v in mae.items()})
    mape_row.update({f"{k} (%)": round(v, 2) for k, v in mape.items()})

    mae_table.append(mae_row)
    mape_table.append(mape_row)

with open(output_file, "w") as f:
    pd.DataFrame(mae_table).to_csv(f, index=False)
    f.write("\n")
    pd.DataFrame(mape_table).to_csv(f, index=False)

print(f"Results saved to {output_file}")
