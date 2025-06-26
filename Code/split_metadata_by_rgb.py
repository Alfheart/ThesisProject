import pandas as pd

metadata = pd.read_csv("true_metadata_cafe1_cleaned.csv")

with open("../dish_ids/splits/rgb_train_ids.txt", "r") as f:
    train_ids = set(line.strip() for line in f if line.strip())

with open("../dish_ids/splits/rgb_test_ids.txt", "r") as f:
    test_ids = set(line.strip() for line in f if line.strip())

print(f"Train IDs: {len(train_ids)}, Test IDs: {len(test_ids)}")

train_df = metadata[metadata["dish_id"].isin(train_ids)]
test_df = metadata[metadata["dish_id"].isin(test_ids)]

print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

output_dir = "Splits"
train_df.to_csv(f"{output_dir}/true_train.csv", index=False)
test_df.to_csv(f"{output_dir}/true_test.csv", index=False)
