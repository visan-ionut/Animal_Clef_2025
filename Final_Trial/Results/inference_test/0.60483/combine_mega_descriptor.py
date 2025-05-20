import pandas as pd

# Load the two prediction files
df1 = pd.read_csv("preds_1_mega_descriptor.csv")
df2 = pd.read_csv("preds_4_mega_descriptor.csv")

# Rename columns to avoid name conflicts
df1 = df1.rename(columns={"identity": "identity_1", "confidence": "confidence_1"})
df2 = df2.rename(columns={"identity": "identity_2", "confidence": "confidence_2"})

# Merge the dataframes on 'image_id'
merged = df1.merge(df2, on="image_id")

# Prepare the final results
result_rows = []

for _, row in merged.iterrows():
    image_id = row["image_id"]

    identity_1 = row["identity_1"]
    identity_2 = row["identity_2"]

    # Weighted confidence: slightly favor the second model
    confidence_1 = row["confidence_1"] * 0.45
    confidence_2 = row["confidence_2"] * 0.55

    if identity_1 == identity_2:
        final_identity = identity_1
        final_confidence = (confidence_1 + confidence_2) * 1.2
    else:
        final_identity = identity_2
        final_confidence = confidence_2 * 1.82

    result_rows.append({
        "image_id": image_id,
        "identity": final_identity,
        "confidence": final_confidence
    })

# Create and save the final DataFrame
result_df = pd.DataFrame(result_rows)
result_df.to_csv("preds_mega_descriptor.csv", index=False)

print("✅ File 'preds_mega_descriptor.csv' has been created successfully.")
