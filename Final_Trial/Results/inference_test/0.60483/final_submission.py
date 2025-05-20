import pandas as pd

# Load predictions from the two models
df_mega = pd.read_csv("preds_mega_descriptor.csv")
df_conv1 = pd.read_csv("preds_convnext_1.csv")

# Rename columns to avoid conflicts
df_mega = df_mega.rename(columns={"identity": "identity_mega", "confidence": "confidence_mega"})
df_conv1 = df_conv1.rename(columns={"identity": "identity_conv1", "confidence": "confidence_conv1"})

# Merge predictions on image_id
merged = df_mega.merge(df_conv1, on="image_id")

# Prepare final results
result_rows = []

for _, row in merged.iterrows():
    image_id = row["image_id"]

    identity_mega = row["identity_mega"]
    identity_conv1 = row["identity_conv1"]

    confidence_mega = row["confidence_mega"]
    confidence_conv1 = row["confidence_conv1"] * 0.25

    if identity_mega == identity_conv1:
        final_identity = identity_mega
        final_confidence = ((confidence_mega + confidence_conv1) / 2) * 1.7
    else:
        if confidence_mega >= confidence_conv1:
            final_identity = identity_mega
            final_confidence = confidence_mega
        else:
            final_identity = identity_conv1
            final_confidence = confidence_conv1 * 1.2

    result_rows.append({
        "image_id": image_id,
        "identity": final_identity,
        "confidence": final_confidence
    })

# Create and save final DataFrame
result_df = pd.DataFrame(result_rows)
result_df.to_csv("final_submission.csv", index=False)

print("✅ The file 'final_submission.csv' has been created using df_mega and df_conv1.")
