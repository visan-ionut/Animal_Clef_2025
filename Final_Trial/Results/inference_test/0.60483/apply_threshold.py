import pandas as pd

# Load the CSV file
df = pd.read_csv("final_submission.csv")

# Apply threshold: if confidence < 0.6, set identity to 'new_individual'
df.loc[df["confidence"] < 0.45, "identity"] = "new_individual"

# Save the updated DataFrame back to CSV
df.to_csv("final_submission_thresholded.csv", index=False)

print("The file 'final_submission_thresholded.csv' has been created with the threshold applied.")
