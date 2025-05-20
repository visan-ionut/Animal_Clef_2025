import os
import numpy as np
import pandas as pd
import timm
import torchvision.transforms as T
from wildlife_tools.features import DeepFeatures
from wildlife_tools.similarity import CosineSimilarity
from wildlife_tools.similarity.wildfusion import SimilarityPipeline, WildFusion
from wildlife_tools.similarity.pairwise.lightglue import MatchLightGlue
from wildlife_tools.features.local import AlikedExtractor
from wildlife_tools.similarity.calibration import IsotonicCalibration
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm
import torch

class InferenceDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.metadata = self.df
        self.col_label = "identity"
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.iloc[idx]['path'])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, idx

# Paths
root = './animal-clef-2025'

# Transforms
transform_display = T.Compose([T.Resize([384, 384])])
transform = T.Compose([
    *transform_display.transforms,
    T.ToTensor(),
    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])
transforms_aliked = T.Compose([
    T.Resize([512, 512]),
    T.ToTensor()
])

# Load metadata
df_database = pd.read_csv("database_metadata.csv")
df_query = pd.read_csv("query_metadata.csv")
df_calibration = pd.read_csv("test_database_metadata.csv")

n_query = len(df_query)

# Datasets
calib_ds = InferenceDataset(df_calibration, root, transform)
query_ds = InferenceDataset(df_query, root, transform)
database_ds = InferenceDataset(df_database, root, transform)

# Load MegaDescriptor model
print("🔄 Loading MegaDescriptor model...")
model = timm.create_model('hf-hub:BVRA/MegaDescriptor-L-384', num_classes=0, pretrained=True)
device = 'cuda'

# Define similarity pipelines
matcher_aliked = SimilarityPipeline(
    matcher=MatchLightGlue(features='aliked', device=device, batch_size=16),
    extractor=AlikedExtractor(),
    transform=transforms_aliked,
    calibration=IsotonicCalibration()
)

matcher_mega = SimilarityPipeline(
    matcher=CosineSimilarity(),
    extractor=DeepFeatures(model=model, device=device, batch_size=16),
    transform=transform,
    calibration=IsotonicCalibration()
)

# Calibrate WildFusion with progress bar
print("🧪 Calibrating WildFusion...")
wildfusion = WildFusion(
    calibrated_pipelines=[matcher_aliked, matcher_mega],
    priority_pipeline=matcher_mega
)

with tqdm(total=1, desc="Fitting Calibration") as pbar:
    wildfusion.fit_calibration(calib_ds, calib_ds)
    pbar.update(1)

# Compute similarities
print("🔍 Computing similarities using WildFusion...")
similarity = wildfusion(query_ds, database_ds, B=None)

# Get predictions
pred_idx = similarity.argmax(axis=1)
pred_scores = similarity[np.arange(n_query), pred_idx]
identities = df_database["identity"].tolist()
predictions = [identities[i] for i in pred_idx]
image_ids = df_query["image_id"].tolist()

# Create submission
submission = pd.DataFrame({
    "image_id": image_ids,
    "identity": predictions,
    "confidence": pred_scores
})

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv has been created successfully.")

# Save the model and WildFusion object
save_dict = {
    'model_state_dict': model.state_dict(),
    'wildfusion': wildfusion,
    'df_database': df_database,
    'df_query': df_query,
    'df_calibration': df_calibration
}

torch.save(save_dict, "wildfusion_model.pth")
print("💾 Model saved to 'wildfusion_model.pth'")
