import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report
import joblib
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pandas as pd

# --- Define label_encoder consistent with extract_embeddings.py ---
train_csv = "train_database_metadata.csv"
train_identities = pd.read_csv(train_csv)['identity'].unique()
label_encoder = {identity: idx for idx, identity in enumerate(sorted(train_identities))}
idx_to_label = {v: k for k, v in label_encoder.items()}

# --- Load all embeddings ---
def load_embeddings(keys, prefix="train"):
    return {
        key: np.load(f"{prefix}_embeddings_{key}.npz") for key in keys
    }

def pool_if_needed(X):
    return X.mean(axis=(2, 3)) if X.ndim == 4 else X

# --- Scheduler helper ---
def linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# --- Keys and data loading ---
keys = ["convnext", "convnext_v2", "maxvit", "swin", "swin_v2", "megadescriptor"]
train_embeddings = load_embeddings(keys, "train")
test_embeddings = load_embeddings(keys, "test")

# --- Preprocess ---
X_train_list, X_test_list = [], []
for k in keys:
    print(f"📦 Using {k}")
    X_train_list.append(pool_if_needed(train_embeddings[k]["X"]))
    X_test_list.append(pool_if_needed(test_embeddings[k]["X"]))

X_train = np.concatenate(X_train_list, axis=1)
X_test = np.concatenate(X_test_list, axis=1)
y_train = train_embeddings[keys[0]]["y"]
y_test = test_embeddings[keys[0]]["y"]

# --- Normalize ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, "mlp_scaler.joblib")

# --- PCA ---
pca = PCA(n_components=0.99, svd_solver='full')
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
joblib.dump(pca, "mlp_pca.joblib")

print(f"🔢 Final input dimension after PCA: {X_train_pca.shape[1]}")

# --- Torch datasets ---
X_train_tensor = torch.tensor(X_train_pca, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_pca, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256)

# --- Model ---
class MLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 4096),
            nn.BatchNorm1d(4096),
            nn.SiLU(),
            nn.Dropout(0.3),

            nn.Linear(4096, 2048),
            nn.BatchNorm1d(2048),
            nn.SiLU(),
            nn.Dropout(0.3),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(),
            nn.Dropout(0.25),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.SiLU(),
            nn.Dropout(0.2),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = MLP(X_train_pca.shape[1], len(np.unique(y_train))).cuda()

# --- Training setup ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-2)

num_epochs = 100
total_steps = len(train_loader) * num_epochs
scheduler = linear_schedule_with_warmup(optimizer, num_warmup_steps=total_steps // 10, num_training_steps=total_steps)

best_loss = float('inf')
early_stop_counter = 0

print("\n🚀 Training MLP model...")
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
    avg_loss_train = running_loss / len(train_loader)

    # --- Evaluation ---
    model.eval()
    running_loss_test = 0.0
    with torch.no_grad():
        outputs = model(X_test_tensor.cuda())
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        acc = accuracy_score(y_test, preds)

        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.cuda(), y_batch.cuda()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            running_loss_test += loss.item()
        avg_loss_test = running_loss_test / len(test_loader)

    print(f"📉 Train Loss: {avg_loss_train:.4f} | 🧪 Test Loss: {avg_loss_test:.4f} | 🎯 Val Accuracy: {acc:.4f}")

    if avg_loss_test < best_loss:
        best_loss = avg_loss_test
        torch.save(model.state_dict(), "mlp_classifier_best.pt")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= 10:
            print("🛑 Early stopping triggered (based on test loss).")
            break

# --- Final evaluation ---
print("\n🔍 Evaluating best model...")
model.load_state_dict(torch.load("mlp_classifier_best.pt"))
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor.cuda())
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    acc = accuracy_score(y_test, preds)
    print(f"\n✅ Accuracy: {acc:.4f}")
    print("\n📊 Classification Report:")
    print(classification_report(y_test, preds, digits=4))

# --- Save classification report to CSV ---
report_dict = classification_report(y_test, preds, output_dict=True)
df_report = pd.DataFrame(report_dict).transpose()

# Păstrăm doar clasele numerice și mapăm înapoi la etichetele originale
df_report = df_report[df_report.index.to_series().str.isdigit()]
df_report["label_index"] = df_report.index.astype(int)
df_report["identity"] = df_report["label_index"].map(idx_to_label)
df_report["underrepresented"] = df_report["support"] < 5
df_report["not_predicted"] = df_report["recall"] == 0.0

# Rearanjăm coloanele
df_report = df_report[[
    "identity", "label_index", "support", "precision", "recall", "f1-score",
    "underrepresented", "not_predicted"
]]

# Salvăm în CSV
df_report.to_csv("class_wise_metrics.csv", index=False)
print("📁 Saved class-wise classification report to class_wise_metrics.csv")

# --- Query inference ---
print("\n🔍 Running inference on query set...")

# Load query embeddings
query_embeddings = []
query_ids = []

for k in keys:
    data = np.load(f"query_embeddings_{k}.npz")
    X_query_k = pool_if_needed(data["X"])
    query_embeddings.append(X_query_k)
    if len(query_ids) == 0:
        query_ids = data["image_ids"]

X_query = np.concatenate(query_embeddings, axis=1)

# Apply scaler and PCA
scaler = joblib.load("mlp_scaler.joblib")
pca = joblib.load("mlp_pca.joblib")

X_query_scaled = scaler.transform(X_query)
X_query_pca = pca.transform(X_query_scaled)

# Predict with MLP
X_query_tensor = torch.tensor(X_query_pca, dtype=torch.float32).cuda()
model.eval()
with torch.no_grad():
    outputs = model(X_query_tensor)
    probs = torch.softmax(outputs, dim=1).cpu().numpy()
    preds = np.argmax(probs, axis=1)

# Map predicted class indices back to labels
pred_labels = [idx_to_label[p] for p in preds]
confidences = probs.max(axis=1)

# Save predictions
df_out = pd.DataFrame({
    "image_id": query_ids,
    "identity": pred_labels,
    "confidence": confidences
})
df_out.to_csv("query_predictions.csv", index=False)
print(f"📁 Saved query predictions to query_predictions.csv")
