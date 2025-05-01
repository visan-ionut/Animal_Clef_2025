import torch
import timm
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_path = "best_model.pth"
checkpoint = torch.load(model_path, map_location=device)

label_encoder = checkpoint['label_encoder']
idx_to_label = {v: k for k, v in label_encoder.items()}

model = timm.create_model(
    'vit_large_patch16_384',
    pretrained=False,
    num_classes=len(label_encoder)
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((384, 384)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
])

def predict(image_path, threshold=0.68):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        conf, preds = probs.max(dim=1)
        confidence = conf.item()
        predicted_idx = preds.item()

    if confidence >= threshold:
        predicted_label = idx_to_label[predicted_idx]
    else:
        predicted_label = "new_individual"

    return predicted_label, confidence

query_csv = "query_metadata.csv"
query_data = pd.read_csv(query_csv)

root = "./animal-clef-2025"

predictions = []
confidences = []

for idx, row in tqdm(query_data.iterrows(), total=len(query_data), desc="Predicting"):
    img_rel_path = row['path']
    img_full_path = os.path.join(root, img_rel_path)

    try:
        pred_label, conf = predict(img_full_path)
    except Exception as e:
        print(f"❌ Eroare la imaginea {img_full_path}: {e}")
        pred_label, conf = "error", 0.0

    predictions.append(pred_label)
    confidences.append(conf)

query_data['identity'] = predictions
query_data['confidence'] = confidences

output_csv = "query_with_predictions.csv"
query_data.to_csv(output_csv, index=False)

print(f"✅ File saved at {output_csv}")
