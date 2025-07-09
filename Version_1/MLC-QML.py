# ====================================================================
# STEP 1: IMPORTS
# ====================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from datasets import load_dataset
import pennylane as qml
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ====================================================================
# STEP 2: LOAD & PREPROCESS GOEMOTIONS DATASET
# ====================================================================
dataset = load_dataset("go_emotions", "raw")
df = dataset['train'].to_pandas()

# Only emotion columns (excluding 'neutral' and non-emotion columns)
emotion_columns = [col for col in df.columns[10:] if df[col].sum() > 0 and col != "neutral"]

# Filter out rows with no emotion labels
df = df[df[emotion_columns].sum(axis=1) > 0]

# Create multilabel targets
labels = df[emotion_columns].values

# Save MultiLabelBinarizer to access emotion class names
mlb = MultiLabelBinarizer()
mlb.fit([emotion_columns])

# Tokenization
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
max_len = 128

class GoEmotionsDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', max_length=max_len)
        self.labels = torch.tensor(labels, dtype=torch.float32)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

# Train-test split
split = int(0.8 * len(df))
train_texts = df['text'][:split].tolist()
test_texts = df['text'][split:].tolist()
train_labels = labels[:split]
test_labels = labels[split:]

train_dataset = GoEmotionsDataset(train_texts, train_labels)
test_dataset = GoEmotionsDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# ====================================================================
# STEP 3: CLASSICAL BERT ENCODER
# ====================================================================
class BERTEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("bert-base-uncased")
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

# ====================================================================
# STEP 4: DEFINE QUANTUM LAYER
# ====================================================================
n_qubits = 4
n_layers = 1
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weight_shapes = {"weights": (n_layers, n_qubits, 3)}

class QuantumLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)
    def forward(self, x):
        return self.qlayer(x)

# ====================================================================
# STEP 5: HYBRID MODEL
# ====================================================================
class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = BERTEncoder()
        self.fc1 = nn.Linear(768, n_qubits)
        self.q_layer = QuantumLayer()
        self.fc2 = nn.Linear(n_qubits, len(emotion_columns))
    def forward(self, batch):
        x = self.encoder(batch['input_ids'].to(device), batch['attention_mask'].to(device))
        x = torch.tanh(self.fc1(x))
        x = self.q_layer(x)
        x = self.fc2(x)
        return torch.sigmoid(x)

model = HybridModel().to(device)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# ====================================================================
# STEP 6: TRAINING LOOP
# ====================================================================
epochs = 1
y_true, y_pred = [], []

for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, leave=True, desc=f"Epoch {epoch+1}")
    for batch in loop:
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch['labels'].to(device))
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

# ====================================================================
# STEP 7: EVALUATION
# ====================================================================
model.eval()
with torch.no_grad():
    for batch in test_loader:
        outputs = model(batch)
        y_pred.extend(outputs.cpu().numpy())
        y_true.extend(batch['labels'].cpu().numpy())

y_pred_bin = (np.array(y_pred) > 0.5).astype(int)

print("Classification Report:")
print(classification_report(np.array(y_true), y_pred_bin, target_names=mlb.classes_))

macro_f1 = f1_score(y_true, y_pred_bin, average="macro")
micro_f1 = f1_score(y_true, y_pred_bin, average="micro")
print(f"\nMacro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f}")

# ====================================================================
# STEP 8: VISUALIZATION
# ====================================================================
label_f1s = f1_score(y_true, y_pred_bin, average=None)
plt.figure(figsize=(10, 8))
sns.barplot(x=label_f1s, y=mlb.classes_, orient="h", palette="viridis")
plt.xlabel("F1 Score")
plt.title("F1 Score per Emotion Label")
plt.tight_layout()
plt.show()

all_confidences = np.array(y_pred).flatten()
plt.figure(figsize=(8, 5))
plt.hist(all_confidences, bins=50, color="skyblue", edgecolor="black")
plt.title("Distribution of Prediction Confidences (Sigmoid Outputs)")
plt.xlabel("Confidence (0-1)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# Top 10 prediction examples
for i in range(10):
    print(f"\nText: {test_texts[i]}")
    pred_indices = np.where(y_pred_bin[i])[0]
    true_indices = np.where(test_labels[i])[0]
    print("Predicted: ", [mlb.classes_[j] for j in pred_indices])
    print("True     : ", [mlb.classes_[j] for j in true_indices])
