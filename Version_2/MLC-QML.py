
# STEP 1: IMPORT 
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pennylane as qml
from tqdm import tqdm


# STEP 2: LOAD AND PREPROCESS 

# Load GoEmotions 
raw_dataset = load_dataset("go_emotions", "raw")
df = raw_dataset['train'].to_pandas()

# Extract emotion columns
emotion_columns = df.columns[10:].tolist()
mlb = MultiLabelBinarizer()
mlb.fit([emotion_columns])

# Prep labels
labels = df[emotion_columns].values
texts = df['text'].tolist()

# Binarizeing multi-labels
y = labels

#  train and test split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    texts, y, test_size=0.2, random_state=42)


# STEP 3: TEXT EMBEDDING BERT (BATCHED)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bert = AutoModel.from_pretrained("bert-base-uncased").to(device)

def bert_embed_batched(texts, batch_size=64):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True, max_length=128
        ).to(device)
        with torch.no_grad():
            outputs = bert(**inputs)
            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu()
            embeddings.append(batch_embeddings)
    return torch.cat(embeddings, dim=0)

X_train = bert_embed_batched(train_texts)
X_test = bert_embed_batched(test_texts)


# STEP 4: DEFINE QUANTUM LAYER

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


# STEP 5: DEFINE HYBRID MODEL

class HybridModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)            # Expanded intermediate layer
        self.relu1 = nn.ReLU()
        self.fc1_reduce = nn.Linear(128, n_qubits)      # Reduce to qubit count
        self.q_layer = QuantumLayer()
        self.fc2_expand = nn.Linear(n_qubits, 128)  # Expand before final
        self.relu2 = nn.ReLU()
        self.fc2_out = nn.Linear(128, output_dim)   # Output layer
    def forward(self, batch):
        x = batch['features']
        x = torch.tanh(self.fc1_reduce(self.relu1(self.fc1(x))))
        x = self.q_layer(x)
        x = self.fc2_expand(x)
        x = self.relu2(x)
        x = self.fc2_out(x)
        return torch.sigmoid(x)


# STEP 6: PREP DATALOADER

class EmotionDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'labels': torch.tensor(self.labels[idx], dtype=torch.float32)
        }

train_data = EmotionDataset(X_train, train_labels)
test_data = EmotionDataset(X_test, test_labels)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)


# STEP 7: TRAIN THE MODEL

model = HybridModel(X_train.shape[1], len(emotion_columns)).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
# class weights computation
label_freq = torch.tensor(train_labels.sum(axis=0), dtype=torch.float32)
class_weights = 1.0 / (label_freq + 1e-5)  # Avoid divide-by-zero
class_weights = class_weights / class_weights.sum()  # Normalize

# move to device
class_weights = class_weights.to(device)

# use weighted loss
criterion = nn.BCELoss(weight=class_weights)

for epoch in range(5):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad()
        outputs = model(batch)
        loss = criterion(outputs, batch['labels'])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())


# STEP 8: EVALUATION & PLOT

model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch).cpu().numpy()
        y_pred.extend(outputs)
        y_true.extend(batch['labels'].cpu().numpy())

y_pred_bin = (np.array(y_pred) > 0.5).astype(int)

# Classification Report
print("Classification Report:")
print(classification_report(np.array(y_true), y_pred_bin, target_names=mlb.classes_))

# Macro/Micro F1
macro_f1 = f1_score(y_true, y_pred_bin, average="macro")
micro_f1 = f1_score(y_true, y_pred_bin, average="micro")
print(f"\nMacro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f}")

# Per-label F1
label_f1s = f1_score(y_true, y_pred_bin, average=None)
plt.figure(figsize=(10, 8))
sns.barplot(x=label_f1s, y=mlb.classes_, orient="h", hue=mlb.classes_, legend=False)
plt.xlabel("F1 Score")
plt.title("F1 Score per Emotion Label")
plt.tight_layout()
plt.show()

# Confidence 
all_confidences = np.array(y_pred).flatten()
plt.figure(figsize=(8, 5))
plt.hist(all_confidences, bins=50, color="skyblue", edgecolor="black")
plt.title("Distribution of Prediction Confidences (Sigmoid Outputs)")
plt.xlabel("Confidence (0-1)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()

# examples
for i in range(10):
    print(f"\nText: {test_texts[i]}")
    pred_indices = np.where(y_pred_bin[i])[0]
    true_indices = np.where(test_labels[i])[0]
    print("Predicted: ", [mlb.classes_[j] for j in pred_indices])
    print("True     : ", [mlb.classes_[j] for j in true_indices])



# BETTER THAN THE PREVIOUS VERSION AT LEAST :)))))
