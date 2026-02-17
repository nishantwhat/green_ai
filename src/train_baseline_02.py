import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from codecarbon import EmissionsTracker
torch.set_num_threads(os.cpu_count() // 2)
# ==========================================
# 1. CONFIGURATION & DEVICE SETUP
# ==========================================
DATA_DIR = "../data"
MODEL_DIR = "../models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Hardware Check: Use your RTX 4050 if available!
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Using Device: {DEVICE}")

# Hyperparameters
MAX_VOCAB_SIZE = 10000
MAX_SEQ_LEN = 50       # We pad/truncate tweets to 50 words
EMBED_DIM = 64         # Size of the word embeddings
BATCH_SIZE = 32
EPOCHS = 15

# ==========================================
# 2. DATASET & TOKENIZER CLASSES
# ==========================================
# A simple word-level tokenizer to convert text to numbers
class SimpleTokenizer:
    def __init__(self, max_vocab=10000):
        self.max_vocab = max_vocab
        self.word2idx = {"<PAD>": 0, "<UNK>": 1} # Special tokens
        
    def fit(self, texts):
        word_counts = {}
        for text in texts:
            for word in str(text).lower().split():
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Keep the most common words up to MAX_VOCAB_SIZE
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        for i, (word, _) in enumerate(sorted_words[:self.max_vocab - 2]):
            self.word2idx[word] = i + 2
            
    def encode(self, text, max_len=50):
        tokens = [self.word2idx.get(w, 1) for w in str(text).lower().split()]
        # Truncate if too long, Pad with 0s if too short
        if len(tokens) > max_len:
            return tokens[:max_len]
        return tokens + [0] * (max_len - len(tokens))

# PyTorch Dataset for our Tweets
class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.x = [tokenizer.encode(t, max_len) for t in texts]
        self.y = labels.tolist()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.x[idx]), torch.tensor(self.y[idx])

# ==========================================
# 3. THE NEURAL NETWORK ARCHITECTURE
# ==========================================
class SentimentCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes=3):
        super(SentimentCNN, self).__init__()
        # Layer 1: Word Embeddings (Turns word IDs into dense vectors)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Layer 2: 1D Convolution (Extracts features from word sequences)
        self.conv1 = nn.Conv1d(in_channels=embed_dim, out_channels=128, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        
        # Layer 3: Global Max Pooling (Shrinks the sequence down to the most important features)
        self.pool = nn.AdaptiveMaxPool1d(1)
        
        # Layer 4: Fully Connected Linear layer (Maps features to the 3 Sentiment Classes)
        self.fc = nn.Linear(128, num_classes)
        # Inside __init__:
        self.dropout = nn.Dropout(0.5)


    def forward(self, x):
        # x shape: (batch_size, seq_len)
        x = self.embedding(x)                 # Output: (batch_size, seq_len, embed_dim)
        x = x.permute(0, 2, 1)                # Conv1d expects: (batch_size, channels, seq_len)
        x = self.conv1(x)                     # Output: (batch_size, 128, seq_len)
        x = self.relu(x)
        x = self.pool(x).squeeze(2)           # Output: (batch_size, 128)
        # Inside forward (before the final fc layer):
        x = self.dropout(x)
        out = self.fc(x)                      # Output: (batch_size, 3)
        return out

# ==========================================
# 4. MAIN PIPELINE
# ==========================================
def main():
    # --- A. Load Data ---
    print("📂 Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_sampled.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_sampled.csv"))

    # --- B. Tokenize and Build DataLoaders ---
    tokenizer = SimpleTokenizer(MAX_VOCAB_SIZE)
    tokenizer.fit(train_df['text'])
    
    train_dataset = TweetDataset(train_df['text'], train_df['label'], tokenizer, MAX_SEQ_LEN)
    test_dataset = TweetDataset(test_df['text'], test_df['label'], tokenizer, MAX_SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- C. Initialize Model, Loss, and Optimizer ---
    model = SentimentCNN(MAX_VOCAB_SIZE, EMBED_DIM, num_classes=3).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001)

    # --- D. Training Loop ---
    print("\n🚀 Starting Training Loop...")
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()           # Clear old gradients
            outputs = model(inputs)         # Forward pass
            loss = criterion(outputs, labels) # Calculate error
            loss.backward()                 # Backpropagation
            optimizer.step()                # Update weights
            
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")

    # Save the trained baseline model
    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "baseline_model.pt"))
    print("💾 Baseline model saved!")

    # --- E. Inference & Energy Measurement ---
    print("\n⚡ Running Inference Test and Measuring Energy...")
    model.eval() # Set model to evaluation mode (turns off dropout/batchnorm if any)
    
    all_preds = []
    all_labels = []
    
    # Initialize CodeCarbon Tracker specifically for Inference!
    # offline_mode=False means it will automatically try to find your local grid emissions
    tracker = EmissionsTracker(project_name="baseline_inference", measure_power_secs=1)
    tracker.start()
    
    with torch.no_grad(): # Disable gradient calculation to save memory/speed up
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    emissions = tracker.stop() # Stop tracking exactly when inference finishes!
    
    # --- F. Calculate and Save Metrics ---
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro') # Macro F1 handles class imbalance
    energy_joules = tracker.final_emissions_data.energy_consumed * 3.6e6 # Convert kWh to Joules
    
    print("\n" + "="*40)
    print("📊 BASELINE RESULTS")
    print(f"Accuracy:        {acc*100:.2f}%")
    print(f"Macro F1-Score:  {f1*100:.2f}%")
    print(f"Energy Consumed: {energy_joules:.4f} Joules")
    print("="*40)

    # Save to our central Results CSV
    results_path = "../results.csv"
    res_df = pd.DataFrame([{
        "Model": "Baseline (FP32)",
        "Accuracy": acc,
        "F1_Score": f1,
        "Energy_Joules": energy_joules
    }])
    
    if os.path.exists(results_path):
        res_df.to_csv(results_path, mode='a', header=False, index=False)
    else:
        res_df.to_csv(results_path, index=False)

if __name__ == "__main__":
    main()