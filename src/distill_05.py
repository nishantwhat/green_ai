import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from codecarbon import EmissionsTracker

from train_baseline_02 import SentimentCNN, SimpleTokenizer, TweetDataset, MAX_VOCAB_SIZE, EMBED_DIM, MAX_SEQ_LEN, BATCH_SIZE
torch.set_num_threads(os.cpu_count() // 2)

DATA_DIR = "../data"
MODEL_DIR = "../models"
RESULTS_FILE = "../results.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. THE HEAVY TEACHER MODEL
# ==========================================
class HeavyTeacherCNN(nn.Module):
    def __init__(self, vocab_size, num_classes=3):
        super(HeavyTeacherCNN, self).__init__()
        # The teacher has double the embedding dimension (128 instead of 64)
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        # The teacher has two Convolutional layers and more channels
        self.conv1 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.embedding(x).permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x).squeeze(2)
        return self.fc(x)

# ==========================================
# 2. DISTILLATION LOSS FUNCTION (THE MATH)
# ==========================================
def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    """
    Computes the Knowledge Distillation Loss.
    - Standard Cross Entropy Loss (learns from the actual hard labels)
    - KL Divergence Loss (learns the 'soft' probabilities from the Teacher using Temperature T)
    """
    # 1. Standard Loss
    standard_loss = F.cross_entropy(student_logits, labels)
    
    # 2. Soft Target Loss (KL Divergence)
    # We soften the probabilities by dividing by the Temperature (T)
    soft_targets = F.softmax(teacher_logits / T, dim=1)
    student_log_soft = F.log_softmax(student_logits / T, dim=1)
    
    # KL Divergence measures how different the student's probability distribution is from the teacher's
    kd_loss = F.kl_div(student_log_soft, soft_targets, reduction='batchmean') * (T * T)
    
    # 3. Combine them using alpha (how much to trust the teacher vs. the real labels)
    return (1. - alpha) * standard_loss + alpha * kd_loss

# ==========================================
# 3. PIPELINE
# ==========================================
def measure_inference(model, test_loader, run_name):
    model.eval()
    all_preds = []
    all_labels = []
    
    print(f"\n⚡ Running Inference for: {run_name}...")
    # Force CPU tracking to match baseline for apples-to-apples comparison
    tracker = EmissionsTracker(project_name=run_name, measure_power_secs=1, tracking_mode="machine")
    tracker.start()
    
    # Force model to CPU for the inference benchmark
    model = model.to(torch.device("cpu"))
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.numpy())
            all_labels.extend(labels.numpy())

    tracker.stop()
    
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    energy_joules = tracker.final_emissions_data.energy_consumed * 3.6e6
    
    print(f"[{run_name}] Acc: {acc*100:.2f}% | F1: {f1*100:.2f}% | Energy: {energy_joules:.4f} J")
    return acc, f1, energy_joules

def main():
    print("📂 Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_sampled.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_sampled.csv"))

    tokenizer = SimpleTokenizer(MAX_VOCAB_SIZE)
    tokenizer.fit(train_df['text'])
    
    train_dataset = TweetDataset(train_df['text'], train_df['label'], tokenizer, MAX_SEQ_LEN)
    test_dataset = TweetDataset(test_df['text'], test_df['label'], tokenizer, MAX_SEQ_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # --- 1. Train the Heavy Teacher ---
    print("\n" + "="*40)
    print("🎓 TRAINING THE HEAVY TEACHER MODEL")
    print("="*40)
    teacher = HeavyTeacherCNN(MAX_VOCAB_SIZE).to(DEVICE)
    optimizer_t = optim.Adam(teacher.parameters(), lr=0.001)
    
    teacher.train()
    for epoch in range(3): # Train teacher for 3 epochs
        total_loss = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer_t.zero_grad()
            outputs = teacher(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer_t.step()
            total_loss += loss.item()
        print(f"Teacher Epoch {epoch+1}/3 - Loss: {total_loss/len(train_loader):.4f}")
    
    teacher.eval() # Teacher must be in eval mode for distillation

    # --- 2. Train the Students (Ablation over Temperature) ---
    temperatures = [2.0, 5.0]
    results = []

    for T in temperatures:
        print("\n" + "="*40)
        print(f"👶 TRAINING STUDENT (Distillation T={T})")
        print("="*40)
        
        # Initialize a fresh, tiny Student model (same exact architecture as our Baseline)
        student = SentimentCNN(MAX_VOCAB_SIZE, EMBED_DIM, num_classes=3).to(DEVICE)
        optimizer_s = optim.Adam(student.parameters(), lr=0.001)
        
        student.train()
        for epoch in range(5):
            total_loss = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                # Get Teacher's soft labels (no gradients needed for teacher)
                with torch.no_grad():
                    teacher_logits = teacher(inputs)
                
                # Get Student's predictions
                optimizer_s.zero_grad()
                student_logits = student(inputs)
                
                # Calculate Distillation Loss
                loss = distillation_loss(student_logits, teacher_logits, labels, T=T, alpha=0.5)
                
                loss.backward()
                optimizer_s.step()
                total_loss += loss.item()
            print(f"Student (T={T}) Epoch {epoch+1}/5 - Loss: {total_loss/len(train_loader):.4f}")

        # --- 3. Measure Inference ---
        run_name = f"Distilled_Student_T{T}"
        # Save the distilled student for potential joint optimization later
        torch.save(student.cpu().state_dict(), os.path.join(MODEL_DIR, f"{run_name}.pt"))
        
        acc, f1, energy = measure_inference(student, test_loader, run_name)
        results.append({
            "Model": run_name,
            "Accuracy": acc,
            "F1_Score": f1,
            "Energy_Joules": energy
        })

    # Save to Results
    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
    print("\n✅ Knowledge Distillation Ablation complete and saved to results.csv!")

if __name__ == "__main__":
    main()