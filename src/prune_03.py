import os
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from codecarbon import EmissionsTracker

# Import our architecture and dataset tools from the baseline script
from train_baseline_02 import SentimentCNN, SimpleTokenizer, TweetDataset, MAX_VOCAB_SIZE, EMBED_DIM, MAX_SEQ_LEN, BATCH_SIZE
torch.set_num_threads(os.cpu_count() // 2)
DATA_DIR = "../data"
MODEL_DIR = "../models"
RESULTS_FILE = "../results.csv"
DEVICE = torch.device("cpu") # Forcing CPU for apples-to-apples comparison

def measure_inference(model, test_loader, run_name):
    """Runs inference, tracks energy, and returns metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    
    print(f"\n⚡ Running Inference for: {run_name}...")
    tracker = EmissionsTracker(project_name=run_name, measure_power_secs=1)
    tracker.start()
    
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

def apply_pruning(model, amount):
    """Applies L1 Unstructured Pruning to the Conv1d and Linear layers"""
    # We prune the Conv1d layer
    prune.l1_unstructured(model.conv1, name="weight", amount=amount)
    # We prune the Fully Connected layer
    prune.l1_unstructured(model.fc, name="weight", amount=amount)
    
    # Make the pruning permanent
    prune.remove(model.conv1, 'weight')
    prune.remove(model.fc, 'weight')
    return model

def main():
    print("📂 Loading test data and tokenizer...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_sampled.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_sampled.csv"))

    tokenizer = SimpleTokenizer(MAX_VOCAB_SIZE)
    tokenizer.fit(train_df['text'])
    test_dataset = TweetDataset(test_df['text'], test_df['label'], tokenizer, MAX_SEQ_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # We will test 3 different levels of sparsity (PnC approach)
    sparsity_levels = [0.20, 0.40, 0.60]
    results = []

    for sparsity in sparsity_levels:
        print(f"\n" + "="*40)
        print(f"✂️  APPLYING {int(sparsity*100)}% PRUNING")
        print("="*40)
        
        # 1. Load a fresh copy of the baseline model for each test
        model = SentimentCNN(MAX_VOCAB_SIZE, EMBED_DIM, num_classes=3)
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "baseline_model.pt")))
        
        # 2. Prune the model
        pruned_model = apply_pruning(model, amount=sparsity)
        
        # 3. Measure it
        run_name = f"Pruned_{int(sparsity*100)}pct"
        acc, f1, energy = measure_inference(pruned_model, test_loader, run_name)
        
        # 4. Save metrics
        results.append({
            "Model": run_name,
            "Accuracy": acc,
            "F1_Score": f1,
            "Energy_Joules": energy
        })

    # Save all results to our central CSV
    res_df = pd.DataFrame(results)
    res_df.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
    print("\n✅ All Pruning Ablation tests complete and saved to results.csv!")

if __name__ == "__main__":
    main()