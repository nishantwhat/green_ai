import os
import warnings
import torch
import torch.ao.quantization as quantization
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from codecarbon import EmissionsTracker

# Ignore the PyTorch torchao deprecation warning for clean terminal output
warnings.filterwarnings("ignore", category=DeprecationWarning)

from train_baseline_02 import SentimentCNN, SimpleTokenizer, TweetDataset, MAX_VOCAB_SIZE, EMBED_DIM, MAX_SEQ_LEN, BATCH_SIZE
torch.set_num_threads(os.cpu_count() // 2)

DATA_DIR = "../data"
MODEL_DIR = "../models"
RESULTS_FILE = "../results.csv"
DEVICE = torch.device("cpu") 

def measure_inference(model, test_loader, run_name):
    model.eval()
    all_preds = []
    all_labels = []
    
    print(f"\n⚡ Running Inference for: {run_name}...")
    
    # We use tracking_mode="machine" to avoid the 1000W GPU driver bug!
    tracker = EmissionsTracker(project_name=run_name, measure_power_secs=1, tracking_mode="machine")
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
    
    # Check physical file size reduction
    torch.save(model.state_dict(), f"../models/{run_name}.pt")
    size_mb = os.path.getsize(f"../models/{run_name}.pt") / (1024 * 1024)
    print(f"[{run_name}] File Size: {size_mb:.2f} MB")
    
    return acc, f1, energy_joules

def main():
    print("📂 Loading test data and tokenizer...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_sampled.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_sampled.csv"))

    tokenizer = SimpleTokenizer(MAX_VOCAB_SIZE)
    tokenizer.fit(train_df['text'])
    test_dataset = TweetDataset(test_df['text'], test_df['label'], tokenizer, MAX_SEQ_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\n" + "="*40)
    print("🗜️  APPLYING INT8 DYNAMIC QUANTIZATION")
    print("="*40)
    
    # 1. Load Baseline
    model = SentimentCNN(MAX_VOCAB_SIZE, EMBED_DIM, num_classes=3)
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "baseline_model.pt")))
    model.eval() 
    
    base_size_mb = os.path.getsize(os.path.join(MODEL_DIR, "baseline_model.pt")) / (1024 * 1024)
    print(f"Baseline File Size: {base_size_mb:.2f} MB")

    # 2. Apply Dynamic Quantization (THE FIX)
    # We explicitly map the correct configurations to the specific layers
    qconfig_dict = {
        torch.nn.Embedding: quantization.float_qparams_weight_only_qconfig,
        torch.nn.Linear: quantization.default_dynamic_qconfig
    }
    
    quantized_model = quantization.quantize_dynamic(
        model, 
        qconfig_spec=qconfig_dict
    )
    
    # 3. Measure it
    run_name = "Quantized_INT8"
    acc, f1, energy = measure_inference(quantized_model, test_loader, run_name)
    
    # 4. Save metrics
    res_df = pd.DataFrame([{
        "Model": run_name,
        "Accuracy": acc,
        "F1_Score": f1,
        "Energy_Joules": energy
    }])
    res_df.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
    print("\n✅ Quantization Ablation test complete and saved to results.csv!")

if __name__ == "__main__":
    main()