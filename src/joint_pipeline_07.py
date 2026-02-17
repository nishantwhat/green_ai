import os
import torch
import torch.nn.utils.prune as prune
import torch.ao.quantization as quantization
import pandas as pd
from torch.utils.data import DataLoader
from codecarbon import EmissionsTracker
import warnings

# Ignore warnings for clean output
warnings.filterwarnings("ignore")

from train_baseline_02 import SentimentCNN, SimpleTokenizer, TweetDataset, MAX_VOCAB_SIZE, EMBED_DIM, MAX_SEQ_LEN, BATCH_SIZE
from prune_03 import apply_pruning
from quantize_04 import measure_inference
torch.set_num_threads(os.cpu_count() // 2)
DATA_DIR = "../data"
MODEL_DIR = "../models"
RESULTS_FILE = "../results.csv"
DEVICE = torch.device("cpu") 

def main():
    print("📂 Loading data...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_sampled.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_sampled.csv"))

    tokenizer = SimpleTokenizer(MAX_VOCAB_SIZE)
    tokenizer.fit(train_df['text'])
    test_dataset = TweetDataset(test_df['text'], test_df['label'], tokenizer, MAX_SEQ_LEN)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("\n" + "="*50)
    print("🧬 THE JOINT PIPELINE: DISTILL -> PRUNE -> QUANTIZE")
    print("="*50)

    # 1. Load the Best Distilled Student (T=5.0)
    print("1️⃣ Loading Distilled Student (T=5.0)...")
    model = SentimentCNN(MAX_VOCAB_SIZE, EMBED_DIM, num_classes=3)
    # We use map_location='cpu' just in case it was saved on GPU
    model.load_state_dict(torch.load(os.path.join(MODEL_DIR, "Distilled_Student_T5.0.pt"), map_location='cpu'))
    
    # 2. Apply a safe 20% Pruning
    print("2️⃣ Applying 20% Structural Pruning...")
    pruned_model = apply_pruning(model, amount=0.20)
    pruned_model.eval() # Must eval before quantizing

    # 3. Apply INT8 Quantization to the Pruned Model
    print("3️⃣ Applying INT8 Dynamic Quantization to Sparse Model...")
    qconfig_dict = {
        torch.nn.Embedding: quantization.float_qparams_weight_only_qconfig,
        torch.nn.Linear: quantization.default_dynamic_qconfig
    }
    final_edge_model = quantization.quantize_dynamic(pruned_model, qconfig_spec=qconfig_dict)

    # 4. Measure the Ultimate Edge Model
    run_name = "Joint_Optimization_Edge_Model"
    acc, f1, energy = measure_inference(final_edge_model, test_loader, run_name)

    # 5. Save metrics
    res_df = pd.DataFrame([{
        "Model": run_name,
        "Accuracy": acc,
        "F1_Score": f1,
        "Energy_Joules": energy
    }])
    res_df.to_csv(RESULTS_FILE, mode='a', header=False, index=False)
    print("\n✅ Joint Optimization complete and saved to results.csv!")

if __name__ == "__main__":
    main()