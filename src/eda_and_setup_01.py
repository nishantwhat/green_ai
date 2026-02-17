import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset


# 1. Setup paths
DATA_DIR = "../data"
PLOT_DIR = "../plots"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def main():
    print("📥 Downloading 'tweet_eval' sentiment dataset from Hugging Face...")
    dataset = load_dataset("tweet_eval", "sentiment")

    # 2. Convert to Pandas for easy manipulation and sampling
    df_train = pd.DataFrame(dataset['train'])
    df_test = pd.DataFrame(dataset['test'])

# 3. Create our "Massive Split" (25000 train, 5000 test)
    print("✂️ Creating 25,000 / 5,000 data split...")
    df_train_sampled = df_train.sample(n=25000, random_state=42).reset_index(drop=True)
    df_test_sampled = df_test.sample(n=5000, random_state=42).reset_index(drop=True)
    # Save to disk
    train_path = os.path.join(DATA_DIR, "train_sampled.csv")
    test_path = os.path.join(DATA_DIR, "test_sampled.csv")
    df_train_sampled.to_csv(train_path, index=False)
    df_test_sampled.to_csv(test_path, index=False)
    print(f"✅ Data saved to {DATA_DIR}")

    # 4. Exploratory Data Analysis (EDA) - Class Distribution
    print("📊 Generating EDA Plots...")
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(data=df_train_sampled, x='label', palette="viridis")
    plt.title('Class Distribution in Training Set (5000 samples)')
    plt.xlabel('Sentiment (0: Negative, 1: Neutral, 2: Positive)')
    plt.ylabel('Count')
    
    # Add count labels on top of bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontsize=10)
        
    plt.savefig(os.path.join(PLOT_DIR, "class_distribution.png"), dpi=300)
    plt.close()

    # 5. EDA - Tweet Length Distribution
    # We calculate word count by splitting the text string by spaces
    df_train_sampled['word_count'] = df_train_sampled['text'].apply(lambda x: len(str(x).split()))
    
    plt.figure(figsize=(8, 5))
    sns.histplot(df_train_sampled['word_count'], bins=30, kde=True, color='coral')
    plt.title('Tweet Word Count Distribution')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(PLOT_DIR, "tweet_length_distribution.png"), dpi=300)
    plt.close()
    from collections import Counter
    import itertools

    print("📊 Generating Top N-Gram Analysis...")
    # Get all words, filter out tiny words like "a" or "the"
    all_words = list(itertools.chain(*[str(t).lower().split() for t in df_train_sampled['text']]))
    filtered_words = [w for w in all_words if len(w) > 3]
    word_freq = Counter(filtered_words).most_common(20)
    
    words, counts = zip(*word_freq)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(counts), y=list(words), palette="magma", hue=list(words), legend=False)
    plt.title('Top 20 Most Frequent Words (Length > 3)', fontsize=14)
    plt.xlabel('Frequency Count')
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "03_top_words_distribution.png"), dpi=300)
    plt.close()
    print(f"✅ EDA Plots saved to {PLOT_DIR}")
    print("🚀 Setup Complete. Ready for Baseline Training!")

if __name__ == "__main__":
    main()