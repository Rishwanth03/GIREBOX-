from src import preprocess

def main():
    RAW_DATA_DIR = "data/raw"
    PROCESSED_PATH = "data/processed/gearbox_dataset.csv"

    print("🔄 Loading and preprocessing dataset...")
    df = preprocess.load_and_label_data(RAW_DATA_DIR)

    print(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    print(df.head())

    print("💾 Saving processed dataset...")
    preprocess.save_processed_data(df, PROCESSED_PATH)

if __name__ == "__main__":
    main()
