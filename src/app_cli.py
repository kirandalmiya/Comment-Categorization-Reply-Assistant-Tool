# src/app_cli.py

import argparse
import pandas as pd
#from src.inference import classify_comment
from inference import classify_comment

def process_file(input_path: str, output_path: str):
    df = pd.read_csv(input_path)
    if "text" not in df.columns:
        raise ValueError("Input CSV must have a 'text' column.")

    results = df["text"].astype(str).apply(classify_comment)
    df["predicted_label"] = results.apply(lambda r: r["label"])
    df["suggested_reply"] = results.apply(lambda r: r["reply"])

    # Sort by label to “bucket” comments
    df.sort_values("predicted_label", inplace=True)

    df.to_csv(output_path, index=False)
    print(f"Saved categorized comments to {output_path}")

def interactive_mode():
    print("Interactive Comment Categorization. Type 'quit' to exit.\n")
    while True:
        text = input("Enter a comment: ")
        if text.strip().lower() == "quit":
            break
        result = classify_comment(text)
        print(f"Label: {result['label']}")
        print(f"Reply: {result['reply']}\n")

def main():
    parser = argparse.ArgumentParser(description="Comment Categorization & Reply Assistant CLI")
    parser.add_argument("--input", type=str, help="Path to CSV with 'text' column")
    parser.add_argument("--output", type=str, default="data/output_categorized.csv")
    args = parser.parse_args()

    if args.input:
        process_file(args.input, args.output)
    else:
        interactive_mode()

if __name__ == "__main__":
    main()
