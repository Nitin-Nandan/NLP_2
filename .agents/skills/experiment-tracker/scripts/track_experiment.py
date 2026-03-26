import argparse
import csv
import os

def main():
    parser = argparse.ArgumentParser(description="Log experiment metrics to CSV.")
    parser.add_argument("--model_name", required=True, type=str)
    parser.add_argument("--accuracy", required=True, type=float)
    parser.add_argument("--f1_score", required=True, type=float)
    parser.add_argument("--train_time", required=True, type=float)
    parser.add_argument("--parameters", required=True, type=int)
    parser.add_argument("--output_file", default="results/results_log.csv", type=str)
    
    args = parser.parse_args()
    
    file_exists = os.path.isfile(args.output_file)
    fields = ["model_name", "accuracy", "f1_score", "train_time", "parameters"]
    
    with open(args.output_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "model_name": args.model_name,
            "accuracy": args.accuracy,
            "f1_score": args.f1_score,
            "train_time": args.train_time,
            "parameters": args.parameters
        })
    print(f"Metrics successfully logged to {args.output_file}")

if __name__ == "__main__":
    main()
