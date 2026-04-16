import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data", type=str, required=True)
    parser.add_argument("--score_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=50000)
    args = parser.parse_args()

    print(f"Loading scores from {args.score_file}...")
    with open(args.score_file, 'r') as f:
        scores = json.load(f)
    
    # scores is a list of [score, index]
    # Sort by score descending
    scores.sort(key=lambda x: x[0], reverse=True)
    top_indices = [item[1] for item in scores[:args.top_k]]
    top_indices_set = set(top_indices)

    print(f"Selecting top {args.top_k} samples from {args.raw_data}...")
    selected_data = []
    with open(args.raw_data, 'r') as f:
        for i, line in enumerate(f):
            if i in top_indices_set:
                selected_data.append(json.loads(line))
    
    print(f"Saving {len(selected_data)} samples to {args.output_file}...")
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    with open(args.output_file, 'w') as f:
        for item in selected_data:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    main()
