import json
import math
import urllib.request
import statistics

def main():
    print("Downloading Wine dataset from UCI Archive (Pure Python)...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    
    try:
        with urllib.request.urlopen(url) as response:
            data = response.read().decode('utf-8')
    except Exception as e:
        print(f"Failed to download data: {e}")
        return

    print("Parsing data...")
    dataset = []
    lines = data.strip().split('\n')
    for line in lines:
        if not line:
            continue
        try:
            parts = [float(x) for x in line.split(',')]
            dataset.append(parts)
        except ValueError:
            continue

    # 1. Separate by Class
    # Class is the first column (index 0)
    separated = {}
    for row in dataset:
        class_val = int(row[0])
        if class_val not in separated:
            separated[class_val] = []
        # Store features only (exclude class)
        separated[class_val].append(row[1:])

    print(f"Data parsed. Found classes: {list(separated.keys())}")

    # 2. Calculate Mean and Stdev for each feature for each class
    # Model Structure: { class_id: { feature_idx: { mean: float, stdev: float } } }
    model = {}

    for class_val, rows in separated.items():
        model[class_val] = {}
        # Transpose rows to columns to calculate stats per feature
        num_features = len(rows[0])
        for i in range(num_features):
            col_values = [r[i] for r in rows]
            mean_val = statistics.mean(col_values)
            stdev_val = statistics.stdev(col_values)
            # Handle zero stdev if any (add epsilon) works for Gaussian NB
            if stdev_val == 0:
                stdev_val = 1e-6
                
            model[class_val][i] = {
                'mean': mean_val,
                'stdev': stdev_val
            }

    print("Training complete (Gaussian Naive Bayes parameters calculated).")

    # 3. Save Model to JSON
    with open('model.json', 'w') as f:
        json.dump(model, f, indent=4)
    
    print("Model saved to model.json")

if __name__ == "__main__":
    main()
