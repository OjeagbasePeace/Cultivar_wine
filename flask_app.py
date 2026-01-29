from flask import Flask, request, jsonify, render_template
import json
import math
import os

app = Flask(__name__)

# Load model (Pure Python / JSON)
MODEL_PATH = 'model.json'
model = None

def load_assets():
    global model
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'r') as f:
                # Convert keys back to int (JSON stores keys as strings)
                raw_model = json.load(f)
                model = {}
                for class_key, features in raw_model.items():
                    model[int(class_key)] = {}
                    for feat_key, stats in features.items():
                        model[int(class_key)][int(feat_key)] = stats
            print("Model (Naive Bayes) loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
    else:
        print("Model not found. Please run train_model.py first.")

def calculate_probability(x, mean, stdev):
    # Gaussian Probability Density Function
    exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global model
    
    if not model:
        load_assets()
        if not model:
             return jsonify({'error': 'Model not loaded. Please contact administrator.'}), 500

    try:
        data = request.get_json()
        # Ensure order matches dataset (13 features)
        # Alcohol, Malic acid, Ash, Alcalinity, Magnesium, Total phenols, Flavanoids, 
        # Nonflavanoid phenols, Proanthocyanins, Color intensity, Hue, OD280, Proline
        input_vector = [
            float(data.get('alcohol')),
            float(data.get('malic_acid')),
            float(data.get('ash')),
            float(data.get('alcalinity_of_ash')),
            float(data.get('magnesium')),
            float(data.get('total_phenols')),
            float(data.get('flavanoids')),
            float(data.get('nonflavanoid_phenols')),
            float(data.get('proanthocyanins')),
            float(data.get('color_intensity')),
            float(data.get('hue')),
            float(data.get('od280')),
            float(data.get('proline'))
        ]
        
        # Naive Bayes Inference (Pure Python)
        probabilities = {}
        for class_val, class_stats in model.items():
            probabilities[class_val] = 1 # Initialize product
            for i, x in enumerate(input_vector):
                stats = class_stats.get(i)
                if stats:
                    prob = calculate_probability(x, stats['mean'], stats['stdev'])
                    probabilities[class_val] *= prob
        
        # Find max probability (unnormalized likelihood)
        best_class = None, 
        max_prob = -1
        
        # Normalize for display confidence
        total_prob = sum(probabilities.values())
        normalized_probs = {}
        
        best_class = max(probabilities, key=probabilities.get)
        if total_prob > 0:
            confidence = (probabilities[best_class] / total_prob) * 100
        else:
            confidence = 0

        # Mapping class index (1, 2, 3 in Wine Dataset)
        # 1: Barolo, 2: Grignolino, 3: Barbera (Usually)
        # Or simple Cultivar 1, 2, 3
        class_names = {
            1: 'Cultivar 1 (e.g. Barolo)',
            2: 'Cultivar 2 (e.g. Grignolino)',
            3: 'Cultivar 3 (e.g. Barbera)'
        }
        
        predicted_name = class_names.get(best_class, f"Cultivar {best_class}")
        
        return jsonify({
            'prediction': predicted_name,
            'confidence': f"{confidence:.2f}%"
        })

    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 400

if __name__ == '__main__':
    load_assets()
    app.run(debug=True)
