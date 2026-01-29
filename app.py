import streamlit as st
import json
import math
import os

# Page setup
st.set_page_config(page_title="Wine Cultivar Prediction", layout="centered")
st.title("🍷 Wine Cultivar Prediction System")
st.write("Predicts the wine cultivar using a Naive Bayes model.")

MODEL_PATH = "model.json"
model = None

# ---------- MODEL LOADING ----------
def load_assets():
    global model
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "r") as f:
            raw_model = json.load(f)
            model = {
                int(class_key): {
                    int(feat_key): stats
                    for feat_key, stats in features.items()
                }
                for class_key, features in raw_model.items()
            }
    else:
        st.error("Model file not found.")

def calculate_probability(x, mean, stdev):
    exponent = math.exp(-((x - mean) ** 2 / (2 * stdev ** 2)))
    return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

if model is None:
    load_assets()

# ---------- INPUT UI ----------
st.subheader("Enter Wine Chemical Properties")

inputs = [
    st.number_input("Alcohol"),
    st.number_input("Malic Acid"),
    st.number_input("Ash"),
    st.number_input("Alcalinity of Ash"),
    st.number_input("Magnesium"),
    st.number_input("Total Phenols"),
    st.number_input("Flavanoids"),
    st.number_input("Nonflavanoid Phenols"),
    st.number_input("Proanthocyanins"),
    st.number_input("Color Intensity"),
    st.number_input("Hue"),
    st.number_input("OD280"),
    st.number_input("Proline"),
]

# ---------- PREDICTION ----------
if st.button("Predict Cultivar"):
    if not model:
        st.error("Model not loaded.")
    else:
        probabilities = {}

        for class_val, class_stats in model.items():
            probabilities[class_val] = 1
            for i, x in enumerate(inputs):
                stats = class_stats.get(i)
                if stats:
                    probabilities[class_val] *= calculate_probability(
                        x, stats["mean"], stats["stdev"]
                    )

        best_class = max(probabilities, key=probabilities.get)
        total_prob = sum(probabilities.values())

        confidence = (
            (probabilities[best_class] / total_prob) * 100
            if total_prob > 0
            else 0
        )

        class_names = {
            1: "Cultivar 1 (e.g. Barolo)",
            2: "Cultivar 2 (e.g. Grignolino)",
            3: "Cultivar 3 (e.g. Barbera)",
        }

        st.success(f"Prediction: {class_names.get(best_class)}")
        st.info(f"Confidence: {confidence:.2f}%")