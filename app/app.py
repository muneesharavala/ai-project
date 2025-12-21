from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import os

# Optional imports – use what you actually have
import pickle
import tensorflow as tf

# If you want X-ray image support, you also need:
import cv2
from PIL import Image
import io

# Optional OpenAI for summaries – set OPENAI_API_KEY in env
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
except Exception:
    client = None

app = Flask(__name__)
CORS(app)  # allow calls from your HTML/JS

# -----------------------------
# Load your existing models
# -----------------------------
MODELS = {
    "xray": None,
    "diabetes_model": None,
    "diabetes_scaler": None,
    "heart_model": None,
    "heart_scaler": None,
    "cancer_model": None,
    "cancer_scaler": None,
}

def load_all_models():
    """Load Keras / sklearn models from your models/ folder."""
    base = "models"

    # Chest X-ray TF model (same as in your Streamlit app)
    try:
        MODELS["xray"] = tf.keras.models.load_model(os.path.join(base, "chest_xray_model.h5"))
        print("Loaded chest_xray_model.h5")
    except Exception as e:
        print("X-ray model load failed:", e)

    # Diabetes
    try:
        MODELS["diabetes_model"] = pickle.load(open(os.path.join(base, "diabetes_model.pkl"), "rb"))
        MODELS["diabetes_scaler"] = pickle.load(open(os.path.join(base, "diabetes_scaler.pkl"), "rb"))
        print("Loaded diabetes model + scaler.")
    except Exception as e:
        print("Diabetes model load failed:", e)

    # Heart
    try:
        MODELS["heart_model"] = pickle.load(open(os.path.join(base, "heart_model.pkl"), "rb"))
        MODELS["heart_scaler"] = pickle.load(open(os.path.join(base, "heart_scaler.pkl"), "rb"))
        print("Loaded heart model + scaler.")
    except Exception as e:
        print("Heart model load failed:", e)

    # Cancer
    try:
        MODELS["cancer_model"] = pickle.load(open(os.path.join(base, "cancer_model.pkl"), "rb"))
        MODELS["cancer_scaler"] = pickle.load(open(os.path.join(base, "cancer_scaler.pkl"), "rb"))
        print("Loaded cancer model + scaler.")
    except Exception as e:
        print("Cancer model load failed:", e)

load_all_models()

# -----------------------------
# Optional AI summary helper
# -----------------------------
def make_summary(condition: str, label: str, score: float) -> str:
    """
    Try to use OpenAI for a short medical summary.
    If not available, return a fallback string.
    """
    if client is None:
        return (
            f"Condition: {condition}\n"
            f"Result: {label} ({score:.1f}% confidence).\n"
            "This is an AI-based prediction. Please consult a qualified doctor "
            "for clinical interpretation and next steps."
        )

    prompt = f"""
You are a senior doctor AI.

Condition: {condition}
Prediction label: {label}
Score: {score:.1f}%

Explain in simple language:
1. What this result generally means
2. Common risks or concerns
3. 3–4 precaution tips
4. Which specialist to consult
    """

    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"AI summary failed: {e}"

# -----------------------------
# API: Chest X-ray
# -----------------------------
@app.route("/api/xray", methods=["POST"])
def api_xray():
    if MODELS["xray"] is None:
        return jsonify({"status": "error", "message": "X-ray model not loaded"}), 500

    if "file" not in request.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    f = request.files["file"]
    try:
        # Read image with OpenCV-like pipeline
        file_bytes = np.frombuffer(f.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise ValueError("Cannot decode image")

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(img_gray, (224, 224)) / 255.0
        inp = np.expand_dims(resized, (0, -1)).astype(np.float32)

        pred = MODELS["xray"].predict(inp)
        prob = float(pred[0][0])
        label = "PNEUMONIA" if prob > 0.5 else "NORMAL"
        conf = prob if label == "PNEUMONIA" else 1 - prob
        conf_pct = conf * 100.0

        summary = make_summary("Chest X-Ray / Pneumonia Detection", label, conf_pct)

        return jsonify({
            "status": "ok",
            "label": label,
            "confidence": conf_pct,
            "summary": summary,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# -----------------------------
# API: Diabetes
# -----------------------------
@app.route("/api/diabetes", methods=["POST"])
def api_diabetes():
    if MODELS["diabetes_model"] is None or MODELS["diabetes_scaler"] is None:
        return jsonify({"status": "error", "message": "Diabetes model/scaler not loaded"}), 500

    data = request.get_json(force=True, silent=True) or {}
    try:
        X = np.array([[ 
            float(data["pregnancies"]),
            float(data["glucose"]),
            float(data["bp"]),
            float(data["skin"]),
            float(data["insulin"]),
            float(data["bmi"]),
            float(data["dpf"]),
            float(data["age"]),
        ]])
    except KeyError as e:
        return jsonify({"status": "error", "message": f"Missing field {e}"}), 400
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid numeric values"}), 400

    try:
        scaler = MODELS["diabetes_scaler"]
        model = MODELS["diabetes_model"]
        Xs = scaler.transform(X)
        pred = model.predict(Xs)[0]
        prob = float(model.predict_proba(Xs)[0][1])
        label = "DIABETIC" if pred == 1 else "NON-DIABETIC"
        risk_pct = prob * 100.0
        summary = make_summary("Diabetes Risk", label, risk_pct)
        return jsonify({
            "status": "ok",
            "label": label,
            "risk": risk_pct,
            "summary": summary,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# -----------------------------
# API: Heart Disease
# -----------------------------
@app.route("/api/heart", methods=["POST"])
def api_heart():
    if MODELS["heart_model"] is None or MODELS["heart_scaler"] is None:
        return jsonify({"status": "error", "message": "Heart model/scaler not loaded"}), 500

    data = request.get_json(force=True, silent=True) or {}
    try:
        X = np.array([[ 
            float(data["age"]),
            float(data["sex"]),
            float(data["cp"]),
            float(data["bp"]),
            float(data["chol"]),
            float(data["fbs"]),
            float(data["restecg"]),
            float(data["thalach"]),
            float(data["exang"]),
            float(data["oldpeak"]),
            float(data["slope"]),
            float(data["ca"]),
            float(data["thal"]),
        ]])
    except KeyError as e:
        return jsonify({"status": "error", "message": f"Missing field {e}"}), 400
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid numeric values"}), 400

    try:
        scaler = MODELS["heart_scaler"]
        model = MODELS["heart_model"]
        Xs = scaler.transform(X)
        pred = model.predict(Xs)[0]
        prob = float(model.predict_proba(Xs)[0][1])
        label = "HEART DISEASE DETECTED" if pred == 1 else "HEALTHY HEART"
        risk_pct = prob * 100.0
        summary = make_summary("Heart Disease Risk", label, risk_pct)
        return jsonify({
            "status": "ok",
            "label": label,
            "risk": risk_pct,
            "summary": summary,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# -----------------------------
# API: Cancer (Breast)
# -----------------------------
@app.route("/api/cancer", methods=["POST"])
def api_cancer():
    if MODELS["cancer_model"] is None or MODELS["cancer_scaler"] is None:
        return jsonify({"status": "error", "message": "Cancer model/scaler not loaded"}), 500

    data = request.get_json(force=True, silent=True) or {}
    try:
        X = np.array([[ 
            float(data["radius"]),
            float(data["texture"]),
            float(data["perimeter"]),
            float(data["area"]),
            float(data["smooth"]),
        ]])
    except KeyError as e:
        return jsonify({"status": "error", "message": f"Missing field {e}"}), 400
    except ValueError:
        return jsonify({"status": "error", "message": "Invalid numeric values"}), 400

    try:
        scaler = MODELS["cancer_scaler"]
        model = MODELS["cancer_model"]
        Xs = scaler.transform(X)
        pred = model.predict(Xs)[0]
        prob = float(model.predict_proba(Xs)[0][1])
        label = "MALIGNANT" if pred == 1 else "BENIGN"
        conf_pct = prob * 100.0
        summary = make_summary("Breast Cancer Prediction", label, conf_pct)
        return jsonify({
            "status": "ok",
            "label": label,
            "confidence": conf_pct,
            "summary": summary,
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# -----------------------------
# Root test
# -----------------------------
@app.route("/")
def root():
  return "LIFE-LEN AI backend is running."

if __name__ == "__main__":
    # Run backend on port 5000
    app.run(host="0.0.0.0", port=5000, debug=True)
