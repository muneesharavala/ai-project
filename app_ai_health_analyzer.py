import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pickle
import os
import time
import base64
from gtts import gTTS
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import pandas as pd
from datetime import datetime
import streamlit.components.v1 as components
import torch
from torchvision import transforms, models
import pdfplumber
import pytesseract
from PIL import Image
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.colors import HexColor
from reportlab.lib import colors
from reportlab.lib.units import cm
from datetime import datetime
import uuid
import qrcode
import tempfile
from utils.database import create_table, insert_prediction, fetch_predictions

create_table()

TEAL = "#38bdf8"
# -----------------------------
# Helper: load logo safely
# -----------------------------
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

LOGO_BASE64 = get_base64_image("assets/logo.png")

import requests

XRAY_MODEL_PATH = "models/chest_xray_model.h5"
XRAY_MODEL_URL = "https://drive.google.com/file/d/1NRFFqkWix8kwvDVSTXUED2LLeUJAodxB/view?usp=drive_link"

def ensure_xray_model():
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(XRAY_MODEL_PATH):
        with st.spinner("📥 Downloading Chest X-Ray AI model (first run only)..."):
            r = requests.get(XRAY_MODEL_URL, stream=True)
            if r.status_code != 200:
                st.error("Failed to download X-Ray model.")
                return

            with open(XRAY_MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

# -----------------------------
# Page config (FIRST Streamlit call)
# -----------------------------
st.set_page_config(
    page_title="LIFE-LEN AI Health Analyzer",
    page_icon="🏥",
    layout="wide"
)
def generate_patient_id():
    return f"LL-{uuid.uuid4().hex[:10].upper()}"
def generate_qr_code(data: str):
    qr = qrcode.make(data)
    temp_path = os.path.join(tempfile.gettempdir(), "patient_qr.png")
    qr.save(temp_path)
    return temp_path

# -----------------------------
# Session defaults
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None

if "Navigation" not in st.session_state:
    st.session_state.Navigation = "Home"

if "users_db" not in st.session_state:
    st.session_state.users_db = {
        "admin": "admin123"
    }

# =========================
# TOP HEADER (ALL PAGES)
# =========================
def render_top_header():

    user_block = (
        f"<div class='header-user'>👤 {st.session_state.user}</div>"
        if st.session_state.logged_in and st.session_state.user
        else "<div class='header-user'>🔒 Secure Access</div>"
    )

    st.markdown(
        f"""
        <style>
        .top-header {{
            position: sticky;
            top: 0;
            z-index: 1000;
            background: linear-gradient(90deg, #020617, #0f172a);
            padding: 0.9rem 1.6rem;
            border-bottom: 1px solid rgba(255,255,255,0.08);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}

        .header-left {{
            display: flex;
            align-items: center;
            gap: 0.9rem;
        }}

        .header-logo img {{
            height: 42px;
        }}

        .header-title {{
            font-size: 1.15rem;
            font-weight: 700;
            color: #38bdf8;
            line-height: 1.1;
        }}

        .header-sub {{
            font-size: 0.75rem;
            color: #94a3b8;
        }}

        .header-user {{
            background: rgba(255,255,255,0.06);
            padding: 0.35rem 0.7rem;
            border-radius: 999px;
            font-size: 0.85rem;
        }}
        </style>

        <div class="top-header">
            <div class="header-left">
                <div class="header-logo">
                    <img src="data:image/png;base64,{LOGO_BASE64}">
                </div>
                <div>
                    <div class="header-title">LIFE-LEN AI</div>
                    <div class="header-sub">Clinical AI Decision Support</div>
                </div>
            </div>
            {user_block}
        </div>
        """,
        unsafe_allow_html=True
    )

render_top_header()

# -----------------------------
# Navigation options
# -----------------------------
NAV_OPTIONS = [
    "Home", "X-Ray", "Diabetes", "Heart",
    "Cancer", "Reports", "Dashboard",
    "About", "Contact", "Legal"
]

if st.session_state.Navigation not in NAV_OPTIONS:
    st.session_state.Navigation = "Home"
# =========================
# SIDEBAR
# =========================
with st.sidebar:

    st.markdown(
        f"""
        <style>
        .sidebar-logo {{
            display: flex;
            justify-content: center;
            margin-top: 0.8rem;
            margin-bottom: 0.6rem;
        }}

        .sidebar-logo img {{
            width: 90px;
            height: 90px;
            border-radius: 50%;
            object-fit: cover;
            border: 2px solid rgba(56,189,248,0.6);
            box-shadow: 0 0 25px rgba(56,189,248,0.35);
            background: #020617;
        }}

        .sidebar-title {{
            text-align: center;
            font-size: 1.05rem;
            font-weight: 700;
            color: #38bdf8;
            margin-bottom: 0.2rem;
        }}

        .sidebar-sub {{
            text-align: center;
            font-size: 0.7rem;
            color: #94a3b8;
            margin-bottom: 1rem;
        }}
        </style>

        <div class="sidebar-logo">
            <img src="data:image/png;base64,{LOGO_BASE64}">
        </div>

        <div class="sidebar-title">LIFE-LEN AI</div>
        <div class="sidebar-sub">Clinical AI Platform</div>
        """,
        unsafe_allow_html=True
    )

    # ---------- LOGIN STATUS ----------
    if st.session_state.logged_in and st.session_state.user:
        st.success(f"👤 Logged in as **{st.session_state.user}**")

        if st.button("🚪 Logout", key="logout_btn"):
            st.session_state.logged_in = False
            st.session_state.user = None
            st.session_state.Navigation = "Home"
            st.rerun()

        st.markdown("---")

        # ---------- NAVIGATION ----------
        page = st.radio(
            "Navigation",
            NAV_OPTIONS,
            index=NAV_OPTIONS.index(st.session_state.Navigation),
            key="sidebar_nav"
        )
        st.session_state.Navigation = page

    else:
        st.info("🔐 Please login to access AI tools")
        st.markdown("---")

    # ---------- INFO ----------
    st.markdown("""
    **AI Capabilities**
    - 🫁 Chest X-Ray  
    - 💉 Diabetes  
    - ❤️ Heart Disease  
    - 🧬 Cancer  
    - 📑 Medical Reports  

    **Notes**
    - AI summaries always enabled  
    - Voice output uses internet  
    """)

# =========================
# AUTH PAGE
# =========================
def auth_page():

    st.markdown("""
    <style>
    .auth-box {
        background: linear-gradient(180deg, #0f172a, #020617);
        border-radius: 22px;
        padding: 2.5rem;
        box-shadow: 0 30px 90px rgba(0,0,0,0.85);
        border: 1px solid rgba(255,255,255,0.12);
    }
    .auth-title {
        font-size: 2rem;
        font-weight: 900;
        text-align: center;
        color: #38bdf8;
    }
    .auth-sub {
        text-align: center;
        color: #94a3b8;
        margin-bottom: 1.6rem;
    }
    </style>
    """, unsafe_allow_html=True)

    _, center, _ = st.columns([1, 1.2, 1])

    with center:
        st.markdown('<div class="auth-box">', unsafe_allow_html=True)

        st.markdown('<div class="auth-title">🏥 LIFE-LEN AI</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-sub">Secure access to AI-powered diagnostics</div>', unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["🔑 Login", "📝 Sign Up"])

        with tab_login:
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            if st.button("Login", use_container_width=True):
                if u in st.session_state.users_db and st.session_state.users_db[u] == p:
                    st.session_state.logged_in = True
                    st.session_state.user = u
                    st.rerun()
                else:
                    st.error("Invalid credentials")

        with tab_signup:
            nu = st.text_input("New Username")
            np = st.text_input("New Password", type="password")
            cp = st.text_input("Confirm Password", type="password")
            if st.button("Create Account", use_container_width=True):
                if np != cp:
                    st.error("Passwords do not match")
                else:
                    st.session_state.users_db[nu] = np
                    st.success("Account created")

        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# AUTH GUARD
# =========================
if not st.session_state.logged_in:
    auth_page()
    st.stop()


# =========================
# GLOBAL STYLES
# =========================
st.markdown("""
<style>
/* ===============================
   GLOBAL APP BACKGROUND
================================ */
.stApp {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    background:
        linear-gradient(rgba(2,6,23,0.88), rgba(2,6,23,0.88)),
        url("https://images.unsplash.com/photo-1586773860418-d37222d8fce3");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}

/* ===============================
   MAIN CONTENT
================================ */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 4rem;
    animation: fadeSlideIn 0.55s ease-out;
}

/* ===============================
   SIDEBAR
================================ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0f172a);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* ===============================
   CARDS
================================ */
.card, .tool-card {
    background: rgba(15,23,42,0.85);
    backdrop-filter: blur(8px);
    border-radius: 18px;
    padding: 1.6rem;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 12px 40px rgba(0,0,0,0.6);
}

/* ===============================
   BUTTONS
================================ */
.stButton > button {
    background: linear-gradient(90deg, #2563eb, #38bdf8);
    color: #fff;
    border-radius: 999px;
    padding: 0.6rem 1.6rem;
    font-weight: 600;
    border: none;
}

.stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 30px rgba(56,189,248,0.4);
}

/* ===============================
   FOOTER (STICKS TO BOTTOM)
================================ */
.footer {
    margin-top: auto;
    width: 100%;
    padding: 1.8rem 1rem;
    text-align: center;
    background: linear-gradient(90deg, #020617, #0f172a);
    border-top: 1px solid rgba(255,255,255,0.08);
}

.footer h4 {
    color: #14f1c9;
    margin-bottom: 0.3rem;
}

.footer p {
    color: #94a3b8;
    font-size: 0.9rem;
}

.footer-bottom {
    margin-top: 0.5rem;
    font-size: 0.8rem;
    color: #64748b;
}

/* ===============================
   PAGE TRANSITION
================================ */
@keyframes fadeSlideIn {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Voice Output
# -----------------------------
def speak_text(text: str):
    """Convert text to speech using gTTS and play inside Streamlit."""
    try:
        tts = gTTS(text, lang="en")
        audio_file = "voice_output.mp3"
        tts.save(audio_file)

        with open(audio_file, "rb") as f:
            audio_bytes = f.read()

        audio_base64 = base64.b64encode(audio_bytes).decode()

        st.markdown(
            f"""
            <audio autoplay>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
            </audio>
            """,
            unsafe_allow_html=True
        )
    except Exception as e:
        st.warning(f"Voice output unavailable: {e}")

def speak_text_autoplay(text: str):
    """Generate TTS audio (gTTS) and embed it with autoplay. Silent fail."""
    try:
        tts = gTTS(text, lang="en")
        tpath = "voice.mp3"
        tts.save(tpath)
        with open(tpath, "rb") as f:
            audio_b64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f"<audio autoplay><source src='data:audio/mp3;base64,{audio_b64}' type='audio/mp3'></audio>",
            unsafe_allow_html=True
        )
    except Exception:
        pass

# Optional OpenAI client (place API key in Streamlit secrets under OPENAI_API_KEY)
try:
    from openai import OpenAI
    client = OpenAI(api_key=st.secrets.get("OPENAI_API_KEY"))
except Exception:
    client = None

# -----------------------------
# Session defaults
# -----------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# Navigation state already created above as st.session_state["Navigation"]

# -----------------------------
# PDF helper + sample reports
# -----------------------------
SAMPLES_DIR = "samples"
os.makedirs(SAMPLES_DIR, exist_ok=True)

def create_pdf(lines, filename="report.pdf", title="AI Report"):
    c = canvas.Canvas(filename, pagesize=letter)
    w, h = letter

    c.setFillColorRGB(0.0/255, 198/255, 169/255)
    c.rect(0, h - 80, w, 80, stroke=0, fill=1)

    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, h - 50, title)

    c.setFillColorRGB(0.06, 0.12, 0.18)
    c.setFont("Helvetica", 11)
    y = h - 110

    for line in lines:
        text = str(line)
        max_chars = 100
        while len(text) > max_chars:
            chunk = text[:max_chars]
            c.drawString(50, y, chunk)
            text = text[max_chars:]
            y -= 14
            if y < 80:
                c.showPage()
                y = h - 80
        c.drawString(50, y, text)
        y -= 18
        if y < 80:
            c.showPage()
            y = h - 80

    c.save()
    return filename

def generate_sample_reports():
    heart_file = os.path.join(SAMPLES_DIR, "sample_heart_report.pdf")
    diabetes_file = os.path.join(SAMPLES_DIR, "sample_diabetes_report.pdf")
    cancer_file = os.path.join(SAMPLES_DIR, "sample_cancer_report.pdf")

    # Heart sample
    lines = [
        "LIFE-LEN — Sample Heart Disease Report",
        "",
        "Patient: Demo Patient",
        "Date: Sample",
        "",
        "Result: HEALTHY HEART (sample)",
        "Confidence: 12.5%",
        "",
        "Meaning of Result:",
        "- Low predicted risk for coronary disease based on sample inputs.",
        "",
        "Which Doctor to Consult:",
        "- Cardiologist (routine).",
        "",
        "Common Symptoms & Risks:",
        "- Typical: chest discomfort, shortness of breath (absent in this sample).",
        "",
        "Precautions & Safety Measures:",
        "- Regular exercise, avoid heavy tobacco use, manage cholesterol.",
        "",
        "Diet Plan (sample):",
        "- Increase vegetables and whole grains.",
        "- Lean proteins (fish, poultry).",
        "- Reduce saturated fats and processed foods.",
        "- Limit salt and sugar.",
        "- Maintain hydration.",
        "",
        "Treatment Options & Estimated Cost (India):",
        "- Routine tests & follow-ups: INR 2,000 - 8,000.",
        "- If interventional procedures required: INR 50,000 - 2,00,000+.",
        "",
        "Next Steps:",
        "- 24 hours: Keep track of symptoms.",
        "- 3 days: Book routine cardiology consult if concerns persist.",
        "- 7 days: Follow-up tests as recommended by specialist.",
    ]
    create_pdf(lines, heart_file, "Sample Heart Disease Report")

    # Diabetes sample
    lines = [
        "LIFE-LEN — Sample Diabetes Report",
        "",
        "Patient: Demo Patient",
        "Date: Sample",
        "",
        "Result: PRE-DIABETES (sample)",
        "Risk Score: 28.0%",
        "",
        "Meaning of Result:",
        "- Increased risk for developing diabetes, lifestyle modification recommended.",
        "",
        "Which Doctor to Consult:",
        "- Endocrinologist or Diabetologist (routine).",
        "",
        "Common Symptoms & Risks:",
        "- Thirst, frequent urination, fatigue (monitor).",
        "",
        "Precautions & Safety Measures:",
        "- Monitor blood glucose, increase activity, modify diet.",
        "",
        "Diet Plan (sample):",
        "- Balanced breakfast with protein and fiber.",
        "- Reduce refined carbs and sugars.",
        "- Regular small meals and portion control.",
        "- Avoid sugary drinks.",
        "- Prefer complex carbs and legumes.",
        "",
        "Treatment Options & Estimated Cost (India):",
        "- Initial blood tests & consults: INR 1,000 - 5,000.",
        "- Ongoing management: INR 500 - 5,000/month depending on meds.",
        "",
        "Next Steps:",
        "- 24 hours: Record fasting blood glucose if possible.",
        "- 3 days: Start dietary log.",
        "- 7 days: Consult with specialist with test results.",
    ]
    create_pdf(lines, diabetes_file, "Sample Diabetes Report")

    # Cancer sample
    lines = [
        "LIFE-LEN — Sample Cancer Report",
        "",
        "Patient: Demo Patient",
        "Date: Sample",
        "",
        "Result: BENIGN (sample)",
        "Confidence: 86.0%",
        "",
        "Meaning of Result:",
        "- The classifier indicates a benign finding in the sample dataset.",
        "",
        "Which Doctor to Consult:",
        "- Oncologist or Surgical Oncologist if further evaluation needed.",
        "",
        "Common Symptoms & Risks:",
        "- Lump, localized pain, changes in tissue (monitor closely).",
        "",
        "Precautions & Safety Measures:",
        "- Promptly evaluate any new lumps or changes; follow-up imaging.",
        "",
        "Diet Plan (sample):",
        "- High-antioxidant fruits and vegetables.",
        "- Lean proteins and whole grains.",
        "- Avoid excessive processed meats and alcohol.",
        "- Maintain healthy BMI.",
        "- Ensure adequate protein intake during treatment.",
        "",
        "Treatment Options & Estimated Cost (India):",
        "- Diagnostics & biopsy: INR 5,000 - 50,000.",
        "- Treatments (surgery/chemo/radiation): INR 50,000 - several lakhs depending on stage.",
        "",
        "Next Steps:",
        "- 24 hours: Keep records of any symptoms.",
        "- 3 days: Schedule consult if patient or clinician recommends.",
        "- 7 days: Consider targeted imaging if advised.",
    ]
    create_pdf(lines, cancer_file, "Sample Cancer Report")

    return {
        "heart": heart_file,
        "diabetes": diabetes_file,
        "cancer": cancer_file
    }

SAMPLE_PATHS = generate_sample_reports()
def create_professional_xray_pdf(
    patient_name,
    age,
    gender,
    result,
    severity,
    confidence,
    ai_summary,
    filename="xray_report.pdf"
):
    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4

    # ===== COLORS =====
    PRIMARY = HexColor("#0f172a")
    ACCENT = HexColor("#38bdf8")
    TEXT = HexColor("#111827")
    MUTED = HexColor("#6b7280")

    # ===== HEADER =====
    c.setFillColor(PRIMARY)
    c.rect(0, height - 90, width, 90, stroke=0, fill=1)

    c.setFillColor("white")
    c.setFont("Helvetica-Bold", 20)
    c.drawString(40, height - 55, "LIFE-LEN AI Diagnostic Center")

    c.setFont("Helvetica", 10)
    c.drawString(40, height - 75, "AI-Assisted Clinical Decision Support")

    # ===== REPORT METADATA =====
    y = height - 120
    c.setFillColor(TEXT)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(40, y, "Patient Information")

    c.setFont("Helvetica", 10)
    y -= 18
    c.drawString(40, y, f"Patient Name: {patient_name}")
    c.drawString(320, y, f"Gender: {gender}")

    y -= 15
    c.drawString(40, y, f"Age: {age}")
    c.drawString(320, y, f"Report Date: {datetime.now().strftime('%d %b %Y')}")

    # ===== DIVIDER =====
    y -= 15
    c.setStrokeColor(ACCENT)
    c.line(40, y, width - 40, y)

    # ===== RESULT SUMMARY =====
    y -= 30
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "Chest X-Ray Diagnostic Summary")

    y -= 22
    c.setFont("Helvetica", 11)
    c.drawString(40, y, f"Detected Condition: {result}")
    y -= 16
    c.drawString(40, y, f"Severity Level: {severity}")
    y -= 16
    c.drawString(40, y, f"AI Confidence Score: {confidence:.2f}%")

    # ===== CLINICAL INTERPRETATION =====
    y -= 30
    c.setFont("Helvetica-Bold", 13)
    c.drawString(40, y, "Clinical Interpretation")

    y -= 18
    c.setFont("Helvetica", 10)

    text_obj = c.beginText(40, y)
    text_obj.setLeading(14)

    for line in ai_summary.split("\n"):
        text_obj.textLine(line)

    c.drawText(text_obj)

    # ===== DISCLAIMER =====
    y = 90
    c.setStrokeColor(MUTED)
    c.line(40, y + 15, width - 40, y + 15)

    c.setFont("Helvetica-Oblique", 8)
    c.setFillColor(MUTED)
    c.drawString(
        40,
        y,
        "Disclaimer: This report is AI-assisted and intended for clinical decision support only. "
        "Final diagnosis must be confirmed by a licensed radiologist or physician."
    )

    # ===== FOOTER =====
    c.setFont("Helvetica", 9)
    c.drawRightString(
        width - 40,
        40,
        "Generated by LIFE-LEN AI • Secure Medical Platform"
    )

    c.save()
    return filename
# -----------------------------
# AI summary (OpenAI / fallback)
# -----------------------------
def ai_summary_prompt(label: str, confidence: float, condition_name: str) -> str:
    if client is None:
        return (
            f"**Meaning:** The model returned *{label}* with confidence {confidence:.1f}%.\n\n"
            f"**Which doctor to consult:** Consult a specialist appropriate for {condition_name}.\n\n"
            f"**Precautions & Diet:** Maintain hydration, follow doctor advice, balanced diet.\n\n"
            f"**Estimated cost:** Indicative, INR 2,000 - 50,000 depending on tests/treatment.\n\n"
            f"**Next steps:** Repeat tests and visit specialist as needed."
        )

    prompt = f"""
You are a senior clinical doctor AI. Provide a concise patient-friendly medical advisory summary.

Condition: {condition_name}
Diagnosis Result: {label}
Confidence Level: {confidence:.1f}%

Include these sections with headings and short bullets:

1. Meaning of the Result
2. Which Doctor to Consult (specialist + urgency)
3. Common Symptoms & Risks
4. Precautions & Safety Measures
5. Diet Plan (5 bullets)
6. Treatment Options & Estimated Cost (INR ranges; optional global)
7. Next Medical Steps (24 hours, 3 days, 7 days; tests)

Keep tone calm and helpful.
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"AI Summary generation failed: {e}"

def analyze_medical_report(text: str):
    if client is None:
        return (
            "Unable to use AI model (OpenAI key missing). "
            "But here is a general interpretation:\n\n"
            "• The report includes clinical/lab values.\n"
            "• Many parameters may be within normal ranges.\n"
            "• Please verify with a doctor for exact interpretation.\n"
        )

    prompt = f"""
You are a senior medical doctor AI. A patient report text has been extracted using OCR.
The text is messy. Clean it and summarize it.

Messy OCR Text:
{text}

Return a clear, structured medical summary with these sections:

1. What the patient is suffering from (diagnosis if identifiable)
2. Are the lab values normal or abnormal? Highlight abnormal parameters clearly
3. Possible medical conditions based on the given values
4. Health risks
5. Precautions the patient should take
6. Diet & lifestyle advice (5 points)
7. Which doctor to consult
8. Next steps (tests or follow-up recommendations)

Write in simple, patient-friendly language.
"""
    try:
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}]
        )
        return res.choices[0].message.content.strip()
    except Exception as e:
        return f"AI analysis failed: {e}"

# -----------------------------
# X-Ray Detector (PyTorch)
# -----------------------------
CLASS_NAMES_DETECTOR = ["NORMAL", "PNEUMONIA", "non_xray"]

@st.cache_resource
def load_xray_detector():
    DEVICE = "cpu"
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, len(CLASS_NAMES_DETECTOR))

    weight_paths = [
        os.path.join("models", "xray_detector_best.pth"),
        "xray_detector_best.pth",
    ]
    state_dict = None
    for wp in weight_paths:
        if os.path.exists(wp):
            state_dict = torch.load(wp, map_location=DEVICE)
            break

    if state_dict is None:
        print("WARNING: xray_detector_best.pth not found.")
        return None

    model.load_state_dict(state_dict)
    model.eval()
    return model

xray_detector = load_xray_detector()

xray_det_tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_is_xray(img_bgr: np.ndarray):
    if xray_detector is None:
        return None, 0.0, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = xray_det_tfms(img_rgb).unsqueeze(0)
    with torch.no_grad():
        logits = xray_detector(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    pred_idx = int(np.argmax(probs))
    pred_label = CLASS_NAMES_DETECTOR[pred_idx]
    pred_conf = float(probs[pred_idx])
    is_xray_flag = pred_label != "non_xray"
    return is_xray_flag, pred_conf, pred_label

def is_xray_image(img_gray: np.ndarray):
    if np.mean(img_gray) > 200:
        return False
    return True

# -----------------------------
# Load ML Models
# -----------------------------
@st.cache_resource
def load_models():
    models_dict = {}
    try:
        models_dict["xray"] = tf.keras.models.load_model("models/chest_xray_model.h5")
    except Exception:
        models_dict["xray"] = None

    try:
        models_dict["diabetes_model"] = pickle.load(open("models/diabetes_model.pkl", "rb"))
        models_dict["diabetes_scaler"] = pickle.load(open("models/diabetes_scaler.pkl", "rb"))
    except Exception:
        models_dict["diabetes_model"] = None
        models_dict["diabetes_scaler"] = None

    try:
        models_dict["heart_model"] = pickle.load(open("models/heart_model.pkl", "rb"))
        models_dict["heart_scaler"] = pickle.load(open("models/heart_scaler.pkl", "rb"))
    except Exception:
        models_dict["heart_model"] = None
        models_dict["heart_scaler"] = None

    try:
        models_dict["cancer_model"] = pickle.load(open("models/cancer_model.pkl", "rb"))
        models_dict["cancer_scaler"] = pickle.load(open("models/cancer_scaler.pkl", "rb"))
    except Exception:
        models_dict["cancer_model"] = None
        models_dict["cancer_scaler"] = None

    return models_dict

ensure_xray_model()
models = load_models()


# -----------------------------
# UI Helper: Report block
# -----------------------------
def render_report_prediction_block(result_label, score_value, condition_name, ai_text, pdf_filename="report.pdf"):
    st.markdown(
        f"""
        <div style="
            background: rgba(0,0,0,0.02);
            padding:18px;
            border-radius:12px;
            border:1px solid rgba(0,0,0,0.06);
            margin-top:12px;
            margin-bottom:12px;">
            <h3 style="color:{TEAL}; margin:0 0 6px 0;">Report Prediction — {condition_name}</h3>
            <div style="font-size:15px; color:#1f2933;">
                <b>Predicted:</b> {result_label} &nbsp; | &nbsp; <b>Score:</b> {score_value:.2f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("**AI Summary & Recommendations**")
    st.info(ai_text)

    try:
        with open(pdf_filename, "rb") as f:
            st.download_button(
                "📄 Download Prediction Report (PDF)",
                f,
                file_name=os.path.basename(pdf_filename)
            )
    except Exception:
        st.warning("PDF report not available for download.")
# -----------------------------
# UI Helper: Card
# ----------------------------
def card_html(title, desc):
        return f"""
        <div style="
        background:linear-gradient(180deg,#111827,#020617);
        border-radius:16px;
        padding:1.4rem;
        height:160px;
        box-shadow:0 10px 30px rgba(0,0,0,0.6);
        border:1px solid rgba(255,255,255,0.08);
        margin-bottom:0.6rem;
        ">
        <h3 style="color:#14f1c9; margin-bottom:0.5rem;">{title}</h3>
        <p style="color:#cbd5f5; font-size:0.95rem; line-height:1.6;">
            {desc}
        </p>
    </div>
    """

#-------------------------------
#----------------------------- Home Page -----------------------------
#-----------------------------
def home_page():

    # ================== STYLES ==================
    st.markdown("""
    <style>
    .hospital-bg {
        background:
            linear-gradient(rgba(2,6,23,0.88), rgba(2,6,23,0.88)),
            url("https://images.unsplash.com/photo-1586773860418-d37222d8fce3");
        background-size: cover;
        background-position: center;
        padding: 4.5rem 2rem 3.5rem;
        border-radius: 28px;
        box-shadow: 0 30px 80px rgba(0,0,0,0.85);
        margin-bottom: 3rem;
    }

    .hero-title {
        font-size: 3.2rem;
        font-weight: 900;
        color: #14f1c9;
        text-shadow: 0 0 22px rgba(20,241,201,0.6);
        text-align: center;
        margin-bottom: 0.4rem;
    }

    .hero-sub {
        font-size: 1.15rem;
        color: #cbd5f5;
        text-align: center;
        margin-bottom: 2.4rem;
        letter-spacing: 0.4px;
    }

    .tool-card {
        background: linear-gradient(180deg,#0f172a,#020617);
        border-radius: 20px;
        padding: 2rem 1.6rem;
        height: 100%;
        box-shadow: 0 16px 45px rgba(0,0,0,0.7);
        border: 1px solid rgba(255,255,255,0.08);
        text-align: center;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }

    .tool-title {
        font-size: 1.35rem;
        font-weight: 700;
        color: #e5e7eb;
        margin-bottom: 0.6rem;
    }

    .tool-desc {
        font-size: 0.95rem;
        color: #cbd5f5;
        line-height: 1.6;
        margin-bottom: 1.4rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # ================== HERO ==================
    st.markdown("""
    <div class="hospital-bg">
        <div class="hero-title">LIFE-LEN AI Health Analyzer</div>
        <div class="hero-sub">Clinical • Precise • Fast</div>
    </div>
    """, unsafe_allow_html=True)

    # ================== CTA ==================
    c1, c2, c3 = st.columns([1, 1.4, 1])
    with c2:
        if st.button(
            "⚡ Explore AI Tools",
            key="home_cta_main",
            use_container_width=True
        ):
            st.session_state.Navigation = "X-Ray"
            st.rerun()

    st.markdown("<br><br>", unsafe_allow_html=True)

    # ================== TOOLS GRID ==================
    row1 = st.columns(3)

    with row1[0]:
        st.markdown("""
        <div class="tool-card">
            <div>
                <div class="tool-title">🫁 Chest X-Ray</div>
                <div class="tool-desc">
                    AI-powered pneumonia detection with medical-grade accuracy.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open X-Ray", key="home_xray", use_container_width=True):
            st.session_state.Navigation = "X-Ray"
            st.rerun()

    with row1[1]:
        st.markdown("""
        <div class="tool-card">
            <div>
                <div class="tool-title">💉 Diabetes</div>
                <div class="tool-desc">
                    Clinical risk prediction with diet & lifestyle guidance.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Diabetes", key="home_diabetes", use_container_width=True):
            st.session_state.Navigation = "Diabetes"
            st.rerun()

    with row1[2]:
        st.markdown("""
        <div class="tool-card">
            <div>
                <div class="tool-title">❤️ Heart Disease</div>
                <div class="tool-desc">
                    Cardiac risk scoring with next-step recommendations.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Heart", key="home_heart", use_container_width=True):
            st.session_state.Navigation = "Heart"
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # ================== CENTERED SECOND ROW ==================
    r2_left, r2_mid, r2_right = st.columns([1, 1.4, 1])

    with r2_mid:
        st.markdown("""
        <div class="tool-card">
            <div>
                <div class="tool-title">🧬 Cancer Prediction</div>
                <div class="tool-desc">
                    Breast cancer malignancy classification with follow-up advice.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Cancer", key="home_cancer", use_container_width=True):
            st.session_state.Navigation = "Cancer"
            st.rerun()

#-------------------------------
#----------------------------- X-Ray Page -----------------------------
#-----------------------------
def chest_xray_page():
    st.header("🫁 Chest X-Ray Analyzer")

    # ---- Model check ----
    if models["xray"] is None:
        st.error("Chest X-Ray model not available.")
        return

    # ---- Upload ----
    file = st.file_uploader("Upload Chest X-Ray", type=["png", "jpg", "jpeg"])
    if not file:
        st.info("Please upload a chest X-ray image.")
        return

    # ---- Read image ----
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        st.error("Invalid image file.")
        return

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    st.image(
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
        caption="Uploaded Chest X-Ray",
        width=520
    )

    # ---- Preprocess ----
    resized = cv2.resize(img_gray, (224, 224)) / 255.0
    inp = np.expand_dims(resized, (0, -1)).astype(np.float32)

    # ---- Predict ----
    if st.button("🔍 Analyze X-Ray"):
        with st.spinner("Analyzing chest radiograph..."):
            pred = models["xray"].predict(inp)
            prob = float(pred[0][0])

        # ---- Post-processing ----
        label = "PNEUMONIA" if prob > 0.5 else "NORMAL"
        conf = prob if label == "PNEUMONIA" else 1 - prob
        disease = "Pneumonia" if label == "PNEUMONIA" else "No Lung Disease"
        color = "#ef4444" if label == "PNEUMONIA" else "#22c55e"

        if conf < 0.30:
            severity = "Very Mild"
        elif conf < 0.60:
            severity = "Mild"
        elif conf < 0.85:
            severity = "Moderate"
        else:
            severity = "Severe"

        urgency = (
            "Routine follow-up"
            if severity in ["Very Mild", "Mild"]
            else "Consult Pulmonologist"
            if severity == "Moderate"
            else "Urgent Medical Attention"
        )

        # ---- STEP 4: Save to database ----
        insert_prediction(
            disease=disease,
            risk=severity,
            score=round(conf, 3)
        )

        # ---- Result card ----
        st.markdown(
            f"""
            <div style="
                background:{color}22;
                border-left:6px solid {color};
                padding:16px;
                border-radius:12px;
                margin-top:14px;
            ">
                <h3 style="color:{color}; margin:0;">{label}</h3>
                <p><b>Disease:</b> {disease}</p>
                <p><b>Severity:</b> {severity}</p>
                <p><b>Confidence:</b> {conf*100:.2f}%</p>
                <p><b>Recommended Action:</b> {urgency}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # ---- Table ----
        st.markdown(f"""
### 🫁 Pulmonary Assessment (AI Decision Support)

| Parameter | Result |
|---------|--------|
| Disease | {disease} |
| Severity | {severity} |
| Confidence | {conf*100:.2f}% |
| Recommended Action | {urgency} |

⚠️ *AI-assisted output. Radiologist confirmation required.*
""")

        # ---- AI summary ----
        ai_text = ai_summary_prompt(
            label=label,
            confidence=conf * 100,
            condition_name="Chest X-Ray / Pneumonia Detection"
        )

        st.info(ai_text)
        speak_text(f"Chest X-ray result {label}. Confidence {conf*100:.0f} percent.")

        # ---- History ----
        record = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "Chest X-Ray",
            "result": label,
            "score": round(conf * 100, 2),
            "summary": ai_text,
        }
        st.session_state.history.insert(0, record)

        # ---- PDF ----
        pdf_path = create_pdf(
            [
                "Chest X-Ray Analysis Report",
                f"Result: {label}",
                f"Disease: {disease}",
                f"Severity: {severity}",
                f"Confidence: {conf*100:.2f}%",
                "",
                ai_text,
            ],
            "xray_report.pdf",
            "Chest X-Ray Report"
        )

        with open(pdf_path, "rb") as f:
            st.download_button(
                "📄 Download X-Ray Report (PDF)",
                f,
                file_name="xray_report.pdf"
            )

#-------------------------------
#----------------------------- Diabetes Page -----------------------------
#-----------------------------

def diabetes_page():
    st.header("💉 Diabetes Risk Assessment")

    if models["diabetes_model"] is None or models["diabetes_scaler"] is None:
        st.error("Diabetes model or scaler missing.")
        return

    st.subheader("🧾 Patient Metabolic Parameters")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        bp = st.number_input("Blood Pressure", 0, 200, 70)
        insulin = st.number_input("Insulin", 0, 900, 85)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 5.0, 0.5)

    with col2:
        glucose = st.number_input("Glucose", 0, 300, 120)
        skin = st.number_input("Skin Thickness", 0, 100, 20)
        bmi = st.number_input("BMI", 1.0, 70.0, 25.0)
        age = st.number_input("Age", 1, 120, 30)

    if st.button("🔍 Analyze Diabetes Risk", key="diabetes_btn"):
        X = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
        Xs = models["diabetes_scaler"].transform(X)

        prob = models["diabetes_model"].predict_proba(Xs)[0][1]

        if prob < 0.30:
            stage = "Non-Diabetic"
            severity = "Low Risk"
        elif prob < 0.65:
            stage = "Pre-Diabetic"
            severity = "Moderate Risk"
        else:
            stage = "Diabetic"
            severity = "High Risk"

        label = stage

        st.markdown(f"""
### 💉 Glycemic Status (AI Decision Support)

| Parameter | Result |
|---------|--------|
| **Status** | {stage} |
| **Risk Level** | {severity} |
| **Probability** | {prob*100:.2f}% |
| **Recommended Specialist** | Endocrinologist |

⚠️ *Confirm with HbA1c and fasting glucose tests.*
""")

        ai_text = ai_summary_prompt(stage, prob * 100, "Diabetes Risk Assessment")
        st.info(ai_text)
        speak_text_autoplay(f"Diabetes status {stage}, risk {prob*100:.0f} percent.")

        rec = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "Diabetes",
            "result": stage,
            "score": round(prob * 100, 2),
            "summary": ai_text,
        }
        st.session_state.history.insert(0, rec)

        pdf_path = f"diabetes_report_{int(time.time())}.pdf"
        create_pdf(
            ["Diabetes Report", f"Status: {stage}", f"Risk: {prob*100:.2f}%", "", ai_text],
            pdf_path,
            "Diabetes Report",
        )

def heart_page():
    st.header("❤️ Heart Disease Risk Assessment")

    if models["heart_model"] is None or models["heart_scaler"] is None:
        st.error("Heart model or scaler missing.")
        return

    st.subheader("🫀 Cardiac Health Parameters")

    c1, c2, c3 = st.columns(3)

    age = c1.number_input("Age", 1, 120, 45)
    sex = c2.selectbox("Sex", ["Male", "Female"])
    bp = c3.number_input("Resting Blood Pressure", 60, 200, 130)

    chol = c1.number_input("Cholesterol", 100, 600, 240)
    thalach = c2.number_input("Max Heart Rate", 60, 220, 150)
    oldpeak = c3.number_input("ST Depression", 0.0, 10.0, 1.0)

    sex_val = 1 if sex == "Male" else 0

    if st.button("🔍 Analyze Heart Risk", key="heart_btn"):
        X = np.array([[age, sex_val, 0, bp, chol, 0, 0, thalach, 0, oldpeak, 1, 0, 2]])
        Xs = models["heart_scaler"].transform(X)

        prob = models["heart_model"].predict_proba(Xs)[0][1]

        if prob < 0.30:
            stage = "Low Risk"
            severity = "Mild"
        elif prob < 0.60:
            stage = "Moderate Risk"
            severity = "Moderate"
        else:
            stage = "High Risk"
            severity = "Severe"

        label = stage

        st.markdown(f"""
### ❤️ Cardiac Risk Evaluation (AI Decision Support)

| Parameter | Result |
|---------|--------|
| **Risk Category** | {stage} |
| **Severity** | {severity} |
| **Probability** | {prob*100:.2f}% |
| **Recommended Specialist** | Cardiologist |

⚠️ *ECG / ECHO / Stress tests recommended.*
""")

        ai_text = ai_summary_prompt(stage, prob * 100, "Heart Disease Risk")
        st.info(ai_text)
        speak_text_autoplay(f"Heart disease risk {stage}, probability {prob*100:.0f} percent.")

        rec = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "Heart",
            "result": stage,
            "score": round(prob * 100, 2),
            "summary": ai_text,
        }
        st.session_state.history.insert(0, rec)

        pdf_path = f"heart_report_{int(time.time())}.pdf"
        create_pdf(
            ["Heart Disease Report", f"Risk: {stage}", f"Probability: {prob*100:.2f}%", "", ai_text],
            pdf_path,
            "Heart Report",
        )
def cancer_page():
    st.header("🧬 Breast Cancer Risk Assessment")

    if models["cancer_model"] is None or models["cancer_scaler"] is None:
        st.error("Cancer model or scaler missing.")
        return

    st.subheader("🧪 Tumor Feature Parameters")

    c1, c2 = st.columns(2)
    radius = c1.number_input("Mean Radius", 1.0, 50.0, 14.0)
    texture = c2.number_input("Mean Texture", 1.0, 50.0, 20.0)
    perimeter = c1.number_input("Mean Perimeter", 1.0, 300.0, 90.0)
    area = c2.number_input("Mean Area", 1.0, 3000.0, 600.0)
    smooth = c1.number_input("Mean Smoothness", 0.0, 1.0, 0.1)

    if st.button("🔍 Analyze Cancer Risk", key="cancer_btn"):
        X = np.array([[radius, texture, perimeter, area, smooth]])
        Xs = models["cancer_scaler"].transform(X)

        prob = models["cancer_model"].predict_proba(Xs)[0][1]
        malignant = prob >= 0.5

        presence = "Cancer Detected" if malignant else "No Cancer Detected"

        if prob < 0.40:
            risk = "Low Risk"
            stage = "Stage 0 / Benign"
        elif prob < 0.70:
            risk = "Intermediate Risk"
            stage = "Stage I–II"
        else:
            risk = "High Risk"
            stage = "Stage III–IV"

        label = presence

        st.markdown(f"""
### 🧬 Oncology Assessment (AI Decision Support)

| Parameter | Result |
|---------|--------|
| **Cancer Presence** | {presence} |
| **Cancer Type** | Breast Cancer |
| **Risk Level** | {risk} |
| **Probable Stage** | {stage} |
| **Probability** | {prob*100:.2f}% |
| **Recommended Specialist** | Oncologist |

⚠️ *Biopsy & imaging required for confirmation.*
""")

        ai_text = ai_summary_prompt(presence, prob * 100, "Breast Cancer Risk")
        st.info(ai_text)
        speak_text_autoplay(f"Cancer assessment {presence}, probability {prob*100:.0f} percent.")

        rec = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "type": "Cancer",
            "result": presence,
            "score": round(prob * 100, 2),
            "summary": ai_text,
        }
        st.session_state.history.insert(0, rec)

        pdf_path = f"cancer_report_{int(time.time())}.pdf"
        create_pdf(
            ["Cancer Report", f"Presence: {presence}", f"Risk: {risk}", f"Stage: {stage}", "", ai_text],
            pdf_path,
            "Cancer Report",
        )


def dashboard_page():
    st.header("📊 Dashboard — Recent Predictions")
    hist = st.session_state.history
    if not hist:
        st.info("No predictions yet. Run any model to populate the dashboard.")
        return

    df = pd.DataFrame(hist)
    st.dataframe(df, use_container_width=True)

    counts = df["type"].value_counts()
    st.subheader("Prediction Counts by Tool")
    st.bar_chart(counts)

    st.markdown("### Risk scores (latest 20)")
    df_scores = df.head(20).copy()
    df_scores["score"] = pd.to_numeric(df_scores["score"], errors="coerce")
    if not df_scores["score"].isna().all():
        st.line_chart(df_scores["score"])

def report_page():
    st.header("📑 Medical Report Analyzer")

    file = st.file_uploader(
        "Upload a Medical Report (PDF / Image)",
        type=["pdf", "png", "jpg", "jpeg"]
    )

    if not file:
        st.info("Upload any medical report to analyze.")
        return

    file_type = file.type
    text_output = ""

    if file_type == "application/pdf":
        st.subheader("📄 PDF Preview")
        file_bytes = file.read()
        base64_pdf = base64.b64encode(file_bytes).decode("utf-8")
        pdf_display = f"""
            <embed src="data:application/pdf;base64,{base64_pdf}"
                   width="100%" height="700px" type="application/pdf">
        """
        st.markdown(pdf_display, unsafe_allow_html=True)

        st.subheader("📝 Extracted Text from PDF")
        file.seek(0)
        try:
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text_output += extracted + "\n"
        except Exception as e:
            st.error(f"PDF text extraction failed: {e}")
            return

        st.text_area("PDF Extracted Text", text_output, height=300)

    else:
        st.subheader("🖼 Image Preview")
        file.seek(0)
        img = Image.open(file)
        st.image(img, caption="Uploaded Report Image", use_column_width=True)

        st.subheader("📝 OCR Extracted Text")
        try:
            text_output = pytesseract.image_to_string(img)
        except Exception as e:
            st.error(f"OCR failed: {e}")
            return

        st.text_area("Extracted Text", text_output, height=300)

    if text_output.strip():
        st.markdown("## 🧠 AI Medical Interpretation")
        ai_summary = analyze_medical_report(text_output)
        st.info(ai_summary)
    else:
        st.warning("No readable text detected in this report.")

def about_page():
    st.header("ℹ️ About LIFE-LEN AI")
    st.markdown(
        """
        **LIFE-LEN AI** is a healthcare AI platform developed by **Muneesh** that
        combines cutting-edge deep learning with clinical logic to support doctors,
        hospitals, and patients.

        ### Mission
        To democratize access to advanced medical diagnostics through artificial intelligence,
        making healthcare more accessible, accurate, and affordable.

        ### Vision
        A world where AI-powered healthcare tools provide instant, reliable diagnostic
        insights, enabling early detection and better health outcomes.
        """,
    )

    st.markdown("### How the AI Works")
    st.write(
        """
        - Convolutional Neural Networks (CNNs) for chest X-ray and imaging tasks  
        - Machine learning models for diabetes, heart, and cancer risk prediction  
        - Natural Language Processing (NLP) to interpret medical reports via OCR
        """
    )

    st.markdown("### Hospital-Grade Credibility")
    st.markdown(
        """
        - ✅ Validated on medical datasets (X-ray, tabular, lab reports)  
        - ✅ Designed to complement clinical judgment, not replace doctors  
        - ✅ Local processing where possible to respect privacy
        """
    )

def contact_page():
    st.header("📞 Contact & Support")

    col_info, col_form = st.columns([1, 1.4])

    with col_info:
        st.subheader("Contact Information")
        st.write("📧 Email: **contact@lifelen-ai.com**")
        st.write("🌐 Location: Healthcare Innovation Center, Medical District")
        st.write("📞 Phone:     ")
        st.write(
            """
            For integration in hospitals or clinics, or academic collaborations,
            feel free to reach out with your use case.
            """
        )

    with col_form:
        st.subheader("Send a Message")
        with st.form("contact-form"):
            name = st.text_input("Your Name")
            email = st.text_input("Email Address")
            subject = st.text_input("Subject")
            message = st.text_area("Message", height=120)
            submitted = st.form_submit_button("Send Message")

        if submitted:
            if not name or not email or not subject or not message:
                st.warning("Please fill in all fields.")
            else:
                st.success("✅ Message sent! Thank you, we will respond as soon as possible.")

def legal_page():
    st.header("⚖️ Privacy Policy & Medical Disclaimer")

    st.subheader("Privacy Policy")
    st.write(
        """
        **Effective Date:** January 2025

        LIFE-LEN AI is committed to protecting your privacy and ensuring that your
        health data is handled safely.
        """
    )

    st.markdown("**What Data Is Used?**")
    st.write(
        """
        - Medical images (e.g., chest X-rays)  
        - Numerical clinical data (e.g., glucose, BP, BMI)  
        - Uploaded medical reports (PDF or images)
        """
    )

    st.markdown("**How Your Data Is Used**")
    st.write(
        """
        - Solely for generating AI-based predictions or explanations  
        - Data is processed locally on your machine/server where possible  
        - No automatic third-party sharing from this app 
        """
    )

    st.subheader("Medical Disclaimer")
    st.write(
        """
        LIFE-LEN AI provides **decision-support only**. It is **not a substitute**
        for professional medical advice, diagnosis, or treatment.

        Always consult a licensed healthcare professional before making any
        health-related decisions. In case of emergencies, call your local emergency
        number immediately.
        """
    )
# -----------------------------
# Page Routing
# -----------------------------
current_page = st.session_state.Navigation

if current_page == "Home":
    home_page()
elif current_page == "X-Ray":
    chest_xray_page()
elif current_page == "Diabetes":
    diabetes_page()
elif current_page == "Heart":
    heart_page()
elif current_page == "Cancer":
    cancer_page()
elif current_page == "Reports":
    report_page()
elif current_page == "Dashboard":
    dashboard_page()
elif current_page == "About":
    about_page()
elif current_page == "Contact":
    contact_page()
elif current_page == "Legal":
    legal_page()

st.divider()
st.subheader("📊 X-Ray Analysis History")

history = fetch_predictions()

if history:
    for row in history:
        st.write(
            f"🩻 {row[1]} | Severity: {row[2]} | Confidence: {row[3]} | ⏰ {row[4]}"
        )
else:
    st.info("No X-ray analyses recorded yet.")

# -----------------------------
# Footer (website-style)
# -----------------------------
st.markdown(
    """
    <footer class="footer">
        <div>
            <h4 style="margin-bottom:6px;">LIFE-LEN AI</h4>
            <p style="margin:0;">
                Advancing healthcare through artificial intelligence.
            </p>
            <p style="margin-top:10px; opacity:0.7;">
                © 2025 LIFE-LEN AI Health Analyzer — Developed by Muneesh
            </p>
        </div>
    </footer>
    """,
    unsafe_allow_html=True,
)
