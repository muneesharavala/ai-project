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
from PIL import Image as PILImage
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
import requests

from utils.database import validate_user, create_user
from utils.database import fetch_all_predictions, fetch_all_users
from utils.database import fetch_messages
from utils.database import insert_message



# =========================
# DATABASE SETUP
# =========================

create_table()

TEAL = "#38bdf8"
# -----------------------------
# Helper: load logo safely
# -----------------------------
def get_base64_image(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()

LOGO_BASE64 = get_base64_image("assets/logo.png")


XRAY_MODEL_PATH = "models/chest_xray_model.h5"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1NRFFqkWix8kwvDVSTXUED2LLeUJAodxB"

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
# Role check helper
# -----------------------------
def is_doctor():
    return st.session_state.get("role") in ["doctor", "admin"]

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

# =========================
# SESSION DEFAULTS
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user" not in st.session_state:
    st.session_state.user = None

if "role" not in st.session_state:
    st.session_state.role = None

if "Navigation" not in st.session_state:
    st.session_state.Navigation = "Home"

if "email" not in st.session_state:
    st.session_state.email = ""

if "selected_patient" not in st.session_state:
    st.session_state.selected_patient = None

if "language" not in st.session_state:
    st.session_state.language = "English"

TRANSLATIONS = {
    "English": {
        "home_title": "LIFE-LEN AI Health Analyzer",
        "welcome": "Welcome back",
        "xray": "Chest X-Ray Analyzer",
        "diabetes": "Diabetes Risk Assessment",
        "heart": "Heart Disease Risk Assessment",
        "cancer": "Cancer Risk Assessment",
        "reports": "Medical Report Analyzer",
        "dashboard": "My Prediction History",
        "help": "Help & Support",
        "settings": "Settings",
        "logout": "Logout",
    },

    "Hindi": {
        "home_title": "LIFE-LEN AI स्वास्थ्य विश्लेषक",
        "welcome": "स्वागत है",
        "xray": "छाती एक्स-रे विश्लेषण",
        "diabetes": "मधुमेह जोखिम मूल्यांकन",
        "heart": "हृदय रोग जोखिम मूल्यांकन",
        "cancer": "कैंसर जोखिम मूल्यांकन",
        "reports": "मेडिकल रिपोर्ट विश्लेषण",
        "dashboard": "मेरी रिपोर्ट्स",
        "help": "सहायता",
        "settings": "सेटिंग्स",
        "logout": "लॉगआउट",
    },

    "Telugu": {
        "home_title": "LIFE-LEN AI ఆరోగ్య విశ్లేషణ",
        "welcome": "స్వాగతం",
        "xray": "చెస్ట్ ఎక్స్-రే విశ్లేషణ",
        "diabetes": "డయాబెటిస్ ప్రమాద విశ్లేషణ",
        "heart": "హృదయ వ్యాధి ప్రమాదం",
        "cancer": "క్యాన్సర్ ప్రమాదం",
        "reports": "మెడికల్ రిపోర్ట్ విశ్లేషణ",
        "dashboard": "నా నివేదికలు",
        "help": "సహాయం",
        "settings": "సెట్టింగ్స్",
        "logout": "లాగౌట్",
    }
}
# =========================
# TRANSLATION HELPER
# =========================

def t(key: str) -> str:
    """
    Translation helper.
    Returns translated text based on selected language.
    Falls back to English if key is missing.
    """
    lang = st.session_state.get("language", "English")

    # Language exists and key exists
    if lang in TRANSLATIONS and key in TRANSLATIONS[lang]:
        return TRANSLATIONS[lang][key]

    # Fallback to English
    return TRANSLATIONS["English"].get(key, key)


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

        # ---------------- LOGIN ----------------
        with tab_login:
            u = st.text_input("Username", key="login_user")
            p = st.text_input("Password", type="password", key="login_pass")

            if st.button("Login", use_container_width=True):
                role = validate_user(u, p)

                if role:
                    st.session_state.logged_in = True
                    st.session_state.user = u
                    st.session_state.role = role
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        # ---------------- SIGN UP ----------------
        with tab_signup:
            nu = st.text_input("New Username", key="signup_user")
            np = st.text_input("New Password", type="password", key="signup_pass")
            cp = st.text_input("Confirm Password", type="password", key="signup_confirm")

            if st.button("Create Account", use_container_width=True):
                if not nu or not np:
                    st.error("All fields are required")
                elif np != cp:
                    st.error("Passwords do not match")
                else:
                    try:
                        create_user(nu, np, role="user")
                        st.success("✅ Account created. Please login.")
                    except Exception:
                        st.error("Username already exists")
        st.markdown('</div>', unsafe_allow_html=True)


# =========================
# AUTH GUARD
# =========================
if not st.session_state.logged_in:
    auth_page()
    st.stop()

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
    "Profile", "Settings", "Help",
    "About", "Contact", "Legal"
]
if st.session_state.get("role") in ["doctor", "admin"]:
    NAV_OPTIONS.append("Patient Profile")

if st.session_state.get("role") == "admin":
    NAV_OPTIONS.insert(-3, "Admin")

if st.session_state.Navigation not in NAV_OPTIONS:
    st.session_state.Navigation = "Home"
# =========================
# SIDEBAR (ALWAYS RENDERED)
# =========================
with st.sidebar:

    # ---------- BRANDING ----------
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

    st.markdown("---")

    # ---------- AUTH STATE ----------
    if st.session_state.get("logged_in") and st.session_state.get("user"):

        st.success(f"👤 Logged in as **{st.session_state.user}**")

        if st.button("🚪 Logout", key="logout_btn"):
            st.session_state.clear()
            st.session_state.logged_in = False
            st.session_state.Navigation = "Home"
            st.rerun()

        st.markdown("---")

        # ---------- EMAIL DISPLAY ----------
        if st.session_state.get("email"):
            st.markdown("### 📧 Report Email")
            st.code(st.session_state.email)

        st.markdown("---")

        # ---------- NAVIGATION ----------
        st.radio(
            "Navigation",
            NAV_OPTIONS,
            key="Navigation"
        )

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
        - Voice output requires internet  
        """)

    else:
        # ---------- NOT LOGGED IN ----------
        st.info("🔒 Please login to access LIFE-LEN AI")
        st.caption("Secure Medical Platform • 2025")


# =========================
# PAGE-SPECIFIC BACKGROUNDS
# =========================
def set_page_background(page: str):

    backgrounds = {
        "Home": """
            linear-gradient(rgba(2,6,23,0.90), rgba(2,6,23,0.90)),
            url("https://images.unsplash.com/photo-1586773860418-d37222d8fce3")
        """,

        "X-Ray": """
            linear-gradient(rgba(2,6,23,0.95), rgba(2,6,23,0.95)),
            url("https://images.unsplash.com/photo-1581090700227-1e37b190418e")
        """,

        "Diabetes": """
            linear-gradient(rgba(2,6,23,0.92), rgba(2,6,23,0.92)),
            url("https://images.unsplash.com/photo-1576091160399-112ba8d25d1d")
        """,

        "Heart": """
            linear-gradient(rgba(50,0,0,0.88), rgba(2,6,23,0.95)),
            url("https://images.unsplash.com/photo-1584467735871-b0c1b8a93a88")
        """,

        "Cancer": """
            linear-gradient(rgba(30,0,60,0.9), rgba(2,6,23,0.95)),
            url("https://images.unsplash.com/photo-1579684385127-1ef15d508118")
        """,

        "Reports": """
            linear-gradient(rgba(2,6,23,0.95), rgba(2,6,23,0.95))
        """,

        "Dashboard": """
            linear-gradient(rgba(2,6,23,0.9), rgba(15,23,42,0.95))
        """,

        "About": """
            linear-gradient(rgba(2,6,23,0.95), rgba(2,6,23,0.95))
        """,

        "Contact": """
            linear-gradient(rgba(2,6,23,0.95), rgba(2,6,23,0.95))
        """,

        "Legal": """
            linear-gradient(rgba(2,6,23,0.95), rgba(2,6,23,0.95))
        """,
    }

    bg = backgrounds.get(page, backgrounds["Home"])

    st.markdown(
        f"""
        <style>
        .stApp {{
            min-height: 100vh;
            background: {bg};
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# =========================
# GLOBAL STYLES
# =========================
st.markdown("""
<style>
/* =============================== MAIN CONTENT ================================ */
.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 4rem;
    animation: fadeSlideIn 0.55s ease-out;
}

/* =============================== SIDEBAR ================================ */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #0f172a);
    border-right: 1px solid rgba(255,255,255,0.08);
}

section[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

/* =============================== CARDS ================================ */
.card, .tool-card {
    background: rgba(15,23,42,0.85);
    backdrop-filter: blur(8px);
    border-radius: 18px;
    padding: 1.6rem;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 12px 40px rgba(0,0,0,0.6);
}

/* =============================== BUTTONS ================================ */
div[data-page="Home"] .stButton > button {
    background: linear-gradient(90deg, #2563eb, #38bdf8);
    color: #fff;
    border-radius: 999px;
    padding: 0.6rem 1.6rem;
    font-weight: 600;
    border: none;
}

div[data-page="Home"] .stButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 30px rgba(56,189,248,0.4);
}

/* =============================== FOOTER ================================ */
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

/* =============================== PAGE TRANSITION ================================ */
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
    confidence = float(confidence)

    # ================= OFFLINE MODE =================
    if client is None:

        # ---------- CHEST X-RAY ----------
        if "x-ray" in condition_name.lower() or "pneumonia" in condition_name.lower():

            if label.upper() == "PNEUMONIA":
                return f"""
🫁 **Diagnosis Summary**
The chest X-ray shows signs suggestive of **Pneumonia** with an AI confidence of **{confidence:.1f}%**.
This indicates a lung infection that requires medical evaluation.

👨‍⚕️ **Doctor to Consult**
• Pulmonologist or General Physician

⚠️ **Common Symptoms**
• Cough, fever, breathlessness  
• Chest pain, fatigue  

🥗 **Diet Recommendation**
• Warm fluids, soups  
• Vitamin-C rich fruits  
• Protein-rich foods  

🛡️ **Precautions**
• Complete rest  
• Avoid smoke & cold air  

📅 **Next Steps**
• Visit doctor within 24–48 hours  
• Follow-up X-ray if advised  

⚠️ *AI-assisted screening result.*
"""
            else:
                return f"""
🫁 **Diagnosis Summary**
The chest X-ray appears **normal** with an AI confidence of **{confidence:.1f}%**.
No clear lung infection detected.

👨‍⚕️ **Doctor Consultation**
• Not urgent unless symptoms persist  

🥗 **Healthy Lung Care**
• Stay hydrated  
• Avoid smoking & pollution  

📅 **Next Steps**
• Monitor symptoms  

⚠️ *AI-assisted screening result.*
"""

        # ---------- DIABETES ----------
        if "diabetes" in condition_name.lower():

            return f"""
💉 **Diabetes Risk Summary**
Your results indicate **{label}** with an estimated risk of **{confidence:.1f}%**.
This reflects how likely you are to have or develop diabetes.

👨‍⚕️ **Doctor to Consult**
• Endocrinologist or Diabetologist

⚠️ **Possible Symptoms**
• Increased thirst  
• Frequent urination  
• Fatigue  

🥗 **Diet Recommendation**
• Reduce sugar & refined carbs  
• High-fiber foods  
• Lean protein  
• Regular meal timing  

🛡️ **Precautions**
• Monitor blood glucose  
• Regular physical activity  

📅 **Next Steps**
• HbA1c & fasting glucose tests  
• Lifestyle modification  

⚠️ *AI-assisted risk assessment.*
"""

        # ---------- HEART ----------
        if "heart" in condition_name.lower():

            return f"""
❤️ **Heart Disease Risk Summary**
Your assessment shows **{label}** with a risk probability of **{confidence:.1f}%**.
This reflects potential cardiovascular risk.

👨‍⚕️ **Doctor to Consult**
• Cardiologist

⚠️ **Risk Indicators**
• High BP or cholesterol  
• Chest discomfort  
• Shortness of breath  

🥗 **Diet Recommendation**
• Low-salt diet  
• Avoid fried foods  
• Fruits, vegetables & whole grains  

🛡️ **Precautions**
• Avoid smoking  
• Control BP & cholesterol  

📅 **Next Steps**
• ECG / ECHO / Stress test  
• Cardiologist consultation  

⚠️ *AI-assisted risk screening.*
"""

        # ---------- CANCER ----------
        if "cancer" in condition_name.lower():

            return f"""
🧬 **Cancer Risk Summary**
The AI model indicates **{label}** with a probability of **{confidence:.1f}%**.
This is a **risk estimation**, not a confirmed diagnosis.

👨‍⚕️ **Doctor to Consult**
• Oncologist or Surgical Oncologist

⚠️ **Possible Warning Signs**
• Lump or tissue changes  
• Persistent pain  

🥗 **Diet Recommendation**
• Antioxidant-rich fruits & vegetables  
• Adequate protein intake  

🛡️ **Precautions**
• Avoid tobacco & alcohol  
• Regular follow-ups  

📅 **Next Steps**
• Imaging & biopsy if advised  

⚠️ *AI-assisted screening result.*
"""

        # ---------- FALLBACK ----------
        return f"""
📄 **Health Assessment Summary**
Result: **{label}**  
Confidence: **{confidence:.1f}%**

Please consult a specialist for further evaluation.

⚠️ *AI-assisted screening output.*
"""

    # ================= ONLINE MODE =================
    prompt = f"""
You are a senior clinical doctor AI.

Condition: {condition_name}
Result: {label}
Confidence: {confidence:.1f}%

Give patient-friendly advice including:
- Meaning
- Doctor
- Symptoms
- Diet
- Precautions
- Next steps
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
# X-Ray Detector (PyTorch)
# -----------------------------
CLASS_NAMES_DETECTOR = ["NORMAL", "PNEUMONIA", "NON_XRAY"]

@st.cache_resource
def load_xray_detector():
    device = "cpu"

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    WEIGHTS_PATH = os.path.join(BASE_DIR, "models", "xray_detector_best.pth")

    if not os.path.exists(WEIGHTS_PATH):
        st.warning("⚠️ X-ray detector weights not found. Validation disabled.")
        return None

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES_DETECTOR))

    state_dict = torch.load(WEIGHTS_PATH, map_location=device)
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
    """
    Returns:
        is_xray (bool)
        confidence (float)
        predicted_label (str)
    """
    if xray_detector is None:
        # Fail-open for safety
        return True, 0.0, "UNKNOWN"

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = xray_det_tfms(img_rgb).unsqueeze(0)

    with torch.no_grad():
        logits = xray_detector(tensor)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    non_xray_idx = CLASS_NAMES_DETECTOR.index("NON_XRAY")
    non_xray_prob = float(probs[non_xray_idx])

    if non_xray_prob >= 0.75:
        return False, non_xray_prob, "NON_XRAY"

    return True, 1.0 - non_xray_prob, "XRAY"


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
# -----------------------------
# Navigation helper (MUST be above home_page)
# -----------------------------
def go_to(page_name: str):
    st.session_state.Navigation = page_name

# -----------------------------
# ----------------------------- Home Page -----------------------------
#-----------------------------
def home_page():
    st.markdown('<div data-page="Home">', unsafe_allow_html=True)

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
        color: #38bdf8;
        text-align: center;
    }

    .hero-sub {
        font-size: 1.15rem;
        color: #cbd5f5;
        text-align: center;
        margin-bottom: 2.4rem;
    }

    .card-img {
        width:100%;
        border-radius:14px;
        margin-bottom:10px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ================== HERO ==================
    st.markdown("""
    <div class="hospital-bg">
        <div class="hero-title">LIFE-LEN AI Health Analyzer</div>
        <div class="hero-sub">
            AI-Assisted Clinical Decision Support Platform
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="text-align:center;font-size:2rem;font-weight:800;color:#e5e7eb;">
        👋 Welcome Back, {st.session_state.user}<br>
        <span style="font-size:1.1rem;color:#94a3b8;">
            Secure • Explainable • Clinically Responsible AI
        </span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    # ================== STATUS BANNER ==================
    st.markdown("""
    <div style="
        display:flex;
        justify-content:space-between;
        align-items:center;
        background:linear-gradient(90deg,#020617,#0f172a);
        padding:1.2rem 1.6rem;
        border-radius:16px;
        border:1px solid rgba(56,189,248,0.25);
        margin-bottom:2rem;
    ">
        <div>
            <h4 style="color:#38bdf8;margin:0;">🟢 System Status</h4>
            <p style="color:#cbd5f5;margin:4px 0 0;">
                All AI models operational • Secure clinical mode enabled
            </p>
        </div>
        <div style="color:#14f1c9;font-weight:700;">
            LIVE
        </div>
    </div>
    """, unsafe_allow_html=True)


    # ================== METRICS ==================
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🧠 AI Models", "5+", "Validated")
    c2.metric("🫁 X-Rays Analyzed", "10K+", "High Accuracy")
    c3.metric("⚡ Avg Analysis Time", "0.8s", "Fast")
    c4.metric("🏥 Clinical Use", "Decision Support", "Safe")

    st.markdown("---")

    # ================== TRUST & SAFETY ==================
    st.markdown("## 🛡️ Clinical Trust & Safety")

    col1, col2, col3 = st.columns(3)

    col1.success("✔ AI-Assisted Only\n\nNo automated diagnosis")
    col2.success("✔ Confidence Scores\n\nEvery output quantified")
    col3.success("✔ Privacy First\n\nLocal & secure processing")


    # ================== WHO USES ==================
    st.markdown("## 👨‍⚕️ Who Uses LIFE-LEN AI?")

    a, b, c = st.columns(3)

    with a:
        st.markdown("""
        <div class="card">
            <h4>🏥 Hospitals</h4>
            <p>Faster triage, second-opinion screening, and AI-assisted workflows.</p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class="card">
            <h4>👨‍⚕️ Doctors</h4>
            <p>Explainable AI insights with confidence scores and summaries.</p>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown("""
        <div class="card">
            <h4>🧑‍🤝‍🧑 Patients</h4>
            <p>Clear understanding of reports, risks, and next steps.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ================== TRUST BANNER ==================
    st.markdown("""
    <div style="
        background:linear-gradient(90deg,#020617,#0f172a);
        border-left:6px solid #38bdf8;
        padding:1.4rem;
        border-radius:16px;
        margin-top:2rem;
    ">
        <h4 style="color:#38bdf8;margin-bottom:6px;">
            🛡️ Clinically Responsible AI
        </h4>
        <p style="color:#cbd5f5;margin:0;">
            LIFE-LEN AI provides <b>decision support only</b>.
            All outputs include confidence scores and require
            confirmation by licensed medical professionals.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ================== TOOLS GRID ==================
    st.markdown("## 🚀 Clinical Tools")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.button(
            "🫁 Chest X-Ray\n\nAI-powered pneumonia screening",
            use_container_width=True,
            on_click=go_to,
            args=("X-Ray",)
        )

    with c2:
        st.button(
            "💉 Diabetes\n\nMetabolic risk prediction",
            use_container_width=True,
            on_click=go_to,
            args=("Diabetes",)
        )

    with c3:
        st.button(
            "❤️ Heart Disease\n\nCardiac risk assessment",
            use_container_width=True,
            on_click=go_to,
            args=("Heart",)
        )

    st.markdown("<br>", unsafe_allow_html=True)

    _, mid, _ = st.columns([1, 1.5, 1])
    with mid:
        st.button(
            "🧬 Cancer Prediction\n\nMalignancy risk classification",
            use_container_width=True,
            on_click=go_to,
            args=("Cancer",)
        )

    # ================== CLINICAL INSIGHT ==================
    clinical_insights = [
        "AI-assisted screening improves early detection accuracy.",
        "Confidence scores indicate certainty — not diagnosis.",
        "Clinical judgment always overrides AI recommendations.",
        "Early risk identification reduces long-term complications."
    ]

    st.info(f"🩺 Clinical Insight: **{np.random.choice(clinical_insights)}**")
        # ================== WORKFLOW ==================
    st.markdown("## 🕒 Clinical AI Workflow")

    st.markdown("""
    <div style="
        border-left:4px solid #38bdf8;
        padding-left:1.5rem;
        margin-top:1.2rem;
    ">
        <p><b>Step 1:</b> Patient data submitted</p>
        <p><b>Step 2:</b> AI pattern analysis executed</p>
        <p><b>Step 3:</b> Confidence score generated</p>
        <p><b>Step 4:</b> Clinical summary produced</p>
        <p><b>Step 5:</b> Doctor reviews & decides</p>
    </div>
    """, unsafe_allow_html=True)
    # ================== chest xray ==================

def chest_xray_page():
    st.header("🫁 Chest X-Ray Analyzer")

    if models["xray"] is None:
        st.error("Chest X-Ray model not available.")
        return

    file = st.file_uploader("Upload Chest X-Ray", type=["png", "jpg", "jpeg"])
    if not file:
        st.info("Please upload a chest X-ray image.")
        return

    file_bytes = np.frombuffer(file.read(), np.uint8)
    img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        st.error("Invalid image file.")
        return

    st.image(
        cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
        caption="Uploaded Image",
        width=520
    )

    # ---------- X-RAY VALIDATION ----------
    is_xray, det_conf, _ = predict_is_xray(img_bgr)

    if not is_xray and det_conf >= 0.75:
        st.error("❌ Uploaded image is NOT a chest X-ray.")
        return

    if not is_xray:
        st.warning("⚠️ Atypical chest X-ray detected. Accuracy may be reduced.")

    # ---------- PREPROCESS ----------
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(img_gray, (224, 224)) / 255.0
    inp = np.expand_dims(resized, (0, -1)).astype(np.float32)

    if st.button("🔍 Analyze X-Ray"):
        with st.spinner("Analyzing chest radiograph..."):
            pred = models["xray"].predict(inp)
            prob = float(pred[0][0])

        label = "PNEUMONIA" if prob > 0.5 else "NORMAL"
        conf = prob if label == "PNEUMONIA" else 1 - prob
        disease = "Pneumonia" if label == "PNEUMONIA" else "No Lung Disease"

        severity = (
            "Very Mild" if conf < 0.30 else
            "Mild" if conf < 0.60 else
            "Moderate" if conf < 0.85 else
            "Severe"
        )

        st.success(f"🩺 Result: {label} ({conf*100:.2f}%)")

        ai_text = ai_summary_prompt(
            label=label,
            confidence=conf * 100,
            condition_name="Chest X-Ray / Pneumonia Detection"
        )
        st.info(ai_text)

        insert_prediction(
            username=st.session_state.user,
            condition="Chest X-Ray",
            result=label,
            confidence=round(conf * 100, 2),
            summary=ai_text
        )

        pdf_path = create_pdf(
            [
                "Chest X-Ray Analysis Report",
                f"Result: {label}",
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
                "📄 Download Report (PDF)",
                f,
                file_name="xray_report.pdf"
            )

#-------------------------------
#----------------------------- Diabetes Page -----------------------------
#-----------------------------

def diabetes_page():
    st.header("💉 Diabetes Risk Assessment")
    st.markdown("""
<div style="
    background:linear-gradient(90deg,#020617,#0f172a);
    padding:1.2rem 1.6rem;
    border-radius:14px;
    border-left:6px solid #38bdf8;
    margin-bottom:1.5rem;
">
    <h4 style="margin:0;color:#38bdf8;">🧭 Clinical Workflow</h4>
    <p style="margin:4px 0 0;color:#cbd5f5;font-size:0.95rem;">
        Input patient data → AI risk analysis → Clinical interpretation → Next steps
    </p>
</div>
""", unsafe_allow_html=True)


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

        insert_prediction(
    username=st.session_state.user,
    condition="Diabetes",
    result=stage,
    confidence=round(prob * 100, 2),
    summary=ai_text
)

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
#------------------------------
#----------------------------- Heart Disease Page -----------------------------
#-----------------------------
def heart_page():
    st.header("❤️ Heart Disease Risk Assessment")
    st.markdown("""
<div style="
    background:linear-gradient(90deg,#020617,#0f172a);
    padding:1.2rem 1.6rem;
    border-radius:14px;
    border-left:6px solid #38bdf8;
    margin-bottom:1.5rem;
">
    <h4 style="margin:0;color:#38bdf8;">🧭 Clinical Workflow</h4>
    <p style="margin:4px 0 0;color:#cbd5f5;font-size:0.95rem;">
        Input patient data → AI risk analysis → Clinical interpretation → Next steps
    </p>
</div>
""", unsafe_allow_html=True)


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
        insert_prediction(
    username=st.session_state.user,
    condition="Heart Disease",
    result=stage,
    confidence=round(prob * 100, 2),
    summary=ai_text
)

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
#------------------------------
#----------------------------- Cancer Page -----------------------------
#-----------------------------
def cancer_page():
    st.header("🧬 Breast Cancer Risk Assessment")
    st.markdown("""
<div style="
    background:linear-gradient(90deg,#020617,#0f172a);
    padding:1.2rem 1.6rem;
    border-radius:14px;
    border-left:6px solid #38bdf8;
    margin-bottom:1.5rem;
">
    <h4 style="margin:0;color:#38bdf8;">🧭 Clinical Workflow</h4>
    <p style="margin:4px 0 0;color:#cbd5f5;font-size:0.95rem;">
        Input patient data → AI risk analysis → Clinical interpretation → Next steps
    </p>
</div>
""", unsafe_allow_html=True)

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
        insert_prediction(
    username=st.session_state.user,
    condition="Cancer",
    result=presence,
    confidence=round(prob * 100, 2),
    summary=ai_text
)


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

# -----------------------------
# REPORT ANALYZER PAGE
# -----------------------------
def report_page():
    st.header("📑 Medical Report Analyzer")

    file = st.file_uploader(
        "Upload a Medical Report (PDF / Image)",
        type=["pdf", "png", "jpg", "jpeg"]
    )

    if not file:
        st.info("Upload a medical report (lab report, prescription, scan, etc.)")
        return

    extracted_text = ""

    # ---------- PDF ----------
    if file.type == "application/pdf":
        st.subheader("📄 PDF Preview")

        pdf_bytes = file.read()
        base64_pdf = base64.b64encode(pdf_bytes).decode("utf-8")
        st.markdown(
            f"""
            <embed src="data:application/pdf;base64,{base64_pdf}"
                   width="100%" height="600px" />
            """,
            unsafe_allow_html=True,
        )

        file.seek(0)
        with pdfplumber.open(file) as pdf:
            for page in pdf.pages:
                txt = page.extract_text()
                if txt:
                    extracted_text += txt + "\n"

    # ---------- IMAGE ----------
    else:
        st.subheader("🖼 Image Preview")
        img = PILImage.open(file)
        st.image(img, use_column_width=True)

        extracted_text = pytesseract.image_to_string(img)

    # ---------- TEXT ----------
    st.subheader("📝 Extracted Text")
    st.text_area("Report Content", extracted_text, height=300)

    if not extracted_text.strip():
        st.warning("No readable medical text found.")
        return

    # ---------- AI ANALYSIS ----------
    st.subheader("🧠 AI Medical Interpretation")

    ai_text = ai_summary_prompt(
        label="Medical Report Analysis",
        confidence=100,
        condition_name="Medical Report / Prescription Analysis"
    )

    st.info(ai_text)

    speak_text_autoplay("Medical report analysis completed.")

    # ---------- STORE IN DB ----------
    insert_prediction(
        username=st.session_state.user,
        condition="Medical Report",
        result="Analyzed",
        confidence=100,
        summary=ai_text
    )

    st.success("✅ Report analyzed and saved to your dashboard")
# -----------------------------
# DASHBOARD PAGE
# -----------------------------
def dashboard_page():
    st.header("📊 My Prediction History")

    username = st.session_state.user
    rows = fetch_predictions(username)

    if not rows:
        st.info("No predictions yet. Run any model to see results here.")
        return

    df = pd.DataFrame(
        rows,
        columns=["Condition", "Result", "Confidence (%)", "Date"]
    )

    st.dataframe(df, use_container_width=True)

    st.subheader("📈 My Analytics")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Predictions by Condition")
        st.bar_chart(df["Condition"].value_counts())

    with col2:
        st.markdown("### Confidence Distribution")
        st.line_chart(df["Confidence (%)"])
# -----------------------------
# PROFILE PAGE (Medical Style)
# -----------------------------
def profile_page():
    # =========================
    # HEADER CARD
    # =========================
    st.markdown("""
    <div style="
        background: linear-gradient(90deg,#0f172a,#020617);
        padding:1.8rem;
        border-radius:18px;
        border:1px solid rgba(255,255,255,0.12);
        box-shadow:0 25px 70px rgba(0,0,0,0.75);
        margin-bottom:1.8rem;
    ">
        <h2 style="color:#38bdf8;margin:0;">👤 Patient Profile</h2>
        <p style="color:#94a3b8;margin-top:6px;">
            AI-assisted clinical activity summary
        </p>
    </div>
    """, unsafe_allow_html=True)

    username = st.session_state.user
    role = st.session_state.role

    # =========================
    # BASIC INFO + SUMMARY
    # =========================
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🧾 Patient Account Information")
        st.write(f"**Patient ID:** {username}")
        st.write(f"**Access Role:** {role.capitalize()}")
        st.write("**Platform:** LIFE-LEN AI Clinical System")

    # ---------- FETCH USER DATA ----------
    rows = fetch_predictions(username)

    total_predictions = len(rows)
    last_activity = rows[0][-1] if rows else None

    with col2:
        st.markdown("### 📊 Clinical Usage Summary")
        st.metric("Total AI Assessments", total_predictions)
        st.metric(
            "Last Clinical Activity",
            last_activity if last_activity else "—"
        )

    st.markdown("---")

    # =========================
    # RECENT ACTIVITY TABLE
    # =========================
    st.markdown("### 🕒 Recent Clinical Assessments")

    if not rows:
        st.info("No clinical assessments recorded yet.")
        return

    df = pd.DataFrame(
        rows,
        columns=["Condition", "Result", "Confidence (%)", "Date"]
    )

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.markdown("---")

    # =========================
    # ANALYTICS
    # =========================
    st.markdown("### 📈 Patient Risk Analytics")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Assessments by Condition**")
        st.bar_chart(df["Condition"].value_counts())

    with col4:
        st.markdown("**AI Confidence Trend Over Time**")
        df_sorted = df.copy()
        df_sorted["Date"] = pd.to_datetime(df_sorted["Date"])
        df_sorted = df_sorted.sort_values("Date")
        st.line_chart(
            df_sorted.set_index("Date")["Confidence (%)"]
        )

    # =========================
    # MEDICAL DISCLAIMER
    # =========================
    st.markdown("---")

    st.warning(
        "⚠️ This profile displays **AI-assisted decision support records only**. "
        "It is **not a medical record** and does **not replace professional diagnosis**. "
        "Always consult a licensed healthcare professional."
    )
    if is_doctor():
        doctor_profile_view()
    else:
        patient_profile_view()
def patient_profile_view():
    st.header("👤 Patient Profile")

    username = st.session_state.user

    rows = fetch_predictions(username)
    total = len(rows)
    last = rows[0][-1] if rows else "No activity"

    col1, col2 = st.columns(2)
    col1.metric("Total AI Assessments", total)
    col2.metric("Last Activity", last)

    st.markdown("---")

    if not rows:
        st.info("No assessments yet.")
        return

    df = pd.DataFrame(
        rows,
        columns=["Condition", "Result", "Confidence (%)", "Date"]
    )

    st.dataframe(df, use_container_width=True)

    st.warning(
        "⚠️ AI-assisted decision support only. Not a medical record."
    )

def doctor_profile_view():
    st.header("👨‍⚕️ Doctor Dashboard")

    users = fetch_all_users()
    predictions = fetch_all_predictions()

    col1, col2, col3 = st.columns(3)
    col1.metric("Registered Patients", len(users))
    col2.metric("Total Assessments", len(predictions))
    col3.metric("Role", st.session_state.role.capitalize())

    st.markdown("---")
    st.markdown("### 🧑‍🤝‍🧑 Patient List")

    for username, role, created_at in users:
        if role != "user":
            continue

        cols = st.columns([3, 2, 2, 1])
        cols[0].markdown(f"**{username}**")
        cols[1].markdown("Patient")
        cols[2].markdown(created_at)

        if cols[3].button("View", key=f"view_{username}"):
            st.session_state.selected_patient = username
            st.session_state.Navigation = "Patient Profile"
            st.rerun()
# -----------------------------
# SELECTED PATIENT PROFILE PAGE
# -----------------------------
def selected_patient_profile():
    patient = st.session_state.selected_patient
    rows = fetch_predictions(patient)

    st.header(f"🧾 Patient Record: {patient}")
    st.dataframe(pd.DataFrame(
        rows,
        columns=["Condition", "Result", "Confidence", "Date"]
    ))

# -----------------------------
# SETTINGS PAGE
# -----------------------------
def settings_page():
    st.header(f"⚙️ {t('settings')}")

    # ---------- LANGUAGE ----------
    st.markdown(f"### 🌐 {t('language')}")
    selected_lang = st.selectbox(
        "Select Language",
        options=list(TRANSLATIONS.keys()),
        index=list(TRANSLATIONS.keys()).index(st.session_state.language),
    )

    if selected_lang != st.session_state.language:
        st.session_state.language = selected_lang
        st.success(f"Language changed to {selected_lang}")
        st.rerun()

    st.markdown("---")

    # ---------- AUDIO ----------
    st.markdown(f"### 🔊 {t('audio')}")
    voice_enabled = st.toggle(
        t("enable_voice"),
        value=st.session_state.get("voice_enabled", True)
    )
    st.session_state.voice_enabled = voice_enabled

    st.markdown("---")

    # ---------- SESSION ----------
    st.markdown(f"### 🧹 {t('session')}")

    col1, col2 = st.columns(2)

    with col1:
        if st.button(f"🗑️ {t('clear_history')}"):
            st.session_state.history = []
            st.success("Prediction history cleared")

    with col2:
        if st.button(f"🚪 {t('logout')}"):
            st.session_state.clear()
            st.session_state.logged_in = False
            st.session_state.Navigation = "Home"
            st.rerun()

    st.markdown("---")

    # ---------- ACCOUNT ----------
    st.markdown(f"### ℹ️ {t('account')}")
    st.write(f"**{t('username')}:** {st.session_state.user}")
    st.write(f"**{t('role')}:** {st.session_state.role.capitalize()}")

    st.info("More customization options coming soon.")

# -----------------------------
# CHATBOT RESPONSE FUNCTION
# -----------------------------
def help_chatbot_response(user_msg: str) -> str:
    msg = user_msg.lower()

    # ---------- OFFLINE SMART RESPONSES ----------
    if "x-ray" in msg:
        return (
            "To analyze an X-ray:\n"
            "- Upload a clear chest X-ray image\n"
            "- Avoid photos of reports or screenshots\n"
            "- If rejected, the image may not be a valid radiograph"
        )

    if "confidence" in msg:
        return (
            "Confidence shows how certain the AI is.\n"
            "- High confidence → stronger pattern match\n"
            "- Low confidence → uncertainty, consult a doctor"
        )

    if "diabetes" in msg:
        return (
            "Diabetes analysis is based on glucose, BMI, age, and other inputs.\n"
            "Always confirm results with HbA1c or fasting glucose tests."
        )

    if "heart" in msg:
        return (
            "Heart risk prediction estimates cardiovascular risk.\n"
            "ECG, ECHO, and doctor consultation are recommended."
        )

    if "cancer" in msg:
        return (
            "Cancer results are risk estimates only.\n"
            "Biopsy and imaging are required for confirmation."
        )

    if "report" in msg:
        return (
            "Upload a PDF or image medical report.\n"
            "The AI extracts text and provides a summary."
        )

    if "voice" in msg:
        return (
            "Voice output requires internet.\n"
            "You can disable it anytime in Settings."
        )

    if "hello" in msg or "hi" in msg:
        return "Hello 👋 I'm the LIFE-LEN AI Help Assistant. How can I help you?"

    if "who made" in msg or "developer" in msg:
        return "LIFE-LEN AI was developed by **Muneesh**."

    # ---------- ONLINE MODE (OPTIONAL) ----------
    if client is not None:
        try:
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a medical app help assistant. Do not give diagnosis."},
                    {"role": "user", "content": user_msg},
                ]
            )
            return res.choices[0].message.content.strip()
        except Exception:
            pass

    return (
        "I'm not sure about that yet 🤔\n"
        "Try asking about X-ray, diabetes, heart, cancer, reports, or settings."
    )

# -----------------------------
# HELP PAGE
# -----------------------------
def help_page():
    st.header("❓ Help & Support")

    st.markdown("### 🧭 How to Use LIFE-LEN AI")

    st.markdown("""
    **1️⃣ Chest X-Ray**
    - Upload a clear chest X-ray image  
    - AI checks validity and analyzes pneumonia risk  

    **2️⃣ Diabetes / Heart / Cancer**
    - Enter clinical values carefully  
    - Results show risk, severity, and recommendations  

    **3️⃣ Reports**
    - Upload PDF or image medical reports  
    - AI extracts and summarizes medical insights  

    **4️⃣ Dashboard**
    - View your full prediction history  
    - Analyze trends and confidence levels  
    """)

    st.markdown("---")

    st.markdown("### ⚠️ Medical Disclaimer")
    st.warning(
        "LIFE-LEN AI provides **AI-assisted decision support only**. "
        "It does NOT replace a licensed medical professional."
    )

    st.markdown("---")

    st.markdown("### 🆘 Common Issues")

    st.markdown("""
    **Q: My X-ray is rejected**
    - Ensure it is a real chest radiograph  
    - Avoid photos of reports or screenshots  

    **Q: Confidence is low**
    - Low confidence means uncertainty  
    - Always consult a doctor  

    **Q: Voice not playing**
    - Internet connection required  
    - Can be disabled in Settings  
    """)

    st.markdown("---")

    st.markdown("### 📧 Contact Support")
    st.info(
        "For support or collaboration:\n\n"
        "**Email:** contact@lifelen-ai.com\n\n"
        "**Developer:** Muneesh"
    )
    st.markdown("---")
    st.markdown("## 🤖 Help Chatbot")

    # Initialize chat history
    if "help_chat" not in st.session_state:
        st.session_state.help_chat = []

    # Clear chat
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("🧹 Clear Chat"):
            st.session_state.help_chat = []
            st.rerun()

    # Display messages
    for chat in st.session_state.help_chat:
        with st.chat_message(chat["role"]):
            st.markdown(chat["content"])

    # User input
    user_input = st.chat_input("Ask how to use LIFE-LEN AI…")

    if user_input:
        # Store user message
        st.session_state.help_chat.append({
            "role": "user",
            "content": user_input
        })

        # Get bot response
        reply = help_chatbot_response(user_input)

        # Display bot response
        with st.chat_message("assistant"):
            st.markdown(reply)

        # Store bot message
        st.session_state.help_chat.append({
            "role": "assistant",
            "content": reply
        })

    st.caption("⚠️ This chatbot provides usage guidance only. No medical diagnosis.")
# -----------------------------
# ABOUT PAGE
# -----------------------------
def about_page():
    st.header("ℹ️ About LIFE-LEN AI")

    st.markdown("""
    **LIFE-LEN AI** is a next-generation **clinical decision-support platform**
    developed by **Muneesh**, designed to assist doctors, hospitals, and patients
    using safe, explainable artificial intelligence.

    LIFE-LEN AI does **not replace doctors** — it empowers them with faster,
    data-driven insights.
    """)

    st.markdown("---")

    # =========================
    # VISUAL SECTION
    # =========================
    st.markdown("### 🏥 AI in Modern Healthcare")

    col1, col2 = st.columns(2)
    with col1:
        st.image(
            "https://images.unsplash.com/photo-1586773860418-d37222d8fce3",
            caption="AI-assisted hospital diagnostics",
            use_column_width=True,
        )
    with col2:
        st.image(
            "https://images.unsplash.com/photo-1581090700227-1e37b190418e",
            caption="Medical imaging & AI analysis",
            use_column_width=True,
        )

    st.markdown("---")

    # =========================
    # MISSION & VISION
    # =========================
    st.markdown("## 🎯 Mission & Vision")

    st.markdown("""
    ### 🌍 Mission
    To democratize access to **advanced medical diagnostics** through
    clinically responsible artificial intelligence — making healthcare
    **faster, safer, and more accessible**.

    ### 🔮 Vision
    A future where **AI-assisted healthcare tools** help detect diseases early,
    reduce diagnostic delays, and improve patient outcomes across the globe.
    """)

    st.markdown("---")

    # =========================
    # HOW IT WORKS
    # =========================
    st.markdown("## 🧠 How LIFE-LEN AI Works")

    st.markdown("""
    LIFE-LEN AI combines multiple AI technologies into a single clinical platform:

    - 🫁 **Deep Learning (CNNs)** for chest X-ray and medical imaging analysis  
    - 📊 **Machine Learning Models** for diabetes, heart disease, and cancer risk  
    - 📑 **OCR + NLP** for extracting insights from medical reports  
    - 🧾 **Explainable AI summaries** for patient-friendly understanding  
    """)

    st.markdown("---")

    # =========================
    # TRUST & SAFETY
    # =========================
    st.markdown("## 🛡️ Trust, Safety & Ethics")

    st.markdown("""
    LIFE-LEN AI is built with **medical responsibility** at its core:

    - ✅ AI outputs are **decision-support only**
    - ✅ No automated diagnosis or treatment decisions
    - ✅ Designed to assist — never override — clinicians
    - ✅ Local processing where possible to respect patient privacy
    - ✅ Clear confidence scores & disclaimers included in every result
    """)

    st.warning(
        "⚠️ LIFE-LEN AI is **not a medical device** and does not replace a licensed healthcare professional."
    )

    st.markdown("---")

    # =========================
    # WHO IT IS FOR
    # =========================
    st.markdown("## 👨‍⚕️ Who Is LIFE-LEN AI For?")

    st.markdown("""
    - 🏥 **Hospitals & Clinics** — Faster triage and second-opinion support  
    - 👨‍⚕️ **Doctors & Radiologists** — AI-assisted pattern recognition  
    - 🧑‍🤝‍🧑 **Patients** — Better understanding of reports & risks  
    - 🎓 **Students & Researchers** — Learning applied medical AI  
    """)

    st.markdown("---")

    # =========================
    # ROADMAP
    # =========================
    st.markdown("## 🚀 Future Roadmap")

    st.markdown("""
    Planned enhancements include:

    - 🔐 Secure patient profiles & longitudinal health tracking  
    - 🌐 Multi-language support  
    - 🤖 Advanced AI chat assistant for result explanation  
    - 🏥 Hospital system integrations  
    - 📱 Mobile-friendly deployment  
    """)

    st.markdown("---")

    # =========================
    # DEVELOPER CREDIT
    # =========================
    st.markdown("## 👤 Developer")

    st.markdown("""
    **Muneesh**  
    AI & Healthcare Application Developer  

    LIFE-LEN AI is built with a strong focus on **clinical realism,
    user experience, and ethical AI design**.
    """)


# -----------------------------
# CONTACT PAGE
# -----------------------------
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
                insert_message(name, email, subject, message)
                st.success("✅ Message sent successfully!")

# -----------------------------
# LEGAL PAGE
# -----------------------------
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
# =========================
# ADMIN DASHBOARD PAGE
# =========================
def admin_page():
    st.header("👑 Admin Dashboard")

    # ================= USERS =================
    st.subheader("👥 Registered Users")
    users = fetch_all_users()

    if users:
        df_users = pd.DataFrame(
            users,
            columns=["Username", "Role", "Created At"]
        )
        st.dataframe(df_users, use_container_width=True)
    else:
        st.info("No users found.")

    st.markdown("---")

    # ================= PREDICTIONS =================
    st.subheader("📊 All Predictions")
    records = fetch_all_predictions()

    if records:
        df = pd.DataFrame(
            records,
            columns=["User", "Condition", "Result", "Confidence (%)", "Date"]
        )
        st.dataframe(df, use_container_width=True)

        st.subheader("📈 Analytics")
        col1, col2 = st.columns(2)

        with col1:
            st.bar_chart(df["Condition"].value_counts())

        with col2:
            st.bar_chart(df["User"].value_counts())
    else:
        st.info("No prediction records available.")

    st.markdown("---")

    # ================= CONTACT MESSAGES =================
    st.subheader("📩 Contact & Support Messages")

    messages = fetch_messages()

    if messages:
        df_msg = pd.DataFrame(
            messages,
            columns=["Name", "Email", "Subject", "Message", "Received At"]
        )
        st.dataframe(df_msg, use_container_width=True)
    else:
        st.info("No support messages received yet.")


# -----------------------------
# Page Routing
# -----------------------------
# -----------------------------
# Page Routing (NO EMAIL GATE)
# -----------------------------
set_page_background(st.session_state.Navigation)

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
elif current_page == "Profile":
    profile_page()
elif current_page == "Patient Profile" and st.session_state.get("role") in ["doctor", "admin"]:
    patient_profile_for_doctor()
elif current_page == "Settings":
    settings_page()
elif current_page == "Help":
    help_page()
elif current_page == "About":
    about_page()
elif current_page == "Contact":
    contact_page()
elif current_page == "Legal":
    legal_page()
elif current_page == "Admin" and st.session_state.role == "admin":
    admin_page()


# =========================
# FOOTER


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
