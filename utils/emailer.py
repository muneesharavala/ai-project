import smtplib
import ssl
from email.message import EmailMessage
import streamlit as st
import os

def send_report_email(
    to_email: str,
    subject: str,
    body: str,
    attachment_path: str | None = None
):
    sender = st.secrets["EMAIL_ADDRESS"]
    password = st.secrets["EMAIL_PASSWORD"]

    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = to_email
    msg["Subject"] = subject
    msg.set_content(body)

    # ---------- ATTACH PDF ----------
    if attachment_path and os.path.exists(attachment_path):
        with open(attachment_path, "rb") as f:
            file_data = f.read()
            file_name = os.path.basename(attachment_path)

        msg.add_attachment(
            file_data,
            maintype="application",
            subtype="pdf",
            filename=file_name
        )

    context = ssl.create_default_context()

    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender, password)
        server.send_message(msg)
