# ========================
# IMPORTS
# ========================
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import google.generativeai as genai
import boto3
import psycopg2
import requests
import os
from dotenv import load_dotenv
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware


# ========================
# LOAD .env (ABSOLUTE PATH → WORKS EVEN ON USB)
# ========================
ENV_PATH = Path(__file__).resolve().parent / ".env"
print("Loading .env from:", ENV_PATH)

load_dotenv(dotenv_path=ENV_PATH)

# DEBUG PRINT (REMOVE LATER)
print("AWS_REGION =", os.getenv("AWS_REGION"))

# ========================
# ENV VARIABLES
# ========================
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION") or "us-east-1"     # <---- FIXED
SIGHTENGINE_USER = os.getenv("SIGHTENGINE_USER")
SIGHTENGINE_SECRET = os.getenv("SIGHTENGINE_SECRET")

DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")
print("========== ENV DEBUG ==========")
print("GEMINI =", os.getenv("GEMINI_API_KEY"))
print("AWS_ACCESS_KEY =", os.getenv("AWS_ACCESS_KEY_ID"))
print("AWS_SECRET =", os.getenv("AWS_SECRET_ACCESS_KEY"))
print("AWS_REGION =", os.getenv("AWS_REGION"))
print("SIGHTENGINE_USER =", os.getenv("SIGHTENGINE_USER"))
print("================================")

# ========================
# AWS Rekognition Client
# ========================
rekognition = boto3.client(
    "rekognition",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

# ========================
# Gemini Setup
# ========================
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# ========================
# FastAPI App
# ========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # allow all websites
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# TEXT API
# ========================
class TextReq(BaseModel):
    user_id: int
    text: str

@app.post("/moderate-text")
async def moderate_text_api(req: TextReq):
    prompt = f"""
    Classify this message as toxic or safe:
    "{req.text}"

    Respond ONLY JSON:
    {{
        "classification": "...",
        "explanation": "..."
    }}
    """

    out = gemini_model.generate_content(prompt).text
    return {"analysis": out}

# ========================
# IMAGE API
# ========================
@app.post("/moderate-image")
async def moderate_image_api(file: UploadFile = File(...)):
    contents = await file.read()
    temp_path = "uploaded_image.jpg"

    with open(temp_path, "wb") as f:
        f.write(contents)

    # --- SightEngine AI detection ---
    ai_json = requests.post(
        "https://api.sightengine.com/1.0/check.json",
        data={
            "models": "genai",
            "api_user": SIGHTENGINE_USER,
            "api_secret": SIGHTENGINE_SECRET
        },
        files={"media": open(temp_path, "rb")}
    ).json()

    ai_score = ai_json.get("type", {}).get("ai_generated", 0)

    # --- AWS Rekognition content moderation ---
    rek = rekognition.detect_moderation_labels(Image={"Bytes": contents})
    labels = rek.get("ModerationLabels", [])

    return {
        "ai_generated": ai_score > 0.5,
        "aws_labels": labels
    }
