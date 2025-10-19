from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from google.cloud import vision
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
import os
import uuid
import tempfile
import json

app = FastAPI(title="Handwriting OCR API v8 (Arabic + Hausa + French Supported)")

# ---------- STEP 1: ENABLE CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to your frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- STEP 2: GOOGLE VISION SETUP ----------
client = None  # Default placeholder
GOOGLE_CREDENTIALS_JSON = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")

try:
    if GOOGLE_CREDENTIALS_JSON:
        # Write credentials JSON (from env var) to a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w") as tmp_file:
            json.dump(json.loads(GOOGLE_CREDENTIALS_JSON), tmp_file)
            tmp_file.flush()
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = tmp_file.name
        client = vision.ImageAnnotatorClient()
        print("✅ Google Vision client initialized from environment variable.")
    else:
        # Local dev fallback (only if running locally)
        local_key = "backend/write2doc-862dc6095d89.json"
        if os.path.exists(local_key):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = local_key
            client = vision.ImageAnnotatorClient()
            print("⚙️ Using local Google Vision credentials.")
        else:
            print("⚠️ No Google Vision credentials found. OCR may be disabled.")
except Exception as e:
    print(f"❌ Error initializing Google Vision client: {e}")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------- STEP 3: OCR FUNCTIONS ----------
def extract_text_from_image(content: bytes) -> str:
    """Extract text from image (Arabic, Hausa, English, French)"""
    if not client:
        raise RuntimeError("Google Vision client not initialized.")
    image = vision.Image(content=content)
    image_context = {"language_hints": ["ar", "en", "ha", "fr"]}
    response = client.document_text_detection(image=image, image_context=image_context)
    if response.error.message:
        raise Exception(response.error.message)
    return response.full_text_annotation.text.strip()

def extract_text_from_pdf(content: bytes) -> str:
    """Extract text from PDF using Google OCR"""
    if not client:
        raise RuntimeError("Google Vision client not initialized.")
    input_doc = vision.InputConfig(content=content, mime_type="application/pdf")
    feature = vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)
    image_context = {"language_hints": ["ar", "en", "ha", "fr"]}
    request = vision.AnnotateFileRequest(
        features=[feature],
        input_config=input_doc,
        image_context=image_context
    )
    result = client.batch_annotate_files(requests=[request])
    extracted_text = ""
    for response in result.responses:
        for page in response.responses:
            extracted_text += page.full_text_annotation.text + "\n\n"
    return extracted_text.strip()

# ---------- STEP 4: MAIN OCR ROUTE ----------
@app.post("/ocr/")
async def ocr_file(file: UploadFile = File(...)):
    try:
        filename = file.filename.lower()
        content = await file.read()

        if filename.endswith((".png", ".jpg", ".jpeg")):
            extracted_text = extract_text_from_image(content)
        elif filename.endswith(".pdf"):
            extracted_text = extract_text_from_pdf(content)
        else:
            return JSONResponse({"error": "Unsupported file type"}, status_code=400)

        if not extracted_text.strip():
            return JSONResponse({"message": "❌ No readable text found."}, status_code=400)

        uid = str(uuid.uuid4())[:8]
        pdf_path = os.path.join(OUTPUT_DIR, f"ocr_{uid}.pdf")
        docx_path = os.path.join(OUTPUT_DIR, f"ocr_{uid}.docx")

        # Create DOCX
        doc = Document()
        doc.add_heading("Extracted Handwritten Text", level=1)
        doc.add_paragraph(extracted_text)
        doc.save(docx_path)

        # Create PDF
        pdf_buffer = io.BytesIO()
        c = canvas.Canvas(pdf_buffer, pagesize=letter)
        c.setFont("Helvetica", 10)
        y = 750
        for line in extracted_text.split("\n"):
            if any("\u0600" <= ch <= "\u06FF" for ch in line):  # Detect Arabic
                c.drawRightString(550, y, line)
            else:
                c.drawString(72, y, line)
            y -= 15
            if y < 50:
                c.showPage()
                y = 750
        c.save()
        with open(pdf_path, "wb") as f:
            f.write(pdf_buffer.getvalue())

        base_url = os.getenv("FLY_APP_URL", os.getenv("RENDER_EXTERNAL_URL", "http://127.0.0.1:8000"))

        return {
            "message": "✅ OCR extraction successful!",
            "extracted_text": extracted_text,
            "pdf_url": f"{base_url}/download/pdf/{uid}",
            "docx_url": f"{base_url}/download/docx/{uid}",
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ---------- STEP 5: DOWNLOAD ROUTES ----------
@app.get("/download/pdf/{uid}")
async def download_pdf(uid: str):
    path = os.path.join(OUTPUT_DIR, f"ocr_{uid}.pdf")
    if not os.path.exists(path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(path, filename=f"ocr_{uid}.pdf", media_type="application/pdf")

@app.get("/download/docx/{uid}")
async def download_docx(uid: str):
    path = os.path.join(OUTPUT_DIR, f"ocr_{uid}.docx")
    if not os.path.exists(path):
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(
        path,
        filename=f"ocr_{uid}.docx",
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )

@app.get("/")
def home():
    return {"message": "🚀 Handwriting OCR API v8 (Arabic + Hausa + French supported) is running securely!"}
