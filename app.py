import os
import json
from flask import Flask, render_template, request, flash, redirect
from werkzeug.utils import secure_filename
import google.genai as genai
from google.genai import types

# Load environment variables from a local .env file if present (optional).
# This is where GEMINI_API_KEY lives - the app uses Gemini for all diagnosis.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ------------------------------
# Flask App Setup
# ------------------------------
app = Flask(__name__)
UPLOAD_FOLDER = 'static/shots'
app.secret_key = 'cropdisease'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Last uploaded filename (used by the /display route)
filename = ""

# ------------------------------
# Gemini setup
# ------------------------------
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
try:
    GEMINI_CLIENT = genai.Client(api_key=os.getenv("GEMINI_API_KEY")) if os.getenv("GEMINI_API_KEY") else None
    if GEMINI_CLIENT is None:
        print("GEMINI_API_KEY not set - diagnosis will be disabled until it is configured.")
except Exception as e:
    GEMINI_CLIENT = None
    print(f"Gemini is unavailable: {e}")


def allowed_file(fname):
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def _mime_for(path):
    ext = path.rsplit('.', 1)[-1].lower() if '.' in path else ''
    return {
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'webp': 'image/webp',
    }.get(ext, 'image/jpeg')


# The exact JSON shape we ask Gemini to return.
ANALYSIS_PROMPT = (
    "You are an expert plant pathologist analysing a photo of a plant leaf for a "
    "crop-disease detection app. Look carefully at the image and respond with ONLY a "
    "single JSON object (no markdown, no code fences, no text before or after it) "
    "using exactly these keys:\n"
    "{\n"
    '  "is_leaf": boolean,        // true only if the main subject is a plant/crop leaf\n'
    '  "crop": string,            // the plant/crop name, e.g. "Tomato", "Maize (corn)", "Cassava"; "" if not a leaf\n'
    '  "healthy": boolean,        // true if the leaf looks healthy with no disease\n'
    '  "disease": string,         // most likely disease name, or "Healthy", or "" if not a leaf\n'
    '  "confidence": integer,     // 0-100, your confidence in the crop + condition identification\n'
    '  "severity": string,        // one of "None", "Mild", "Moderate", "Severe"\n'
    '  "symptoms": string,        // short description of the visible symptoms you see\n'
    '  "treatment": string,       // concise, safe management/treatment steps\n'
    '  "prevention": string,      // concise prevention advice\n'
    '  "summary": string          // one or two plain-language sentences for a farmer\n'
    "}\n"
    "Rules: If the image is NOT a plant leaf (e.g. a person, object, screenshot, logo, "
    "food, landscape), set is_leaf=false, crop and disease to \"\", confidence=0, and "
    "explain in summary that the image is not a plant leaf. Never invent a disease for a "
    "non-leaf image. Recommend following product labels and local agricultural guidance, "
    "and do not give unsafe chemical advice."
)


def _parse_json(text):
    """Parse Gemini's reply into a dict, tolerating stray text or code fences."""
    if not text:
        return None
    text = text.strip()
    # Strip ```json ... ``` fences if present
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    # Grab the outermost {...} if there is surrounding prose
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]
    try:
        return json.loads(text)
    except Exception:
        return None


def analyze_with_gemini(image_path):
    """
    Send the image to Gemini and return a normalised result dict:
        status: 'ok' | 'not_leaf' | 'unconfigured' | 'error'
        plus crop, disease, confidence, severity, symptoms, treatment,
        prevention, summary / message depending on status.
    """
    if GEMINI_CLIENT is None:
        return {"status": "unconfigured", "confidence": 0,
                "message": "Gemini is not configured on the server. Add your "
                           "GEMINI_API_KEY to a .env file to enable diagnosis."}

    if not os.path.exists(image_path):
        return {"status": "error", "confidence": 0,
                "message": "The uploaded image could not be found. Please try again."}

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        response = GEMINI_CLIENT.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=_mime_for(image_path)),
                ANALYSIS_PROMPT,
            ],
        )
        data = _parse_json(getattr(response, "text", None))
    except Exception as e:
        print(f"Gemini analysis failed: {e}")
        return {"status": "error", "confidence": 0,
                "message": "The AI service could not analyse this image right now. "
                           "Please check your internet connection and try again."}

    if not isinstance(data, dict):
        return {"status": "error", "confidence": 0,
                "message": "The AI returned an unexpected response. Please try again."}

    if not data.get("is_leaf", False):
        return {"status": "not_leaf", "confidence": 0,
                "message": data.get("summary") or
                "This doesn't look like a plant leaf. Please upload a clear, well-lit "
                "close-up of a single crop leaf."}

    # Confident, structured diagnosis
    try:
        confidence = int(round(float(data.get("confidence", 0))))
    except (TypeError, ValueError):
        confidence = 0
    confidence = max(0, min(100, confidence))

    disease = (data.get("disease") or "").strip()
    if data.get("healthy") and (not disease or disease.lower() == "healthy"):
        disease = "Healthy"

    return {
        "status": "ok",
        "crop": (data.get("crop") or "Unknown plant").strip(),
        "disease": disease or "Unspecified condition",
        "healthy": bool(data.get("healthy")),
        "confidence": confidence,
        "severity": (data.get("severity") or "").strip(),
        "symptoms": (data.get("symptoms") or "").strip(),
        "treatment": (data.get("treatment") or "").strip(),
        "prevention": (data.get("prevention") or "").strip(),
        "summary": (data.get("summary") or "").strip(),
    }


# ------------------------------
# Routes
# ------------------------------
@app.route('/')
def index():
    return render_template('index.html')


@app.route("/input", methods=['GET', 'POST'])
def input():
    return render_template("input.html")


def render_result(fname):
    """Analyse a saved file with Gemini and render the results page."""
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    result = analyze_with_gemini(image_path)
    return render_template('display.html', variable_name=fname, **result)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    global filename

    if request.method == 'GET':
        # Nothing to analyse on a bare GET - send the user to the upload form.
        flash("Please upload an image to analyse.")
        return redirect('/input')

    # POST (file upload)
    if 'file' not in request.files:
        flash("No file part")
        return redirect('/input')
    file = request.files['file']
    if file.filename == '':
        flash("No image selected for uploading")
        return redirect('/input')
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        return render_result(filename)
    else:
        flash("Allowed image types are - png, jpg, jpeg, webp.")
        return redirect('/input')


@app.route('/display')
def display_image():
    if not filename:
        flash("No image to display.")
        return redirect('/input')
    return render_result(filename)


if __name__ == "__main__":
    app.run(debug=True)
