import os
import json
import time
from flask import Flask, render_template, request, flash, redirect
from werkzeug.utils import secure_filename
import ollama

import db

# Load environment variables from a local .env file if present (optional).
# This is where the Ollama settings live - the app uses a local Ollama
# vision model for all diagnosis.
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

# Create the scans database/table on startup (safe to run every time).
db.init_db()

# Last uploaded filename (used by the /display route)
filename = ""

# ------------------------------
# Ollama setup (Ollama Cloud)
# ------------------------------
# Uses Ollama Cloud by default. Get an API key from https://ollama.com
# (Account -> Keys) and put it in .env as OLLAMA_API_KEY. Cloud vision models
# are named with a "-cloud" suffix, e.g. "qwen2.5vl:7b-cloud".
# To use a local server instead, set OLLAMA_HOST=http://localhost:11434 and
# leave OLLAMA_API_KEY blank.
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5vl:7b-cloud")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "https://ollama.com")
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY", "").strip()
try:
    _headers = {"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else None
    OLLAMA_CLIENT = ollama.Client(host=OLLAMA_HOST, headers=_headers)
    _auth = "with API key" if OLLAMA_API_KEY else "no API key set"
    print(f"Ollama configured: host={OLLAMA_HOST} model={OLLAMA_MODEL} ({_auth})")
    if not OLLAMA_API_KEY and "ollama.com" in OLLAMA_HOST:
        print("OLLAMA_API_KEY is not set - Ollama Cloud requests will fail until "
              "you add it to your .env file.")
except Exception as e:
    OLLAMA_CLIENT = None
    print(f"Ollama is unavailable: {e}")


def allowed_file(fname):
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# The exact JSON shape we ask the model to return.
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
    """Parse the model's reply into a dict, tolerating stray text or code fences."""
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


def analyze_with_ollama(image_path):
    """
    Send the image to a local Ollama vision model and return a normalised
    result dict:
        status: 'ok' | 'not_leaf' | 'unconfigured' | 'error'
        plus crop, disease, confidence, severity, symptoms, treatment,
        prevention, summary / message depending on status.
    """
    if OLLAMA_CLIENT is None or ("ollama.com" in OLLAMA_HOST and not OLLAMA_API_KEY):
        return {"status": "unconfigured", "confidence": 0,
                "message": "Ollama Cloud is not configured on the server. Add your "
                           "OLLAMA_API_KEY to a .env file to enable diagnosis."}

    if not os.path.exists(image_path):
        return {"status": "error", "confidence": 0,
                "message": "The uploaded image could not be found. Please try again."}

    try:
        # Retry transient errors (server starting up / model loading) with backoff.
        response = None
        for attempt in range(3):
            try:
                response = OLLAMA_CLIENT.chat(
                    model=OLLAMA_MODEL,
                    messages=[{
                        "role": "user",
                        "content": ANALYSIS_PROMPT,
                        "images": [image_path],
                    }],
                    format="json",  # ask Ollama to return strict JSON
                    options={"temperature": 0},
                )
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise

        text = (response or {}).get("message", {}).get("content")
        data = _parse_json(text)
    except Exception as e:
        print(f"Ollama analysis failed: {e}", flush=True)
        msg = ("The AI model could not analyse this image right now. "
               "Please check your Ollama Cloud API key and connection, then try again.")
        return {"status": "error", "confidence": 0, "message": msg}

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


# Cache of the most recent analysis so re-viewing /display does not call Ollama again.
_last = {"filename": None, "result": None}


def analyze_and_cache(fname):
    """Analyse a saved file with Ollama and remember the result for /display."""
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    result = analyze_with_ollama(image_path)
    # Only cache meaningful results. A transient 'error' (rate limit / network)
    # is not cached, so re-opening /display will retry instead of showing a stale error.
    if result.get("status") != "error":
        _last["filename"] = fname
        _last["result"] = result
    return result


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
        result = analyze_and_cache(filename)  # fresh upload -> one Ollama call
        # Save every real diagnosis to the database (skip transient failures
        # like network/'error' or an unconfigured API key).
        if result.get("status") in ("ok", "not_leaf"):
            try:
                db.save_scan(filename, result, model_used=OLLAMA_MODEL)
            except Exception as e:
                print(f"Could not save scan to database: {e}", flush=True)
        return render_template('display.html', variable_name=filename, **result)
    else:
        flash("Allowed image types are - png, jpg, jpeg, webp.")
        return redirect('/input')


@app.route('/display')
def display_image():
    if not filename:
        flash("No image to display.")
        return redirect('/input')
    # Reuse the cached result for the current image instead of calling Ollama again.
    if _last["filename"] == filename and _last["result"] is not None:
        result = _last["result"]
    else:
        result = analyze_and_cache(filename)
    return render_template('display.html', variable_name=filename, **result)


@app.route('/history')
def history():
    """Show every saved diagnosis, newest first, with a small stats summary."""
    scans = db.get_all_scans()
    stats = db.get_stats()
    return render_template('history.html', scans=scans, stats=stats)


if __name__ == "__main__":
    app.run(debug=True)
