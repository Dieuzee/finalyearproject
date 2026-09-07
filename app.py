import tensorflow as tf
from flask import Flask, render_template, request, Response, flash, redirect
import cv2
import os

# Load environment variables from a local .env file if present (optional).
# This is where GEMINI_API_KEY lives so the Gemini leaf-check / guidance work.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass
import google.genai as genai
from google.genai import types
from werkzeug.utils import secure_filename
from random import randint
from tensorflow.keras.models import load_model
import numpy as np

# Globals
global capture, switch, filename
capture = 0
switch = 0
filename = ""

# Flask App Setup
app = Flask(__name__)
UPLOAD_FOLDER = 'static/shots'
app.secret_key = 'cropdisease'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure shots folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Camera - only initialize if not in production (Render has no webcam)
# Camera disabled
camera = None

# Random name for captured image
variable_name = str(randint(0, 100))
size = len(variable_name)

def allowed_file(fname):
    return '.' in fname and fname.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_frames():
    global capture
    if camera is None:
        return
    while True:
        success, frame = camera.read()
        if success:
            if capture:
                capture = 0
                p = os.path.sep.join([UPLOAD_FOLDER, f"{variable_name}.png"])
                cv2.imwrite(p, frame)
            ret, buffer = cv2.imencode('.jpg', cv2.flip(frame, 1))
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/input", methods=['GET', 'POST'])
def input():
    return render_template("input.html")

@app.route('/video')
def video():
    if camera is None:
        # Return a 1x1 transparent pixel instead of hanging
        return Response(b'', status=204)

@app.route('/requests', methods=['POST', 'GET'])
def tasks():
    global switch, camera, capture
    if request.method == 'POST':
        if request.form.get('click') == 'Capture Image':
            capture = 1
            # After capture, jump to display using the camera file
            return redirect("/upload")  # handled as GET below
        elif request.form.get('stop') == 'Stop/Start':
            if switch == 1:
                switch = 0
                if camera is not None:
                    camera.release()
                cv2.destroyAllWindows()
            else:
                camera = cv2.VideoCapture(0)
                switch = 1
        return redirect("/upload")
    return render_template('display.html')

# ------------------------------
# Model and Classes
# ------------------------------
MODEL_PATH = "best_model.h5"
CONFIDENCE_THRESHOLD = 0.40  # 40%
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

try:
    GEMINI_CLIENT = genai.Client(api_key=os.getenv("GEMINI_API_KEY")) if os.getenv("GEMINI_API_KEY") else None
except Exception as e:
    GEMINI_CLIENT = None
    print(f"Gemini is unavailable: {e}")

NEW_CLASS_NAMES = [
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

TREATMENT_DICT = {
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Use fungicides like mancozeb and rotate crops.",
    "Corn_(maize)___Common_rust_": "Apply fungicides at early infection stage and plant resistant varieties.",
    "Corn_(maize)___Northern_Leaf_Blight": "Use resistant hybrids and apply fungicides.",
    "Corn_(maize)___healthy": "No disease detected. Continue good practices.",
    "Grape___Black_rot": "Remove infected fruits, prune vines, and apply fungicides.",
    "Grape___Esca_(Black_Measles)": "Prune infected wood and avoid water stress.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Apply fungicides and prune infected leaves.",
    "Grape___healthy": "No disease detected. Maintain good vineyard hygiene.",
    "Tomato___Bacterial_spot": "Use copper-based sprays and remove infected leaves.",
    "Tomato___Early_blight": "Apply fungicides and rotate crops annually.",
    "Tomato___Late_blight": "Remove infected plants and apply fungicides promptly.",
    "Tomato___Leaf_Mold": "Increase ventilation and apply fungicides.",
    "Tomato___Septoria_leaf_spot": "Remove affected leaves and use fungicides.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Use miticides or neem oil.",
    "Tomato___Target_Spot": "Apply fungicides and ensure proper plant spacing.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Use insecticides to control whiteflies, remove infected plants, and use resistant varieties."
}

def _infer_model_img_size(model, fallback=(224, 224)):
    """
    Try to read (H, W) from the loaded model.
    Fallback to 224x224 (matches the 36,864 Flatten -> 12x12x256 signature).
    """
    try:
        ishape = model.input_shape
        if isinstance(ishape, (list, tuple)):
            if isinstance(ishape[0], (list, tuple)):
                ishape = ishape[0]
        h, w = ishape[1], ishape[2]
        if isinstance(h, int) and isinstance(w, int):
            return (h, w)
    except Exception:
        pass
    return fallback

try:
    MODEL = load_model(MODEL_PATH)
    MODEL_IMG_SIZE = _infer_model_img_size(MODEL, fallback=(224, 224))
    print(f"Loaded model: {MODEL_PATH}")
    print(f"Inferred model input size: {MODEL_IMG_SIZE}")
except Exception as e:
    MODEL = None
    MODEL_IMG_SIZE = (224, 224)
    print(f"Error loading model: {e}")

def _safe_softmax(x):
    # If last layer already softmax, values will sum ~1 in [0,1]; otherwise apply softmax.
    x = np.asarray(x).astype("float32")
    s = x.sum()
    if np.any(x > 1.0001) or s <= 0.0 or s > 1.0001:
        return tf.nn.softmax(x).numpy()
    return x

def humanize_label(label):
    """Turn a raw class label like 'Tomato___Late_blight' into ('Tomato', 'Late blight')."""
    crop, _, disease = label.partition("___")
    crop = crop.replace("_", " ").strip()
    disease = disease.replace("_", " ").strip()
    disease = " ".join(disease.split())  # collapse repeated spaces
    if disease.lower() == "healthy":
        disease = "Healthy"
    return crop, disease


def is_leaf_image(image_path, min_plant_ratio=0.12):
    """
    Coarse check for whether an image plausibly shows a crop leaf, using HSV colour
    distribution. This is a lightweight filter, NOT a guarantee: a heavily browned
    leaf can score low and a green non-leaf (e.g. a lawn) can score high. It exists
    to reject obviously non-plant uploads (icons, documents, screenshots, portraits).
    Returns (is_leaf: bool, plant_ratio: float, reason: str)
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return False, 0.0, "Could not read the image file."

        # Resize for faster processing
        h, w = img.shape[:2]
        if max(h, w) > 512:
            scale = 512 / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        total_pixels = hsv.shape[0] * hsv.shape[1]

        # Leaf-specific hues in OpenCV HSV (H is 0-179):
        # true leaf green ~ H 30-85, and yellow/brown diseased tissue ~ H 15-30.
        # Narrower than before so cyan/teal UI colours no longer count as "plant".
        green_mask = cv2.inRange(hsv, (30, 40, 40), (85, 255, 255))
        yellow_brown_mask = cv2.inRange(hsv, (15, 40, 40), (30, 255, 255))

        plant_mask = green_mask | yellow_brown_mask
        plant_ratio = cv2.countNonZero(plant_mask) / total_pixels

        if plant_ratio >= min_plant_ratio:
            return True, plant_ratio, f"Plant content detected ({plant_ratio:.0%})."

        return False, plant_ratio, (
            "This doesn't look like a crop leaf. Please upload a clear, well-lit "
            "close-up of a single corn, grape, or tomato leaf."
        )
    except Exception as e:
        # If validation fails, allow the image through (fail-open) rather than block a real leaf.
        return True, 0.0, f"Validation skipped: {e}"


def gemini_is_crop_leaf(image_path):
    """
    Use Gemini (a real vision model) to decide whether the image is a crop leaf.
    This is the reliable gate: it rejects people, objects, food, logos, screenshots,
    landscapes, etc. Returns True (accept), False (reject), or None when Gemini is
    unavailable or errors out (caller then falls back to the local colour check).
    """
    if GEMINI_CLIENT is None:
        return None
    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        prompt = (
            "You are the input validator for a crop-disease tool that only supports "
            "corn (maize), grape, and tomato leaves. Reply with EXACTLY one word: "
            "ACCEPT or REJECT.\n"
            "Reply ACCEPT if the image is a clear, close-up photo of a plant leaf "
            "(ideally corn, grape, or tomato) where the leaf is the main subject.\n"
            "Reply REJECT if the main subject is NOT a single plant leaf - for example "
            "a person, animal, hand, object, food dish, packaging, logo, icon, "
            "screenshot, drawing, document, building, or wide landscape/field."
        )
        response = GEMINI_CLIENT.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt,
            ],
        )
        text = (response.text or "").strip().upper()
        if "ACCEPT" in text:
            return True
        if "REJECT" in text:
            return False
        return None
    except Exception as e:
        print(f"Gemini leaf check failed: {e}")
        return None


def passes_leaf_gate(image_path):
    """
    Decide whether an image should be analysed by the disease model.
    Prefers Gemini's vision judgement; falls back to the local colour heuristic
    when Gemini is unavailable. Returns (ok: bool, message: str).
    """
    verdict = gemini_is_crop_leaf(image_path)
    if verdict is True:
        return True, ""
    if verdict is False:
        return False, (
            "This doesn't look like a crop leaf. Please upload a clear, well-lit "
            "close-up of a single corn, grape, or tomato leaf."
        )
    # Gemini unavailable -> local colour fallback
    is_leaf, _ratio, reason = is_leaf_image(image_path)
    return is_leaf, ("" if is_leaf else reason)


def processing(fname):
    """
    Analyse an uploaded image and return a structured result dict:
        status: 'ok' | 'not_leaf' | 'uncertain' | 'error'
        label:  raw class label (only for 'ok')
        crop, disease: human-readable pieces (only for 'ok')
        confidence: 0-100 float
        treatment:  treatment text (only for 'ok')
        message:    user-facing explanation for non-'ok' states
    """
    global MODEL
    if MODEL is None:
        return {"status": "error", "confidence": 0.0,
                "message": "The detection model could not be loaded on the server. "
                           "Please contact the administrator."}

    image_path = os.path.join(UPLOAD_FOLDER, fname)
    if not os.path.exists(image_path):
        return {"status": "error", "confidence": 0.0,
                "message": "The uploaded image could not be found. Please try again."}

    # Step 1: leaf gate (Gemini vision if available, else local colour heuristic)
    ok, reason = passes_leaf_gate(image_path)
    if not ok:
        return {"status": "not_leaf", "confidence": 0.0, "message": reason}

    # Step 2: run the model
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=MODEL_IMG_SIZE)
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.expand_dims(input_arr, axis=0).astype("float32") / 255.0

    preds = MODEL.predict(input_arr, verbose=0)
    probs = _safe_softmax(preds[0])
    idx = int(np.argmax(probs))
    label = NEW_CLASS_NAMES[idx] if idx < len(NEW_CLASS_NAMES) else f"Class {idx}"
    confidence = round(float(probs[idx]) * 100.0, 2)

    if confidence < (CONFIDENCE_THRESHOLD * 100.0):
        return {"status": "uncertain", "confidence": confidence,
                "message": "The model is not confident about this image. It may not be a "
                           "corn, grape, or tomato leaf, or the photo may be blurry. "
                           "Try a clearer, close-up photo of a single leaf."}

    crop, disease = humanize_label(label)
    return {
        "status": "ok",
        "label": label,
        "crop": crop,
        "disease": disease,
        "confidence": confidence,
        "treatment": TREATMENT_DICT.get(label, "No treatment information available."),
    }

def generate_gemini_guidance(image_path, result):
    """Ask Gemini for practical context. Optional: only runs for a confident result."""
    if GEMINI_CLIENT is None or result.get("status") != "ok":
        return None

    try:
        with open(image_path, "rb") as image_file:
            image_bytes = image_file.read()

        prompt = (
            "You are an agricultural assistant. Review the crop-leaf image and the local model result below. "
            "Return concise, plain-text guidance with exactly these headings: Assessment, Immediate steps, "
            "Prevention, When to seek expert help. Do not claim certainty, do not recommend unsafe chemical use, "
            "and advise following the product label and local agricultural guidance.\n\n"
            f"Local model result: {result.get('label')}\n"
            f"Confidence: {result.get('confidence', 0):.2f}%\n"
            f"Existing treatment suggestion: {result.get('treatment')}"
        )
        response = GEMINI_CLIENT.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt,
            ],
        )
        return response.text.strip() if response.text else None
    except Exception as e:
        print(f"Gemini guidance failed: {e}")
        return None

# ---------- Upload / Display ----------
def render_result(fname):
    """Run analysis on a saved file and render the results page."""
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    result = processing(fname)
    gemini_guidance = generate_gemini_guidance(image_path, result)
    return render_template('display.html',
                           variable_name=fname,
                           status=result.get("status"),
                           label=result.get("label"),
                           crop=result.get("crop"),
                           disease=result.get("disease"),
                           confidence=result.get("confidence", 0.0),
                           treatment=result.get("treatment"),
                           message=result.get("message"),
                           gemini_guidance=gemini_guidance)


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """
    GET  -> show last camera capture (variable_name.png) if present
    POST -> handle file upload
    """
    global filename

    if request.method == 'GET':
        # Camera path
        cam_file = f"{variable_name}.png"
        cam_path = os.path.join(app.config['UPLOAD_FOLDER'], cam_file)
        if os.path.exists(cam_path):
            filename = cam_file
            return render_result(filename)
        # no camera file -> back to input
        flash("No captured image found. Please upload an image.")
        return redirect('/input')

    # POST (file upload)
    if 'file' not in request.files:
        flash("No file part")
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash("No image selected for uploading")
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        return render_result(filename)
    else:
        flash("Allowed image types are - png, jpg, jpeg, gif.")
        return redirect('/input')

@app.route('/display')
def display_image():
    # Use the last known filename
    if not filename:
        flash("No image to display.")
        return redirect('/input')
    return render_result(filename)

if __name__ == "__main__":
    app.run(debug=True)