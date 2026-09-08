"""
Populate the scan history with SAMPLE data for demonstrations / screenshots.

This is a development helper only — the rows it inserts are illustrative
placeholders, NOT real AI diagnoses. Use it when you want to show the History
page and its statistics without having to run live scans (e.g. for report
screenshots), then clear it again when you are done.

Usage (from the project root):

    python seed_demo.py           # insert sample scans
    python seed_demo.py --clear   # delete ALL scans (real and sample)

The images referenced are taken from whatever files already exist in
static/shots/; if that folder is empty the script tells you to upload a photo
first so the thumbnails have something to show.
"""

import os
import sys

import db

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "static", "shots")

# Illustrative sample diagnoses (crop, disease, severity, healthy, confidence).
SAMPLE_DIAGNOSES = [
    ("Tomato", "Late blight", "Severe", False, 72),
    ("Tomato", "Early blight", "Moderate", False, 84),
    ("Maize (corn)", "Common rust", "Mild", False, 90),
    ("Maize (corn)", "Healthy", "None", True, 96),
    ("Grape", "Black rot", "Moderate", False, 80),
    ("Potato", "Late blight", "Severe", False, 68),
    ("Pepper", "Bacterial spot", "Mild", False, 77),
    ("Cassava", "Mosaic disease", "Moderate", False, 74),
    ("Grape", "Healthy", "None", True, 93),
    ("Apple", "Apple scab", "Mild", False, 81),
]


def seed():
    if not os.path.isdir(UPLOAD_FOLDER):
        print("static/shots/ does not exist yet. Run the app and upload at least "
              "one image first, then re-run this script.")
        return

    images = sorted(
        f for f in os.listdir(UPLOAD_FOLDER)
        if f.lower().rsplit(".", 1)[-1] in ("jpg", "jpeg", "png", "webp")
    )
    if not images:
        print("No images found in static/shots/. Upload at least one photo "
              "through the app first so the sample scans have a thumbnail.")
        return

    db.init_db()
    inserted = 0
    for i, (crop, disease, severity, healthy, confidence) in enumerate(SAMPLE_DIAGNOSES):
        image = images[i % len(images)]
        db.save_scan(
            image,
            {
                "status": "ok",
                "crop": crop,
                "disease": disease,
                "healthy": healthy,
                "confidence": confidence,
                "severity": severity,
                "summary": f"[SAMPLE] {disease} identified on the {crop.lower()} leaf.",
                "treatment": "Sample data — apply appropriate, label-directed management.",
                "symptoms": "Sample data — visible lesions on the leaf surface.",
                "prevention": "Sample data — rotate crops and avoid overhead watering.",
            },
            model_used="sample-data",
        )
        inserted += 1

    # One rejected (non-leaf) example so the 'Not a leaf' state is visible too.
    db.save_scan(
        images[0],
        {"status": "not_leaf",
         "message": "[SAMPLE] This does not look like a plant leaf."},
        model_used="sample-data",
    )
    inserted += 1

    print(f"Inserted {inserted} SAMPLE scans. Open /history to view them.")
    print("Run 'python seed_demo.py --clear' to remove all scans when finished.")


def clear():
    removed = db.clear_scans()
    print(f"Cleared {removed} scan(s) from the database.")


if __name__ == "__main__":
    if "--clear" in sys.argv:
        clear()
    else:
        seed()
