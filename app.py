from flask import Flask, request, render_template, send_file
import cv2, numpy as np, os, uuid
from tensorflow.keras.models import load_model
from datetime import datetime
from generate_pdf import generate_pdf

# --------------------------------------------------
# APP SETUP
# --------------------------------------------------
app = Flask(__name__)

UPLOAD_DIR = "uploads"
REPORT_DIR = "reports"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
ultrasound_model = load_model("models/ultrasound_unet.h5")
endoscopy_model  = load_model("models/endoscopy_unet.h5")

IMG_SIZE = 224

# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def calculate_symptom_score(form):
    score = 0

    yes_fields = [
        "pelvic_pain", "menstrual_pain",
        "lower_back_pain", "activity_related_pain"
    ]
    for f in yes_fields:
        if form.get(f) == "yes":
            score += 1

    if form.get("pain_frequency") in ["daily", "constant"]:
        score += 2

    if form.get("cycle_irregularity") in ["frequently irregular", "very irregular"]:
        score += 2

    if form.get("menstrual_flow") in ["heavy", "very heavy"]:
        score += 1

    if form.get("bleeding_duration") in ["8-10 days", "more than 10 days"]:
        score += 1

    try:
        if int(form.get("fatigue", 0)) >= 6:
            score += 1
        if int(form.get("digestive_severity", 0)) >= 6:
            score += 1
    except:
        pass

    if form.get("daily_impact") in ["significant impact", "severe impact"]:
        score += 2

    if "yes" in form.get("family_history", ""):
        score += 1

    if "yes" in form.get("previous_diagnosis", ""):
        score += 2

    score += len(form.getlist("pain_triggers"))
    score += len(form.getlist("digestive_symptoms"))

    return score

# --------------------------------------------------
# ULTRASOUND ANALYSIS (TRAINED MODEL)
# --------------------------------------------------
def analyze_ultrasound(path):
    img = preprocess_image(path)
    pred = ultrasound_model.predict(img)[0]

    lesion_pixels = np.sum(pred > 0.5)
    total_pixels = pred.shape[0] * pred.shape[1]
    lesion_ratio = lesion_pixels / total_pixels

    confidence = round(lesion_ratio * 100, 2)

    if lesion_ratio >= 0.05:
        return "Ultrasound abnormality detected", confidence, True
    else:
        return "Ultrasound appears normal", confidence, False

# --------------------------------------------------
# ENDOSCOPY ANALYSIS (TRAINED MODEL – KVASIR-SEG)
# --------------------------------------------------
def analyze_endoscopy(path):
    img = preprocess_image(path)
    pred = endoscopy_model.predict(img)[0]

    lesion_pixels = np.sum(pred > 0.5)
    total_pixels = pred.shape[0] * pred.shape[1]
    lesion_ratio = lesion_pixels / total_pixels

    confidence = round(lesion_ratio * 100, 2)

    if lesion_ratio >= 0.05:
        return "Endoscopic lesion detected", confidence, True
    else:
        return "No significant endoscopic lesion detected", confidence, False

# --------------------------------------------------
# ROUTES
# --------------------------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("symptoms.html")

@app.route("/symptoms", methods=["POST"])
def symptoms():
    symptom_score = calculate_symptom_score(request.form)

    symptom_risk = (
        "Low symptom-based risk"
        if symptom_score < 6 else
        "Elevated symptom-based risk"
    )

    return render_template(
        "upload_images.html",
        symptom_score=symptom_score,
        symptom_risk=symptom_risk
    )

@app.route("/upload-images", methods=["POST"])
def upload_images():
    symptom_score = int(request.form.get("symptom_score"))
    total_score = symptom_score

    findings = []
    confidences = []

    # ---------------- ULTRASOUND ----------------
    us_file = request.files.get("ultrasound")
    if us_file and us_file.filename:
        us_path = os.path.join(UPLOAD_DIR, us_file.filename)
        us_file.save(us_path)

        finding, conf, abnormal = analyze_ultrasound(us_path)
        findings.append(finding)
        confidences.append(f"Ultrasound confidence: {conf}%")

        if abnormal:
            total_score += 3

    # ---------------- ENDOSCOPY ----------------
    endo_file = request.files.get("endoscopy")
    if endo_file and endo_file.filename:
        endo_path = os.path.join(UPLOAD_DIR, endo_file.filename)
        endo_file.save(endo_path)

        finding, conf, abnormal = analyze_endoscopy(endo_path)
        findings.append(finding)
        confidences.append(f"Endoscopy confidence: {conf}%")

        if abnormal:
            total_score += 3

    # ---------------- FINAL RISK ----------------
    risk_level = (
        "High Risk – Immediate Clinical Evaluation Recommended"
        if total_score >= 12 else
        "Moderate Risk – Medical Consultation Advised"
        if total_score >= 6 else
        "Low Risk"
    )

    report_id = f"EHS-{uuid.uuid4().hex[:6]}"

    data = {
        "report_id": report_id,
        "date": datetime.now().strftime("%d %B %Y"),
        "risk_level": risk_level,
        "risk_score": total_score,
        "findings": findings,
        "confidences": confidences
    }

    pdf_path = f"{REPORT_DIR}/{report_id}.pdf"
    generate_pdf(pdf_path, data)

    return render_template(
        "report.html",
        data=data,
        pdf_file=pdf_path
    )

@app.route("/download-report")
def download_report():
    file = request.args.get("file")
    return send_file(file, as_attachment=True)

# --------------------------------------------------
# RUN
# --------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
