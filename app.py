import os
from flask import Flask, render_template, request
from predict import predict_image

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

REASONS = {
    "Normal":
        "The ECG shows a well-defined P wave followed by a narrow QRS complex and a normal T wave. "
        "The P – QRS – T sequence is regular, indicating normal atrial and ventricular conduction.",

    "Myocardial Infarction":
        "The ECG exhibits abnormal Q waves and ST-segment deviations with altered T-wave morphology, "
        "indicating myocardial tissue damage and impaired ventricular depolarization.",

    "History of MI":
        "The ECG shows persistent pathological Q waves with relatively stabilized ST segments and T waves, "
        "suggesting residual structural changes due to a previous myocardial infarction.",

    "Abnormal Heartbeat":
        "The ECG demonstrates irregular P wave timing and inconsistent QRS complexes, "
        "indicating abnormal cardiac rhythm and conduction disturbances.",

    "Image not supported":
        "The uploaded image does not contain recognizable ECG waveform patterns (P, QRS, T)."
}

SUGGESTIONS = {
    "Normal":
        "Maintain a heart-healthy lifestyle with regular exercise.\n"
        "Continue periodic check-ups with a healthcare provider.\n"
        "Monitor for new symptoms such as chest pain or palpitations.",

    "Myocardial Infarction":
        "Seek immediate medical attention if experiencing chest pain or shortness of breath.\n"
        "Avoid heavy physical exertion until clinically evaluated.\n"
        "Ensure timely access to emergency medical services and follow treatment guidance.",

    "History of MI":
        "Attend regular follow-ups with a cardiologist.\n"
        "Maintain a balanced, heart-healthy lifestyle (low sodium, low fat, moderate activity).\n"
        "Track symptoms like chest tightness, fatigue, or reduced exercise tolerance.\n"
        "Discuss preventive strategies such as stress control and medication adherence.",

    "Abnormal Heartbeat":
        "Keep a diary of irregular heartbeat episodes and associated symptoms.\n"
        "Reduce stimulants (caffeine, energy drinks, alcohol) if sensitive.\n"
        "Consult a clinician if irregular rhythms occur often or worsen.\n"
        "Consider a Holter monitor for extended cardiac rhythm evaluation.",

    "Image not supported":
        "Upload a clear ECG image with visible waveforms.\n"
        "Ensure the ECG is not blurry, rotated, or cropped.\n"
        "Use a higher-resolution image for better interpretation."
}

@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    reason = None
    suggestions = None

    if request.method == "POST":
        file = request.files.get("image")
        if file and file.filename:
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)
            try:
                result = predict_image(path)
                reason = REASONS.get(result, "No reason available.")
                suggestions = SUGGESTIONS.get(result, "No suggestions available.")
            except:
                result = "Image not supported"
                reason = REASONS[result]
                suggestions = SUGGESTIONS[result]

    return render_template("index.html", result=result, reason=reason, suggestions=suggestions)

if __name__ == "__main__":
    app.run(debug=True)
