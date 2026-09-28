import joblib
import gradio as gr
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "models" / "sms_spam_model.pkl"

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        "The trained model was not found. Run the notebook first so it creates "
        "models/sms_spam_model.pkl."
    )

model = joblib.load(MODEL_PATH)

def predict_sms(message):
    if not message or not message.strip():
        return "Please enter an SMS message.", 0.0, 0.0

    prediction = int(model.predict([message])[0])
    probabilities = model.predict_proba([message])[0]

    ham_probability = float(probabilities[0])
    spam_probability = float(probabilities[1])

    if prediction == 1:
        result = "🚨 SPAM MESSAGE"
    else:
        result = "✅ HAM / LEGITIMATE MESSAGE"

    return result, ham_probability, spam_probability

with gr.Blocks(title="SMS Spam Detector") as demo:
    demo.css = """
    :root {
        --bg-main: #0b1020;
        --bg-panel: #111827;
        --bg-soft: #171f32;
        --card: rgba(17, 24, 39, 0.82);
        --card-2: rgba(30, 41, 59, 0.9);
        --line: rgba(148, 163, 184, 0.14);
        --text: #e5eefb;
        --muted: #a9b7cf;
        --primary: #8b5cf6;
        --primary-2: #3b82f6;
        --accent: #22c55e;
        --warning: #f59e0b;
        --shadow: 0 22px 60px rgba(15, 23, 42, 0.45);
    }
    .gradio-container {
        max-width: 1200px !important;
        background: radial-gradient(circle at top left, #141d35 0%, var(--bg-main) 35%, #0a0f1d 100%);
        color: var(--text);
        padding-top: 2rem !important;
    }
    .app-header {
        background: linear-gradient(135deg, rgba(139, 92, 246, 0.9), rgba(59, 130, 246, 0.85));
        border-radius: 26px;
        padding: 2rem 2.25rem;
        margin-bottom: 1.2rem;
        box-shadow: var(--shadow);
        border: 1px solid rgba(255, 255, 255, 0.12);
    }
    .app-header h1 {
        color: #ffffff !important;
        margin: 0 0 0.45rem 0;
        font-size: clamp(2.2rem, 3vw, 3.1rem);
        font-weight: 800;
        letter-spacing: -0.05em;
        line-height: 1.05;
    }
    .app-header p {
        margin: 0;
        color: rgba(255, 255, 255, 0.88);
        font-size: 1rem;
        font-weight: 600;
    }
    .panel {
        background: linear-gradient(180deg, rgba(17, 24, 39, 0.92), rgba(15, 23, 42, 0.82));
        border: 1px solid var(--line);
        border-radius: 22px;
        padding: 1.4rem;
        box-shadow: 0 18px 32px rgba(2, 6, 23, 0.28);
    }
    .panel .markdown {
        color: var(--text);
    }
    .submit-btn {
        background: linear-gradient(135deg, #8b5cf6 0%, #3b82f6 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        font-weight: 800 !important;
        letter-spacing: 0.02em;
        box-shadow: 0 18px 30px rgba(91, 33, 182, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    .submit-btn:hover {
        transform: translateY(-1px);
        filter: brightness(1.08);
    }
    .result-box {
        background: linear-gradient(180deg, rgba(15, 23, 42, 0.96), rgba(17, 24, 39, 0.9)) !important;
        border-radius: 14px !important;
        border: 1px solid rgba(148, 163, 184, 0.2) !important;
        color: var(--text) !important;
    }
    .result-box input {
        font-weight: 700 !important;
        color: var(--text) !important;
    }
    .info-note {
        color: var(--muted);
        font-size: 0.95rem;
        margin-top: 1rem;
        padding: 0.3rem 0.2rem;
    }
    .gradio-container .block {
        border-radius: 16px;
    }
    .gradio-container .button-primary {
        padding: 0.8rem 1.5rem;
    }
    .gradio-container textarea, .gradio-container input, .gradio-container .output-text {
        background: rgba(15, 23, 42, 0.9) !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        color: var(--text) !important;
    }
    """
    gr.HTML(
        """
        <div class="app-header">
            <h1>📱 SMS Spam Detector</h1>
            <p>Natural Language Processing Project</p>
        </div>
        """
    )

    with gr.Row(equal_height=True):
        with gr.Column(scale=2, elem_classes=["panel"]):
            gr.Markdown(
                "Enter an SMS message and the trained NLP model will classify it as "
                "**Spam** or **Ham (Not Spam)**.\n\n"
                "**Pipeline:** SMS → TF-IDF → Multinomial Naive Bayes → Prediction"
            )

            message = gr.Textbox(
                label="Enter an SMS message",
                placeholder="Example: Congratulations! You have won a free prize!",
                lines=5,
            )

            check_button = gr.Button("🔍 Check Message", elem_classes=["submit-btn"])

            gr.Examples(
                examples=[
                    ["Congratulations! You have won a free prize. Call now!"],
                    ["Hey, are we still meeting tomorrow?"],
                    ["URGENT! You have won $1000. Claim your reward now!"],
                    ["Can you send me the notes from today's class?"]
                ],
                inputs=message,
                label="Try an example",
            )

        with gr.Column(scale=1, elem_classes=["panel"]):
            prediction = gr.Textbox(
                label="Prediction",
                elem_classes=["result-box"],
            )
            ham_probability = gr.Number(
                label="Ham probability",
                elem_classes=["result-box"],
                precision=4,
            )
            spam_probability = gr.Number(
                label="Spam probability",
                elem_classes=["result-box"],
                precision=4,
            )

    check_button.click(
        predict_sms,
        inputs=message,
        outputs=[prediction, ham_probability, spam_probability],
    )

    gr.Markdown(
        "---\n"
        "**How the system works:** The model was trained in the notebook and "
        "the app loads the same `models/sms_spam_model.pkl` file.",
        elem_classes=["info-note"],
    )

if __name__ == "__main__":
    demo.launch()
