# SMS Spam Detection Using NLP

## Assignment
Build an NLP text-classification system that predicts whether an SMS is Spam or Ham.

## Dataset
UCI SMS Spam Collection (Dataset ID 228).
Dataset placeholder included; official dataset must be added locally.

## Project structure
- `SMS_Spam_Detection.ipynb` — complete step-by-step notebook.
- `app.py` — Gradio website.
- `models/sms_spam_model.pkl` — trained model used by the website.
- `data/SMSSpamCollection` — dataset.
- `presentation/PRESENTATION_NOTES.txt` — 5–7 minute presentation guide.
- `REPORT.md` — report template.
- `requirements.txt` — required Python packages.

## How the notebook and website are connected
1. Run the notebook.
2. The notebook trains the final model.
3. The notebook saves it as `models/sms_spam_model.pkl`.
4. `app.py` loads that exact file.
5. The website therefore uses the model trained in the notebook.

## Run it
Open a terminal in this project folder and run:

    python -m pip install -r requirements.txt

Then start Jupyter:

    jupyter notebook

Open `SMS_Spam_Detection.ipynb` and run all cells from top to bottom.

After the notebook successfully creates `models/sms_spam_model.pkl`, run:

    python app.py

Open the local Gradio URL shown in the terminal, normally:

    http://127.0.0.1:7860

If the browser does not open automatically, copy that address into your browser.
