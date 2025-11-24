from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# Try to import langformers
try:
    from langformers import tasks

    langformers_available = True
except ImportError:
    langformers_available = False
    tasks = None

# Initialize classifier if langformers is available
classifier = None
if langformers_available:
    try:
        classifier = tasks.load_classifier(r'C:\Users\Curtis\PycharmProjects\PhishingModels\langformers-classifier-d20251023-t200514\best_model')
    except Exception as e:
        classifier = None
        print(f"Error loading model: {e}")


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/classify', methods=['POST'])
def classify():
    if not langformers_available:
        return jsonify({
            "error": "The 'langformers' library is not installed. Please install it with 'pip install langformers'."
        }), 500

    if classifier is None:
        return jsonify({
            "error": "Classifier model failed to load. Check the model path or try retraining."
        }), 500

    data = request.get_json()
    sender = data.get("sender", "")
    subject = data.get("subject", "")
    body = data.get("body", "")

    combined_text = f"From: {sender}\nSubject: {subject}\n\n{body}"

    try:
        result = classifier.classify([combined_text])
    except Exception as e:
        return jsonify({"error": f"Error during classification: {str(e)}"}), 500

    # Parse model output
    if isinstance(result, list) and len(result) > 0 and isinstance(result[0], dict):
        label = result[0].get("label", "Unknown")
        confidence_score = result[0].get("prob", "N/A")
    else:
        label, confidence_score = "Unknown", "N/A"

    return jsonify({
        "label": label,
        "confidence": confidence_score
    })


if __name__ == '__main__':
    app.run(debug=True)
