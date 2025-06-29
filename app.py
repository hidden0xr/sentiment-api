from flask import Flask, request, jsonify
from flask_cors import CORS

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from langdetect import detect
import torch

app = Flask(__name__)
CORS(app)

# تحميل موديلات اللغة العربية
arabic_model_name = "Ammar-alhaj-ali/arabic-MARBERT-sentiment"
arabic_tokenizer = AutoTokenizer.from_pretrained(arabic_model_name)
arabic_model = AutoModelForSequenceClassification.from_pretrained(arabic_model_name)
arabic_labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

# تحميل موديلات اللغة الإنجليزية
english_model_name = "distilbert-base-uncased-finetuned-sst-2-english"
english_tokenizer = AutoTokenizer.from_pretrained(english_model_name)
english_model = AutoModelForSequenceClassification.from_pretrained(english_model_name)
english_labels = {0: "Negative", 1: "Positive"}  # No Neutral in English model

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    text = data.get("text", "")

    # fallback لكشف اللغة في حال الجملة قصيرة
    if len(text.split()) <= 3 and all(ord(c) < 128 for c in text):
        lang = "en"
    else:
        try:
            lang = detect(text)
        except:
            return jsonify({"error": "Language detection failed"}), 400

    if lang == "ar":
        tokenizer = arabic_tokenizer
        model = arabic_model
        labels = arabic_labels
    elif lang == "en":
        tokenizer = english_tokenizer
        model = english_model
        labels = english_labels
    else:
        return jsonify({"error": "Unsupported language"}), 400

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        scores = torch.nn.functional.softmax(outputs.logits, dim=1)
        prediction = torch.argmax(scores, dim=1).item()
        sentiment = labels[prediction]

    # فلتر داخلي للكلمات السلبية (للعربية فقط)
    force_negative_keywords = [
        "متضايق", "زعلان", "طفشان", "مالي خلق", "مخنوق", "مهموم", "ضايق صدري", "محبط", "تعبان نفسياً"
    ]

    if lang == "ar" and sentiment == "Neutral":
        for word in force_negative_keywords:
            if word in text:
                sentiment = "Negative"
                break

    return jsonify({
        "language": lang,
        "sentiment": sentiment
    })

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)