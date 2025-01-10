import os
import re
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.naive_bayes import MultinomialNB
import PyPDF2
import docx
# Import necessary libraries for handling files, text preprocessing, model training, Flask API, and file formats.


# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing
app.config["UPLOAD_FOLDER"] = "./uploads"  # Directory to store uploaded files

# Categories and training data
categories = ["Mathematics", "Science", "Literature", "History", "Biology", "Technology", "Geography", "Fungi", "Philosophy", "Astronomy", "Medicine", "Engineering", "Economics", "Psychology", "Sociology", "Architecture", "Music", "Art", "Linguistics", "Political Science"]
training_data = {
    "Mathematics": ["algebra", "geometry", "calculus", "equations", "matrices", "integration"],
    "Science": ["physics", "chemistry", "laws of motion", "quantum mechanics", "thermodynamics"],
    "Literature": ["novel", "poetry", "sonnet", "fiction", "drama", "prose"],
    "History": ["ancient empires", "medieval history", "renaissance", "war chronicles", "dynasties"],
    "Biology": ["cells", "genetics", "evolution", "organisms", "photosynthesis"],
    "Technology": ["artificial intelligence", "blockchain", "software development", "data science", "IoT"],
    "Geography": ["continents", "oceans", "climate", "topography", "earthquakes"],
    "Fungi": ["mycology", "mushrooms", "fungal spores", "decomposition", "symbiosis"],
    "Philosophy": ["ethics", "metaphysics", "epistemology", "logic", "aesthetics"],
    "Astronomy": ["stars", "planets", "galaxies", "black holes", "cosmology"],
    "Medicine": ["anatomy", "pharmacology", "surgery", "diseases", "diagnosis"],
    "Engineering": ["mechanical", "civil", "electrical", "software", "chemical"],
    "Economics": ["supply", "demand", "markets", "trade", "finance"],
    "Psychology": ["cognition", "behavior", "mental health", "therapy", "emotion"],
    "Sociology": ["society", "culture", "social norms", "demographics", "institutions"],
    "Architecture": ["design", "buildings", "urban planning", "landscaping", "construction"],
    "Music": ["composition", "instrument", "melody", "harmony", "rhythm"],
    "Art": ["painting", "sculpture", "drawing", "photography", "ceramics"],
    "Linguistics": ["syntax", "semantics", "phonology", "morphology", "pragmatics"],
    "Political Science": ["governance", "policy", "elections", "diplomacy", "law"]
}

# Preprocess text
stop_words = set(stopwords.words('english')).union(ENGLISH_STOP_WORDS)
lemmatizer = WordNetLemmatizer()

def preprocess_text(text, vocabulary):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower()  # Convert to lowercase
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    words = [word for word in words if word in vocabulary]  # Keep only words in vocabulary
    return ' '.join(words)

# Prepare training data
X_train = []
y_train = []
for category, keywords in training_data.items():
    for keyword in keywords:
        X_train.append(keyword)
        y_train.append(category)

# Train the model
vectorizer = TfidfVectorizer(preprocessor=None)  # Preprocessing is done manually
model = MultinomialNB()
X_train_vec = vectorizer.fit_transform(X_train)
model.fit(X_train_vec, y_train)

# Extract vocabulary
vocabulary = set(vectorizer.get_feature_names_out())

# Helper functions to extract text from files
def extract_text_from_pdf(filepath):
    try:
        with open(filepath, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            text = ''
            for page in reader.pages:
                text += page.extract_text() + '\n'
        return text
    except Exception as e:
        raise ValueError(f"Error extracting text from PDF: {e}")

def extract_text_from_docx(filepath):
    try:
        doc = docx.Document(filepath)
        text = ''
        for paragraph in doc.paragraphs:
            text += paragraph.text + '\n'
        return text
    except Exception as e:
        raise ValueError(f"Error extracting text from DOCX: {e}")

def extract_text(filepath):
    ext = os.path.splitext(filepath)[-1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(filepath)
    elif ext == '.docx':
        return extract_text_from_docx(filepath)
    elif ext == '.txt':
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            raise ValueError(f"Error reading TXT file: {e}")
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def calculate_category_confidence(content):
    word_counts = {word: content.split().count(word) for word in content.split()}
    category_scores = {
        category: sum(word_counts[word] for word in keywords if word in word_counts)
        for category, keywords in training_data.items()
    }
    best_category = max(category_scores, key=category_scores.get)
    confidence = (
        (category_scores[best_category] / sum(category_scores.values())) * 100
        if sum(category_scores.values()) > 0
        else 0
    )
    return best_category, round(confidence, 2)

# Classification endpoint
@app.route('/classify', methods=['POST'])
def classify():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    try:
        # Extract text content from the file
        content = extract_text(filepath)
        content = preprocess_text(content, vocabulary)

        # Predict category and confidence
        best_category, confidence = calculate_category_confidence(content)

        # Log classification result
        log_entry = {"filename": file.filename, "category": best_category, "confidence": confidence}
        with open("classification_log.json", "a") as log:
            log.write(json.dumps(log_entry) + "\n")

        return jsonify({"filename": file.filename, "category": best_category, "confidence": confidence, "filtered_text": content})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up the uploaded file
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == "__main__":
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])
    app.run(debug=True)