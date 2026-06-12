from flask import Flask, render_template, request, redirect, url_for, jsonify, session, flash
from functools import wraps
from typing import Any
from werkzeug.security import generate_password_hash, check_password_hash
from cryptography.fernet import Fernet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import mysql.connector
from mysql.connector import Error as MySQLError
import re
from datetime import datetime
import os
import json

app = Flask(__name__)
app.secret_key = "super_secret_key_for_sessions"

# ─────────────────────────────────────────────────────────────
#  SECURITY: ENCRYPTION
# ─────────────────────────────────────────────────────────────
# Using a fixed key for the prototype. In production, load from ENV.
ENCRYPTION_KEY = b'wL2U1w_P19T-_YcZJk9J4F7kK9U7X2E2_Y_w8R5jXj8='
cipher_suite = Fernet(ENCRYPTION_KEY)

def encrypt_data(text):
    if not text: return text
    return cipher_suite.encrypt(text.encode('utf-8')).decode('utf-8')

def decrypt_data(text):
    if not text: return text
    try:
        return cipher_suite.decrypt(text.encode('utf-8')).decode('utf-8')
    except Exception:
        return text  # Fallback for plain text rows

# ─────────────────────────────────────────────────────────────
#  SECURITY: AUTHENTICATION
# ─────────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


# ─────────────────────────────────────────────────────────────
#  DATABASE CONNECTION  (MySQL / XAMPP)
# ─────────────────────────────────────────────────────────────
DB_CONFIG = {
    "host":       "127.0.0.1",
    "user":       "root",
    "password":   "",
    "database":   "defaultdb",
    "port":       3306,
    "ssl_ca":     None,
    "autocommit": False,
}

# Override config from environment variables (standard for Render cloud deployment)
for key, env_var in [("host", "DB_HOST"), ("user", "DB_USER"), ("password", "DB_PASSWORD"), ("database", "DB_NAME"), ("ssl_ca", "DB_SSL_CA")]:
    val = os.environ.get(env_var)
    if val is not None:
        DB_CONFIG[key] = val

if os.environ.get("DB_PORT"):
    try:
        DB_CONFIG["port"] = int(os.environ.get("DB_PORT"))
    except ValueError:
        pass

# Fallback/Override config from local file if it exists (for local development, NOT uploaded to GitHub)
local_config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "db_config.local.json")
if os.path.exists(local_config_path):
    try:
        with open(local_config_path, "r") as f:
            local_config = json.load(f)
            for k, v in local_config.items():
                if k in DB_CONFIG:
                    DB_CONFIG[k] = v
    except Exception as e:
        print(f"Error loading local config: {e}")

def get_db():
    """Return a live MySQL connection, creating one if needed."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        if conn.is_connected():
            return conn
    except MySQLError as e:
        raise RuntimeError(f"[ERROR] MySQL connection failed: {e}") from e

def get_cursor(conn):
    """Return a dict cursor for the given connection."""
    return conn.cursor()

def init_db():
    try:
        # Connect directly to the database and create tables
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        try:
            cursor.execute("ALTER TABLE sessions ADD COLUMN username VARCHAR(50)")
            conn.commit()
        except MySQLError:
            pass
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                username VARCHAR(50) UNIQUE,
                password VARCHAR(255)
            )
        """)
        # Insert default admin user if not exists
        cursor.execute("SELECT * FROM users WHERE username='admin'")
        if not cursor.fetchone():
            cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", 
                           ('admin', generate_password_hash('password')))
                           
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS adaptive_learning (
                id INT AUTO_INCREMENT PRIMARY KEY,
                symptoms TEXT,
                condition_label VARCHAR(100),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INT AUTO_INCREMENT PRIMARY KEY,
                session_id INT,
                user_text TEXT,
                structured TEXT,
                summary TEXT,
                response TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
               )
        """)
        conn.commit()
        cursor.close()
        conn.close()
        print(f"[OK] Database '{DB_CONFIG['database']}' and tables initialized successfully on Aiven.")
    except MySQLError as e:
        print(f"[ERROR] Database initialization failed: {e}")
        print("[WARNING] Check your Aiven cloud database connection credentials.")

# Initialize the database on startup
init_db()

# ─────────────────────────────────────────────────────────────
#  MACHINE LEARNING MODEL  (Multinomial Naïve Bayes)
# ─────────────────────────────────────────────────────────────
training_data = [
    # Malaria
    ("fever headache chills sweating body pain", "Malaria"),
    ("high fever shivering chills headache fatigue", "Malaria"),
    ("fever nausea vomiting chills", "Malaria"),
    # COVID-19
    ("cough fever breathing difficulty loss of smell", "COVID-19"),
    ("dry cough fever shortness of breath fatigue", "COVID-19"),
    ("fever cough sore throat breathing", "COVID-19"),
    # Diabetes
    ("frequent urination fatigue blurred vision", "Diabetes"),
    ("excessive thirst frequent urination weight loss", "Diabetes"),
    ("fatigue hunger frequent urination", "Diabetes"),
    # Heart Disease
    ("chest pain shortness of breath dizziness", "Heart Disease"),
    ("chest tightness pain left arm fatigue", "Heart Disease"),
    ("palpitations chest pain sweating shortness breath", "Heart Disease"),
    # Typhoid
    ("fever body pain weakness diarrhoea", "Typhoid"),
    ("high fever abdominal pain weakness nausea", "Typhoid"),
    ("prolonged fever headache weakness loss appetite", "Typhoid"),
    # Tuberculosis
    ("persistent cough blood sputum night sweats", "Tuberculosis"),
    ("cough chest pain weight loss fatigue", "Tuberculosis"),
    ("chronic cough fever night sweats", "Tuberculosis"),
    # Hypertension
    ("severe headache dizziness blurred vision", "Hypertension"),
    ("headache nosebleed fatigue chest pain", "Hypertension"),
    # Asthma
    ("wheezing shortness breath chest tightness cough", "Asthma"),
    ("breathing difficulty cough wheezing", "Asthma"),
    # Flu / Influenza
    ("fever chills muscle aches fatigue sore throat", "Influenza"),
    ("runny nose cough fever body aches", "Influenza"),
    # Migraine
    ("severe headache nausea sensitivity to light blurred vision", "Migraine"),
    ("throbbing headache nausea vomiting dizziness", "Migraine"),
    # Gastroenteritis
    ("vomiting diarrhoea abdominal pain fever nausea", "Gastroenteritis"),
    ("stomach pain diarrhoea nausea vomiting chills", "Gastroenteritis"),
    # Pneumonia
    ("fever cough breathing difficulty chest pain chills", "Pneumonia"),
    ("persistent cough shortness of breath chest pain fatigue", "Pneumonia"),
    # Anemia
    ("fatigue weakness pale skin dizziness shortness of breath", "Anemia"),
    ("extreme fatigue weakness cold hands blurred vision", "Anemia"),
    # Peptic Ulcer
    ("abdominal pain heartburn nausea vomiting loss of appetite", "Peptic Ulcer"),
    ("stomach pain burning sensation heartburn nausea", "Peptic Ulcer"),
]

# ─────────────────────────────────────────────────────────────
#  HELPER — NLP TEXT PROCESSING  (Forward Flow)
# ─────────────────────────────────────────────────────────────
CONDITION_LIST = [
    "malaria", "diabetes", "cancer", "typhoid", "covid-19", "covid",
    "influenza", "flu", "hypertension", "asthma", "tuberculosis", "tb",
    "stroke", "arthritis", "hepatitis", "ulcer", "infection", "anemia",
    "pneumonia", "dengue", "cholera", "heart disease", "migraine", 
    "gastroenteritis", "peptic ulcer",
]

SYMPTOM_LIST = [
    "fever", "headache", "pain", "cough", "fatigue", "vomiting",
    "nausea", "dizziness", "weakness", "chills", "sweating",
    "breathing difficulty", "shortness of breath", "chest pain",
    "frequent urination", "weight loss", "blurred vision", "wheezing",
    "loss of smell", "sore throat", "diarrhoea", "rash", "palpitations",
    "night sweats", "loss of appetite", "muscle aches",
    "sensitivity to light", "abdominal pain", "pale skin", "heartburn", 
    "stomach pain", "joint pain", "muscle weakness", "burning sensation",
    "cold hands",
]

vectorizer: Any = None
model: Any = None

def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

def is_close_match(word, target):
    if word == target:
        return True
    if len(target) <= 3:
        return False
    allowed_dist = 1 if len(target) <= 6 else 2
    return levenshtein_distance(word, target) <= allowed_dist

def extract_age_group(text):
    text_lower = text.lower()
    age_patterns = [
        r"(?:age[d]?[:\s]+|i am\s+|i'm\s+|am\s+)(\d{1,3})\s*(?:years?|yrs?)?",
        r"\b(\d{1,3})\s*(?:years?\s*old|yrs?\s*old)",
        r"\b(\d{2,3})\b",
    ]
    for pat in age_patterns:
        m = re.search(pat, text_lower)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 120:
                if val < 18:
                    return "age_child"
                elif val >= 60:
                    return "age_elderly"
                else:
                    return "age_adult"
    return "age_unknown"

def smart_extract_symptoms(text):
    text_lower = text.lower()
    
    # Define smart mapping rules for compound/synonym symptoms
    compound_rules = {
        "Abdominal Pain": [["abdominal", "pain"], ["pain", "abdomen"], ["pain", "stomach"]],
        "Stomach Pain": [["stomach", "pain"], ["pain", "stomach"]],
        "Chest Pain": [["chest", "pain"], ["pain", "chest"]],
        "Heartburn": [["heartburn"], ["heart", "burn"]],
        "Shortness Of Breath": [["shortness", "breath"], ["short", "breath"]],
        "Breathing Difficulty": [["breathing", "difficulty"], ["breath", "difficulty"], ["difficulty", "breathing"]],
        "Frequent Urination": [["frequent", "urination"], ["frequently", "urinate"]],
        "Blurred Vision": [["blurred", "vision"], ["blur", "vision"]],
        "Loss Of Smell": [["loss", "smell"]],
        "Loss Of Appetite": [["loss", "appetite"]],
        "Night Sweats": [["night", "sweat"], ["night", "sweating"]],
        "Muscle Aches": [["muscle", "ache"], ["muscle", "aches"], ["muscle", "pain"]],
        "Muscle Weakness": [["muscle", "weakness"], ["muscle", "weak"]],
        "Sensitivity To Light": [["sensitivity", "light"], ["sensitive", "light"]],
        "Pale Skin": [["pale", "skin"]],
        "Cold Hands": [["cold", "hands"], ["cold", "hand"]],
        "Joint Pain": [["joint", "pain"]],
        "Burning Sensation": [["burning", "sensation"]],
        "Sore Throat": [["sore", "throat"]],
        "Weakness": [["weakness"], ["weak"]],
    }
    
    found_symptoms = set()
    clauses = re.split(r'[,.;]|\band\b|\bbut\b', text_lower)
    negation_words = {"no", "not", "never", "without", "free of", "denies", "negative for"}
    
    for clause in clauses:
        clause = clause.strip()
        if not clause:
            continue
        clause_words = re.findall(r'\b\w+\b', clause)
        has_negation = any(neg in clause_words for neg in negation_words) or "no " in clause or "not " in clause
        
        # Check compound rules first
        for symptom_name, phrase_list in compound_rules.items():
            for words_req in phrase_list:
                matched_all = True
                for req_w in words_req:
                    if not any(is_close_match(cw, req_w) for cw in clause_words):
                        matched_all = False
                        break
                if matched_all:
                    if not has_negation:
                        found_symptoms.add(symptom_name)
                        break
        
        # Check simple symptoms
        for s in SYMPTOM_LIST:
            s_words = s.split()
            if len(s_words) == 1:
                if any(is_close_match(cw, s) for cw in clause_words):
                    if not has_negation:
                        found_symptoms.add(s.title())
            else:
                matched_all = True
                for req_w in s_words:
                    if not any(is_close_match(cw, req_w) for cw in clause_words):
                        matched_all = False
                        break
                if matched_all and not has_negation:
                    found_symptoms.add(s.title())
                    
    return sorted(list(found_symptoms))

def train_model():
    global vectorizer, model
    all_texts  = []
    all_labels = []
    
    # Process static training data through the symptom extractor
    for raw_text, label in training_data:
        extracted = smart_extract_symptoms(raw_text)
        all_texts.append(" ".join(extracted) if extracted else "")
        all_labels.append(label)
        
    # Adaptive Data
    try:
        conn = get_db()
        cursor = get_cursor(conn)
        cursor.execute("SELECT symptoms, condition_label FROM adaptive_learning")
        rows = cursor.fetchall()
        for row in rows:
            extracted = smart_extract_symptoms(row[0])
            age_group = extract_age_group(row[0])
            symptom_tokens = extracted.copy()
            if age_group != "age_unknown":
                symptom_tokens.append(age_group)
            extracted_str = " ".join(symptom_tokens)
            # Weight user feedback by adding it 5 times to the training data
            for _ in range(5):
                all_texts.append(extracted_str)
                all_labels.append(row[1])
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"[WARNING] Could not load adaptive learning data: {e}")

    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(all_texts)
    
    model = MultinomialNB()
    model.fit(X, all_labels)
    print(f"[OK] Model trained on {len(all_texts)} samples.")

train_model()

# ─────────────────────────────────────────────────────────────
#  HELPER — PREDICT DISEASE
# ─────────────────────────────────────────────────────────────
def predict_disease(symptoms_text, age_group="age_unknown"):
    extracted = smart_extract_symptoms(symptoms_text)
    feature_tokens = extracted.copy()
    if age_group != "age_unknown":
        feature_tokens.append(age_group)
    feature_str = " ".join(feature_tokens)
    
    X_test     = vectorizer.transform([feature_str])
    prediction = model.predict(X_test)[0]
    proba      = model.predict_proba(X_test)[0]
    confidence = round(max(proba) * 100, 1)
    return str(prediction), float(confidence)

# ─────────────────────────────────────────────────────────────
#  HELPER — NLP TEXT PROCESSING  (Forward Flow)
# ─────────────────────────────────────────────────────────────

def process_text(text):
    raw        = text.strip()
    text_lower = raw.lower()
    # ── normalise: remove special chars except spaces/hyphens
    cleaned = re.sub(r"[^a-zA-Z0-9\s\-]", " ", text_lower)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # ── Name detection
    name = "Not specified"
    name_patterns = [
        r"(?:my name is|name[:\s]+|patient[:\s]+|i am|i'm)\s+([A-Za-z]+(?:\s[A-Za-z]+)?)",
        r"^([A-Z][a-z]+(?:\s[A-Z][a-z]+)?)\b",
    ]
    for pat in name_patterns:
        m = re.search(pat, raw, re.IGNORECASE)
        if m:
            name = m.group(1).strip().title()
            break

    # ── Age detection
    age = "Not specified"
    age_patterns = [
        r"(?:age[d]?[:\s]+|i am\s+|i'm\s+|am\s+)(\d{1,3})\s*(?:years?|yrs?)?",
        r"\b(\d{1,3})\s*(?:years?\s*old|yrs?\s*old)",
        r"\b(\d{2,3})\b",
    ]
    for pat in age_patterns:
        m = re.search(pat, text_lower)
        if m:
            val = int(m.group(1))
            if 1 <= val <= 120:
                age = str(val)
                break

    # ── Condition detection
    condition = "Unknown"
    for c in CONDITION_LIST:
        if c in text_lower:
            condition = c.title()
            break

    # ── Symptom detection (collect ALL)
    extracted_symptoms_list = smart_extract_symptoms(text_lower)
    symptom = ", ".join(extracted_symptoms_list) if extracted_symptoms_list else "Not specified"

    # ── Structured pipe string
    structured = f"{name}|{age}|{condition}|{symptom}"

    # ── Human-readable summary
    summary = (
        f"{name} ({age} yrs) is presenting with {condition}, "
        f"showing symptoms of {symptom}."
    )

    # ── Bag-of-Words feature vector (for display)
    feature_names = vectorizer.get_feature_names_out()
    vec           = vectorizer.transform([text_lower]).toarray()[0]
    bow_pairs     = [(feature_names[i], int(vec[i])) for i in range(len(vec)) if vec[i] > 0]

    return {
        "raw":        raw,
        "cleaned":    cleaned,
        "structured": structured,
        "summary":    summary,
        "bow":        bow_pairs,
        "fields": {
            "name":      name,
            "age":       age,
            "condition": condition,
            "symptom":   symptom,
        }
    }

# ─────────────────────────────────────────────────────────────
#  HELPER — BACKWARD FLOW: structured → human explanation
# ─────────────────────────────────────────────────────────────
def backward_interpret(name, age, condition, symptom, prediction, confidence):
    """Turn structured data back into a rich human-readable report."""
    lines = []
    lines.append(f"📋 <strong>Patient Report</strong>")
    lines.append(f"Patient <em>{name}</em>, aged <em>{age}</em>, has submitted health data.")
    lines.append(
        f"The extracted condition is <strong>{condition}</strong> with symptoms: "
        f"<em>{symptom}</em>."
    )
    lines.append(
        f"Based on the symptom profile, the Multinomial Naïve Bayes model predicts "
        f"<strong>{prediction}</strong> with a confidence of <strong>{confidence}%</strong>."
    )
    # Recommendations
    recs = {
        "Malaria":       "🔬 Recommend rapid diagnostic test (RDT) and antimalarial therapy.",
        "COVID-19":      "🦠 Isolate the patient; PCR test and supportive care advised.",
        "Diabetes":      "🩸 Blood glucose testing and dietary assessment are recommended.",
        "Heart Disease": "❤️ Urgent ECG and cardiac enzyme panel required.",
        "Typhoid":       "💊 Widal/blood culture test and antibiotic therapy advised.",
        "Tuberculosis":  "🫁 Sputum culture and chest X-ray recommended.",
        "Hypertension":  "💊 Monitor BP; lifestyle modification and medication review needed.",
        "Asthma":        "💨 Bronchodilator therapy and spirometry test recommended.",
        "Influenza":     "🤒 Rest, hydration and antiviral therapy where applicable.",
        "Migraine":      "💊 Rest in a quiet, dark room. Consider pain relief and stay hydrated.",
        "Gastroenteritis":"💧 Maintain fluid intake. Consider oral rehydration salts and rest.",
        "Pneumonia":     "🫁 Chest X-ray and antibiotic evaluation recommended.",
        "Anemia":        "🩸 Full blood count test and iron supplements evaluation advised.",
        "Peptic Ulcer":  "💊 Endoscopy recommended. Avoid spicy foods and consider antacids.",
    }
    rec = recs.get(prediction, "⚕️ Further clinical evaluation is recommended.")
    lines.append(f"<strong>Recommendation:</strong> {rec}")
    return "<br>".join(lines)

# ─────────────────────────────────────────────────────────────
#  ROUTES
# ─────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        
        conn = get_db()
        cursor = get_cursor(conn)
        cursor.execute("SELECT password FROM users WHERE username=%s", (username,))
        row = cursor.fetchone()
        cursor.close()
        conn.close()
        
        if row and check_password_hash(row[0], password):
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for('home'))
        else:
            flash("Invalid username or password.", "error")
            
    return render_template("index.html", show_login=True)

@app.route("/signup", methods=["POST"])
def signup():
    username = request.form.get("username", "").strip()
    password = request.form.get("password", "")
    
    if not username or not password:
        flash("Username and password are required.", "error")
        return redirect(url_for('login'))
        
    if len(password) < 8 or not re.search(r"[a-z]", password) or not re.search(r"[A-Z]", password) or not re.search(r"[0-9]", password):
        flash("Password must be at least 8 characters long, and contain an uppercase letter, a lowercase letter, and a number.", "error")
        return redirect(url_for('login'))
        
    conn = get_db()
    cursor = get_cursor(conn)
    cursor.execute("SELECT id FROM users WHERE username=%s", (username,))
    if cursor.fetchone():
        cursor.close()
        conn.close()
        flash("Username already exists.", "error")
        return redirect(url_for('login'))
        
    hashed_pw = generate_password_hash(password)
    cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, hashed_pw))
    conn.commit()
    cursor.close()
    conn.close()
    
    flash("Signup successful! Please log in.", "success")
    return redirect(url_for('login'))

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route("/", methods=["GET", "POST"])
@login_required
def home():
    conn       = get_db()
    cursor     = get_cursor(conn)
    session_id = request.args.get("session_id", type=int)
    username   = session.get("username")

    if session_id:
        cursor.execute("SELECT id FROM sessions WHERE id=%s AND username=%s", (session_id, username))
        if not cursor.fetchone():
            session_id = None

    # Create a new session if none exists 
    # We will only create it when receiving a POST request to avoid cluttering with empty sessions.

    if request.method == "POST":
        if not session_id:
            cursor.execute("INSERT INTO sessions (username) VALUES (%s)", (username,))
            conn.commit()
            session_id = cursor.lastrowid

        file       = request.files.get("file")
        user_input = request.form.get("text", "").strip()

        # File upload takes priority
        if file and file.filename != "":
            try:
                user_input = file.read().decode("utf-8", errors="ignore").strip()
            except Exception:
                user_input = "Error reading file"

        if user_input:
            # Forward Flow
            result               = process_text(user_input)
            age_group            = extract_age_group(user_input)
            prediction, confidence = predict_disease(result["fields"]["symptom"], age_group)

            # Backward Flow 
            f        = result["fields"]
            backward = backward_interpret(
                f["name"], f["age"], f["condition"], f["symptom"],
                prediction, confidence
            )

            # Encode prediction & confidence into the pipe-string so they
            # survive without needing extra DB columns:
            # name|age|condition|symptom|prediction|confidence
            extended_structured = (
                f"{result['structured']}|{prediction}|{confidence}"
            )

            # ── Encrypt unstructured text before saving to DB
            encrypted_user_text = encrypt_data(user_input)

            # ── INSERT — only columns that actually exist in MySQL ─────
            cursor.execute("""
                INSERT INTO messages
                    (session_id, user_text, structured, summary, response)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                session_id,
                encrypted_user_text,
                extended_structured,   # name|age|condition|symptom|pred|conf
                result["summary"],
                backward,              # backward report stored in `response`
            ))
            conn.commit()

    messages = []
    if session_id:
        # ── Fetch messages and rebuild 7-element tuples for the template ──
        # DB columns: user_text[0], structured[1], summary[2], response[3]
        cursor.execute("""
            SELECT user_text, structured, summary, response
            FROM messages
            WHERE session_id = %s
            ORDER BY id ASC
        """, (session_id,))
        raw_rows = cursor.fetchall()
    for row in raw_rows if session_id else []:
        user_text, structured, summary, response = row
        decrypted_user_text = decrypt_data(user_text)
        parts      = (structured or "").split("|")
        prediction = parts[4] if len(parts) > 4 else "Unknown"
        confidence = parts[5] if len(parts) > 5 else "0"
        # Rebuild tuple to match what index.html expects:
        messages.append((
            decrypted_user_text, structured, summary, response,
            prediction, confidence, response
        ))

    #  Chart: derive prediction counts from structured pipe-strings 
    all_structured = []
    if session_id:
        cursor.execute("""
            SELECT structured FROM messages WHERE session_id = %s
        """, (session_id,))
        all_structured = cursor.fetchall()
    from collections import Counter
    pred_counter = Counter(
        row[0].split("|")[4]
        for row in all_structured
        if row[0] and len(row[0].split("|")) > 4
    )
    chart_labels = list(pred_counter.keys())
    chart_values = list(pred_counter.values())

    #  Session list (sidebar) 
    # Fetch sessions and decrypt the first message to use as the label
    cursor.execute("""
        SELECT s.id,
               s.created_at,
               (
                   SELECT m2.user_text
                   FROM messages m2
                   WHERE m2.session_id = s.id
                   ORDER BY m2.id ASC
                   LIMIT 1
               ) AS first_msg,
               COUNT(m.id) AS msg_count
        FROM sessions s
        LEFT JOIN messages m ON m.session_id = s.id
        WHERE s.username = %s
        GROUP BY s.id, s.created_at
        ORDER BY s.id DESC
    """, (username,))
    raw_sessions = cursor.fetchall()
    sessions = []
    for row in raw_sessions:
        s_id = row[0]
        created_at = row[1]
        first_msg_enc = row[2]
        msg_count = row[3]
        if first_msg_enc:
            decrypted = decrypt_data(first_msg_enc)
            label = decrypted[:35] + ("..." if len(decrypted) > 35 else "")
        else:
            label = created_at.strftime('%d %b %Y, %H:%M')
        sessions.append((s_id, label, msg_count))

    cursor.close()
    conn.close()

    return render_template(
        "index.html",
        messages=messages,
        sessions=sessions,
        session_id=session_id,
        chart_labels=chart_labels,
        chart_values=chart_values,
        now=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )


@app.route("/api/analyze", methods=["POST"])
@login_required
def api_analyze():
    """JSON API endpoint – returns full pipeline output."""
    data       = request.get_json(force=True)
    user_input = data.get("text", "").strip()
    if not user_input:
        return jsonify({"error": "No text provided"}), 400

    result     = process_text(user_input)
    age_group  = extract_age_group(user_input)
    prediction, confidence = predict_disease(result["fields"]["symptom"], age_group)
    f          = result["fields"]
    backward   = backward_interpret(
        f["name"], f["age"], f["condition"], f["symptom"],
        prediction, confidence
    )

    return jsonify({
        "raw_input":   result["raw"],
        "cleaned":     result["cleaned"],
        "structured":  result["structured"],
        "fields":      result["fields"],
        "bow":         result["bow"],
        "summary":     result["summary"],
        "prediction":  prediction,
        "confidence":  confidence,
        "backward":    backward,
    })


@app.route("/delete_session/<int:sid>", methods=["POST"])
@login_required
def delete_session(sid):
    username = session.get("username")
    conn   = get_db()
    cursor = get_cursor(conn)
    
    cursor.execute("SELECT id FROM sessions WHERE id=%s AND username=%s", (sid, username))
    if not cursor.fetchone():
        cursor.close()
        conn.close()
        return "Unauthorized", 403

    cursor.execute("DELETE FROM messages WHERE session_id = %s", (sid,))
    cursor.execute("DELETE FROM sessions WHERE id = %s", (sid,))
    conn.commit()
    cursor.close()
    conn.close()
    
    if request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"status": "success"})
        
    current = request.args.get("current_session", type=int)
    if current and current != sid:
        return redirect(url_for("home") + f"?session_id={current}")
    return redirect(url_for("home"))


@app.route("/search")
@login_required
def search():
    query  = request.args.get("q", "").strip()
    username = session.get("username")
    conn   = get_db()
    cursor = get_cursor(conn)
    like   = f"%{query}%"
    # Only query columns that exist in the MySQL messages table
    cursor.execute("""
        SELECT m.id, m.user_text, m.structured, m.summary, m.session_id
        FROM messages m
        JOIN sessions s ON m.session_id = s.id
        WHERE s.username = %s AND (m.user_text LIKE %s OR m.structured LIKE %s OR m.summary LIKE %s)
        ORDER BY m.id DESC LIMIT 50
    """, (username, like, like, like))
    results = cursor.fetchall()
    cursor.close()
    conn.close()
    return jsonify([{
        "id":         r[0],
        "user_text":  decrypt_data(r[1]),
        "structured": r[2],
        "summary":    r[3],
        # Extract prediction from pipe-string index 4
        "prediction": r[2].split("|")[4] if r[2] and len(r[2].split("|")) > 4 else "Unknown",
        "session_id": r[4],
    } for r in results])

@app.route("/api/feedback", methods=["POST"])
@login_required
def api_feedback():
    data = request.get_json(force=True)
    symptoms = data.get("symptoms", "").strip()
    condition = data.get("condition", "").strip()
    
    if not symptoms or not condition:
        return jsonify({"status": "error", "message": "Missing data"}), 400
        
    cleaned_symptoms = re.sub(r"[^a-zA-Z0-9\s\-]", " ", symptoms.lower())
    cleaned_symptoms = re.sub(r"\s+", " ", cleaned_symptoms).strip()
    
    # Normalize common conditions
    condition = condition.title()
    if "Ulcer" in condition: condition = "Peptic Ulcer"
    elif "Covid" in condition: condition = "COVID-19"
    elif "Heart" in condition: condition = "Heart Disease"
    elif "Tb" == condition or "Tuberculosis" in condition: condition = "Tuberculosis"
        
    conn = get_db()
    cursor = get_cursor(conn)
    cursor.execute("""
        INSERT INTO adaptive_learning (symptoms, condition_label)
        VALUES (%s, %s)
    """, (cleaned_symptoms, condition))
    conn.commit()
    cursor.close()
    conn.close()
    
    # Retrain model dynamically
    train_model()
    
    return jsonify({"status": "success"})


@app.route("/api/check_username")
def check_username():
    username = request.args.get("u", "").strip()
    if not username:
        return jsonify({"available": False})
        
    conn = get_db()
    cursor = get_cursor(conn)
    cursor.execute("SELECT id FROM users WHERE username=%s", (username,))
    row = cursor.fetchone()
    cursor.close()
    conn.close()
    
    return jsonify({"available": row is None})


if __name__ == "__main__":
    app.run(debug=True, port=5001)