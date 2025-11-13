from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re, difflib, random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT

app = Flask(__name__)
CORS(app)

# === Load Dataset ===
data = pd.read_csv("Maidan dataset (1)(in) (1).csv")
valid_majors = [m.strip().lower() for m in data["major"].dropna().unique()]

# === Prepare TF-IDF ===
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(data["job_skill_set_simple"].fillna(""))

# === Keyword Model ===
kw_model = KeyBERT(model="all-MiniLM-L6-v2")
last_index_per_major = {}

# ==========================================
# Helper Functions
# ==========================================
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_noise_words(text):
    text = re.sub(
        r"\b(freshworks|aramco|baxter|google|amazon|shell|oracle|microsoft|riyadh|ksa|saudi|"
        r"company|organization|enterprise|firm|corporation|department|manager|director|"
        r"executive|representative|officer|assistant|senior|junior|inc|ltd|corp|"
        r"california|san mateo|denver|qsys|tse|francisco)\b",
        "",
        str(text),
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", text).strip()

def unique_terms(terms, max_items=6):
    seen, out = set(), []
    for t in terms:
        t = clean_text(t.lower())
        if not t or t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_items:
            break
    return out

def limit_words(text, max_words=4):
    words = text.split()
    return " ".join(words[:max_words])

# ==========================================
# Quality Control + Smart Variation
# ==========================================
def evaluate_quality(text):
    """Simple quality score based on clarity and repetition"""
    clarity = 1 if len(text.split()) >= 5 else 0.5
    repetition = 1 - (len(re.findall(r"\b(\w+)( \1\b)+", text)) * 0.2)
    score = max(0, min(clarity * repetition, 1))
    if score > 0.8:
        return "High"
    elif score > 0.5:
        return "Medium"
    else:
        return "Low"

def vary_action_verb(context):
    verbs = {
        "analysis": ["Analyze", "Assess", "Investigate", "Review"],
        "planning": ["Design", "Plan", "Formulate", "Draft"],
        "execution": ["Implement", "Apply", "Execute", "Carry out"],
        "improvement": ["Enhance", "Optimize", "Strengthen", "Upgrade"],
        "creative": ["Build", "Create", "Compose", "Organize"]
    }
    if any(k in context for k in ["data", "analysis", "research"]):
        return random.choice(verbs["analysis"])
    elif any(k in context for k in ["plan", "project", "strategy"]):
        return random.choice(verbs["planning"])
    elif any(k in context for k in ["implement", "system", "software"]):
        return random.choice(verbs["execution"])
    elif any(k in context for k in ["improve", "optimize", "enhance"]):
        return random.choice(verbs["improvement"])
    else:
        return random.choice(sum(verbs.values(), []))

# ==========================================
# Smart Task Generator
# ==========================================
def generate_task(title, desc, skills, enhance_variation=True, evaluate_output=True):
    title = remove_noise_words(title)
    desc = remove_noise_words(desc)
    skills = remove_noise_words(skills)
    text = clean_text(f"{title} {desc} {skills}")

    # Extract keywords via KeyBERT
    try:
        keywords = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=6)
        key_terms = unique_terms([k[0] for k in keywords])
    except:
        key_terms = ["process improvement", "workflow optimization"]

    expanded_terms = []
    for term in key_terms:
        expanded_terms.extend(term.split())
    expanded_terms = unique_terms(expanded_terms, 10)

    context = " ".join(expanded_terms).lower()
    action = vary_action_verb(context) if enhance_variation else "Develop"

    # Title (max 4 words)
    title_words = [action] + expanded_terms[:5]
    title_final = limit_words(" ".join(title_words), 4).title()

    # Description
    goals = [
        "to enhance workflow efficiency",
        "to achieve measurable outcomes",
        "to support operational success",
        "to improve project results",
        "to drive performance improvements"
    ]
    goal = random.choice(goals)
    desc_templates = [
        f"{action} practical tasks involving {' '.join(expanded_terms[:3])} and also {skills} {goal}",
        f"{action} initiatives around {' '.join(expanded_terms[:3])} plus {skills} {goal}",
        f"{action} real projects applying {' '.join(expanded_terms[:3])} and {skills} {goal}",
    ]
    description = random.choice(desc_templates)
    description = re.sub(r"\b(\w+)( \1\b)+", r"\1", description)
    description = re.sub(r"[,.]+", "", description)
    description = re.sub(r"\s+", " ", description).strip()

    # Evaluate quality
    quality = evaluate_quality(description) if evaluate_output else None

    return title_final, description, quality

# ==========================================
# API Endpoint
# ==========================================
@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        user_data = request.get_json()
        major = user_data.get("major", "").strip().lower()
        skills = [s.lower() for s in user_data.get("skills", [])]

        if not major:
            return jsonify({"error": "Missing major"}), 400

        # Map common major names to dataset majors
        major_mapping = {
            "computer science": "informationtechnology",
            "cs": "informationtechnology",
            "it": "informationtechnology",
            "information technology": "informationtechnology",
            "human resources": "hr",
            "hr": "hr",
            "sales": "sales",
            "finance": "finance",
            "business development": "businessdevelopment",
            "business": "businessdevelopment",
            "bd": "businessdevelopment"
        }
        
        # Try mapping first
        mapped_major = major_mapping.get(major, major)
        
        # Then try fuzzy matching with lower cutoff
        closest = difflib.get_close_matches(mapped_major, valid_majors, n=1, cutoff=0.6)
        if not closest:
            # Last resort: return all available majors in error message
            available = ", ".join(sorted(set(valid_majors)))
            return jsonify({
                "error": f"No matching major for '{major}'. Available majors in dataset: {available}",
                "available_majors": list(sorted(set(valid_majors))),
                "suggested": major_mapping.get(major, None)
            }), 400
        correct_major = closest[0]

        filtered = data[data["major"].str.lower().str.strip() == correct_major]
        if filtered.empty:
            return jsonify({"error": f"No data found for {correct_major}"}), 404

        user_input = " ".join(skills)
        user_vec = tfidf.transform([user_input])
        indices = filtered.index
        filtered_tfidf = tfidf_matrix[indices]
        similarity = cosine_similarity(user_vec, filtered_tfidf).flatten()
        filtered = filtered.copy()
        filtered["similarity"] = similarity

        def count_matches(skill_text):
            job_skills = str(skill_text).lower()
            return sum(1 for s in skills if s in job_skills)

        filtered["match_count"] = filtered["job_skill_set_simple"].apply(count_matches)

        def get_level(m):
            if m >= 3:
                return "Beginner"
            elif m >= 1:
                return "Intermediate"
            else:
                return "Advanced"

        filtered["level"] = filtered["match_count"].apply(get_level)
        filtered = filtered.sort_values(by="similarity", ascending=False).reset_index(drop=True)

        start = last_index_per_major.get(correct_major, 0)
        end = start + 3
        if start >= len(filtered):
            start, end = 0, 3
        batch = filtered.iloc[start:end]
        last_index_per_major[correct_major] = end

        recs = []
        for _, row in batch.iterrows():
            title, desc, quality = generate_task(
                row["job_title"], row["job_description_simple"], row["job_skill_set_simple"],
                enhance_variation=True, evaluate_output=True
            )
            recs.append({
                "title": title,
                "description": desc,
                "quality": quality,
                "skills_required": row["job_skill_set_simple"],
                "similarity": round(float(row["similarity"]), 3),
                "level": row["level"],
                "major": correct_major,
            })

        return jsonify(recs)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================
# Run App
# ==========================================
if __name__ == "__main__":
    app.run(debug=True, port=5003)
