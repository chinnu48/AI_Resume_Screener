import streamlit as st
import re
import io
import json
import time
import os
import matplotlib.pyplot as plt
from functools import wraps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai

# ------------------ CONFIG ------------------
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

gemini_available = False
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_available = True
    except Exception:
        gemini_available = False


# ------------------ CACHE DECORATOR ------------------
def cached_ttl(ttl_seconds=300):
    def decorator(fn):
        cache = {}
        @wraps(fn)
        def wrapped(*args, **kwargs):
            key = json.dumps({"args": args, "kwargs": kwargs}, default=str, sort_keys=True)
            now = time.time()
            if key in cache:
                val, ts = cache[key]
                if now - ts < ttl_seconds:
                    return val
            val = fn(*args, **kwargs)
            cache[key] = (val, now)
            return val
        return wrapped
    return decorator


# ------------------ TEXT UTILITIES ------------------
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    return "\n".join(page.extract_text() or "" for page in reader.pages)

def clean_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


# ------------------ GEMINI PROMPTS ------------------
def build_resume_check_prompt(resume_text):
    return f"""
You are a document classification expert.
Determine if the provided text is a *resume/CV* or not.
Respond ONLY with JSON:
{{"is_resume": true/false, "reason": "short explanation"}}

Text:
\"\"\"{resume_text[:4000]}\"\"\"
"""

def build_feedback_prompt(resume_text, job_text, matched_skills, missing_skills):
    return f"""
You are a resume optimization and ATS expert.

Compare this resume to the job description and return ONLY JSON:
{{
  "summary_feedback": "...",
  "skills_feedback": ["...", "..."],
  "projects_feedback": ["..."],
  "keywords_to_add": ["...", "..."],
  "rewritten_summary": "...",
  "confidence": 0-100
}}

Resume:
\"\"\"{resume_text[:6000]}\"\"\"

Job Description:
\"\"\"{job_text[:6000]}\"\"\"

Matched Skills: {matched_skills}
Missing Skills: {missing_skills}
"""


# ------------------ GEMINI CALLS ------------------
@cached_ttl(ttl_seconds=600)
def call_gemini(prompt, model="gemini-2.5-flash"):
    try:
        m = genai.GenerativeModel(model)
        resp = m.generate_content(prompt)
        raw = resp.text.strip()
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            return json.loads(raw[start:end+1])
        return {"error": "Invalid JSON", "raw": raw}
    except Exception as e:
        return {"error": str(e)}


# ------------------ LOCAL FALLBACK ------------------
def local_feedback_fallback(matched_skills, missing_skills):
    return {
        "summary_feedback": "Resume partially matches job description. Add more measurable achievements.",
        "skills_feedback": missing_skills[:3] if missing_skills else ["Add relevant technical details."],
        "projects_feedback": ["Include 1–2 domain-relevant projects."],
        "keywords_to_add": missing_skills[:5],
        "rewritten_summary": "Motivated professional with adaptable skills, eager to contribute effectively.",
        "confidence": 60,
        "mode": "local_fallback"
    }


# ------------------ SCORING ------------------
def compute_skill_match(resume_text, job_text, skills_list):
    resume_skills = [s for s in skills_list if s in resume_text]
    job_skills = [s for s in skills_list if s in job_text]
    matched = [s for s in resume_skills if s in job_skills]
    missing = [s for s in job_skills if s not in resume_skills]
    percent = (len(matched) / len(job_skills) * 100) if job_skills else 0
    return matched, missing, round(percent, 2)

def compute_semantic_similarity(resume_text, job_text):
    if not resume_text.strip() or not job_text.strip():
        return 0.0
    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        vectors = vectorizer.fit_transform([resume_text, job_text])
        cosine_score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        scaled = max(0, min((cosine_score - 0.1) / 0.6 * 100, 100))
        return round(scaled, 2)
    except Exception:
        return 0.0

def compute_ats_score(resume_text):
    score = 0
    txt = resume_text.lower()
    sections = ["summary", "objective", "skills", "project", "experience", "education"]
    score += 20 * (sum(1 for s in sections if s in txt) / len(sections))
    verbs = ["developed","built","designed","managed","tested","optimized","analyzed"]
    score += 15 * (sum(1 for v in verbs if v in txt) / len(verbs))
    word_count = len(txt.split())
    score += 15 * (0.8 if 150 < word_count < 800 else 0.5)
    certs = ["bachelor","master","degree","certificate","certified"]
    score += 10 if any(c in txt for c in certs) else 3
    return round(min(score + 42, 100), 2)

def hybrid_score(skill, semantic, ats):
    return round((0.5 * skill) + (0.3 * semantic) + (0.2 * ats), 2)


# ------------------ PDF GENERATION ------------------
def generate_pdf_report(name, total, skill, matched, missing, ats, ai_feedback=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, "AI Resume Screening Report", ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 8, f"Resume: {name}", ln=True)
    pdf.cell(200, 8, f"Overall Match: {total}%", ln=True)
    pdf.cell(200, 8, f"Skill Match: {skill}%", ln=True)
    pdf.cell(200, 8, f"ATS Score: {ats}%", ln=True)
    pdf.ln(8)
    pdf.multi_cell(0, 8, f"Matched Skills: {', '.join(matched) or 'None'}")
    pdf.multi_cell(0, 8, f"Missing Skills: {', '.join(missing) or 'None'}")
    if ai_feedback:
        pdf.ln(8)
        for key in ["summary_feedback", "skills_feedback", "projects_feedback"]:
            val = ai_feedback.get(key)
            if val:
                text = val if isinstance(val, str) else ", ".join(val)
                pdf.multi_cell(0, 8, f"{key.replace('_',' ').title()}: {text}")
    pdf_bytes = pdf.output(dest="S").encode("latin1")
    return io.BytesIO(pdf_bytes)


# ------------------ STREAMLIT APP ------------------
st.set_page_config(page_title=" Resume Screener by Nikita", layout="wide")
st.title("🤖 ResumeIQ")

if not gemini_available:
    st.warning("⚠️ Gemini API not configured — using **local fallback mode**.")

uploaded_file = st.file_uploader("📂 Upload Resume (TXT or PDF)", type=["txt", "pdf"])
job_description = st.text_area("💼 Paste Job Description", height=200)

if uploaded_file and job_description:
    resume_text = (
        extract_text_from_pdf(uploaded_file)
        if uploaded_file.name.endswith(".pdf")
        else uploaded_file.read().decode("utf-8")
    )
    resume_text, job_text = clean_text(resume_text), clean_text(job_description)

    # Step 1 — Detect Resume
    is_resume = True
    if gemini_available:
        with st.spinner("🔍 Checking if uploaded document is a valid resume..."):
            check = call_gemini(build_resume_check_prompt(resume_text))
            if check.get("is_resume") is False:
                st.error(f"❌ Not a Resume: {check.get('reason')}")
                st.stop()

    # Step 2 — Calculate scores
    skills = ["python","java","sql","tableau","power bi","machine learning","deep learning","excel",
              "data analysis","visualization","django","flask","aws","git","testing","selenium",
              "html","css","javascript","react"]

    matched, missing, skill_score = compute_skill_match(resume_text, job_text, skills)
    semantic_score = compute_semantic_similarity(resume_text, job_text)
    ats_score = compute_ats_score(resume_text)
    total_score = hybrid_score(skill_score, semantic_score, ats_score)

    # Safe score conversions
    def safe_score(value):
        try:
            if value is None or value != value:
                return 0
            return float(value)
        except Exception:
            return 0.0

    skill_score = safe_score(skill_score)
    semantic_score = safe_score(semantic_score)
    ats_score = safe_score(ats_score)
    total_score = safe_score(total_score)

    # Step 3 — Display metrics
    st.subheader("📊 Resume Evaluation Results")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall Score", f"{total_score:.2f}%")
    c2.metric("Skill Match", f"{skill_score:.2f}%")
    c3.metric("Semantic Match", f"{semantic_score:.2f}%")
    c4.metric("ATS Score", f"{ats_score:.2f}%")

    # Step 4 — Pie chart
    categories = ["Skill Match", "Semantic Match", "ATS Score"]
    values = [skill_score, semantic_score, ats_score]
    if sum(values) == 0:
        st.warning("⚠️ Not enough valid data to generate chart.")
    else:
        colors = ["#4CAF50", "#2196F3", "#FFC107"]
        fig, ax = plt.subplots()
        ax.pie(values, labels=categories, autopct="%1.1f%%", startangle=90, colors=colors)
        ax.axis("equal")
        st.pyplot(fig)

    # Step 5 — AI Suggestions
    st.subheader("🧠 AI Suggestions & Resume Improvement")
    ai_feedback = None
    if st.button("✨ Generate AI Feedback"):
        with st.spinner("Analyzing resume with Gemini..."):
            if gemini_available:
                ai_feedback = call_gemini(build_feedback_prompt(resume_text, job_text, matched, missing))
            else:
                ai_feedback = local_feedback_fallback(matched, missing)

        if "error" in ai_feedback:
            st.error(f"Gemini Error: {ai_feedback.get('error')}")
        else:
            st.write("**Summary Feedback:**", ai_feedback.get("summary_feedback"))
            st.write("**Skills Feedback:**", ai_feedback.get("skills_feedback"))
            st.write("**Projects Feedback:**", ai_feedback.get("projects_feedback"))
            st.write("**Keywords to Add:**", ", ".join(ai_feedback.get("keywords_to_add", [])))
            st.code(ai_feedback.get("rewritten_summary", "-"))

    # Step 6 — PDF Report
    st.subheader("📥 Download Report")
    if st.button("Generate PDF Report"):
        pdf = generate_pdf_report(uploaded_file.name, total_score, skill_score, matched, missing, ats_score, ai_feedback)
        st.download_button("📄 Download PDF", pdf, file_name=f"{uploaded_file.name}_AI_Report.pdf", mime="application/pdf")

else:
    st.info("👆 Upload a resume and paste the job description to begin.")
