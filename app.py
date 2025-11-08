import streamlit as st
import re
import io
import json
import time
import os
import hashlib
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from datetime import datetime, timedelta
from functools import wraps
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from fpdf import FPDF
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import google.generativeai as genai
DEFAULT_SKILLS = [
    "python","java","sql","tableau","power bi","machine learning","deep learning","excel",
    "data analysis","visualization","django","flask","aws","git","testing","selenium",
    "html","css","javascript","react"
]

EXPERIENCE_ACTION_VERBS = [
    "developed","built","designed","managed","tested","optimized","analyzed","led",
    "implemented","created","delivered","increased","reduced","improved","launched",
]

ACHIEVEMENT_KEYWORDS = [
    "achieved","improved","reduced","increased","grew","saved","delivered","surpassed",
    "boosted","accelerated","streamlined","enhanced"
]

INTERVIEW_STAGES = [
    "Applied",
    "Screening",
    "Interview Round 1",
    "Interview Round 2",
    "Offer",
    "Hired",
    "Rejected",
]

CANDIDATE_CHECKLIST = [
    "Tailor resume summary to job keywords",
    "Quantify top three achievements",
    "Align skills section to job requirements",
    "Update LinkedIn to match resume language",
    "Prepare STAR stories for interviews",
    "Draft follow-up email template",
]

DATA_FILE = Path("data_store.json")
USERS_FILE = Path("users_store.json")
DEFAULT_DATA = {
    "candidate_records": [],
    "recruiter_sessions": [],
    "messages": [],
    "candidate_notes": [],
    "interview_events": [],
}
DEFAULT_USERS = {
    "candidates": {},
    "recruiters": {},
}


def safe_pdf_text(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        value = ", ".join(str(v) for v in value)
    try:
        return str(value).encode("latin-1", "replace").decode("latin-1")
    except Exception:
        return str(value).encode("latin-1", "ignore").decode("latin-1")


# ------------------ AUTHENTICATION ------------------
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def load_users():
    if USERS_FILE.exists():
        try:
            with USERS_FILE.open("r", encoding="utf-8") as fh:
                users = json.load(fh)
            for key in DEFAULT_USERS:
                users.setdefault(key, {})
            return users
        except Exception:
            return DEFAULT_USERS.copy()
    return DEFAULT_USERS.copy()


def save_users(users):
    try:
        with USERS_FILE.open("w", encoding="utf-8") as fh:
            json.dump(users, fh, indent=2)
    except Exception:
        pass


def register_user(email, password, name, role="candidate", team=None):
    users = load_users()
    role_key = "candidates" if role == "candidate" else "recruiters"
    if email in users[role_key]:
        return False, "Email already registered"
    users[role_key][email] = {
        "email": email,
        "password_hash": hash_password(password),
        "name": name,
        "role": role,
        "team": team,
        "created_at": datetime.utcnow().isoformat(),
    }
    save_users(users)
    return True, "Registration successful"


def authenticate_user(email, password, role="candidate"):
    users = load_users()
    role_key = "candidates" if role == "candidate" else "recruiters"
    if email not in users[role_key]:
        return False, None
    user = users[role_key][email]
    if user["password_hash"] == hash_password(password):
        return True, user
    return False, None


def get_current_user():
    if "authenticated" in st.session_state and st.session_state.authenticated:
        return {
            "email": st.session_state.get("user_email"),
            "name": st.session_state.get("user_name"),
            "role": st.session_state.get("user_role"),
            "team": st.session_state.get("user_team"),
        }
    return None


def logout():
    for key in ["authenticated", "user_email", "user_name", "user_role", "user_team"]:
        if key in st.session_state:
            del st.session_state[key]


def load_datastore():
    if DATA_FILE.exists():
        try:
            with DATA_FILE.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            for key in DEFAULT_DATA:
                data.setdefault(key, [])
            return data
        except Exception:
            return DEFAULT_DATA.copy()
    return DEFAULT_DATA.copy()


def save_datastore(data):
    try:
        with DATA_FILE.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2)
    except Exception:
        pass


def append_candidate_record(record):
    data = load_datastore()
    data.setdefault("candidate_records", []).append(record)
    save_datastore(data)


def append_recruiter_session(record):
    data = load_datastore()
    data.setdefault("recruiter_sessions", []).append(record)
    save_datastore(data)


def append_message(message):
    data = load_datastore()
    data.setdefault("messages", []).append(message)
    save_datastore(data)


def upsert_candidate_note(note):
    data = load_datastore()
    notes = data.setdefault("candidate_notes", [])
    existing_idx = next(
        (
            idx
            for idx, entry in enumerate(notes)
            if entry.get("recruiter_email") == note.get("recruiter_email")
            and entry.get("candidate_id") == note.get("candidate_id")
        ),
        None,
    )
    if existing_idx is not None:
        notes[existing_idx] = note
    else:
        notes.append(note)
    save_datastore(data)


def get_candidate_notes(recruiter_email=None):
    data = load_datastore()
    notes = data.get("candidate_notes", [])
    if recruiter_email:
        notes = [n for n in notes if n.get("recruiter_email") == recruiter_email]
    return notes


def get_recent_candidate_records(limit=10, email=None, workspace_code=None):
    data = load_datastore()
    records = data.get("candidate_records", [])
    if email:
        records = [r for r in records if r.get("candidate_email") == email]
    if workspace_code:
        records = [r for r in records if r.get("workspace_code") == workspace_code]
    return records[-limit:][::-1]


def get_candidate_records_by_user(user_email):
    data = load_datastore()
    records = data.get("candidate_records", [])
    return [r for r in records if r.get("candidate_email") == user_email]


def get_recent_recruiter_sessions(limit=10, recruiter_email=None, workspace_code=None):
    data = load_datastore()
    sessions = data.get("recruiter_sessions", [])
    if recruiter_email:
        sessions = [s for s in sessions if s.get("recruiter_email") == recruiter_email]
    if workspace_code:
        sessions = [s for s in sessions if s.get("workspace_code") == workspace_code]
    return sessions[-limit:][::-1]


def get_recruiter_sessions_by_user(user_email):
    data = load_datastore()
    sessions = data.get("recruiter_sessions", [])
    return [s for s in sessions if s.get("recruiter_email") == user_email]


def get_recent_messages(limit=20, channel=None):
    data = load_datastore()
    messages = data.get("messages", [])
    if channel:
        messages = [m for m in messages if m.get("channel") == channel]
    messages = sorted(messages, key=lambda m: m.get("timestamp", ""))
    return messages[-limit:]


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


def build_section_guidance_prompt(resume_text, job_text, matched_skills, missing_skills):
    return f"""
You are a career coach who specializes in resume rewrites.

Analyze the resume and job description and return ONLY JSON with this structure:
{{
  "summary": ["actionable coaching point", ...],
  "experience": ["..."],
  "skills": ["..."],
  "education": ["..."],
  "keywords_to_emphasize": ["..."],
  "questions_to_reflect": ["..."],
  "next_actions": ["..."]
}}

Focus on incorporating the job requirements, quantifying impact, and aligning to ATS keywords.
Do not include text outside valid JSON.

Resume:
\"\"\"{resume_text[:6000]}\"\"\"

Job Description:
\"\"\"{job_text[:4000]}\"\"\"

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


def local_section_guidance(matched_skills, missing_skills):
    summary_tips = [
        "Use a 2-3 sentence headline that references the target role and blends key matched skills.",
        "Lead with years of experience and one quantified achievement (e.g., % improvement, revenue impact).",
    ]
    if missing_skills:
        summary_tips.append(
            f"Reference at least one of these keywords early in your summary: {', '.join(missing_skills[:3])}."
        )

    experience_tips = [
        "Start each bullet with a strong action verb and follow with measurable outcomes.",
        "Highlight projects or responsibilities that mirror the job description's priorities.",
    ]
    if missing_skills:
        experience_tips.append(
            f"Add a bullet describing hands-on experience with {missing_skills[0]} if applicable."
        )

    skills_tips = [
        "Group technical skills by category (languages, tools, platforms) for readability.",
        "Prioritize matched skills first, then add emerging expertise or certifications.",
    ]
    if missing_skills:
        skills_tips.append(
            f"Consider training or cert prep for: {', '.join(missing_skills[:5])}."
        )

    education_tips = [
        "List relevant coursework, certifications, or bootcamps tied to the job description.",
        "Mention honors, GPA (if strong), or ongoing professional development."
    ]

    keywords_emphasize = matched_skills[:5] + missing_skills[:5]
    questions = [
        "Where can I quantify impact with numbers (%, $, time saved)?",
        "Does each bullet connect to a requirement in the job description?",
        "Am I showcasing recent technology or methodologies demanded by the role?",
    ]
    next_actions = [
        "Update LinkedIn headline to mirror the resume summary.",
        "Reach out to a peer for a quick review focused on clarity and impact.",
        "Prepare STAR stories for the top achievements listed.",
    ]

    return {
        "summary": summary_tips,
        "experience": experience_tips,
        "skills": skills_tips,
        "education": education_tips,
        "keywords_to_emphasize": keywords_emphasize,
        "questions_to_reflect": questions,
        "next_actions": next_actions,
        "mode": "local_fallback",
    }


def get_section_guidance(resume_text, job_text, matched_skills, missing_skills):
    if gemini_available:
        guidance = call_gemini(
            build_section_guidance_prompt(resume_text, job_text, matched_skills, missing_skills)
        )
        if isinstance(guidance, dict) and "error" not in guidance:
            return guidance
    return local_section_guidance(matched_skills, missing_skills)


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

def compute_experience_score(raw_resume_text):
    if not raw_resume_text.strip():
        return 0.0

    lines = [line.strip().lower() for line in raw_resume_text.splitlines() if line.strip()]
    if not lines:
        return 0.0

    total_lines = len(lines)
    action_hits = sum(any(verb in line for verb in EXPERIENCE_ACTION_VERBS) for line in lines)
    quantified_hits = sum(bool(re.search(r"\b\d+(\.\d+)?%?", line)) for line in lines)
    achievement_hits = sum(any(word in line for word in ACHIEVEMENT_KEYWORDS) for line in lines)

    weighted = (
        0.5 * (action_hits / total_lines)
        + 0.3 * (quantified_hits / total_lines)
        + 0.2 * (achievement_hits / total_lines)
    )
    return round(min(weighted * 100, 100), 2)


def hybrid_score(score_map, weight_map):
    total = 0.0
    for key, weight in weight_map.items():
        total += score_map.get(key, 0.0) * weight
    return round(total, 2)


def safe_score(value):
    try:
        if value is None or value != value:
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def parse_skill_input(raw_value):
    if not raw_value:
        return DEFAULT_SKILLS
    parsed = [s.strip().lower() for s in raw_value.split(",") if s.strip()]
    return parsed or DEFAULT_SKILLS


def normalize_weights(weight_inputs):
    total = sum(weight_inputs.values())
    if total <= 0:
        count = len(weight_inputs)
        return {key: (1 / count if count else 0) for key in weight_inputs}
    return {key: weight_inputs[key] / total for key in weight_inputs}


def evaluate_resume(raw_resume_text, job_text, skills, weight_map):
    clean_resume = clean_text(raw_resume_text)
    clean_job = clean_text(job_text)

    matched, missing, skill_score = compute_skill_match(clean_resume, clean_job, skills)
    semantic_score = compute_semantic_similarity(clean_resume, clean_job)
    ats_score = compute_ats_score(clean_resume)
    experience_score = compute_experience_score(raw_resume_text)

    score_map = {
        "Skill Match": safe_score(skill_score),
        "Semantic Match": safe_score(semantic_score),
        "ATS Score": safe_score(ats_score),
        "Experience Score": safe_score(experience_score),
    }
    total_score = safe_score(hybrid_score(score_map, weight_map))

    return {
        "raw_resume": raw_resume_text,
        "clean_resume": clean_resume,
        "clean_job": clean_job,
        "matched": matched,
        "missing": missing,
        "scores": score_map,
        "total_score": total_score,
    }


# ------------------ PDF GENERATION ------------------
def generate_pdf_report(name, total, skill, matched, missing, ats, experience, weights, ai_feedback=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, safe_pdf_text("AI Resume Screening Report"), ln=True, align="C")
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 8, safe_pdf_text(f"Resume: {name}"), ln=True)
    pdf.cell(200, 8, safe_pdf_text(f"Overall Match: {total}%"), ln=True)
    pdf.cell(200, 8, safe_pdf_text(f"Skill Match: {skill}%"), ln=True)
    pdf.cell(200, 8, safe_pdf_text(f"ATS Score: {ats}%"), ln=True)
    pdf.cell(200, 8, safe_pdf_text(f"Experience Score: {experience}%"), ln=True)
    pdf.ln(8)
    pdf.multi_cell(0, 8, safe_pdf_text(f"Matched Skills: {', '.join(matched) or 'None'}"))
    pdf.multi_cell(0, 8, safe_pdf_text(f"Missing Skills: {', '.join(missing) or 'None'}"))
    if weights:
        pdf.ln(4)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(200, 8, safe_pdf_text("Weight Configuration"), ln=True)
        pdf.set_font("Arial", size=12)
        for label, weight in weights.items():
            pdf.cell(200, 6, safe_pdf_text(f"{label}: {weight*100:.1f}%"), ln=True)
    if ai_feedback:
        pdf.ln(8)
        for key in ["summary_feedback", "skills_feedback", "projects_feedback"]:
            val = ai_feedback.get(key)
            if val:
                text = val if isinstance(val, str) else ", ".join(val)
                pdf.multi_cell(0, 8, safe_pdf_text(f"{key.replace('_',' ').title()}: {text}"))
    pdf_bytes = pdf.output(dest="S").encode("latin1", "replace")
    return io.BytesIO(pdf_bytes)


def generate_recruiter_pdf(job_excerpt, df, weights, threshold, recruiter_name=None):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, safe_pdf_text("Recruiter Screening Summary"), ln=True, align="C")
    pdf.set_font("Arial", size=11)
    if recruiter_name:
        pdf.cell(200, 8, safe_pdf_text(f"Prepared by: {recruiter_name}"), ln=True)
    pdf.multi_cell(0, 6, safe_pdf_text(f"Job Summary: {job_excerpt[:400]}..."))
    pdf.ln(4)
    pdf.cell(200, 8, safe_pdf_text("Weight Configuration"), ln=True)
    for label, weight in weights.items():
        pdf.cell(200, 6, safe_pdf_text(f"- {label}: {weight*100:.1f}%"), ln=True)
    pdf.cell(200, 6, safe_pdf_text(f"Shortlist Threshold: {threshold}%"), ln=True)
    pdf.ln(6)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(200, 8, safe_pdf_text("Top Candidates"), ln=True)
    pdf.set_font("Arial", size=11)
    for _, row in df.iterrows():
        pdf.cell(200, 7, safe_pdf_text(f"{row['Candidate']} — Overall {row['Overall Score']:.1f}%"), ln=True)
        pdf.cell(
            200,
            6,
            safe_pdf_text(
                f"Skill {row['Skill Match']:.1f}%, Semantic {row['Semantic Match']:.1f}%, "
                f"ATS {row['ATS Score']:.1f}%, Experience {row['Experience Score']:.1f}%"
            ),
            ln=True,
        )
        pdf.cell(200, 6, safe_pdf_text(f"Matched Skills: {row['Matched Skills']}"), ln=True)
        pdf.cell(200, 6, safe_pdf_text(f"Missing Skills: {row['Missing Skills']}"), ln=True)
        pdf.ln(2)

    pdf_bytes = pdf.output(dest="S").encode("latin1", "replace")
    return io.BytesIO(pdf_bytes)


def build_calendar_invite(title, start_dt, duration_minutes=45, description="", location="Virtual"):
    uid = f"{int(start_dt.timestamp())}-{abs(hash(title))}@resumeiq"
    dtstamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    start_str = start_dt.strftime("%Y%m%dT%H%M%SZ")
    end_str = (start_dt + timedelta(minutes=duration_minutes)).strftime("%Y%m%dT%H%M%SZ")
    ics = f"""BEGIN:VCALENDAR
VERSION:2.0
PRODID:-//ResumeIQ//EN
BEGIN:VEVENT
UID:{uid}
DTSTAMP:{dtstamp}
DTSTART:{start_str}
DTEND:{end_str}
SUMMARY:{title}
DESCRIPTION:{description}
LOCATION:{location}
END:VEVENT
END:VCALENDAR
"""
    return io.BytesIO(ics.encode("utf-8"))


# ------------------ STREAMLIT APP ------------------
st.set_page_config(page_title=" Resume Screener by Nikita", layout="wide")

if not gemini_available:
    st.warning("⚠️ Gemini API not configured — using **local fallback mode**.")

current_user = get_current_user()

if not current_user:
    st.title("🤖 ResumeIQ - Login")
    st.markdown("---")
    
    login_tab, candidate_register_tab, recruiter_register_tab = st.tabs(["Login", "Register as Candidate", "Register as Recruiter"])
    
    with login_tab:
        st.subheader("Login to Your Account")
        role_choice = st.radio("I am a:", ("Candidate", "Recruiter"), key="login_role")
        login_email = st.text_input("Email", key="login_email")
        login_password = st.text_input("Password", type="password", key="login_password")
        
        if st.button("Login", key="login_btn"):
            role = "candidate" if role_choice == "Candidate" else "recruiter"
            success, user = authenticate_user(login_email, login_password, role)
            if success:
                st.session_state.authenticated = True
                st.session_state.user_email = user["email"]
                st.session_state.user_name = user["name"]
                st.session_state.user_role = user["role"]
                st.session_state.user_team = user.get("team")
                st.success(f"Welcome back, {user['name']}!")
                st.experimental_rerun()
            else:
                st.error("Invalid email or password. Please try again or register.")
    
    with candidate_register_tab:
        st.subheader("Create Candidate Account")
        reg_name = st.text_input("Full Name", key="candidate_reg_name")
        reg_email = st.text_input("Email", key="candidate_reg_email")
        reg_password = st.text_input("Password", type="password", key="candidate_reg_password")
        reg_confirm = st.text_input("Confirm Password", type="password", key="candidate_reg_confirm")
        reg_role_goal = st.text_input("Target Role / Title (optional)", key="candidate_reg_role")
        
        if st.button("Register", key="candidate_reg_btn"):
            if not reg_name or not reg_email or not reg_password:
                st.error("Please fill in all required fields.")
            elif reg_password != reg_confirm:
                st.error("Passwords do not match.")
            else:
                success, message = register_user(reg_email, reg_password, reg_name, role="candidate")
                if success:
                    st.success(message + " You can now login.")
                else:
                    st.error(message)
    
    with recruiter_register_tab:
        st.subheader("Create Recruiter Account")
        reg_name = st.text_input("Full Name", key="recruiter_reg_name")
        reg_email = st.text_input("Work Email", key="recruiter_reg_email")
        reg_password = st.text_input("Password", type="password", key="recruiter_reg_password")
        reg_confirm = st.text_input("Confirm Password", type="password", key="recruiter_reg_confirm")
        reg_team = st.text_input("Team / Organization (optional)", key="recruiter_reg_team")
        
        if st.button("Register", key="recruiter_reg_btn"):
            if not reg_name or not reg_email or not reg_password:
                st.error("Please fill in all required fields.")
            elif reg_password != reg_confirm:
                st.error("Passwords do not match.")
            else:
                success, message = register_user(reg_email, reg_password, reg_name, role="recruiter", team=reg_team)
                if success:
                    st.success(message + " You can now login.")
                else:
                    st.error(message)

else:
    st.title("🤖 ResumeIQ")
    
    user = current_user
    st.sidebar.title("Account")
    st.sidebar.write(f"**Logged in as:** {user['name']}")
    st.sidebar.write(f"**Email:** {user['email']}")
    if user.get("team"):
        st.sidebar.write(f"**Team:** {user['team']}")
    if st.sidebar.button("Logout"):
        logout()
        st.experimental_rerun()
    
    if user["role"] == "candidate":
        st.header("Candidate Self-Assessment")
        st.write("Upload your resume, paste the target job description, and personalize how the screening engine scores you.")
        
        candidate_name = user["name"]
        candidate_email = user["email"]
        candidate_role_goal = st.sidebar.text_input("Target Role / Title", key="candidate_role_goal")
        workspace_code = st.sidebar.text_input(
            "Workspace code (share with recruiter)",
            value=st.session_state.get("workspace_code", "general"),
            key="candidate_workspace_code",
        ).strip() or "general"
        st.session_state["workspace_code"] = workspace_code
        st.sidebar.caption("Use the same code as your recruiter to sync notes & messages.")

        uploaded_file = st.file_uploader(
            "📂 Upload Resume (TXT or PDF)",
            type=["txt", "pdf"],
            key="candidate_resume_uploader",
        )
        job_description = st.text_area(
            "💼 Paste Job Description",
            height=200,
            key="candidate_job_description",
        )

        if uploaded_file and job_description:
            if uploaded_file.name.lower().endswith(".pdf"):
                resume_bytes = uploaded_file.read()
                raw_resume_text = extract_text_from_pdf(io.BytesIO(resume_bytes))
            else:
                raw_resume_text = uploaded_file.read().decode("utf-8", errors="ignore")
            uploaded_file.seek(0)

            with st.expander("⚙️ Advanced Scoring Settings", expanded=False):
                skill_input = st.text_area(
                    "Skill Keywords (comma-separated)",
                    value=", ".join(DEFAULT_SKILLS),
                    height=120,
                    key="candidate_skill_keywords",
                )
                weight_defaults = {
                    "Skill Match": 40,
                    "Semantic Match": 25,
                    "ATS Score": 20,
                    "Experience Score": 15,
                }
                weight_cols = st.columns(len(weight_defaults))
                weight_inputs = {}
                for (label, default), col in zip(weight_defaults.items(), weight_cols):
                    weight_inputs[label] = col.slider(
                        f"{label} Weight (%)",
                        min_value=0,
                        max_value=100,
                        value=default,
                        step=5,
                        key=f"candidate_{label.lower().replace(' ', '_')}_weight",
                    )
                normalized_weights = normalize_weights(weight_inputs)
                st.caption(
                    f"Weights normalize automatically (current sum: {sum(weight_inputs.values())}%)."
                )

            skills = parse_skill_input(skill_input)
            evaluation = evaluate_resume(raw_resume_text, job_description, skills, normalized_weights)
            resume_text = evaluation["clean_resume"]
            job_text = evaluation["clean_job"]
            score_map = evaluation["scores"]
            total_score = evaluation["total_score"]
            matched = evaluation["matched"]
            missing = evaluation["missing"]

            if gemini_available:
                with st.spinner("🔍 Checking if uploaded document is a valid resume..."):
                    check = call_gemini(build_resume_check_prompt(resume_text))
                    if check.get("is_resume") is False:
                        st.error(f"❌ Not a Resume: {check.get('reason')}")
                        st.stop()

            st.subheader("📊 Resume Evaluation Results")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Overall Score", f"{total_score:.2f}%")
            c2.metric("Skill Match", f"{score_map['Skill Match']:.2f}%")
            c3.metric("Semantic Match", f"{score_map['Semantic Match']:.2f}%")
            c4.metric("ATS Score", f"{score_map['ATS Score']:.2f}%")
            c5.metric("Experience Score", f"{score_map['Experience Score']:.2f}%")

        st.caption("Weighted blend shown above is based on your configuration.")

        categories = list(score_map.keys())
        values = [score_map[label] for label in categories]
        if sum(values) == 0:
            st.warning("⚠️ Not enough valid data to generate chart.")
        else:
            colors = ["#4CAF50", "#2196F3", "#FFC107", "#9C27B0"]
            fig, ax = plt.subplots()
            bars = ax.bar(categories, values, color=colors)
            ax.set_ylim(0, 100)
            ax.set_ylabel("Score (%)")
            ax.set_title("Score Breakdown")
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 1,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom",
                )
            st.pyplot(fig)

        st.subheader("🔍 Skill Alignment Snapshot")
        col_a, col_b = st.columns(2)
        col_a.write("**Matched Skills:**" if matched else "**Matched Skills:** None detected")
        if matched:
            col_a.write(", ".join(sorted(set(matched))))
        col_b.write("**Missing Skills (from job):**" if missing else "**Missing Skills (from job):** All covered")
        if missing:
            col_b.write(", ".join(sorted(set(missing))))

        feedback_state_key = "candidate_ai_feedback"
        if feedback_state_key not in st.session_state:
            st.session_state[feedback_state_key] = None

        st.subheader("🧠 AI Suggestions & Resume Improvement")
        if st.button("✨ Generate AI Feedback", key="candidate_feedback_btn"):
            with st.spinner("Analyzing resume with Gemini..."):
                if gemini_available:
                    ai_feedback = call_gemini(
                        build_feedback_prompt(resume_text, job_text, matched, missing)
                    )
                else:
                    ai_feedback = local_feedback_fallback(matched, missing)

            if "error" in ai_feedback:
                st.error(f"Gemini Error: {ai_feedback.get('error')}")
            else:
                st.session_state[feedback_state_key] = ai_feedback
                st.success("AI suggestions generated. Scroll down to review them.")

        ai_feedback = st.session_state.get(feedback_state_key)
        if ai_feedback and "error" not in ai_feedback:
            st.write("**Summary Feedback:**", ai_feedback.get("summary_feedback"))
            st.write("**Skills Feedback:**", ai_feedback.get("skills_feedback"))
            st.write("**Projects Feedback:**", ai_feedback.get("projects_feedback"))
            st.write("**Keywords to Add:**", ", ".join(ai_feedback.get("keywords_to_add", [])))
            st.code(ai_feedback.get("rewritten_summary", "-"))

        guidance_state_key = "candidate_section_guidance"
        if guidance_state_key not in st.session_state:
            st.session_state[guidance_state_key] = None

        st.subheader("🎯 Advanced Coaching")
        col_guidance_btn, col_guidance_info = st.columns([1, 2])
        if col_guidance_btn.button("Generate Section Guidance", key="candidate_guidance_btn"):
            with st.spinner("Building personalized coaching plan..."):
                guidance = get_section_guidance(raw_resume_text, job_description, matched, missing)
            st.session_state[guidance_state_key] = guidance
            st.success("Coaching plan ready. Explore the tabs below.")
        else:
            guidance = st.session_state.get(guidance_state_key)

        tabs = st.tabs(["Section Guidance", "Keyword Gaps", "Preparation Checklist"])

        with tabs[0]:
            guidance = st.session_state.get(guidance_state_key)
            if guidance:
                st.caption(
                    "Use these actionable bullet points to fine-tune each section before you apply."
                )
                for section in ["summary", "experience", "skills", "education"]:
                    tips = guidance.get(section) or []
                    if tips:
                        st.markdown(f"**{section.title()} Focus**")
                        for tip in tips:
                            st.markdown(f"- {tip}")
                        st.markdown("---")
                keywords = guidance.get("keywords_to_emphasize", [])
                if keywords:
                    st.write("**Keywords to emphasize:**", ", ".join(dict.fromkeys(keywords)))
                questions = guidance.get("questions_to_reflect", [])
                if questions:
                    st.write("**Reflection prompts:**")
                    for q in questions:
                        st.markdown(f"- {q}")
                next_actions = guidance.get("next_actions", [])
                if next_actions:
                    st.write("**Immediate next actions:**")
                    for action in next_actions:
                        st.markdown(f"- {action}")
            else:
                st.info("Generate the coaching plan to unlock tailored tips.")

        with tabs[1]:
            if missing:
                st.caption("Target these gaps to align fully with the job description.")
                for skill in missing:
                    st.markdown(
                        f"- Add a bullet highlighting `{skill}` experience or training. Include metrics where possible."
                    )
            else:
                st.success("You're covering every key skill from the job description—great work!")

        with tabs[2]:
            st.caption("Track your prep progress as you polish materials and get interview-ready.")
            for idx, item in enumerate(CANDIDATE_CHECKLIST):
                key = f"candidate_checklist_{idx}"
                if key not in st.session_state:
                    st.session_state[key] = False
                st.checkbox(item, value=st.session_state.get(key, False), key=key)

        st.subheader("📥 Download Report")
        if st.button("Generate PDF Report", key="candidate_pdf_btn"):
            pdf = generate_pdf_report(
                uploaded_file.name,
                total_score,
                score_map["Skill Match"],
                matched,
                missing,
                score_map["ATS Score"],
                score_map["Experience Score"],
                normalized_weights,
                ai_feedback,
            )
            st.download_button(
                "📄 Download PDF",
                pdf,
                file_name=f"{uploaded_file.name}_AI_Report.pdf",
                mime="application/pdf",
            )

        if st.button("💾 Save assessment to history", key="candidate_save_btn"):
            candidate_record = {
                "timestamp": datetime.utcnow().isoformat(),
                "candidate_name": candidate_name or uploaded_file.name,
                "candidate_email": candidate_email,
                "target_role": candidate_role_goal,
                "resume_file": uploaded_file.name,
                "scores": score_map,
                "total_score": total_score,
                "matched_skills": matched,
                "missing_skills": missing,
                "weights": normalized_weights,
                "ai_feedback": ai_feedback if ai_feedback and "error" not in ai_feedback else None,
                "job_description_excerpt": job_description[:500],
                "workspace_code": workspace_code,
            }
            append_candidate_record(candidate_record)
            st.success("Assessment saved to your history.")

        with st.expander("📚 Previous Assessments"):
            history = (
                get_recent_candidate_records(email=candidate_email, workspace_code=workspace_code)
                if candidate_email
                else get_recent_candidate_records(workspace_code=workspace_code)
            )
            if history:
                for record in history:
                    st.markdown(
                        f"**{record.get('timestamp', 'N/A')}** — {record.get('resume_file')} (Overall {record.get('total_score', 0):.1f}%)"
                    )
                    st.caption(
                        f"Matched: {', '.join(record.get('matched_skills', [])) or 'None'} | Missing: {', '.join(record.get('missing_skills', [])) or 'None'}"
                    )
            else:
                st.write("No previous assessments yet.")

        st.subheader("💬 Workspace Messages")
        st.caption(
            "Collaborate with your recruiter by sharing updates or questions. Messages sync when you share the same workspace code."
        )
        message_history = get_recent_messages(channel=workspace_code)
        if message_history:
            for msg in message_history:
                role = msg.get("role", "participant").title()
                author = msg.get("author", "Anonymous")
                timestamp = msg.get("timestamp", "")
                st.markdown(f"**[{role}] {author}** — {timestamp}")
                st.write(msg.get("message", ""))
        else:
            st.write("No messages yet. Start the conversation below.")

        with st.form("candidate_message_form"):
            candidate_message = st.text_area("Message", height=120)
            send_message = st.form_submit_button("Send message")
            if send_message and candidate_message.strip():
                append_message(
                    {
                        "timestamp": datetime.utcnow().isoformat(),
                        "channel": workspace_code,
                        "author": candidate_name or "Candidate",
                        "author_email": candidate_email,
                        "role": "candidate",
                        "message": candidate_message.strip(),
                    }
                )
                st.success("Message sent.")
                st.experimental_rerun()

            else:
                st.info("👆 Upload a resume and paste the job description to begin.")
    
    elif user["role"] == "recruiter":
        st.header("Recruiter Screening Workspace")
        st.write(
            "Rank multiple resumes against a single job description, spot skill gaps, and export a shortlist for your hiring team."
        )
        
        recruiter_name = user["name"]
        recruiter_email = user["email"]
        recruiter_team = user.get("team", "")
        workspace_code = st.sidebar.text_input(
            "Workspace code (share with candidates)",
            value=st.session_state.get("workspace_code", "general"),
            key="recruiter_workspace_code",
        ).strip() or "general"
        st.session_state["workspace_code"] = workspace_code
        st.sidebar.caption("Share this code to collaborate on notes and messages.")

        job_description = st.text_area(
            "💼 Job Description",
            height=220,
            key="recruiter_job_description",
        )
        resume_files = st.file_uploader(
            "📂 Upload Candidate Resumes",
            type=["txt", "pdf"],
            accept_multiple_files=True,
            key="recruiter_resume_uploader",
        )

        with st.expander("⚙️ Scoring Configuration", expanded=False):
            skill_input = st.text_area(
                "Core Skill Keywords (comma-separated)",
                value=", ".join(DEFAULT_SKILLS),
                height=120,
                key="recruiter_skill_keywords",
            )
            weight_defaults = {
                "Skill Match": 35,
                "Semantic Match": 25,
                "ATS Score": 20,
                "Experience Score": 20,
            }
            weight_cols = st.columns(len(weight_defaults))
            weight_inputs = {}
            for (label, default), col in zip(weight_defaults.items(), weight_cols):
                weight_inputs[label] = col.slider(
                    f"{label} Weight (%)",
                    min_value=0,
                    max_value=100,
                    value=default,
                    step=5,
                    key=f"recruiter_{label.lower().replace(' ', '_')}_weight",
                )
            normalized_weights = normalize_weights(weight_inputs)
            shortlist_threshold = st.slider(
                "Shortlist threshold (Overall Score %)",
                min_value=0,
                max_value=100,
                value=70,
                step=5,
                key="shortlist_threshold",
            )
            verify_resumes = st.checkbox(
                "Verify resume format with Gemini (slower)",
                value=False,
                key="recruiter_verify_resumes",
            )
            st.caption(
                f"Weight sliders sum to {sum(weight_inputs.values())}% and normalize automatically for scoring."
            )
            if verify_resumes and not gemini_available:
                st.warning(
                    "Gemini API is not configured, so verification will be skipped despite the checkbox.",
                    icon="⚠️",
                )

        if job_description and resume_files:
            skills = parse_skill_input(skill_input)
            results = []
            missing_counter = Counter()
            matched_counter = Counter()
            existing_notes = {
                note.get("candidate_id"): note
                for note in get_candidate_notes(recruiter_email=recruiter_email)
            }

        for file in resume_files:
            if file.name.lower().endswith(".pdf"):
                resume_bytes = file.read()
                raw_resume_text = extract_text_from_pdf(io.BytesIO(resume_bytes))
            else:
                raw_resume_text = file.read().decode("utf-8", errors="ignore")
            file.seek(0)

            if not raw_resume_text.strip():
                st.warning(f"⚠️ Unable to read content from {file.name}. Skipping.")
                continue

            evaluation = evaluate_resume(raw_resume_text, job_description, skills, normalized_weights)

            if verify_resumes and gemini_available:
                check = call_gemini(build_resume_check_prompt(evaluation["clean_resume"]))
                if check.get("is_resume") is False:
                    st.warning(f"❌ {file.name} flagged as non-resume ({check.get('reason')}). Skipping.")
                    continue

            matched = evaluation["matched"]
            missing = evaluation["missing"]
            matched_counter.update(matched)
            missing_counter.update(missing)

            scores = evaluation["scores"]
            candidate_id = f"{file.name}-{abs(hash((file.name, evaluation['clean_resume'][:200]))) % 10**8}"
            note_data = existing_notes.get(candidate_id, {})
            current_stage = note_data.get("stage", INTERVIEW_STAGES[0])
            results.append(
                {
                    "Candidate ID": candidate_id,
                    "Candidate": file.name,
                    "Overall Score": evaluation["total_score"],
                    "Skill Match": scores["Skill Match"],
                    "Semantic Match": scores["Semantic Match"],
                    "ATS Score": scores["ATS Score"],
                    "Experience Score": scores["Experience Score"],
                    "Stage": current_stage,
                    "Matched Skills": ", ".join(sorted(set(matched))) if matched else "-",
                    "Missing Skills": ", ".join(sorted(set(missing))) if missing else "-",
                }
            )

        if results:
            df = pd.DataFrame(results).sort_values("Overall Score", ascending=False).reset_index(drop=True)
            display_df = df.drop(columns=["Candidate ID"]) if "Candidate ID" in df.columns else df

            st.subheader("🏆 Ranked Candidates")
            st.dataframe(display_df, use_container_width=True)

            top_candidate = df.iloc[0]
            st.success(
                f"Top candidate right now: {top_candidate['Candidate']} with an overall match of {top_candidate['Overall Score']:.2f}%"
            )

            shortlisted = df[df["Overall Score"] >= shortlist_threshold]
            if not shortlisted.empty:
                st.info(
                    f"✅ {len(shortlisted)} candidate(s) meet or exceed the {shortlist_threshold}% shortlist threshold."
                )
                st.dataframe(
                    shortlisted[["Candidate", "Overall Score", "Skill Match", "Experience Score", "Stage"]],
                    use_container_width=True,
                )
            else:
                st.warning(
                    "No candidates crossed your shortlist threshold yet. Consider adjusting the weights or threshold."
                )

            csv_data = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "📥 Download ranked candidates (CSV)",
                csv_data,
                file_name="resume_screening_results.csv",
                mime="text/csv",
            )

            recruiter_pdf = generate_recruiter_pdf(
                job_description,
                shortlisted if not shortlisted.empty else df.head(5),
                normalized_weights,
                shortlist_threshold,
                recruiter_name=recruiter_name,
            )
            st.download_button(
                "📄 Download recruiter summary (PDF)",
                recruiter_pdf,
                file_name="recruiter_screening_summary.pdf",
                mime="application/pdf",
            )

            ats_export = df[
                [
                    "Candidate",
                    "Overall Score",
                    "Skill Match",
                    "Semantic Match",
                    "ATS Score",
                    "Experience Score",
                    "Stage",
                    "Matched Skills",
                    "Missing Skills",
                ]
            ]
            st.download_button(
                "📤 Export for ATS (CSV)",
                ats_export.to_csv(index=False).encode("utf-8"),
                file_name="ats_ready_export.csv",
                mime="text/csv",
            )

            st.subheader("🔍 Candidate Comparison")
            candidate_choices = df["Candidate"].tolist()
            selected_candidates = st.multiselect(
                "Select up to 2 candidates to compare",
                candidate_choices,
                max_selections=2,
                key="compare_candidates",
            )
            if selected_candidates:
                comparison_df = df[df["Candidate"].isin(selected_candidates)][
                    [
                        "Candidate",
                        "Overall Score",
                        "Skill Match",
                        "Semantic Match",
                        "ATS Score",
                        "Experience Score",
                        "Stage",
                        "Matched Skills",
                        "Missing Skills",
                    ]
                ].reset_index(drop=True)
                st.dataframe(comparison_df, use_container_width=True)

            stage_counts = df["Stage"].value_counts()
            if not stage_counts.empty:
                fig_stage, ax_stage = plt.subplots()
                stage_counts.plot(kind="bar", color="#607d8b", ax=ax_stage)
                ax_stage.set_title("Pipeline Distribution")
                ax_stage.set_ylabel("Candidates")
                st.pyplot(fig_stage)

            if matched_counter:
                st.subheader("💪 Frequently Matched Skills")
                top_matched = ", ".join(
                    f"{skill} ({count})" for skill, count in matched_counter.most_common(10)
                )
                st.write(top_matched)

            if missing_counter:
                st.subheader("⚠️ Skill Gaps Across Candidates")
                top_missing = ", ".join(
                    f"{skill} ({count})" for skill, count in missing_counter.most_common(10)
                )
                st.write(top_missing)

            with st.expander("📄 Candidate-by-candidate notes"):
                for row in df.to_dict("records"):
                    st.markdown(f"**{row['Candidate']}**")
                    note_key_prefix = f"{row['Candidate ID']}_{recruiter_email or 'anon'}"
                    note_data = existing_notes.get(row["Candidate ID"], {})
                    default_stage = note_data.get("stage", row.get("Stage", INTERVIEW_STAGES[0]))
                    stage_index = (
                        INTERVIEW_STAGES.index(default_stage)
                        if default_stage in INTERVIEW_STAGES
                        else 0
                    )
                    stage_choice = st.selectbox(
                        "Stage",
                        INTERVIEW_STAGES,
                        index=stage_index,
                        key=f"{note_key_prefix}_stage",
                    )
                    note_text = st.text_area(
                        "Notes",
                        value=note_data.get("notes", ""),
                        key=f"{note_key_prefix}_notes",
                        height=120,
                    )
                    next_steps = st.text_input(
                        "Next steps",
                        value=note_data.get("next_steps", ""),
                        key=f"{note_key_prefix}_next",
                    )
                    if st.button("Save note", key=f"{note_key_prefix}_save"):
                        note_payload = {
                            "candidate_id": row["Candidate ID"],
                            "candidate_name": row["Candidate"],
                            "recruiter_email": recruiter_email,
                            "recruiter_name": recruiter_name,
                            "stage": stage_choice,
                            "notes": note_text,
                            "next_steps": next_steps,
                            "updated_at": datetime.utcnow().isoformat(),
                        }
                        upsert_candidate_note(note_payload)
                        st.success("Saved candidate note.")
                        st.experimental_rerun()

            if st.button("💾 Save screening session", key="recruiter_save_session"):
                recruiter_record = {
                    "timestamp": datetime.utcnow().isoformat(),
                    "recruiter_name": recruiter_name,
                    "recruiter_email": recruiter_email,
                    "recruiter_team": recruiter_team,
                    "job_description_excerpt": job_description[:500],
                    "weights": normalized_weights,
                    "shortlist_threshold": shortlist_threshold,
                    "results": df.to_dict("records"),
                    "workspace_code": workspace_code,
                }
                append_recruiter_session(recruiter_record)
                st.success("Screening session saved.")

            st.subheader("📅 Schedule Interviews")
            invite_col1, invite_col2 = st.columns(2)
            candidate_for_invite = invite_col1.selectbox(
                "Candidate",
                df["Candidate"].tolist(),
                key="invite_candidate",
            )
            invite_date = invite_col1.date_input(
                "Date",
                key="invite_date",
            )
            invite_time = invite_col2.time_input(
                "Start time",
                key="invite_time",
            )
            duration_minutes = invite_col2.number_input(
                "Duration (minutes)",
                min_value=15,
                max_value=240,
                value=45,
                step=15,
                key="invite_duration",
            )
            invite_description = st.text_area(
                "Agenda / Notes",
                value="Interview with candidate",
                key="invite_description",
            )
            invite_location = st.text_input(
                "Location / Video link",
                value="Virtual",
                key="invite_location",
            )
            if st.button("Generate calendar invite", key="generate_invite_btn"):
                start_dt = datetime.combine(invite_date, invite_time)
                title = f"Interview: {candidate_for_invite}"
                invite_file = build_calendar_invite(
                    title,
                    start_dt,
                    duration_minutes=duration_minutes,
                    description=invite_description,
                    location=invite_location,
                )
                st.download_button(
                    "📥 Download .ics",
                    invite_file,
                    file_name=f"{candidate_for_invite.replace(' ', '_')}_interview.ics",
                    mime="text/calendar",
                )
            else:
                st.warning("No usable resumes processed yet. Double-check uploaded files or disable Gemini verification.")

        else:
            st.info("👆 Upload at least one resume and paste the job description to analyze your candidate pool.")

        with st.expander("📑 Recent Screening Sessions"):
            sessions = get_recent_recruiter_sessions(
                recruiter_email=recruiter_email,
                workspace_code=workspace_code,
            )
            if sessions:
                for session in sessions:
                    st.markdown(
                        f"**{session.get('timestamp', 'N/A')}** — {len(session.get('results', []))} candidate(s) scored"
                    )
                    st.caption(
                        f"Threshold: {session.get('shortlist_threshold', 0)}% | Top score: {max((r.get('Overall Score', 0) for r in session.get('results', [])), default=0):.1f}%"
                    )
            else:
                st.write("No historical sessions yet.")