import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import re
import matplotlib.pyplot as plt
import numpy as np

st.title("🤖 AI Resume Screener")

# Load model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Clean text
def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip().lower()

# Extract skills (basic list)
def extract_skills(text):
    skill_keywords = [
        'python', 'java', 'sql', 'excel', 'machine learning', 'data analysis',
        'communication', 'problem solving', 'leadership', 'project management',
        'javascript', 'html', 'css', 'power bi', 'tableau', 'deep learning',
        'testing', 'automation', 'devops', 'django', 'flask'
    ]
    found = [skill for skill in skill_keywords if re.search(r'\b' + skill + r'\b', text)]
    return set(found)

# Upload + Job Description input
uploaded_file = st.file_uploader("📄 Upload your Resume (PDF)", type=["pdf"])
job_desc = st.text_area("🧠 Paste the Job Description here:")

if uploaded_file and job_desc:
    resume_text = extract_text_from_pdf(uploaded_file)
    resume_text = clean_text(resume_text)
    job_desc = clean_text(job_desc)

    # Convert to embeddings
    resume_emb = model.encode(resume_text, convert_to_tensor=True)
    job_emb = model.encode(job_desc, convert_to_tensor=True)

    # Calculate similarity
    similarity = util.pytorch_cos_sim(resume_emb, job_emb)
    score = float(similarity[0][0]) * 100

    st.subheader(f"🧮 Match Score: {score:.2f}%")

    # ---- SKILL MATCH ANALYSIS ----
    st.divider()
    st.subheader("🧩 Skill Analysis")

    resume_skills = extract_skills(resume_text)
    job_skills = extract_skills(job_desc)

    matched_skills = resume_skills.intersection(job_skills)
    missing_skills = job_skills - resume_skills

    st.write("✅ **Skills Found in Resume:**", ", ".join(resume_skills) if resume_skills else "None")
    st.write("🧠 **Required Skills (from Job):**", ", ".join(job_skills) if job_skills else "None")
    st.write("❌ **Missing Skills:**", ", ".join(missing_skills) if missing_skills else "None")

    # ---- RECOMMENDATION ----
    st.divider()
    st.subheader("📋 Recommendation")

    if score > 70 and len(missing_skills) < 3:
        st.success("✅ Strong Match! Your resume aligns well with this job.")
    elif score > 40:
        st.warning(f"⚠️ Moderate Match. Consider adding: {', '.join(missing_skills) if missing_skills else 'some domain-related skills.'}")
    else:
        st.error(f"❌ Weak Match. You may want to improve resume sections and add: {', '.join(missing_skills)}")
