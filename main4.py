import os
import PyPDF2
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Streamlit Page Config
st.set_page_config(page_title="Resume Screener", page_icon="📄", layout="wide")

# Custom Styles (Dark Background, White Inputs, and Styled Job Role Text)
st.markdown(
    """
    <style>
        body { background-color: #333; color: white; }
        .stTextInput, .stNumberInput, .stFileUploader { background-color: white !important; color: black !important; border-radius: 10px; }
        .stButton>button { background-color: #555; color: white; border-radius: 10px; font-size: 18px; }
        .stButton>button:hover { background-color: #777; transform: scale(1.05); transition: 0.3s ease-in-out; }
        .custom-text { font-size: 22px; font-weight: bold; color: white; }
        .custom-title { font-size: 32px; font-weight: bold; color: #FFD700; }
        .custom-subtitle { font-size: 20px; font-weight: bold; color: #FFD700; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Paths
RESUME_FOLDER_PATH = "E:/Resume_Screener/resumes"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_source):
    text = ""
    reader = PyPDF2.PdfReader(pdf_source)
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + " "
    return text.lower()

# Function to calculate similarity score using TF-IDF and cosine similarity
def calculate_similarity(job_role, resume_texts):
    vectorizer = TfidfVectorizer()
    documents = [job_role] + resume_texts
    tfidf_matrix = vectorizer.fit_transform(documents)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return cosine_similarities

# Streamlit UI
st.markdown("<h1 class='custom-title'>📄 Resume Screener</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='custom-subtitle'>Find the best candidate for your job role! 🔍</h3>", unsafe_allow_html=True)

# Styled Job Role Input
st.markdown("<p class='custom-text'>🔍 Enter Job Role:</p>", unsafe_allow_html=True)
job_role = st.text_input("", key="job_input", placeholder="Enter job role here")  # Empty label to use custom styling

top_n = st.number_input("📌 Number of Top Candidates to Display:", min_value=1, value=5)
uploaded_files = st.file_uploader("📂 Upload Resume(s) (PDF)", type=["pdf"], accept_multiple_files=True, key="upload_box")

if uploaded_files:
    st.success(f"✅ {len(uploaded_files)} Resume(s) Uploaded Successfully!")

if st.button("🔎 Search Resumes", key="search_btn"):
    if not job_role:
        st.warning("Please enter a job role.")
    elif not uploaded_files:
        st.warning("Please upload at least one resume.")
    else:
        resume_texts = []
        for uploaded_file in uploaded_files:
            resume_text = extract_text_from_pdf(uploaded_file)
            resume_texts.append(resume_text)

        similarity_scores = calculate_similarity(job_role, resume_texts)

        results = list(zip([uploaded_file.name for uploaded_file in uploaded_files], similarity_scores))
        results.sort(key=lambda x: x[1], reverse=True)

        st.markdown("<h3 class='custom-subtitle'>Top Candidates:</h3>", unsafe_allow_html=True)
        for i, (file_name, score) in enumerate(results[:top_n], start=1):
            st.markdown(f"<p class='custom-text'>{i}. {file_name} - Similarity Score: {score * 100:.2f}%</p>", unsafe_allow_html=True)