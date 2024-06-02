import streamlit as st
import PyPDF2
import pdfplumber
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Define custom CSS styles
st.markdown(
    """
    <style>
    .title {
        font-size: 36px;
        color: #1E90FF;
        text-align: center;
    }
    .subheader {
        font-size: 24px;
        color: #4682B4;
        margin-top: 20px;
        text-align:center;
    }
    .caption {
        color: black;
        margin-top: 30px;
        text-align: justify;
    }
    .button {
        background-color: #1E90FF;
        color: white;
        padding: 10px 20px;
        border: none;
        border-radius: 5px;
        cursor: pointer;
        font-size: 18px;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Custom function to process similarity
def getResult(JD_txt, resume_txt):
    content = [JD_txt, resume_txt]
    cv = CountVectorizer()
    matrix = cv.fit_transform(content)
    similarity_matrix = cosine_similarity(matrix)
    match = similarity_matrix[0][1] * 100
    return round(match, 2)

# Page layout
st.markdown("<h1 class='title'>RESUME SCREENING</h1>", unsafe_allow_html=True)
st.markdown("<h2 class='subheader'>ML Based Resume Screening</h2>", unsafe_allow_html=True)
st.markdown(
    "<p class='caption'>Aim of this project is to check whether a candidate is qualified for a role based his or her education, experience, and other information captured on their resume. In a nutshell, it's a form of pattern matching between a job's requirements and the qualifications of a candidate based on their resume.</p>",
    unsafe_allow_html=True
)

# File uploaders and process button
uploadedJD = st.file_uploader("Upload Job Description (PDF)", type="pdf")
uploadedResume = st.file_uploader("Upload Resume (PDF)", type="pdf")
click = st.button("Process")

# Process uploaded files
if uploadedJD and uploadedResume and click:
    try:
        with pdfplumber.open(uploadedJD) as pdf:
            pages = pdf.pages[0]
            job_description = pages.extract_text()
    except:
        st.write("Error processing Job Description PDF.")

    try:
        with pdfplumber.open(uploadedResume) as pdf:
            pages = pdf.pages[0]
            resume = pages.extract_text()
    except:
        st.write("Error processing Resume PDF.")

    match = getResult(job_description, resume)
    st.markdown(f"<h1 class='caption'>Match Percentage: {match}%</h1>", unsafe_allow_html=True)

# Footer
st.markdown("<p class='caption'> ~ made by Ishwari, Priyal, Pallavi, Saloni</p>", unsafe_allow_html=True)
