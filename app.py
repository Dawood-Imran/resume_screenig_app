import pdfplumber
import pickle
import re
from nltk.corpus import stopwords
import string
import streamlit as st

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Ensure NLTK stopwords are available
import nltk
nltk.download('stopwords')

# Text cleaning functions
def remove_url(text):
    pattern = re.compile(r'https?://\S+|www\.\S+')
    return pattern.sub(r'', text)

exclude = string.punctuation
def remove_punc1(text):
    return text.translate(str.maketrans('', '', exclude))

def remove_stopwords(text):
    new_text = []
    for word in text.split():
        if word in stopwords.words('english'):
            new_text.append('')
        else:
            new_text.append(word)
    return " ".join(new_text)

def clean_resume(text):
    clean_text = text.lower()  # Ensure consistency with training data
    clean_text = re.sub('http\S+\s', ' ', clean_text)
    clean_text = re.sub('@\S+', ' ', clean_text)
    clean_text = re.sub('#\S+', ' ', clean_text)
    clean_text = re.sub(r'[^\w\s.,&()\'-]', ' ', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text)
    clean_text = remove_punc1(clean_text)
    clean_text = remove_stopwords(clean_text)
    return clean_text

# Categories
categories = [
    'Advocate', 'Arts', 'Automation Testing', 'Blockchain', 'Business Analyst',
    'Civil Engineer', 'Data Science', 'Database', 'DevOps Engineer', 'DotNet Developer',
    'ETL Developer', 'Electrical Engineering', 'HR', 'Hadoop', 'Health and fitness',
    'Java Developer', 'Mechanical Engineer', 'Network Security Engineer', 'Operations Manager',
    'PMO', 'Python Developer', 'SAP Developer', 'Sales', 'Testing', 'Web Designing'
]

# Prediction function
def make_prediction(text):
    cleaned_text = clean_resume(text)
    transformed_text = vectorizer.transform([cleaned_text])
    prediction = model.predict(transformed_text)
    predicted_category = categories[prediction[0]]
    return f"The predicted Resume is for : '{predicted_category}'"

# Extract text from PDF using pdfplumber
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Streamlit interface
def main():
    st.title("Resume Screening App")
    uploaded_file = st.file_uploader("Upload your Resume", type=["pdf", 'txt'])

    if uploaded_file is not None:
        if uploaded_file.name.endswith('.pdf'):
            text = extract_text_from_pdf(uploaded_file)
        else:
            try:
                text = uploaded_file.read().decode('utf-8')
            except UnicodeDecodeError:
                text = uploaded_file.read().decode('latin-1')

        result = make_prediction(text)
        st.write(result)

    else:
        st.info("Upload a resume to get started")


    
    

if __name__ == '__main__':
    main()
