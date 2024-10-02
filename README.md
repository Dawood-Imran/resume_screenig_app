Welcome to the **Resume Screening App**! This application leverages **Machine Learning** to predict the most suitable job category for a resume. Just upload a PDF or text file, and let the app do the magic! ✨

## 🚀 Features
- Upload a resume in **PDF** or **text** format.
- Cleans and processes resumes using **NLP** techniques.
- Predicts job categories from 25 different fields, like **Data Science**, **Blockchain**, **DevOps**, and more! 💼

## 🛠️ How it works:
1. The app uses **TF-IDF** vectorization to process resumes.
2. A pre-trained **K-Nearest Neighbors (KNN)** model is used to classify the resume into one of the job roles.
3. Get an instant prediction and see the result! 📊

## 📝 Requirements
- `streamlit`
- `pdfplumber` (for PDF extraction)
- `nltk`
- `scikit-learn`
- `pickle` (for loading pre-trained model)

## 📂 Files in the project:
app.py: The main Streamlit app for resume prediction.
model.pkl: The pre-trained KNN model.
vectorizer.pkl: TF-IDF vectorizer used for transforming resume text.
