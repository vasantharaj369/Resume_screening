import streamlit as st
import pickle
import re
import PyPDF2
import io
import base64

def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
set_background('pexels-francesco-ungaro-281260.jpg')


model = pickle.load(open('model.pkl','rb'))
tf = pickle.load(open('tf.pkl', 'rb'))

category = {15: "Java Developer", 23: "Testing", 8: "DevOps Engineer",
             20: "Python Developer", 24: "Web Designing", 12: "HR",
             13: "Hadoop", 3: "Blockchain", 10: "ETL Developer",
             18: "Operations Manager", 6: "Data Science", 22: "Sales",
             16: "Mechanical Engineer", 1: "Arts", 7: "Database",
             11: "Electrical Engineering", 14: "Health and fitness",
             19: "PMO", 4: "Business Analyst", 9: "DotNet Developer",
             2: "Automation Testing", 17: "Network Security Engineer",
             21: "SAP Developer", 5: "Civil Engineer", 0: "Advocate"}

def clean(x):
    xxx = x.lower().split()
    y = []
    for x in xxx:
        x = re.sub('http\S+', '', x)
        x = re.sub('.com\S+', '', x)
        x = re.sub('#\S+', '', x)
        x = re.sub('[^a-zA-Z]', ' ', x)
        y.append(x.strip())
    return (' '.join(y))


col1, col2, col3 = st.columns([2, 4, 1])
with col1:
    st.write('')
with col2:
    st.header('Resume Screening')
    st.write('')
    st.write('')
with col3:
    st.write('')

file = st.file_uploader('Upload Resume', type = ['pdf'])
col1, col2, col3 = st.columns([2,1,1])
with col1:
    st.write('')
with col2:
    st.write('OR')
with col3:
    st.write('')

value = st.text_area('Enter your resume details', height= 150)


if st.button('Predict'):
    if len(value) > 10:
        vectorize = tf.transform([value]).toarray()
        result = model.predict(vectorize)
        st.header('Category ------- ' + str(category[result[0]]))
    elif file is not None:
        content = file.read()
        resume_bytes = io.BytesIO(content)
        resume_text = ""
        reader = PyPDF2.PdfReader(resume_bytes)
        for page_number in range(len(reader.pages)):
            page = reader.pages[page_number]
            resume_text += page.extract_text()
        cleaned = clean(resume_text)
        hello = tf.transform([cleaned]).toarray()
        result = model.predict(hello)
        st.header('Category ------- ' + str(category[result[0]]))

