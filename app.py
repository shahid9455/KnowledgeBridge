import streamlit as st
import speech_recognition as sr
from io import BytesIO
import fitz  # PyMuPDF
from docx import Document
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import requests
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io

# IBM Watson NLU Configuration
api_key_watson = 'IHbYzsY18Sl7i3Wr-_9YrYjpARDKZRnkO2ETjR5mfvnP'
nlu_url = 'https://api.au-syd.natural-language-understanding.watson.cloud.ibm.com/instances/3f07153d-defe-42a0-8215-b0d2d480d44f'

# Initialize IBM Watson NLU
authenticator = IAMAuthenticator(api_key_watson)
nlu = NaturalLanguageUnderstandingV1(
    version='2021-08-01',
    authenticator=authenticator
)
nlu.set_service_url(nlu_url)

# AIML API Configuration
class AIMLClient:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url

    def chat_completions_create(self, model, messages):
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        data = {"model": model, "messages": messages}
        response = requests.post(f"{self.base_url}/chat/completions", headers=headers, json=data)
        response.raise_for_status()
        return response.json()

aiml_client = AIMLClient(api_key="45228194012549f09d70dd18da5ff8a8", base_url="https://api.aimlapi.com")

# Define the filename where text will be stored
filename = 'text_storage_with_keywords.txt'

# Initialize session state
if "is_recording" not in st.session_state:
    st.session_state.is_recording = False

if "transcribed_text" not in st.session_state:
    st.session_state.transcribed_text = ""

if "uploaded_text" not in st.session_state:
    st.session_state.uploaded_text = ""

# Streamlit app
st.title("Knowledge and Experience Storage System")

# Voice Recording Section
st.header("Voice Recording")
start_button = st.button("Start Recording", key="start_recording")

if start_button:
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    with mic as source:
        st.write("Listening...")
        audio = recognizer.listen(source)

        try:
            # Recognize speech using Google Web Speech API
            text = recognizer.recognize_google(audio)
            st.session_state.transcribed_text = text
        except sr.UnknownValueError:
            st.session_state.transcribed_text = "Google Speech Recognition could not understand audio"
        except sr.RequestError:
            st.session_state.transcribed_text = "Could not request results from Google Speech Recognition service"

# Text Input Section
st.header("Text Input")
text_input = st.text_area("Enter text to save:", value=st.session_state.transcribed_text)

if st.button("Save Text", key="save_text"):
    if text_input:
        try:
            response = nlu.analyze(
                text=text_input,
                features=Features(
                    keywords=KeywordsOptions(limit=15)  # Increase the limit for more keywords
                )
            ).get_result()

            keywords = [kw['text'] for kw in response['keywords']]
            keyword_string = ', '.join(keywords)

            with open(filename, 'a') as file:
                file.write(f"Text: {text_input}\nKeywords: {keyword_string}\n\n")

            st.success("Your input and extracted keywords have been saved successfully.")
        except Exception as e:
            st.error(f"An error occurred while processing the text: {str(e)}")
    else:
        st.warning("No text entered to save.")

# Search Section
st.header("Search")
query_input = st.text_area("Enter keyword or full prompt to search:")

if st.button("Search", key="search"):
    if query_input:
        try:
            # Extract keywords from the query input using NLU
            response = nlu.analyze(
                text=query_input,
                features=Features(
                    keywords=KeywordsOptions(limit=15)  # Increase the limit for more keywords
                )
            ).get_result()

            query_keywords = [kw['text'] for kw in response['keywords']]
            keyword_string = ', '.join(query_keywords)

            # Read the text file and get all keywords and texts
            with open(filename, 'r') as file:
                lines = file.readlines()
                all_keywords = []
                all_texts = []

                for line in lines:
                    if 'Keywords:' in line:
                        stored_keywords = [kw.strip() for kw in line.replace('Keywords:', '').split(',')]
                        all_keywords.append(stored_keywords)
                    if 'Text:' in line:
                        all_texts.append(line.replace('Text:', '').strip())

            # Find matching texts based on the extracted keywords
            matching_texts = []
            for text in all_texts:
                for keywords in all_keywords:
                    if any(keyword.lower() in text.lower() for keyword in query_keywords):
                        matching_texts.append(text)
                        break

            # Generate refined text using the AIML model if there are matching texts
            if matching_texts:
                response = aiml_client.chat_completions_create(
                    model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
                    messages=[
                        {"role": "system", "content": "You are an AI assistant who knows everything."},
                        {"role": "user", "content": f"Refine the following text into a professional and polished summary without asking for additional data:\n\n{' '.join(matching_texts)}"}
                    ]
                )
                refined_text = response['choices'][0]['message']['content'].strip()
                if refined_text:
                    st.subheader("Refined Text:")
                    st.write(refined_text)
                    pdf_button = st.button("Download as PDF", key="download_pdf")
                    if pdf_button:
                        buffer = io.BytesIO()
                        c = canvas.Canvas(buffer, pagesize=letter)
                        width, height = letter

                        c.drawString(100, height - 100, "Here is a polished and professional summary:")
                        text_object = c.beginText(100, height - 120)
                        text_object.setFont("Helvetica", 12)
                        text_object.setTextOrigin(100, height - 140)
                        text_object.textLines(refined_text)
                        c.drawText(text_object)
                        c.showPage()
                        c.save()

                        buffer.seek(0)
                        st.download_button(label="Download PDF", data=buffer, file_name="refined_text_summary.pdf", mime="application/pdf")
                else:
                    st.warning("No text generated.")
            else:
                st.warning(f"No matching content found for '{query_input}'.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter your keyword or full prompt to search.")

# File Upload Section
st.header("Upload PDF or DOC File")
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])

if uploaded_file is not None:
    file_type = uploaded_file.type
    if file_type == "application/pdf":
        # Extract text from PDF
        pdf_file = BytesIO(uploaded_file.read())
        pdf_document = fitz.open(stream=pdf_file, filetype="pdf")
        pdf_text = ""
        for page in pdf_document:
            pdf_text += page.get_text()
        pdf_document.close()
        st.session_state.uploaded_text = pdf_text

    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        # Extract text from DOCX
        docx_file = BytesIO(uploaded_file.read())
        doc = Document(docx_file)
        doc_text = ""
        for paragraph in doc.paragraphs:
            doc_text += paragraph.text + "\n"
        st.session_state.uploaded_text = doc_text

    # Update the text input field with the extracted text
    st.session_state.transcribed_text = st.session_state.uploaded_text
