import streamlit as st
import speech_recognition as sr
from io import BytesIO
import fitz  # PyMuPDF
from docx import Document
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import requests
from fpdf import FPDF

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

if "search_results" not in st.session_state:
    st.session_state.search_results = []

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

# Function to generate PDF
def generate_pdf(texts):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for text in texts:
        pdf.multi_cell(0, 10, text)
        pdf.ln(5)
    return pdf

# Apply custom CSS for black boxes
st.markdown("""
    <style>
        .stTextInput, .stButton, .stTextArea, .stDownloadButton, .stFileUploader {
            background-color: black !important;
            color: white !important;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit app with separate pages
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Input", "Search"])

if page == "Input":
    st.title("KnowledgeBridge")

    # Voice Recording Section
    st.header("Voice Recording")
    start_button = st.button("Start Recording", key="start_recording")

    if start_button:
        recognizer = sr.Recognizer()

        try:
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
        except OSError:
            st.error("No Default Input Device Available. Please connect a microphone and try again.")

    # Text Input Section
    st.header("Text Input")
    text_input = st.text_area("Enter text to save:", value=st.session_state.transcribed_text)

    if st.button("Save Text", key="save_text"):
        if text_input:
            try:
                response = nlu.analyze(
                    text=text_input,
                    features=Features(
                        keywords=KeywordsOptions(limit=15)
                    )
                ).get_result()

                keywords = [kw['text'] for kw in response['keywords']]
                keyword_string = ', '.join(keywords)

                with open(filename, 'a') as file:
                    file.write(f"Text: {text_input}\nKeywords: {keyword_string}\n\n")

                st.session_state.transcribed_text = ""  # Clear the input field after saving
                st.success("Your input and extracted keywords have been saved successfully.")
            except Exception as e:
                st.error(f"An error occurred while processing the text: {str(e)}")
        else:
            st.warning("No text entered to save.")

    # File Upload Section
    st.header("Upload PDF or DOC File")
    uploaded_file = st.file_uploader("Choose a file", type=["pdf", "docx"])

    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            try:
                with fitz.open(stream=uploaded_file.read(), filetype="pdf") as pdf:
                    text = ""
                    for page_num in range(pdf.page_count):
                        page = pdf[page_num]
                        text += page.get_text()

                st.session_state.uploaded_text = text
                st.success("PDF content successfully extracted and stored.")
                st.text_area("Extracted Text from PDF:", value=st.session_state.uploaded_text, height=300)
            except Exception as e:
                st.error(f"An error occurred while extracting text from the PDF: {str(e)}")

        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            try:
                doc = Document(uploaded_file)
                text = "\n".join([para.text for para in doc.paragraphs])

                st.session_state.uploaded_text = text
                st.success("DOCX content successfully extracted and stored.")
                st.text_area("Extracted Text from DOCX:", value=st.session_state.uploaded_text, height=300)
            except Exception as e:
                st.error(f"An error occurred while extracting text from the DOCX: {str(e)}")

        # Save extracted text with keywords
        if st.button("Save Extracted Text with Keywords", key="save_extracted_text"):
            if st.session_state.uploaded_text:
                try:
                    response = nlu.analyze(
                        text=st.session_state.uploaded_text,
                        features=Features(
                            keywords=KeywordsOptions(limit=15)
                        )
                    ).get_result()

                    keywords = [kw['text'] for kw in response['keywords']]
                    keyword_string = ', '.join(keywords)

                    with open(filename, 'a') as file:
                        file.write(f"Text: {st.session_state.uploaded_text}\nKeywords: {keyword_string}\n\n")

                    st.session_state.uploaded_text = ""  # Clear the text after saving
                    st.success("Extracted text and keywords have been saved successfully.")
                except Exception as e:
                    st.error(f"An error occurred while processing the text: {str(e)}")
            else:
                st.warning("No extracted text to save.")

elif page == "Search":
    st.title("Search Stored Knowledge")

    st.header("Search")
    query_input = st.text_area("Enter one or more keywords to search:", value=st.session_state.query_input)

    if st.button("Search", key="search"):
        if query_input:
            try:
                query_keywords = [kw.strip() for kw in query_input.split()]
                keyword_string = ', '.join(query_keywords)

                all_keywords = []
                all_texts = []
                
                try:
                    with open(filename, 'r') as file:
                        lines = file.readlines()

                        for line in lines:
                            if 'Keywords:' in line:
                                stored_keywords = [kw.strip().lower() for kw in line.replace('Keywords:', '').split(',')]
                                all_keywords.append(stored_keywords)
                            if 'Text:' in line:
                                all_texts.append(line.replace('Text:', '').strip())
                except FileNotFoundError:
                    st.error("The text file was not found. Please save some text first.")
                    all_keywords = []
                    all_texts = []

                matching_texts = []
                for text, keywords in zip(all_texts, all_keywords):
                    if any(query_kw.lower() in ' '.join(keywords) for query_kw in query_keywords):
                        matching_texts.append(text)

                if not matching_texts:
                    response = aiml_client.chat_completions_create(
                        model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
                        messages=[
                            {"role": "system", "content": "You are an AI assistant who knows everything."},
                            {"role": "user", "content": f"Provide related data for the following keywords: {keyword_string}"}
                        ]
                    )
                    related_data = response['choices'][0]['message']['content'].strip()
                    
                    if related_data:
                        output_text = f"You: {query_input}\n\nKnowledgeBridge:\n\n *Your search result is not in the database but here is the related data:*\n\n{related_data}"
                    else:
                        output_text = f"You: {query_input}\n\nKnowledgeBridge: Sorry, no relevant data found."
                else:
                    output_text = f"You: {query_input}\n\nKnowledgeBridge:\n\n *Your search result is in the database:*\n\n{matching_texts[0]}"

                st.text_area("Search Result:", value=output_text, height=300)

            except Exception as e:
                st.error(f"An error occurred during the search: {str(e)}")
        else:
            st.warning("No keywords entered for search.")

    # Export Results Section
    if st.button("Export Results", key="export_results"):
        if st.session_state.search_results:
            pdf = generate_pdf(st.session_state.search_results)
            pdf_buffer = BytesIO()
            pdf.output(pdf_buffer)
            pdf_buffer.seek(0)

            st.download_button(
                label="Download PDF",
                data=pdf_buffer,
                file_name="search_results.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("No search results to export.")
