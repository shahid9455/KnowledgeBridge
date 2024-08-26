import streamlit as st
from io import BytesIO
import fitz  # PyMuPDF
from docx import Document
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_watson.natural_language_understanding_v1 import Features, KeywordsOptions
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import requests
import sqlite3

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

# Database configuration
conn = sqlite3.connect('text_storage.db', check_same_thread=False)
c = conn.cursor()

# Create table if it doesn't exist
c.execute('''CREATE TABLE IF NOT EXISTS texts
             (text TEXT, keywords TEXT)''')

# Function to save text and keywords to the database
def save_to_db(text, keywords):
    c.execute("INSERT INTO texts (text, keywords) VALUES (?, ?)", (text, ', '.join(keywords)))
    conn.commit()

# Function to search the database
def search_db(query_keywords):
    c.execute("SELECT text, keywords FROM texts")
    all_rows = c.fetchall()
    matching_texts = []

    for row in all_rows:
        text, keywords = row
        stored_keywords = keywords.split(', ')
        if any(query_kw.lower() in stored_keywords for query_kw in query_keywords):
            matching_texts.append(text)

    return matching_texts

# Initialize session state
if "search_results" not in st.session_state:
    st.session_state.search_results = []

if "query_input" not in st.session_state:
    st.session_state.query_input = ""

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

    # Text Input Section
    st.header("Text Input")
    text_input = st.text_area("Enter text to save:")

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
                save_to_db(text_input, keywords)

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

                st.success("PDF content successfully extracted and stored.")
                st.text_area("Extracted Text from PDF:", value=text, height=300)

                if st.button("Save Extracted Text with Keywords", key="save_extracted_text"):
                    try:
                        response = nlu.analyze(
                            text=text,
                            features=Features(
                                keywords=KeywordsOptions(limit=15)
                            )
                        ).get_result()

                        keywords = [kw['text'] for kw in response['keywords']]
                        save_to_db(text, keywords)

                        st.success("Extracted text and keywords have been saved successfully.")
                    except Exception as e:
                        st.error(f"An error occurred while processing the text: {str(e)}")

            except Exception as e:
                st.error(f"An error occurred while extracting text from the PDF: {str(e)}")

        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            try:
                doc = Document(uploaded_file)
                text = "\n".join([para.text for para in doc.paragraphs])

                st.success("DOCX content successfully extracted and stored.")
                st.text_area("Extracted Text from DOCX:", value=text, height=300)

                if st.button("Save Extracted Text with Keywords", key="save_extracted_text"):
                    try:
                        response = nlu.analyze(
                            text=text,
                            features=Features(
                                keywords=KeywordsOptions(limit=15)
                            )
                        ).get_result()

                        keywords = [kw['text'] for kw in response['keywords']]
                        save_to_db(text, keywords)

                        st.success("Extracted text and keywords have been saved successfully.")
                    except Exception as e:
                        st.error(f"An error occurred while processing the text: {str(e)}")

            except Exception as e:
                st.error(f"An error occurred while extracting text from the DOCX: {str(e)}")

elif page == "Search":
    st.title("Search Stored Knowledge")

    st.header("Search")
    query_input = st.text_area("Enter one or more keywords to search:", value=st.session_state.query_input)

    if st.button("Search", key="search"):
        if query_input:
            try:
                query_keywords = [kw.strip() for kw in query_input.split()]
                matching_texts = search_db(query_keywords)

                if not matching_texts:
                    response = aiml_client.chat_completions_create(
                        model="meta-llama/Meta-Llama-3-8B-Instruct-Lite",
                        messages=[
                            {"role": "system", "content": "You are an AI assistant who knows everything."},
                            {"role": "user", "content": f"Provide related data for the following keywords: {', '.join(query_keywords)}"}
                        ]
                    )
                    related_data = response['choices'][0]['message']['content'].strip()
                    
                    if related_data:
                        output_text = f"You: {query_input}\n\nKnowledgeBridge:\n\n*Your search result is not in the database, but here is the related data:*\n\n{related_data}"
                    else:
                        output_text = f"You: {query_input}\n\nKnowledgeBridge:\n\n*No related data found.*"
                else:
                    output_text = f"You: {query_input}\n\nKnowledgeBridge:\n\nYour search result is:\n\n" + '\n\n'.join(matching_texts)

                st.session_state.search_results = output_text
                st.text_area("Search Results", value=st.session_state.search_results, height=300)

            except Exception as e:
                st.error(f"An error occurred during the search: {str(e)}")
        else:
            st.warning("Please enter some keywords to search.")
