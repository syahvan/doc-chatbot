# Multi-Documents ChatBot App

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://syahvan-document-chatbot.streamlit.app/)

## Introduction

The Multi-Documents Chatbot App is a Python application that allows you to engage in conversations with multiple documents simultaneously. You can ask questions about the loaded documents using natural language, and the chatbot will provide relevant responses based on the content of the documents. This app utilizes a language model (Llama 3) to generate accurate answers to your queries. Please note that the app will only respond to questions related to the loaded documents.

## How It Works

<p align="center">
  <img src="https://raw.githubusercontent.com/syahvan/doc-chatbot/main/img/process-diagram.png" width="85%" height="85%">
  <br>
  Fig 1. Multi-Documents Chatbot App Process Diagram
</p>

The application follows these steps to provide responses to your questions:
1. **Document Loading**: The app reads multiple documents (.pdf, .docx, or .txt) and extracts their text content.
2. **Text Chunking**: The extracted text is divided into smaller chunks that can be processed effectively.
3. **Language Model**: The application utilizes a language model to generate vector representations (embeddings) of the text chunks.
4. **Similarity Matching**: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.
5. **Response Generation**: The selected chunks are passed to the language model, which generates a response based on the relevant content of the documents.

## Dependencies and Installation

To install the Multi-Documents Chatbot App, please follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies by running the following command:

   ```bash
   pip install -r requirements.txt
   ```
3. Obtain an API key from Groq and add it as a secret key in your Streamlit deployment settings.

   ```
   GROQ_API_KEY=YOUR_API_TOKEN
   ```

## Usage

### Local Deployment

1. Ensure that you have installed the required dependencies and added Groq API key.
2. Run the `app.py` file using the Streamlit CLI. Execute the following command:

   ```bash
   streamlit run app.py
   ```
3. The application will launch in your default web browser, displaying the user interface.
4. Load multiple documents into the app by following the provided instructions.
5. Ask questions in natural language about the loaded documents using the chat interface.

### Web Deployment

1. Deploy the app on Streamlit Cloud or any other hosting service that supports Streamlit applications.
2. Add your Groq API key as a secret key in the deployment settings of the hosting service.
3. Once deployed, access your app via the provided web link.
4. Share the web link with others to allow them to interact with the chatbot.

<p align="center">
  <img src="https://raw.githubusercontent.com/syahvan/doc-chatbot/main/img/example.png" width="90%" height="90%">
  <br>
  Fig 2. Usage Example
</p>