# Multi-Documents ChatBot App

## Introduction

The Multi-Documents Chatbot App is a Python application that allows you to engage in conversations with multiple documents simultaneously. You can ask questions about the loaded documents using natural language, and the chatbot will provide relevant responses based on the content of the documents. This app utilizes a language model to generate accurate answers to your queries. Please note that the app will only respond to questions related to the loaded PDFs.

## How It Works

![Multi-Documents Chat App Diagram](./img/process-diagram.png)

The application follows these steps to provide responses to your questions:
1. Document Loading: The app reads multiple documents and extracts their text content.
2. Text Chunking: The extracted text is divided into smaller chunks that can be processed effectively.
3. Language Model: The application utilizes a language model to generate vector representations (embeddings) of the text chunks.
4. Similarity Matching: When you ask a question, the app compares it with the text chunks and identifies the most semantically similar ones.
5. Response Generation: The selected chunks are passed to the language model, which generates a response based on the relevant content of the PDFs.

## Dependencies and Installation

To install the MultiPDF Chat App, please follow these steps:

1. Clone the repository to your local machine.
2. Install the required dependencies by running the following command:
   ```
   pip install -r requirements.txt
   ```
3. Obtain an API key from Groq and add it to the `.env` file in the project directory.
   ```
   GROQ_API_KEY=YOUR_API_TOKEN
   ```

## Usage

To use the Multi-Documents Chatbot App, follow these steps:

1. Ensure that you have installed the required dependencies and added the Groq API key to the `.env` file.
2. Run the `main.py` file using the Streamlit CLI. Execute the following command:
   ```
   streamlit run app.py
   ```
3. The application will launch in your default web browser, displaying the user interface.
4. Load multiple documents into the app by following the provided instructions.
5. Ask questions in natural language about the loaded PDFs using the chat interface.

![Example](./img/example.png)