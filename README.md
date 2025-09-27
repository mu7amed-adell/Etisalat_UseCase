# E& AI-Powered Customer Churn Prediction Chatbot

This repository contains the code for an AI-powered chatbot designed for the E& marketing team. The chatbot, named Gulia, helps predict customer churn and answers marketing-related queries through a conversational interface.

## Features

*   **Interactive Chatbot UI:** Built with Streamlit for an intuitive user experience.
*   **Churn Prediction:** Leverages a pre-trained machine learning pipeline to predict customer churn.
*   **FastAPI Backend:** Provides a robust API for integration with other systems.
*   **Conversational AI:** Utilizes LangChain and a Large Language Model (LLM) for natural language interaction.

## Project Structure

*   `chatbot/`: Contains the core application logic, including the Streamlit UI (`app.py`), FastAPI backend (`main.py`), and the churn prediction model (`churn_pipeline.pkl`).
*   `churn_pipeline.pkl`: The serialized machine learning pipeline for churn prediction.
*   `requirements.txt`: Lists all Python dependencies.
*   `Project_Documentation.md`: Detailed documentation of the project.

## Setup and Installation

To get started with the project, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Etisalat_UseCase.git
    cd Etisalat_UseCase
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv .venv
    # On Windows
    .\.venv\Scripts\activate
    # On macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ensure the churn pipeline is available:**
    Make sure `churn_pipeline.pkl` is present in the `chatbot/` directory. If it's not, you might need to train the model or obtain it separately.

## Running the Application

### 1. Run the Streamlit Chatbot UI

Navigate to the `chatbot` directory and run the Streamlit application:

```bash
cd chatbot
streamlit run app.py
```

This will open the chatbot interface in your default web browser.

### 2. Run the FastAPI Backend (Optional, for API access)

If you need to access the churn prediction functionality via an API, you can start the FastAPI server:

```bash
cd chatbot
uvicorn main:app --reload
```

The API will be available at `http://127.0.0.1:8000`. You can access the interactive API documentation (Swagger UI) at `http://127.0.0.1:8000/docs`.

## Usage

Interact with the chatbot through the Streamlit UI to provide customer details and get churn predictions. For API usage, refer to the `/docs` endpoint of the FastAPI application for detailed request/response schemas.

## Created by Mohamed Adel - A Use-Case for E&