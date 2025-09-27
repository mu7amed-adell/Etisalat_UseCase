
## 5. Deployment & Usage Guide

This section provides instructions on how to set up and run the Churn Prediction Assistant, along with a practical use case example.

### A. Prerequisites

Before running the applications, ensure you have the following installed and configured:

1.  **Python 3.9+:** Download and install from [python.org](https://www.python.org/).
2.  **Ollama:** Install Ollama from [ollama.com](https://ollama.com/) and ensure the server is running.
3.  **Ollama Model:** Pull the required LLM model (e.g., `qwen3:4b` or `llama3.2:3b` or `llama3-groq-tool-use:8b`). Open your terminal and run:
    ```bash
    ollama pull llama3.2:3b
    # OR
    ollama pull qwen3:4b
    ```
4.  **Project Dependencies:** Navigate to the project root (`D:\Etisalat_UseCase\`) and install the required Python packages:
    ```bash
    pip install -r requirements.txt
    # Ensure you also install LangChain specific dependencies if not in requirements.txt
    pip install langchain langchain-community langchain-core pydantic uvicorn streamlit pandas joblib python-dotenv
    ```
5.  **Pre-trained ML Pipeline:** Ensure `churn_pipeline.pkl` is present in the `chatbot` directory.

### B. Running the Applications

#### 1. FastAPI Backend (`main.py`)

The FastAPI application provides the API endpoints for both conversational and direct churn prediction.

-   **Navigate to the `chatbot` directory:**
    ```bash
    cd D:\Etisalat_UseCase\chatbot
    ```
-   **Start the FastAPI server:**
    ```bash
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ```
    The API will be accessible at `http://localhost:8000`. You can view the interactive API documentation (Swagger UI) at `http://localhost:8000/docs`.

#### 2. Streamlit Frontend (`app.py`)

The Streamlit application provides the interactive chat interface for Gulia.

-   **Navigate to the `chatbot` directory:**
    ```bash
    cd D:\Etisalat_UseCase\chatbot
    ```
-   **Start the Streamlit application:**
    ```bash
    streamlit run app.py
    ```
    The Streamlit app will open in your web browser, typically at `http://localhost:8501`.

### C. Use Case Example: Customer Churn Prediction

Let's use the following customer data to demonstrate both interfaces:

**Customer Profile:**
"Female, not a senior citizen, married, has dependents, 12 months tenure.
Services: Has phone service, no multiple lines, has online security, no online backup, has device protection, no tech support, has streaming TV, no streaming movies.
Billing: Uses paperless billing, monthly charges $75.50, total charges $800.
Service details: Fiber optic internet, Two year contract, Credit card automatic payment."

#### 1. Using the Streamlit Chat Interface (`app.py`)

1.  Ensure both the FastAPI (`main.py`) and Streamlit (`app.py`) applications are running.
2.  Open the Streamlit app in your browser (`http://localhost:8501`).
3.  In the chat input box, type a message to Gulia, asking her to predict churn, and then provide the customer data. You can paste the entire customer profile above or provide it in parts as Gulia asks for information.
    *Example Chat Input:*
    ```
    Hi Gulia, can you predict churn for a customer with the following details: Female, not a senior citizen, married, has dependents, 12 months tenure. Services: Has phone service, no multiple lines, has online security, no online backup, has device protection, no tech support, has streaming TV, no streaming movies. Billing: Uses paperless billing, monthly charges $75.50, total charges $800. Service details: Fiber optic internet, Two year contract, Credit card automatic payment.
    ```
4.  Gulia (the chatbot) will process the information, potentially ask clarifying questions if needed, and once all 19 data points are collected, she will call the underlying prediction tool. The result will be displayed directly in the chat interface.

#### 2. Using the FastAPI Direct Prediction Endpoint (`/predict`)

This method is ideal for programmatic integration and bypasses the conversational flow.

1.  Ensure the FastAPI application (`main.py`) is running.
2.  You can use `curl` from your terminal or any API client (like Postman, Insomnia, or a Python script) to send a POST request to 
`http://localhost:8000/predict`.

    **Request Body (JSON):**
    ```json
    {
      "gender": 0,
      "Senior_Citizen": 0,
      "Is_Married": 1,
      "Dependents": 1,
      "tenure": 12.0,
      "Phone_Service": 1,
      "Dual": 0,
      "Online_Security": 1,
      "Online_Backup": 0,
      "Device_Protection": 1,
      "Tech_Support": 0,
      "Streaming_TV": 1,
      "Streaming_Movies": 0,
      "Paperless_Billing": 1,
      "Monthly_Charges": 75.50,
      "Total_Charges": 800.0,
      "Internet_Service": "Fiber optic",
      "Contract": "Two year",
      "Payment_Method": "Credit card (automatic)"
    }
    ```

    **Example `curl` command:**
    ```bash
    curl -X POST "http://localhost:8000/predict" \
         -H "Content-Type: application/json" \
         -d 
    {
               "gender": 0,
               "Senior_Citizen": 0,
               "Is_Married": 1,
               "Dependents": 1,
               "tenure": 12.0,
               "Phone_Service": 1,
               "Dual": 0,
               "Online_Security": 1,
               "Online_Backup": 0,
               "Device_Protection": 1,
               "Tech_Support": 0,
               "Streaming_TV": 1,
               "Streaming_Movies": 0,
               "Paperless_Billing": 1,
               "Monthly_Charges": 75.50,
               "Total_Charges": 800.0,
               "Internet_Service": "Fiber optic",
               "Contract": "Two year",
               "Payment_Method": "Credit card (automatic)"
             }
    ```

    **Expected Response (JSON):**
    ```json
    {
      "prediction": "Will Not Churn",
      "risk_level": "Low Risk",
      "prediction_value": 0
    }
    ```
    *(Note: The actual prediction result depends on the trained model in `churn_pipeline.pkl` and may vary.)*
