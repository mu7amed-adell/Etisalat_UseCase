
# Documentation: E& Marketing Team Churn Prediction Chatbot

**Project:** Customer Churn Prediction Assistant ("Gulia")  
**Team:** Marketing  
**Date:** September 27, 2025

---

## 1. Overview

This document provides a comprehensive overview of the Customer Churn Prediction Chatbot, codenamed "Gulia." It details the business and technical motivations behind its development, its alignment with the marketing team's requirements, and the architecture that ensures its reliability and maintainability.

The solution provides two primary interfaces:
1.  **A user-friendly web application** for interactive, conversational analysis.
2.  **A robust API** for programmatic integration with other business systems.

---

## 2. Business & Technical Motivations

### A. Business Rationale: Why a Chatbot?

The marketing team requires a tool to proactively identify customers at risk of churning. A traditional dashboard-based approach can be static and require significant training. A chatbot was chosen as the ideal solution for several key reasons:

-   **Accessibility & Ease of Use:** A conversational interface is intuitive and requires minimal training. Marketing team members can simply "ask" the chatbot to predict churn, making the technology accessible to non-technical users.
-   **Guided Data Entry:** The chatbot, Gulia, guides the user through the data collection process step-by-step. This reduces errors, ensures all 19 required data points are collected, and makes the process feel less like filling out a form.
-   **Immediate & Actionable Insights:** Instead of just presenting a churn score, the chatbot provides a clear, concise prediction ("Will Churn" / "Will Not Churn") and a risk level ("High risk" / "Low risk"). This allows the team to take immediate, targeted action.
-   **Increased Efficiency:** By automating the prediction process, the chatbot frees up the marketing team's time to focus on developing and implementing retention strategies rather than on manual data analysis.

### B. Technical Architecture & Choices

The technical stack was carefully selected to deliver a robust, scalable, and maintainable solution that meets both immediate and future needs.

-   **Core Engine: Pre-trained ML Pipeline (`churn_pipeline.pkl`)**
    -   **Simplification:** The solution leverages a pre-trained machine learning model saved as a `joblib` pipeline (`.pkl` file). This is a critical design choice that decouples the complex data science workflow (data cleaning, preprocessing, feature engineering, model training) from the application logic.
    -   **Efficiency:** By using a pre-trained pipeline, the chatbot does not need to perform these computationally expensive tasks for every prediction. It simply passes the input data through the saved pipeline, resulting in near-instantaneous predictions.
    -   **Consistency:** The pipeline ensures that the exact same preprocessing steps (e.g., encoding categorical variables, scaling numerical features) that were used during model training are applied during prediction, which is crucial for accuracy.

-   **Language Model Integration: LangChain & Ollama**
    -   **Conversational Intelligence:** The chatbot's natural language understanding is powered by a Large Language Model (LLM) running locally via **Ollama** (e.g., `llama3.2:3b`).
    -   **Tool-Based Agency:** We use **LangChain** to create an "agent." This agent can use "tools." The `predict_customer_churn` function is exposed as a tool to the LLM.
    -   **Workflow:** The LLM's primary role is to have a conversation, collect the 19 required pieces of customer data, and then call the prediction tool with that data. This creates a seamless and intelligent user experience.

-   **Dual Interfaces: Streamlit & FastAPI**
    -   **Streamlit (UI):** Provides an interactive, user-friendly web interface (`app.py`). This is perfect for direct use by the marketing team for one-off predictions or for demonstrating the capability to stakeholders. The interface includes a chat history, helpful guides, and quick-test buttons.
    -   **FastAPI (API):** Provides a high-performance, scalable API for programmatic access (`main.py`). This is essential for integrating the churn prediction functionality into other systems, such as a CRM, automated marketing platforms, or batch processing workflows.

-   **Data Validation: Pydantic**
    -   **Robustness:** Pydantic models (`CustomerData`, `ChatRequest`, etc.) are used in both the Streamlit and FastAPI applications.
    -   **Functionality:** Pydantic enforces strict data validation, ensuring that the data passed to the machine learning pipeline is in the correct format (e.g., `int`, `float`, `str`). This prevents runtime errors and ensures the reliability of predictions. It also automatically generates documentation for the API endpoints.

---

## 3. Alignment with Client Requirements

The chatbot solution is directly aligned with the E& marketing team's need for an accessible, reliable, and integrable churn prediction tool.

### A. Meeting Marketing Team Needs

-   **Intuitive Interaction:** The conversational UI allows any team member to get a churn prediction without needing to understand the underlying data science.
-   **Flexibility of Use:** The solution caters to different operational needs by providing both an interactive UI and a powerful API:
    -   **UI (Streamlit):** Ideal for individual customer lookups, ad-hoc analysis, and presentations. A team member can quickly check the churn risk for a customer they are on the phone with.
    -   **API (FastAPI):** Enables large-scale, automated churn analysis. For example, it can be used to run predictions on thousands of customers overnight and flag high-risk individuals in the company's CRM.
-   **Clear, Actionable Results:** The output is not a complex set of probabilities but a simple, direct answer to the team's question: "Is this customer going to leave?"

### B. System Qualities

-   **Reliability:**
    -   The use of Pydantic for data validation and a pre-trained, tested pipeline ensures that predictions are consistent and the system is robust against bad inputs.
    -   The separation of concerns (UI, API, ML model) means that a failure in one component is less likely to affect the others.
-   **Maintainability:**
    -   The code is well-structured and modular. The `churn_pipeline.pkl` can be updated independently by the data science team without requiring changes to the application code.
    -   If a new or better churn model is developed, the only change required is to replace the `.pkl` file.
-   **Ease of Integration:**
    -   The FastAPI provides an industry-standard REST API that is automatically documented (via OpenAPI/Swagger).
    -   This makes it straightforward for developers to integrate the churn prediction service into any other application or workflow within E&'s technical ecosystem. The `/predict` endpoint allows for direct, high-throughput predictions, bypassing the conversational interface entirely for system-to-system communication.
