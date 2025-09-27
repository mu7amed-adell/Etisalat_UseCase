from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
import streamlit as st
import re
import pickle
import joblib
from typing import Annotated, Optional, Type, ClassVar, Literal, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
import os
import pandas as pd
from typing import Annotated
from pydantic import BaseModel, Field


system_prompt = """
# Role and Objective

You are Gulia, a friendly AI assistant for the marketing team specializing in customer churn analysis.

Your MAIN TASK: Help predict customer churn by collecting information through natural conversation, then using the predict_customer_churn tool.

# CONVERSATION FLOW:
1. When user asks about churn prediction, warmly explain you'll help them
2. Start collecting information step-by-step (max 2-3 questions at a time)
3. Keep track of what you've already collected
4. Ask follow-up questions if answers are unclear
5. Once you have ALL 19 pieces of information, call the predict_customer_churn tool
6. Provide insights and actionable recommendations based on results

# REQUIRED INFORMATION TO COLLECT:
## Personal Info:
- Gender (Male/Female) 
- Senior Citizen status (Yes/No)
- Marital status (Married/Single)
- Dependents (Yes/No)
- Tenure (how many months with the company)

## Services:
- Phone Service (Yes/No)
- Multiple lines/Dual lines (Yes/No) 
- Online Security (Yes/No)
- Online Backup (Yes/No)
- Device Protection (Yes/No)
- Tech Support (Yes/No)
- Streaming TV (Yes/No)
- Streaming Movies (Yes/No)

## Billing:
- Paperless Billing (Yes/No)
- Monthly Charges (dollar amount)
- Total Charges (total dollar amount paid)

## Service Details:
- Internet Service type (Fiber optic, DSL, or No internet)
- Contract type (Month-to-month, One year, Two year)
- Payment Method (Credit card automatic, Bank transfer, Electronic check, Mailed check, etc.)

CONVERSION RULES:
- Male → 1, Female → 0
- Yes → 1, No → 0  
- Keep service types and contract types as strings exactly as user provides

CONVERSATION STYLE:
- Be warm and conversational
- Don't overwhelm with too many questions at once
- Acknowledge their responses
- Show progress: "Great! I have X out of 19 pieces of information"
- If user provides multiple answers at once, that's perfect - use them all
- ONLY call the prediction tool when you have ALL 19 pieces of information

IMPORTANT: Do NOT call the predict_customer_churn tool until you have collected ALL required information from the user conversation.
"""

load_dotenv()

os.environ["OLLAMA_BASE_URL"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")

# Load the model (you might want to load this once at startup for better performance)
pipeline = joblib.load("churn_pipeline.pkl")

class CustomerData(BaseModel):
    """Data model for customer information"""
    gender: int = Field(description="Gender (0=Female, 1=Male)")
    Senior_Citizen: int = Field(description="Senior citizen status (0=No, 1=Yes)")
    Is_Married: int = Field(description="Marital status (0=No, 1=Yes)")
    Dependents: int = Field(description="Has dependents (0=No, 1=Yes)")
    tenure: float = Field(description="Number of months with the company")
    Phone_Service: int = Field(description="Phone service (0=No, 1=Yes)")
    Dual: int = Field(description="Multiple lines (0=No, 1=Yes)")
    Online_Security: int = Field(description="Online security (0=No, 1=Yes)")
    Online_Backup: int = Field(description="Online backup (0=No, 1=Yes)")
    Device_Protection: int = Field(description="Device protection (0=No, 1=Yes)")
    Tech_Support: int = Field(description="Tech support (0=No, 1=Yes)")
    Streaming_TV: int = Field(description="Streaming TV (0=No, 1=Yes)")
    Streaming_Movies: int = Field(description="Streaming movies (0=No, 1=Yes)")
    Paperless_Billing: int = Field(description="Paperless billing (0=No, 1=Yes)")
    Monthly_Charges: float = Field(description="Monthly charges amount")
    Total_Charges: float = Field(description="Total charges amount")
    Internet_Service: str = Field(description="Internet service type (e.g., 'Fiber optic', 'DSL', 'No')")
    Contract: str = Field(description="Contract type (e.g., 'Two year', 'One year', 'Month-to-month')")
    Payment_Method: str = Field(description="Payment method (e.g., 'Credit card (automatic)', 'Bank transfer', etc.)")

# Create the churn prediction tool using @tool decorator for better LLM integration
@tool
def predict_customer_churn(
    gender: Annotated[Literal[0,1], "Gender (0=Female, 1=Male)"],
    Senior_Citizen: Annotated[Literal[0,1], "Senior citizen status (0=No, 1=Yes)"],
    Is_Married: Annotated[Literal[0,1], "Marital status (0=No, 1=Yes)"],
    Dependents: Annotated[Literal[0,1], "Has dependents (0=No, 1=Yes)"],
    tenure: Annotated[float, "Number of months with the company"],
    Phone_Service: Annotated[Literal[0,1], "Phone service (0=No, 1=Yes)"],
    Dual: Annotated[Literal[0,1], "Multiple lines (0=No, 1=Yes)"],
    Online_Security: Annotated[Literal[0,1], "Online security (0=No, 1=Yes)"],
    Online_Backup: Annotated[Literal[0,1], "Online backup (0=No, 1=Yes)"],
    Device_Protection: Annotated[Literal[0,1], "Device protection (0=No, 1=Yes)"],
    Tech_Support: Annotated[Literal[0,1], "Tech support (0=No, 1=Yes)"],
    Streaming_TV: Annotated[Literal[0,1], "Streaming TV (0=No, 1=Yes)"],
    Streaming_Movies: Annotated[Literal[0,1], "Streaming movies (0=No, 1=Yes)"],
    Paperless_Billing: Annotated[Literal[0,1], "Paperless billing (0=No, 1=Yes)"],
    Monthly_Charges: Annotated[float, "Monthly charges amount"],
    Total_Charges: Annotated[float, "Total charges amount"],
    Internet_Service: Annotated[Literal['Fiber optic', 'DSL', 'No'], "Internet service type (e.g., 'Fiber optic', 'DSL', 'No')"],
    Contract: Annotated[Literal['Two year', 'One year', 'Month-to-month'], "Contract type (e.g., 'Two year', 'One year', 'Month-to-month')"],
    Payment_Method: Annotated[Literal['Credit card (automatic)', 'Bank transfer (automatic)','Electronic check', 'Mailed check'], "Payment method (e.g., 'Credit card (automatic)', 'Bank transfer (automatic)','Electronic check', 'Mailed check'.)"]
) -> str:
    """
    Predicts customer churn probability using a trained machine learning model.
    This tool requires ALL customer information to make a prediction.
    """
    try:
        # Create customer data model
        customer_data = CustomerData(
            gender=gender,
            Senior_Citizen=Senior_Citizen,
            Is_Married=Is_Married,
            Dependents=Dependents,
            tenure=tenure,
            Phone_Service=Phone_Service,
            Dual=Dual,
            Online_Security=Online_Security,
            Online_Backup=Online_Backup,
            Device_Protection=Device_Protection,
            Tech_Support=Tech_Support,
            Streaming_TV=Streaming_TV,
            Streaming_Movies=Streaming_Movies,
            Paperless_Billing=Paperless_Billing,
            Monthly_Charges=Monthly_Charges,
            Total_Charges=Total_Charges,
            Internet_Service=Internet_Service,
            Contract=Contract,
            Payment_Method=Payment_Method
        )

        # Convert to DataFrame
        df = pd.DataFrame([customer_data.model_dump()])

        # Make prediction
        prediction = pipeline.predict(df)

        # Format result
        churn_status = "Will Churn" if prediction[0] == 1 else "Will Not Churn"
        probability = "High risk" if prediction[0] == 1 else "Low risk"

        return f"🎯 Customer Churn Prediction Results:\n✅ Status: {churn_status}\n📊 Risk Level: {probability}\n🔢 Prediction Value: {prediction[0]}"

    except Exception as e:
        return f"❌ Prediction failed: {str(e)}"

# Create the model
model = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "qwen3:4b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
    reasoning=False
)

# Enhanced prompt for conversational data collection and churn prediction
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Create tools list
tools = [predict_customer_churn]

# Create agent
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def get_langchain_messages():
    messages = []
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    return messages

# Streamlit app
def main():
    st.title("E& - Churn Prediction Assistant")
    st.subheader("Telcom Use Case - By: Mohamed Adel Ismaiel")
    st.markdown("*Your AI marketing assistant for customer churn analysis*")

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        welcome_msg = {
            "role": "assistant",
            "content": (
                "Hi! I'm Gulia, your customer churn analysis assistant! 👋\n\n"
                "I can help you predict if a customer is likely to churn. Just ask me something like:\n"
                "- 'Can you predict churn for a customer?'\n"
                "- 'I need to analyze churn risk'\n"
                "- 'Help me check if a customer will leave'\n\n"
                "I'll guide you through collecting the necessary information step by step!"
            )
        }
        st.session_state.messages.append(welcome_msg)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt_input := st.chat_input("Ask me about customer churn prediction..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt_input})
        with st.chat_message("user"):
            st.markdown(prompt_input)

        # Prepare chat history for agent
        chat_history = get_langchain_messages()

        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Gulia is thinking..."):
                try:
                    response = agent_executor.invoke({
                        "input": prompt_input,
                        "chat_history": chat_history
                    })
                    # Clean up response if needed
                    ai_response = re.sub(r"<think>.*?</think>", "", response["output"], flags=re.DOTALL).strip()
                    st.markdown(ai_response)

                    # Add AI response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

    # Sidebar with helpful information
    with st.sidebar:
        st.header("💡 How to Start")
        st.markdown("""
        **Try saying:**
        - "Can you predict churn for a customer?"
        - "I need to analyze customer retention risk"
        - "Help me check if a customer will churn"
        - "Predict churn risk"
        """)

        st.header("🧪 Quick Test Example")
        if st.button("📋 Copy Test Customer Data"):
            test_data = """Here's a customer for churn prediction:

Female, not a senior citizen, married, has dependents, 12 months tenure.

Services: Has phone service, no multiple lines, has online security, no online backup, has device protection, no tech support, has streaming TV, no streaming movies.

Billing: Uses paperless billing, monthly charges $75.50, total charges $800.

Service details: Fiber optic internet, Two year contract, Credit card automatic payment."""

            st.code(test_data, language=None)
            st.success("Copy this text and paste it in the chat to test the complete flow!")

        st.header("🔍 Expected Result")
        st.markdown("""
        After providing all info, Gulia should:
        1. ✅ Collect all 19 pieces of data
        2. 🔧 Call the prediction tool
        3. 📊 Show results with risk level
        4. 💡 Provide recommendations
        """)

        st.header("📋 Information I'll Need")
        st.markdown("""
        I'll ask you for 19 pieces of customer information:
        
        **Personal (5):**
        - Gender, Age group, Marital status, Dependents, Tenure
        
        **Services (8):** 
        - Phone, Multiple lines, Online security, Online backup, Device protection, Tech support, Streaming TV, Streaming movies
        
        **Billing (3):**
        - Paperless billing, Monthly charges, Total charges
        
        **Service Details (3):**
        - Internet service type, Contract type, Payment method
        """)

        st.header("🔄 Process")
        st.markdown("""
        1. **Ask me** to predict churn
        2. **Answer my questions** as I collect info
        3. **Get prediction** once I have everything
        4. **Receive insights** and recommendations
        """)

        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = [st.session_state.messages[0]]  # Keep welcome message
            st.rerun()
        
        st.markdown("---")
        st.caption("💡 Tip: You can provide multiple answers at once to speed up the process!")

if __name__ == "__main__":
    # Run the Streamlit app
    main()