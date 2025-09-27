from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import uvicorn
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
import joblib
import pandas as pd
from typing import Annotated
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

os.environ["OLLAMA_BASE_URL"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")

# Pydantic models for API requests and responses
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message sender (user/assistant)")
    content: str = Field(..., description="Content of the message")

class ChatRequest(BaseModel):
    message: str = Field(..., description="User's message")
    chat_history: Optional[List[ChatMessage]] = Field(default=[], description="Previous chat history")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Assistant's response")
    chat_history: List[ChatMessage] = Field(..., description="Updated chat history")
    tool_called: bool = Field(default=False, description="Whether the prediction tool was called")

class DirectPredictionRequest(BaseModel):
    """Direct prediction request with all customer data"""
    gender: int = Field(..., description="Gender (0=Female, 1=Male)")
    Senior_Citizen: int = Field(..., description="Senior citizen status (0=No, 1=Yes)")
    Is_Married: int = Field(..., description="Marital status (0=No, 1=Yes)")
    Dependents: int = Field(..., description="Has dependents (0=No, 1=Yes)")
    tenure: float = Field(..., description="Number of months with the company")
    Phone_Service: int = Field(..., description="Phone service (0=No, 1=Yes)")
    Dual: int = Field(..., description="Multiple lines (0=No, 1=Yes)")
    Online_Security: int = Field(..., description="Online security (0=No, 1=Yes)")
    Online_Backup: int = Field(..., description="Online backup (0=No, 1=Yes)")
    Device_Protection: int = Field(..., description="Device protection (0=No, 1=Yes)")
    Tech_Support: int = Field(..., description="Tech support (0=No, 1=Yes)")
    Streaming_TV: int = Field(..., description="Streaming TV (0=No, 1=Yes)")
    Streaming_Movies: int = Field(..., description="Streaming movies (0=No, 1=Yes)")
    Paperless_Billing: int = Field(..., description="Paperless billing (0=No, 1=Yes)")
    Monthly_Charges: float = Field(..., description="Monthly charges amount")
    Total_Charges: float = Field(..., description="Total charges amount")
    Internet_Service: str = Field(..., description= "Internet service type (e.g., 'Fiber optic', 'DSL', 'No')")
    Contract: str = Field(..., description="Contract type (e.g., 'Two year', 'One year', 'Month-to-month')")
    Payment_Method: str = Field(..., description="Payment method (e.g., 'Credit card (automatic)', 'Bank transfer (automatic)','Electronic check', 'Mailed check'.)")

class DirectPredictionResponse(BaseModel):
    prediction: str = Field(..., description="Churn prediction result")
    risk_level: str = Field(..., description="Risk level assessment")
    prediction_value: int = Field(..., description="Raw prediction value (0 or 1)")

# Customer data model (same as before)
class CustomerData(BaseModel):
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
    Internet_Service: str = Field(description= "Internet service type (e.g., 'Fiber optic', 'DSL', 'No')")
    Contract: str = Field(description="Contract type (e.g., 'Two year', 'One year', 'Month-to-month')")
    Payment_Method: str = Field(description="Payment method (e.g., 'Credit card (automatic)', 'Bank transfer (automatic)','Electronic check', 'Mailed check'.)")

# Churn prediction tool (same as before)
@tool
def predict_customer_churn(
    gender: Annotated[int, "Gender (0=Female, 1=Male)"],
    Senior_Citizen: Annotated[int, "Senior citizen status (0=No, 1=Yes)"],
    Is_Married: Annotated[int, "Marital status (0=No, 1=Yes)"],
    Dependents: Annotated[int, "Has dependents (0=No, 1=Yes)"],
    tenure: Annotated[float, "Number of months with the company"],
    Phone_Service: Annotated[int, "Phone service (0=No, 1=Yes)"],
    Dual: Annotated[int, "Multiple lines (0=No, 1=Yes)"],
    Online_Security: Annotated[int, "Online security (0=No, 1=Yes)"],
    Online_Backup: Annotated[int, "Online backup (0=No, 1=Yes)"],
    Device_Protection: Annotated[int, "Device protection (0=No, 1=Yes)"],
    Tech_Support: Annotated[int, "Tech support (0=No, 1=Yes)"],
    Streaming_TV: Annotated[int, "Streaming TV (0=No, 1=Yes)"],
    Streaming_Movies: Annotated[int, "Streaming movies (0=No, 1=Yes)"],
    Paperless_Billing: Annotated[int, "Paperless billing (0=No, 1=Yes)"],
    Monthly_Charges: Annotated[float, "Monthly charges amount"],
    Total_Charges: Annotated[float, "Total charges amount"],
    Internet_Service: Annotated[str,  "Internet service type (e.g., 'Fiber optic', 'DSL', 'No')"],
    Contract: Annotated[str, "Contract type (e.g., 'Two year', 'One year', 'Month-to-month')"],
    Payment_Method: Annotated[str, "Payment method (e.g., 'Credit card (automatic)', 'Bank transfer (automatic)','Electronic check', 'Mailed check'.)"]
) -> str:
    """Predicts customer churn probability using a trained machine learning model."""
    try:
        pipeline = joblib.load("churn_pipeline.pkl")
        
        customer_data = CustomerData(
            gender=gender, Senior_Citizen=Senior_Citizen, Is_Married=Is_Married,
            Dependents=Dependents, tenure=tenure, Phone_Service=Phone_Service,
            Dual=Dual, Online_Security=Online_Security, Online_Backup=Online_Backup,
            Device_Protection=Device_Protection, Tech_Support=Tech_Support,
            Streaming_TV=Streaming_TV, Streaming_Movies=Streaming_Movies,
            Paperless_Billing=Paperless_Billing, Monthly_Charges=Monthly_Charges,
            Total_Charges=Total_Charges, Internet_Service=Internet_Service,
            Contract=Contract, Payment_Method=Payment_Method
        )
        
        df = pd.DataFrame([customer_data.model_dump()])
        prediction = pipeline.predict(df)
        
        churn_status = "Will Churn" if prediction[0] == 1 else "Will Not Churn"
        probability = "High risk" if prediction[0] == 1 else "Low risk"
        
        return f"🎯 Customer Churn Prediction Results:\n✅ Status: {churn_status}\n📊 Risk Level: {probability}\n🔢 Prediction Value: {prediction[0]}"
        
    except Exception as e:
        return f"❌ Prediction failed: {str(e)}"

# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for conversational customer churn prediction using Gulia AI assistant",
    version="1.0.0"
)

# Initialize LLM and agent
model = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are Gulia, a friendly AI assistant for the marketing team specializing in customer churn analysis.

Your MAIN TASK: Help predict customer churn by collecting information through natural conversation, then using the predict_customer_churn tool.

CONVERSATION FLOW:
1. When user asks about churn prediction, warmly explain you'll help them
2. Start collecting information step-by-step (max 2-3 questions at a time)
3. Keep track of what you've already collected
4. Ask follow-up questions if answers are unclear
5. Once you have ALL 19 pieces of information, call the predict_customer_churn tool
6. Provide insights and actionable recommendations based on results

REQUIRED INFORMATION TO COLLECT:
✅ Personal Info: Gender, Senior Citizen status, Marital status, Dependents, Tenure
✅ Services: Phone Service, Multiple lines, Online Security, Online Backup, Device Protection, Tech Support, Streaming TV, Streaming Movies
✅ Billing: Paperless Billing, Monthly Charges, Total Charges
✅ Service Details: Internet Service type, Contract type, Payment Method

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

IMPORTANT: Do NOT call the predict_customer_churn tool until you have collected ALL required information from the user conversation."""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

tools = [predict_customer_churn]
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Customer Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "/chat": "POST - Conversational churn prediction with Gulia",
            "/predict": "POST - Direct churn prediction with all data",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "message": "API is running"}

@app.post("/chat", response_model=ChatResponse)
async def chat_with_gulia(request: ChatRequest):
    """
    Chat with Gulia for conversational churn prediction.
    She'll collect customer information step by step and make predictions.
    """
    try:
        from langchain_core.messages import HumanMessage, AIMessage
        
        # Handle None chat_history by defaulting to empty list
        chat_history_list = request.chat_history or []
        
        # Convert chat history to LangChain message format
        langchain_chat_history = []
        for msg in chat_history_list:
            if msg.role == "user":
                langchain_chat_history.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_chat_history.append(AIMessage(content=msg.content))
        
        # Get response from agent
        response = agent_executor.invoke({
            "input": request.message,
            "chat_history": langchain_chat_history
        })
        
        ai_response = response["output"]
        
        # Check if tool was called (simple check for prediction results)
        tool_called = "🎯 Customer Churn Prediction Results:" in ai_response
        
        # Update chat history - handle None case
        updated_history = chat_history_list + [
            ChatMessage(role="user", content=request.message),
            ChatMessage(role="assistant", content=ai_response)
        ]
        
        return ChatResponse(
            response=ai_response,
            chat_history=updated_history,
            tool_called=tool_called
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.post("/predict", response_model=DirectPredictionResponse)
async def direct_prediction(request: DirectPredictionRequest):
    """
    Direct churn prediction with all customer data provided at once.
    Bypasses the conversational flow.
    """
    try:
        # Load model and make prediction directly
        pipeline = joblib.load("churn_pipeline.pkl")
        
        # Create customer data
        customer_data = CustomerData(**request.model_dump())
        df = pd.DataFrame([customer_data.model_dump()])
        
        # Make prediction
        prediction = pipeline.predict(df)
        
        # Format results
        churn_status = "Will Churn" if prediction[0] == 1 else "Will Not Churn"
        risk_level = "High Risk" if prediction[0] == 1 else "Low Risk"
        
        return DirectPredictionResponse(
            prediction=churn_status,
            risk_level=risk_level,
            prediction_value=int(prediction[0])
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/test-data")
async def get_test_data():
    """
    Returns sample test data for API testing
    """
    return {
        "message": "Sample customer data for testing",
        "test_customer": {
            "description": "Female, not senior citizen, married, has dependents, 12 months tenure",
            "chat_message": "Here's a customer for churn prediction: Female, not a senior citizen, married, has dependents, 12 months tenure. Services: Has phone service, no multiple lines, has online security, no online backup, has device protection, no tech support, has streaming TV, no streaming movies. Billing: Uses paperless billing, monthly charges $75.50, total charges $800. Service details: Fiber optic internet, Two year contract, Credit card automatic payment.",
            "direct_api_data": {
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
                "Monthly_Charges": 75.5,
                "Total_Charges": 800.0,
                "Internet_Service": "Fiber optic",
                "Contract": "Two year",
                "Payment_Method": "Credit card (automatic)"
            }
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )