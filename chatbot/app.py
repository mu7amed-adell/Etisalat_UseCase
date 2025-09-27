from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import pickle
import joblib
from typing import Annotated, Optional, Type, ClassVar, Literal, Any
from langchain.tools import BaseTool
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from dotenv import load_dotenv
import os
import pandas as pd
from typing import Annotated
from pydantic import BaseModel, Field


load_dotenv()

os.environ["OLLAMA_BASE_URL"] = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")



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

class CustomerChurnTool(BaseTool):
    name: str = "customer_churn_predictor"
    description: str = """
    Predicts customer churn probability using a trained machine learning model.
    
    Input parameters:
    - gender: 0 for Female, 1 for Male
    - Senior_Citizen: 0 for No, 1 for Yes
    - Is_Married: 0 for No, 1 for Yes
    - Dependents: 0 for No, 1 for Yes
    - tenure: Number of months with the company (float)
    - Phone_Service: 0 for No, 1 for Yes
    - Dual: 0 for No multiple lines, 1 for Yes
    - Online_Security: 0 for No, 1 for Yes
    - Online_Backup: 0 for No, 1 for Yes
    - Device_Protection: 0 for No, 1 for Yes
    - Tech_Support: 0 for No, 1 for Yes
    - Streaming_TV: 0 for No, 1 for Yes
    - Streaming_Movies: 0 for No, 1 for Yes
    - Paperless_Billing: 0 for No, 1 for Yes
    - Monthly_Charges: Monthly charges amount (float)
    - Total_Charges: Total charges amount (float)
    - Internet_Service: Internet service type (string)
    - Contract: Contract type (string)
    - Payment_Method: Payment method (string)
    
    Returns: Churn prediction (0 = No Churn, 1 = Churn)
    """

    _pipeline: Any = PrivateAttr()  # Private attribute for the model pipeline

    def __init__(self, model_path: str = "churn_pipeline.pkl", **kwargs):
        super().__init__(**kwargs)
        try:
            self._pipeline = joblib.load(model_path)
            print(f"Successfully loaded model from {model_path}")
        except Exception as e:
            raise ValueError(f"Failed to load model from {model_path}: {e}")

    def _run(
        self,
        gender: int,
        Senior_Citizen: int,
        Is_Married: int,
        Dependents: int,
        tenure: float,
        Phone_Service: int,
        Dual: int,
        Online_Security: int,
        Online_Backup: int,
        Device_Protection: int,
        Tech_Support: int,
        Streaming_TV: int,
        Streaming_Movies: int,
        Paperless_Billing: int,
        Monthly_Charges: float,
        Total_Charges: float,
        Internet_Service: str,
        Contract: str,
        Payment_Method: str,
        **kwargs
    ) -> str:
        """Synchronous prediction using the saved pipeline."""
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
            prediction = self._pipeline.predict(df)
            
            # Format result
            churn_status = "Will Churn" if prediction[0] == 1 else "Will Not Churn"
            
            return f"Customer Churn Prediction: {churn_status} (Value: {prediction[0]})"
            
        except Exception as e:
            return f"Prediction failed: {str(e)}"

    async def _arun(self, **kwargs) -> str:
        """Asynchronous version - just calls the sync version."""
        return self._run(**kwargs)


model = ChatOllama(
    model=os.getenv("OLLAMA_MODEL", "llama3.2:3b"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that helps our marketing team with regular tasks and churn queries. Your name is Gulia. You can use the tool add_tool to add two numbers."),
        ("user", "Question: {question}")
    ]
)


if __name__ == "__main__":
    # Initialize the tool
    churn_tool = CustomerChurnTool(model_path="churn_pipeline.pkl")
    
    # Test with your sample data
    result = churn_tool._run(
        gender=1,
        Senior_Citizen=0,
        Is_Married=1,
        Dependents=0,
        tenure=12.0,
        Phone_Service=1,
        Dual=0,
        Online_Security=1,
        Online_Backup=0,
        Device_Protection=1,
        Tech_Support=0,
        Streaming_TV=1,
        Streaming_Movies=0,
        Paperless_Billing=1,
        Monthly_Charges=75.5,
        Total_Charges=800.0,
        Internet_Service="Fiber optic",
        Contract="Two year",
        Payment_Method="Credit card (automatic)"
    )
    
    print(result)

# chain = prompt | model | StrOutputParser()


# st.title("Churn assistant - E&")
# input_text = st.text_area("Enter your question here", height=100)
# if input_text:
#     st.write(chain.invoke({"question": input_text}))
