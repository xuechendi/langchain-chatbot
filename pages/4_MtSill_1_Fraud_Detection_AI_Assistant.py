import utils
import streamlit as st
from streaming import StreamHandler

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub
import pytesseract
import os
from PIL import Image
from io import BytesIO
import base64

from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain.tools import BaseTool
    
st.set_page_config(page_title="Chatbot", page_icon="💬")
st.header('Insurance Fraud Detection AI Assistant')
st.write('Using Mt.Sill Expert Model for Insurance Fraud Detection')

def get_prediction(features):
    return "insurance_fraud_detection: \n This report shows high risk of Fraud"

class FraudDetectionToolInput(BaseModel):
    # Check the input for Weather
    features: str = Field(
        ..., description="Json String of input content"
    )
    
class FraudDetectionTool(BaseTool):
    name = "insurance_fraud_detection"
    description = "Used to do fraud detection based on input json string data"

    def _run(self, features: str):
        return get_prediction(features)
    
    args_schema: Optional[Type[BaseModel]] = FraudDetectionToolInput

    
class Basic:

    def __init__(self):
        self.openai_model = "mistral-7b-instruct-v0.2"
        self.history_messages = utils.enable_chat_history('mtsill_fraud_detection_airbnb')

    def main(self):
        for message in self.history_messages: # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])

        with st.form("my-form", clear_on_submit=True):
            image_input = st.file_uploader('Upload a jpg image')
            submitted = st.form_submit_button()

        if submitted and image_input is not None:
            base64_image = base64.b64encode(image_input.getvalue()).decode("utf-8")
            base64_image = f"data:image/png;base64,{base64_image}"
            image_md = f"![image]({base64_image})"
            self.history_messages.append({"role": "user", "content": image_md})
            with st.chat_message('user'):
                st.write(image_md)
            text = pytesseract.image_to_string(Image.open(BytesIO(image_input.getvalue())))
            
            user_query = f"Please convert 'Input' into a markdown table with columns of ['Name', 'Gender', 'Age', 'region', 'bmi', 'bloodpressure', 'diabetic', 'smoker']. Don't provide additional explaination besides markdown.\n Input: {text}"
            message = [
                HumanMessage(
                    content=[
                        {"type": "text", "text": user_query},
                    ]
                )
            ]
                        
            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container())
                llm = ChatOpenAI(openai_api_base = "http://10.0.2.14:8000/v1", model_name=self.openai_model, openai_api_key="not_needed", streaming=True, max_tokens=512, callbacks=[st_callback])
                response = llm(message)
                self.history_messages.append({"role": "assistant", "content": response.content})
                image_input = None
        
                tools = [FraudDetectionTool()]
                agent = create_openai_tools_agent(tools=tools, llm=llm, prompt=hub.pull("hwchase17/openai-tools-agent"))
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
                
                user_query = f"Is this report a fraud or not according to 'insurance_fraud_detection'.\n Input: {response.content}"
                
                response = agent_executor.invoke({"input": user_query}, {"callbacks": [st_callback]})
                
                self.history_messages.append({"role": "assistant", "content": response['output']})
                st.write(response['output'])
        

if __name__ == "__main__":
    obj = Basic()
    obj.main()
