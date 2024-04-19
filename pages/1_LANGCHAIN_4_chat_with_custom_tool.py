import utils
import streamlit as st

from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub
    
st.set_page_config(page_title="ChatWeb", page_icon="🌐")
st.header('Chatbot with AI Agent & Custom Tool')

from pydantic import BaseModel, Field
from typing import Optional, Type
from langchain.tools import BaseTool


def get_current_weather(location, unit):
    # Call an external API to get relevant information (like serpapi, etc)
    # Here for the demo we will send a mock response
    weather_info = {
        "location": location,
        "temperature": "78",
        "unit": unit,
        "forecast": ["sunny", "with a chance of rain"],
    }
    return weather_info


class GetCurrentWeatherCheckInput(BaseModel):
    # Check the input for Weather
    location: str = Field(
        ..., description="The name of the location name for which we need to find the weather"
    )
    unit: str = Field(..., description="The unit for the temperature value")


class GetCurrentWeatherTool(BaseTool):
    name = "get_current_weather"
    description = "Used to find the weather for a given location in said unit"

    def _run(self, location: str, unit: str):
        # print("I am running!")
        weather_response = get_current_weather(location, unit)
        return weather_response

    def _arun(self, location: str, unit: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = GetCurrentWeatherCheckInput


# ================================================ #

class ChatbotTools:

    def __init__(self):
        #utils.configure_openai_api_key()
        self.openai_model = "mistral-7b-instruct-v0.2"
        self.history_messages = utils.enable_chat_history('custom_tool_chat')

    def main(self):
        for message in self.history_messages: # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])
        tools = [GetCurrentWeatherTool()]
        llm = ChatOpenAI(openai_api_base = "http://localhost:8000/v1", model_name=self.openai_model, openai_api_key="not_needed", streaming=True, max_tokens=512,)
        agent = create_openai_tools_agent(tools=tools, llm=llm, prompt=hub.pull("hwchase17/openai-tools-agent"))
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        if user_query := st.chat_input(placeholder="What is the weather in Autin, Texas?"):
            self.history_messages.append({"role": "user", "content": user_query})
            with st.chat_message('user'):
                st.write(user_query)
            with st.chat_message("assistant"):
                st_callback = StreamlitCallbackHandler(st.container())
                response = agent_executor.invoke({"input": user_query}, {"callbacks": [st_callback]})
                self.history_messages.append({"role": "assistant", "content": response['output']})
                st.write(response['output'])

if __name__ == "__main__":
    obj = ChatbotTools()
    obj.main()
