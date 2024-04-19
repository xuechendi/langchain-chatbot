import utils
import streamlit as st

from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.tools import DuckDuckGoSearchRun
from langchain.agents import initialize_agent, Tool, load_tools
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain import hub
    
st.set_page_config(page_title="ChatWeb", page_icon="🌐")
st.header('Chatbot with Internet Access')
st.write('Equipped with internet access, enables users to ask questions about recent events')

class ChatbotTools:

    def __init__(self):
        #utils.configure_openai_api_key()
        self.openai_model = "mistral-7b-instruct-v0.2"
        self.history_messages = utils.enable_chat_history('web_chat')

    def main(self):
        tools = [DuckDuckGoSearchRun(max_results=1)]
        #tools = load_tools(["ddg-search"])
        llm = ChatOpenAI(openai_api_base = "http://localhost:8000/v1", model_name=self.openai_model, openai_api_key="not_needed", streaming=True, max_tokens=512,)
        agent = create_openai_tools_agent(tools=tools, llm=llm, prompt=hub.pull("hwchase17/openai-tools-agent"))
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
        for message in self.history_messages: # Display the prior chat messages
            with st.chat_message(message["role"]):
                st.write(message["content"])
        if user_query := st.chat_input(placeholder="Ask me anything!"):
            self.history_messages.append({"role": "user", "content": user_query})
            with st.chat_message('user'):
                st.write(user_query)

            with st.chat_message("assistant"):
                st_cb = StreamlitCallbackHandler(st.container())
                response = agent_executor.invoke({"input": user_query}, {"callbacks": [st_cb]})
                self.history_messages.append({"role": "assistant", "content": response['output']})
                st.write(response['output'])

if __name__ == "__main__":
    obj = ChatbotTools()
    obj.main()
