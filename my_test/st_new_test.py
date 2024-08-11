import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from langchain_experimental.tools import PythonREPLTool
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langgraph.graph import StateGraph, END
from typing import Annotated, Sequence, TypedDict
import random
import functools
import operator
import matplotlib.pyplot as plt
import io
import re

load_dotenv()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.set_page_config(page_title="Streaming Bot", page_icon="🤖")

st.title("Streamlit Bot")

def get_response(query, chat_history):
    template = """
    You are a helpful assistant. Answer the following question considering the history of the conversation:
    
    Chat history:{chat_history}
    
    User question:{user_question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI()
    chain = prompt | llm | StrOutputParser()
    return chain.stream({
        "chat_history": chat_history,
        "user_question": query
    })

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    else:
        with st.chat_message("AI"):
            st.markdown(message.content)


# File uploader
# if uploaded_file is not None:
user_query = st.chat_input("您的消息")


if user_query is not None and user_query != "":
    st.session_state.chat_history.append(HumanMessage(user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)
    with st.chat_message("AI"):
        ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history))
    st.session_state.chat_history.append(AIMessage(ai_response))

python_repl_tool = PythonREPLTool()

@tool("lower_case", return_direct=False)
def to_lower_case(input: str) -> str:
    """Returns the input as all lower case."""
    return input.lower()

@tool("random_number", return_direct=False)
def random_number_maker(input: str) -> str:
    """Returns a random number between 0-100. input the word 'random'"""
    return random.randint(0, 100)

tools = [to_lower_case, random_number_maker, python_repl_tool]

def create_agent(llm: ChatOpenAI, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        MessagesPlaceholder(variable_name="agent_scratchpad")
    ])
    agent = create_openai_tools_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor

def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}

members = ["Lotto_Manager", "Coder"]
system_prompt = (
    "You are a supervisor tasked with managing a conversation between the"
    " following workers: {members}. Given the following user request,"
    " respond with the worker to act next. Each worker will perform a"
    " task and respond with their results and status. When finished,"
    " respond with FINISH."
)
options = ["FINISH"] + members
function_def = {
    "name": "route",
    "description": "Select the next role.",
    "parameters": {
        "title": "routeSchema",
        "type": "object",
        "properties": {
            "next": {
                "title": "Next",
                "anyOf": [
                    {"enum": options},
                ],
            }
        },
        "required": ["next"],
    },
}

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    (
        "system",
        "Given the conversation above, who should act next?"
        " Or should we FINISH? Select one of: {options}",
    ),
]).partial(options=str(options), members=", ".join(members))

llm = ChatOpenAI()

supervisor_chain = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Given the conversation above, who should act next? Or should we FINISH? Select one of: {options}"),
]).partial(options=str(options), members=", ".join(members)) | \
    llm.bind_functions(functions=[function_def], function_call="route") | \
    JsonOutputFunctionsParser()

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str

lotto_agent = create_agent(llm, tools, "You are a senior lotto manager. you run the lotto and get random numbers")
lotto_node = functools.partial(agent_node, agent=lotto_agent, name="Lotto_Manager")

code_agent = create_agent(llm, [python_repl_tool], "You can generate secure python code to analyze data and produce graphs using matplotlib, and download the resulting graphs to your computer.")
code_node = functools.partial(agent_node, agent=code_agent, name="Coder")


workflow = StateGraph(AgentState)
workflow.add_node("Lotto_Manager", lotto_node)
workflow.add_node("Coder", code_node)
workflow.add_node("supervisor", supervisor_chain)

for member in members:
    workflow.add_edge(member, "supervisor")

conditional_map = {k: k for k in members}
conditional_map["FINISH"] = END
workflow.add_conditional_edges("supervisor", lambda x: x["next"], conditional_map)

workflow.set_entry_point("supervisor")

graph = workflow.compile()

if user_query:
    config = {"recursion_limit": 20}
    for s in graph.stream({
        "messages": [
            HumanMessage(content=user_query)
        ]
    }, config=config):
        if "__end__" not in s:
            print(s)
            print("----")
            st.write(s)

    final_response = graph.invoke({
        "messages": [
            HumanMessage(content=user_query)
        ]
    }, config=config)
    

    st.write("Final response structure:", final_response)
    "st.session_state object:", st.session_state
    