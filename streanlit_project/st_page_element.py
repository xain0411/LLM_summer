import streamlit as st  
import numpy as np
import pandas as pd
from PIL import Image
import random
import matplotlib.pyplot as plt
import datetime
from vega_datasets import data
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain.tools import tool
from langchain_core.output_parsers import StrOutputParser
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
import os
import importlib
import importlib.util
import sys
from langchain_experimental.agents import create_pandas_dataframe_agent


load_dotenv()

left, middle = st.columns([4,6], gap="medium",vertical_alignment="top")



### left ###

# DataFrame chart
left.subheader("DataFrame chart")
df = pd.DataFrame(np.random.randn(5, 1),columns=("col %d" % i for i in range(1)))
left.dataframe(df.style.highlight_max(axis=0))

# metric
left.subheader("Metric")
left.metric(label="Temperature", value="70 °F", delta="1.2 °F")


# buttons
left.subheader("buttons")
# link button
left.link_button("GO TO GOOGLE",
               url="https://www.google.com",
               type ="primary",
               help="Click me to go to Google"
)

left.link_button("GO TO spotify",
               url="https://open.spotify.com",
               disabled= True,
               help="Click me to go to spotify"
)




#------------------------------------使用HTML排版版本------------------------------------#
# Function to create a link button
# def link_button(label, url, disabled=False, type="primary", help=None):
#     button_style = {
#         "primary": "background-color: #007bff; color: white;",
#         "secondary": "background-color: #6c757d; color: white;",
#         "success": "background-color: #28a745; color: white;",
#         "danger": "background-color: #dc3545; color: white;",
#         "warning": "background-color: #ffc107; color: black;",
#         "info": "background-color: #17a2b8; color: white;",
#         "light": "background-color: #f8f9fa; color: black;",
#         "dark": "background-color: #343a40; color: white;",
#     }
    
#     style = button_style.get(type, button_style["primary"])
#     if disabled:
#         st.markdown(f'<button style="{style} width:100%;" disabled title="{help}">{label}</button>', unsafe_allow_html=True)
#     else:
#         st.markdown(f'<a href="{url}" target="_blank"><button style="{style} width:100%;" title="{help}">{label}</button></a>', unsafe_allow_html=True)

# # List of buttons with labels and URLs
# buttons = [
#     {"label": "GO TO GOOGLE", "url": "https://www.google.com", "type": "danger", "help": "Click me to go to Google", "disabled": False},
#     {"label": "GO TO SPOTIFY", "url": "https://open.spotify.com", "type": "primary", "help": "Click me to go to Spotify", "disabled": True},
#     {"label": "GO TO YOUTUBE", "url": "https://www.youtube.com", "type": "primary", "help": "Click me to go to YouTube", "disabled": False},
#     {"label": "GO TO TWITTER", "url": "https://www.twitter.com", "help": "Click me to go to Twitter", "disabled": False},
# ]

# # Create a new row of columns for each button
# for i in range(0, len(buttons), 4):
#     cols = st.columns(4)  # Adjust the number of columns based on how many buttons you want per row
#     for col, button in zip(cols, buttons[i:i+4]):
#         with col:
#             link_button(
#                 label=button["label"],
#                 url=button["url"],
#                 disabled=button.get("disabled", False),
#                 type=button.get("type", "primary"),
#                 help=button.get("help", "")
#             )
#------------------------------------使用HTML排版版本------------------------------------#



# page link
left.subheader("page link")
left.page_link("st_page_element.py", label="Home", icon="🏠")
left.page_link("pages/page_1.py", label="Page 1", icon="1️⃣")
left.page_link("pages/page_2.py", label="Page 2", icon="2️⃣", disabled=True)



### middle ###

container = middle.container(border=True)
container.subheader("Streamlit Bot")
user_query = st.chat_input("Your message")

# chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
    
# Processed files
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
    
# File uploader
if "uploader_count" not in st.session_state:
    st.session_state.uploader_count = 0
    
    
# get response
def get_response(query, chat_history):
    template = """
    You are a helpful assistant. Answer the following question considering the history of the conversation:
    
    Chat history:{chat_history}
    
    User question:{user_question}
    
    """

    prompt = ChatPromptTemplate.from_template(template)

    llm = ChatOpenAI()

    chain = prompt | llm | StrOutputParser()

    # return chain.invoke({
    #     "chat_history": chat_history,
    #     "user_question": query
    # })
    
    return chain.stream({
        "chat_history": chat_history,
        "user_question": query
    })
    
    # response_content = chain.stream({
    #     "chat_history": chat_history,
    #     "user_question": query,
    # })
    # return response_content

# Handle file uploader toggle
def show_upload():
    st.session_state.uploader_count += 1
    st.session_state.chat_history.append({"type": "file", "index": st.session_state.uploader_count})

if middle.button("切換文件上傳器"):
    show_upload()
   
# pandas_agent = create_pandas_dataframe_agent(ChatOpenAI(), df, verbose=True)   

# def function_agent():
#     st.write("**Data Overview**")
#     st.write("The first rows of your dataset look like this:")
#     st.write(df.head())
#     st.write("**Data Cleaning**")
#     columns_df = pandas_agent.run("What are the meaning of the columns?")
#     st.write(columns_df)
#     missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
#     st.write(missing_values)
#     duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
#     st.write(duplicates)
#     st.write("**Data Summarisation**")
#     st.write(df.describe())
#     correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
#     st.write(correlation_analysis)
#     outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
#     st.write(outliers)
#     new_features = pandas_agent.run("What new features would be interesting to create?")
#     st.write(new_features)


# # conversation
# for message in st.session_state.chat_history:
#     if isinstance(message, HumanMessage):
#         with container.chat_message("Human"):
#             st.markdown(message.content)
#     elif isinstance(message, AIMessage):
#         with container.chat_message("AI"):
#             st.markdown(message.content)
#     elif message['type'] == 'file':
#         with container.chat_message("AI"):
#             st.markdown("請上傳您的數據:")
#             file = st.file_uploader(f"上傳您的數據 {message['index']}", key=f"file_uploader_{message['index']}", type=["xlsx", "csv"])
#             if file is not None and file.name not in st.session_state.processed_files:
#                 with st.spinner("正在處理您的文件"):
#                     time.sleep(5)
#                     try:
#                         # 判断文件的扩展名
#                         if file.name.endswith('.xlsx'):
#                             # 读取 Excel 文件为 DataFrame
#                             df = pd.read_excel(file)
#                             st.success("Excel 文件處理成功！")
#                             function_agent()
#                         elif file.name.endswith('.csv'):
#                             # 读取 CSV 文件为 DataFrame
#                             df = pd.read_csv(file, encoding="utf-8")
#                             st.success("CSV 文件處理成功！")
#                             function_agent()
#                         else:
#                             st.error("不支持的文件格式。請上傳 .xlsx 或 .csv 文件。")
#                             continue

#                     except UnicodeDecodeError:
#                         st.error("文件编码错误。请检查文件编码或尝试其他文件。")
#                     except Exception as e:
#                         st.error(f"處理文件時出錯: {e}")
#                         continue

#                     # 将 DataFrame 存储在 session_state 中
#                     st.session_state[f'dataframe_{message["index"]}'] = df
#                     st.session_state.processed_files.append(file.name)
            
# Define pandas agent and function for data analysis
def process_file_and_create_agent(file):
    if file.name.endswith('.xlsx'):
        file.seek(0)
        df = pd.read_excel(file,low_memory=False)
    elif file.name.endswith('.csv'):
        file.seek(0)
        df = pd.read_csv(file, encoding="utf-8",low_memory=False)
    else:
        st.error("Unsupported file format.")
        return None, None

    # Create pandas agent
    pandas_agent = create_pandas_dataframe_agent(ChatOpenAI(), df, verbose=True)
    return df, pandas_agent

def function_agent(df, pandas_agent):
    # Data Overview
    st.write("**Data Overview**")
    st.write("The first rows of your dataset look like this:")
    st.write(df.head())
    
    # Data Cleaning
    st.write("**Data Cleaning**")
    try:
        columns_df = pandas_agent.run("What are the meaning of the columns?")
        st.write(columns_df)
    except Exception as e:
        st.error(f"Error while retrieving column information: {e}")
    
    try:
        missing_values = pandas_agent.run("How many missing values does this dataframe have? Start the answer with 'There are'")
        st.write(missing_values)
    except Exception as e:
        st.error(f"Error while retrieving missing values information: {e}")
    
    try:
        duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
        st.write(duplicates)
    except Exception as e:
        st.error(f"Error while retrieving duplicate values information: {e}")
    
    # Data Summarisation
    st.write("**Data Summarisation**")
    st.write(df.describe())
    
    try:
        correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
        st.write(correlation_analysis)
    except Exception as e:
        st.error(f"Error while retrieving correlation analysis: {e}")
    
    try:
        outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
        st.write(outliers)
    except Exception as e:
        st.error(f"Error while retrieving outliers information: {e}")
    
    try:
        new_features = pandas_agent.run("What new features would be interesting to create?")
        st.write(new_features)
    except Exception as e:
        st.error(f"Error while retrieving new features information: {e}")


# conversation
for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with container.chat_message("Human"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with container.chat_message("AI"):
            st.markdown(message.content)
    elif message['type'] == 'file':
        with container.chat_message("AI"):
            st.markdown("請上傳您的數據")
            file = st.file_uploader(f"上傳您的數據 {message['index']}", key=f"file_uploader_{message['index']}", type=["xlsx", "csv"])
            if file is not None and file.name not in st.session_state.processed_files:
                with st.spinner("正在處理您的文件"):
                    time.sleep(5)
                    try:
                        df, pandas_agent = process_file_and_create_agent(file)
                        if df is not None and pandas_agent is not None:
                            st.success(f"{file.name} 文件處理成功！")
                            function_agent(df, pandas_agent)
                        else:
                            st.error("文件處理失敗。")
                    except UnicodeDecodeError:
                        st.error("文件編碼錯誤。請檢查文件編碼或嘗試其他文件。")
                    except Exception as e:
                        st.error(f"處理文件時出錯: {e}")
                        continue

                    # 将 DataFrame 存储在 session_state 中
                    st.session_state[f'dataframe_{message["index"]}'] = df
                    st.session_state.processed_files.append(file.name)          
            
      
def clear_plot_script():
    # 確保使用絕對路徑來清空 plot_script.py 的內容並新增空的 gen_plot 函數
    script_path = "C:/Users/LinChengXain/LLM_summer/plot_script.py"
    with open(script_path, "w") as file:
        # file.write("# 此文件已清空以做為新的輸入\n")
        file.write("def gen_plot():\n")
        file.write("    pass\n")  # 空函數內容

def dynamic_import_and_execute_plot():
    # 添加包含 plot_script.py 的目錄到 sys.path
    script_dir = os.path.dirname("C:/Users/LinChengXain/LLM_summer/plot_script.py")
    if script_dir not in sys.path:
        sys.path.append(script_dir)
    
    # 動態導入 plot_script.py 模組並呼叫 gen_plot 函式
    plot_module = importlib.import_module("plot_script")
    plot_module.gen_plot()

# user input
if user_query:
    # 清空 plot_script.py 的所有內容，並保留 gen_plot 函數
    clear_plot_script()
    
    st.session_state.chat_history.append(HumanMessage(user_query))
    with container.chat_message("Human"):
        st.markdown(user_query)
    
    with container.chat_message("AI"):
        ai_response = st.write_stream(get_response(user_query, st.session_state.chat_history))
        st.session_state.chat_history.append(AIMessage(ai_response))
        
        st.write("test")
        
        # 動態導入和執行 gen_plot 函數
        dynamic_import_and_execute_plot()




        
    

            

    

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

code_agent = create_agent(
    llm, [python_repl_tool], 
        "You are a skilled agent tasked with writing and saving [Streamlit code] to generate plots. Use the PythonREPL tool to achieve this. Ensure that the code is saved in the function `gen_plot()` and is stored in 'plot_script.py'.")
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

flow =[]

if "flow" not in st.session_state:
    st.session_state.flow = []

if "final_response" not in st.session_state:
    st.session_state.final_response = None
    
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
            st.session_state.flow.append(s)

    st.session_state.final_response = graph.invoke({
        "messages": [
            HumanMessage(content=user_query)
        ]
    }, config=config)
       

    
### right ###  



### sidebar ###

                 
# Using object notation
add_selectbox = st.sidebar.selectbox(
    "How would you like to be contacted?",
    ("working flow", "Final response structure", "st.session_state object")
)


# Using a form
with st.sidebar:
    if add_selectbox == "working flow":
        st.write(st.session_state.flow)
    elif add_selectbox == "Final response structure":
            if st.session_state.final_response:
                st.write(st.session_state.final_response)
    elif add_selectbox == "st.session_state object":
        st.write(st.session_state)
