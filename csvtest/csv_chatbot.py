import streamlit as st
import pandas as pd
import time
import importlib
from langchain_core.messages import HumanMessage, AIMessage
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.tools import PythonREPLTool
from langchain import hub
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler



st.set_page_config(page_title="Streamlit Agent", page_icon="🤖")
st.title("Streamlit Agent🤖")
user_query = st.chat_input("Your message")
python_repl_tool = PythonREPLTool()

# Chat history and state initialization
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
if "uploader_count" not in st.session_state:
    st.session_state.uploader_count = 0
if "pandas_agent" not in st.session_state:
    st.session_state.pandas_agent = None

# Get response
def get_response(query, chat_history, processed_files):
    template = """
    You are a helpful assistant. Answer the following question considering the history of the conversation:

    Chat history: {chat_history}

    User question: {user_question}

    Processed Files: {processed_files}

    All responses should be presented in a Streamlit-compatible format. Use markdown, tables, or other relevant Streamlit components as appropriate to ensure a clear and concise presentation.

    [Streamlit Example]
    - st.dataframe(df)
    - st.area_chart(df)
    - st.bar_chart(df)
    - st.line_chart(df)

    [Case 1]
    [Execution Condition]
    If the question can be answered using historical data:
    - Answer the question using information from chat history or processed files.

    [Case 2]
    [Execution Condition]
    If the question cannot be answered from historical data:
    - First, say: "Sorry, this has nothing to do with the data you uploaded."
    - Then break away from the framework of historical data, find answers based on the question, and respond.

    [Case 3]
    [Execution Condition]
    If the question requires drawing a chart:
    - First, say: "Here is your chart!"
    - You are an excellent agent designed to write and save Streamlit code to complete the chart.

    - Standard Procedure:
    1. Python code: Write and save the Streamlit code (design the functionality in the `def gen_plot()` function). Below are the programming and calling principles:
        {streamlit_principle}
    - Notes:
        - Once you write the Streamlit code, it must be saved using Python REPL.
        - Prefer using native Streamlit drawing components.
        - Ignore errors if they occur.
        - Default to UTF-8 encoding.
        - Do not call `gen_plot` in the code.
        - After saving, do not suggest running the Streamlit script.

    [Q&A Examples]
    8. User Input: Please DEMO chart (line, 3D or pie)
        - Analysis: "chart" usually indicates a need for plotting. Since it's just a DEMO and no other related data is provided, there is no need to use data from a database or Tavily. Please assume variables as needed and use Streamlit’s plotting capabilities. Remember, the final code must be saved using Python REPL.
        a. Write Streamlit code and save it using Python REPL:
            {streamlit_example}
    """
    streamlit_example = """
    Example of calling Python REPL, use the input "__arg1": "code"=..., avoid using "arguments".
    Example of drawing a line chart:
    {
    "__arg1": "code = \"\"\"\nimport streamlit as st\nimport numpy as np\nimport pandas as pd\n\ndef gen_plot():\n    st.header('Line Chart Example')\n\n    # Create sample data\n    x = np.linspace(0, 10, 100)\n    y = np.sin(x)\n    df = pd.DataFrame({'x': x, 'y': y})\n\n    # Create line chart\n    st.line_chart(df.set_index('x'))\n\"\"\"\n\nwith open('plot_script.py', 'w', encoding='utf-8') as f:\n    f.write(code)"
    }
    Example of drawing a 3D chart:
    {
    "__arg1": "code = \"\"\"\nimport streamlit as st\nimport numpy as np\nimport pandas as pd\nimport plotly.express as px\n\n\ndef gen_plot():\n    st.header('3D Scatter Plot Example')\n\n    # Create sample data\n    df = pd.DataFrame({\n        'x': np.random.rand(100),\n        'y': np.random.rand(100),\n        'z': np.random.rand(100),\n        'label': np.random.choice(['A', 'B', 'C'], 100)\n    })\n\n    # Create 3D scatter plot\n    fig = px.scatter_3d(df, x='x', y='y', z='z', color='label')\n    st.plotly_chart(fig)\n\"\"\"\n\nwith open('plot_script.py', 'w', encoding='utf-8') as f:\n    f.write(code)"
    }
    Example of drawing a pie chart:
    {
    "__arg1": "code = \"\"\"\nimport streamlit as st\nimport matplotlib.pyplot as plt\n\ndef gen_plot():\n    # Pie chart, where the slices will be ordered and plotted counter-clockwise:\n    labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'\n    sizes = [15, 30, 45, 10]\n    explode = (0, 0.1, 0, 0)  # only \"explode\" the 2nd slice (i.e. 'Hogs')\n\n    fig1, ax1 = plt.subplots()\n    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',\n            shadow=True, startangle=90)\n    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.\n\n    st.pyplot(fig1)\n\"\"\"\n\nwith open('plot_script.py', 'w', encoding='utf-8') as f:\n    f.write(code)"
    }
    """

    streamlit_principle = """
    {
    "__arg1": "code = \"\"\"\n import .... def gen_plot():\n Place Streamlit code here\n\"\"\"\n\nwith open('plot_script.py', 'w', encoding='utf-8') as f:\n    f.write(code)"
    }
    """

    instructions = template.format(
        chat_history=chat_history,
        user_question=query,
        processed_files=processed_files,
        streamlit_example=streamlit_example,
        streamlit_principle=streamlit_principle
    )

    base_prompt = hub.pull("langchain-ai/openai-functions-template")
    prompt = base_prompt.partial(instructions=instructions)
    llm = ChatOpenAI(temperature=0)
    chain = prompt | llm | StrOutputParser()

    return chain.invoke({
        "input": {
            "chat_history": chat_history,
            "user_question": query,
            "processed_files": processed_files
        },
        "agent_scratchpad": [HumanMessage(content=query)]  # Here we ensure agent_scratchpad is a list of messages
    })



# Show file upload
def show_upload():
    st.session_state.uploader_count += 1
    st.session_state.chat_history.append({"type": "file", "index": st.session_state.uploader_count})

# Function agent for data analysis
def function_agent(df, pandas_agent):
    results = []

    all_data = df
    results.append({"type": "alldata", "content": all_data})
    
    # Data Overview
    st.write("**Data Overview**")
    overview = df.head().to_markdown()
    st.write(df.head())
    results.append({"type": "overview", "content": overview})
    
    # Data Cleaning
    st.write("**Data Cleaning**")
    try:
        columns_df = pandas_agent.run("What are the meaning of the columns?")
        st.write(columns_df)
        results.append({"type": "data_cleaning", "content": columns_df})
    except Exception as e:
        st.error(f"Error retrieving column descriptions: {e}")
        results.append({"type": "error", "content": str(e)})

    try:
        missing_values = pandas_agent.run("Check for missing values in the data. If any, answer with 'There are'")
        st.write(missing_values)
        results.append({"type": "missing_values", "content": missing_values})
    except Exception as e:
        st.error(f"Error retrieving missing values: {e}")
        results.append({"type": "error", "content": str(e)})

    try:
        duplicates = pandas_agent.run("Are there any duplicate values and if so where?")
        st.write(duplicates)
        results.append({"type": "duplicates", "content": duplicates})
    except Exception as e:
        st.error(f"Error retrieving duplicate values: {e}")
        results.append({"type": "error", "content": str(e)})
    
    # Data Summarisation
    st.write("**Data Summarisation**")
    summarization = df.describe().to_markdown()
    st.write(df.describe())
    results.append({"type": "summarization", "content": summarization})
    
    try:
        correlation_analysis = pandas_agent.run("Calculate correlations between numerical variables to identify potential relationships.")
        st.write(correlation_analysis)
        results.append({"type": "correlation_analysis", "content": correlation_analysis})
    except Exception as e:
        st.error(f"Error retrieving correlation analysis: {e}")
        results.append({"type": "error", "content": str(e)})

    try:
        outliers = pandas_agent.run("Identify outliers in the data that may be erroneous or that may have a significant impact on the analysis.")
        st.write(outliers)
        results.append({"type": "outliers", "content": outliers})
    except Exception as e:
        st.error(f"Error retrieving outliers: {e}")
        results.append({"type": "error", "content": str(e)})
    
    try:
        new_features = pandas_agent.run("What new features would be interesting to create?")
        st.write(new_features)
        results.append({"type": "new_features", "content": new_features})
    except Exception as e:
        st.error(f"Error suggesting new features: {e}")
        results.append({"type": "error", "content": str(e)})
    
    return results

def clear_plot_script():
    with open("plot_script.py", "w") as file:
        # file.write("# 此文件已清空以做為新的輸入\n")
        file.write("def gen_plot():\n")
        file.write("    pass\n")  # 空函數內容

def dynamic_import_and_execute_plot():
    # 动态导入 plot_script.py 模块并调用 gen_plot 函数
    try:
        plot_module = importlib.import_module("plot_script")
        plot_module.gen_plot()
    except AttributeError as e:
        st.error(f"Error executing gen_plot: {e}")
    except Exception as e:
        st.error(f"Error importing plot_script: {e}")


file = None

if st.button("切換文件上傳器"):
    show_upload()

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, dict) and message.get('type') == 'file':
        with st.chat_message("AI"):
            file = st.file_uploader(f"上傳您的數據 {message['index']}", key=f"file_uploader_{message['index']}", type=["xlsx", "csv"])
            if file is not None and file.name not in st.session_state.processed_files:
                with st.spinner("正在處理您的文件"):
                    time.sleep(2)  
                    try:
                        df = pd.read_csv(file, low_memory=False)
                        pandas_agent = create_pandas_dataframe_agent(ChatOpenAI(), df, verbose=True)
                        if df is not None and pandas_agent is not None:
                            st.success(f"{file.name} 文件處理成功！")
                            results = function_agent(df, pandas_agent)
                            file_name = file.name
                            if st.session_state.uploader_count != 0:
                                st.session_state.processed_files.append(file_name)
                                st.session_state.chat_history.append({"type": "file_name", "content":file_name})
                                st.session_state.chat_history.extend(results)
                                st.session_state.pandas_agent = pandas_agent
                                
                        else:
                            st.error("文件處理失敗。")
                    except UnicodeDecodeError:
                        st.error("文件编码错误。请检查文件编码或尝试其他文件。")
                    except Exception as e:
                        st.error(f"处理文件时出错: {e}")

# Handle user query
if user_query:
    # 清空 plot_script.py 的所有内容，并保留 gen_plot 函数
    clear_plot_script()
    
    st.session_state.chat_history.append(HumanMessage(user_query))
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        ai_response = get_response(user_query, st.session_state.chat_history, st.session_state.processed_files)
        st.session_state.chat_history.append(AIMessage(ai_response))
        st.markdown(ai_response)
        st.write("------------")
        if file is not None:
            df = pd.read_csv(file, low_memory=False)
            pandas_agent = create_pandas_dataframe_agent(ChatOpenAI(), df, verbose=True)
            st_callback = StreamlitCallbackHandler(st.container())
            response = pandas_agent.invoke(
                {"input": user_query}, {"callbacks": [st_callback]}
            )   
            st.write(response["output"])
            st.write("------------")
            # 之後呼叫動態導入的 gen_plot
            dynamic_import_and_execute_plot()
            st.write("------------")
    st.write("------------")
    for message in st.session_state.chat_history:
        st.write(message)
    st.write("------------")
