import importlib
import os
import streamlit as st
from python_agent import PythonAgent
from langchain_community.callbacks.streamlit import StreamlitCallbackHandler

st.set_page_config(page_title="Streamlit Agent", page_icon="🤖")
st.title("Streamlit Agent🤖")
agent_executor = PythonAgent()


def clear_plot_script():
    # 清空 plot_script.py 的內容並新增空的 gen_plot 函數
    with open("plot_script.py", "w") as file:
        file.write("# 此文件已清空以做為新的輸入\n")
        file.write("def gen_plot():\n")
        file.write("    pass\n")  # 空函數內容


def dynamic_import_and_execute_plot():
    # 動態導入 plot_script.py 模組並呼叫 gen_plot 函式
    plot_module = importlib.import_module("plot_script")
    plot_module.gen_plot()


# Streamlit 介面處理
if prompt := st.chat_input():
    # 清空 plot_script.py 的所有內容，並保留 gen_plot 函數
    clear_plot_script()

    # 寫入用戶輸入
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        st_callback = StreamlitCallbackHandler(st.container())
        # 呼叫 agent_executor 以處理用戶請求
        response = agent_executor.invoke(
            {"input": prompt}, {"callbacks": [st_callback]}
        )
        st.write(response["output"])

        # 之後呼叫動態導入的 gen_plot
        dynamic_import_and_execute_plot()

        # 如果需要，可以在這裡選擇重啟應用
        # st.experimental_rerun()
