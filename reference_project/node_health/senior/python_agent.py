from langchain import hub
from langchain.agents import AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import create_openai_functions_agent, create_openai_tools_agent
from langchain_openai import ChatOpenAI


class PythonAgent(AgentExecutor):
    def __init__(self):
        tools = [PythonREPLTool()]

        instructions = """你是一個優秀的代理，旨在編寫與存檔streamlit程式碼來完成繪圖。
        [streamlit繪圖能力]
        - 使用說明：使用pythonREPL工具 實現編寫與存檔streamlit程式碼來完成繪圖
        - 使用時機：當最終需要繪圖時使用這個能力
        - 標準流程：
            1. python code: 撰寫並儲存streamlit code (功能請設計在def gen_plot()中)。下方為編程與調用原則
                {streamlit_priciple}
        - 注意事項：
                - 一旦編寫streamlit code，必須用python PREL 將程式儲存
                - 優先使用streamlit原生繪圖的元件
                - 如果出現錯誤，不用理會
                - 預設使用utf-8進行編碼
                - 禁止在code中呼叫gen_plot
                - 存檔後禁止建議執行streamlit run python檔案。
        [問答範例]
            8. 使用者輸入:請DEMO  chart (line, 3d or pie)
                - 解析：chart通常代表需要繪圖，且因為只是DEMO 並無提供其他相關資料，所以無需使用到資料庫的資料或是tavily的尋找。請直接按照需求假設變數並使用streamlit繪圖能力即可，記住最終需要使用python prel 儲存此段程式碼
                a. 撰寫streamlit code 並使用 REPL python 儲存檔案
                    {streamlit_example}
        """
        streamlit_example = """
        調用python REPL範例，請按照輸入"__arg1": "code"=...，禁止使用"arguments"
        繪畫 line chart的範例:
        {
        "__arg1": "code = \"\"\"\nimport streamlit as st\nimport numpy as np\nimport pandas as pd\n\ndef gen_plot():\n    st.header(\\\"Line Chart Example\\\")\n\n    # Create sample data\n    x = np.linspace(0, 10, 100)\n    y = np.sin(x)\n    df = pd.DataFrame({'x': x, 'y': y})\n\n    # Create line chart\n    st.line_chart(df.set_index('x'))\n\"\"\"\n\nwith open('plot_script.py', 'w', encoding='utf-8') as f:\n    f.write(code)"
        }
        繪畫 3d chart的範例:

        {
        "__arg1": "code = \"\"\"\nimport streamlit as st\nimport numpy as np\nimport pandas as pd\nimport plotly.express as px\n\n\ndef gen_plot():\n    st.header('3D Scatter Plot Example')\n\n    # Create sample data\n    df = pd.DataFrame({\n        'x': np.random.rand(100),\n        'y': np.random.rand(100),\n        'z': np.random.rand(100),\n        'label': np.random.choice(['A', 'B', 'C'], 100)\n    })\n\n    # Create 3D scatter plot\n    fig = px.scatter_3d(df, x='x', y='y', z='z', color='label')\n    st.plotly_chart(fig)\n\"\"\"\n\nwith open('plot_script.py', 'w', encoding='utf-8') as f:\n    f.write(code)"
        }

        繪畫 pie chart的範例:
        {
        "__arg1": "code = \"\"\"\nimport streamlit as st\nimport matplotlib.pyplot as plt\n\ndef gen_plot():\n    # Pie chart, where the slices will be ordered and plotted counter-clockwise:\n    labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'\n    sizes = [15, 30, 45, 10]\n    explode = (0, 0.1, 0, 0)  # only \"explode\" the 2nd slice (i.e. 'Hogs')\n\n    fig1, ax1 = plt.subplots()\n    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',\n            shadow=True, startangle=90)\n    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.\n\n    st.pyplot(fig1)\n\"\"\"\n\nwith open('plot_script.py', 'w', encoding='utf-8') as f:\n    f.write(code)"
        }
        """

        streamlit_priciple = """
        {
        "__arg1": "code = \"\"\"\n import .... def gen_plot():\n 這裡放入streamlit程式內容\n\"\"\"\n\nwith open('plot_script.py', 'w', encoding='utf-8') as f:\n    f.write(code)"
        }

        """
        instructions = instructions.format(streamlit_example=streamlit_example,
                                           streamlit_priciple=streamlit_priciple)
        base_prompt = hub.pull("langchain-ai/openai-functions-template")
        prompt = base_prompt.partial(instructions=instructions)

        # agent = create_openai_functions_agent(
        agent = create_openai_tools_agent(
            # ChatOpenAI(model='gpt-4o-mini', temperature=0), tools, prompt
            ChatOpenAI(model='gpt-4o', temperature=0), tools, prompt
            # ChatOpenAI(model='gpt-3.5-turbo', temperature=0), tools, prompt# 目前0728的prompt做不出來 理解力不足
        )

        super().__init__(agent=agent, tools=tools, verbose=True)


# 使用範例
if __name__ == "__main__":
    agent = PythonAgent()
    response = agent.invoke({
        # "input": "我現在要使用streamlit 繪製 bar chart ，請用def gen_plot(): 此函式將所有code包覆。完成後將其存檔為plot_script.py"
        "input": "prompt"
    })
    print(response)
