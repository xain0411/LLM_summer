import streamlit as st  
import numpy as np
import pandas as pd
from PIL import Image
import random
import matplotlib.pyplot as plt
import datetime


# st.title('Hello World')
# st.write("This is a simple example of a Streamlit app.")



### 表格 ###
# df = pd.DataFrame(np.random.randn(10, 10),columns=("col %d" % i for i in range(10)))
# st.dataframe(df) # same as st.write(df)
# st.dataframe(df.style.highlight_max(axis=0))



### 圖表 ###
#面積圖
chart_data = pd.DataFrame(np.random.randn(20, 3),columns=['a', 'b', 'c'])

st.area_chart(chart_data)

st.area_chart(
    chart_data, x = "a", y = ["b","c"],color=["#FF0000","#00FF00"]
)


#長條圖
st.bar_chart(chart_data)

st.bar_chart(
    chart_data, x = "a", y = ["b","c"],color=["#00FF00","#0000FF"]
)


#折線圖
date_rng = pd.date_range(start="2024-07-08",end="2024-07-15",freq="D")
data = {
    "日期":date_rng,
    "A":np.random.randint(1000,5000,len(date_rng)),
    "B":np.random.randint(1000,5000,len(date_rng))
}
st.line_chart(data,x="日期",y=["A","B"],color=["#FF0000","#00FF00"])


#散點圖
chart_data = pd.DataFrame(np.random.randn(20, 3),columns=['a', 'b', 'c'])
st.scatter_chart(chart_data)

#加上size大小變化
chart_data = pd.DataFrame(np.random.randn(20, 3),columns=['col1', 'col2', 'col3'])
chart_data["col4"] = np.random.choice(["A","B","C"],20)

st.scatter_chart(chart_data,
                 x="col1",
                 y="col2",
                 size="col4",
                 color="col3")

#學生身高和體重的散點圖
stds = ["stdA","stdB","stdC","stdD","stdE"]
height = np.random.randint(150,190,len(stds))
weight = np.random.randint(50,100,len(stds))

df = pd.DataFrame({"學生姓名":stds,"身高":height,"體重":weight})
df = pd.DataFrame(df)
st.scatter_chart(df,x="身高",y="體重",color="學生姓名")

# 普通長條圖
arr = np.random.normal(1,1,size=100)
fig,ax = plt.subplots()
ax.hist(arr,bins=20)
st.pyplot(fig)

# 台北市人口地圖
taipei_boundaries = {
    "區域":["中正區","大同區","中山區","松山區","大安區","萬華區","信義區","士林區","北投區","內湖區","南港區","文山區"],
    "西經度":[121.519706,121.513042,121.526553,121.577004,121.543445,121.497986,121.571520,121.570458,121.508904,121.575512,121.618622,121.570037],
    "東經度":[121.526554,121.532599,121.553252,121.577005,121.575502,121.507463,121.592559,121.589997,121.540436,121.615142,121.634303,121.573608],
    "北緯度":[25.032404,25.063424,25.068697,25.051234,25.026515,25.046790,25.040609,25.127392,25.149780,25.083672,25.055033,24.989840],
    "南緯度":[25.032404,25.063424,25.068697,25.051234,25.026515,25.046790,25.040609,25.127392,25.149780,25.083672,25.055033,24.989840]
}

df_boundaries = pd.DataFrame(taipei_boundaries)

# 生成隨機人口數
df_boundaries["人口數"] = np.random.randint(100,1000,len(df_boundaries))

# 創建一個包含隨機顏色的數據列
df_boundaries["顏色"] = ["#{:02x}{:02x}{:02x}".format(random.randint(0,255),random.randint(0,255),random.randint(0,255)) for i in range(len(df_boundaries))]

# 繪製地圖
st.map(df_boundaries,
       latitude="南緯度",
       longitude="西經度",
       size="人口數",
       color="顏色",
)

# 下載按鈕
@st.cache_data #讓電腦做快取，檔案如果很大，第一次會載比較久，之後就會直接從快取拿，以加快下載及轉檔的速度
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")

csv = convert_df(df_boundaries)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name="df_boundaries.csv",
    mime="text/csv"
)


# 圓餅圖
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

st.pyplot(fig1)


### 互動工具 ###
# 按鈕
st.button("重設",type="primary")

if (st.button("Click me")):
    st.write("You clicked me!")
    
if (st.button("Click me too")):
    st.write("You clicked me too!")
    
# link button
st.link_button("GO TO GOOGLE",
               url="https://www.google.com",
               type ="primary",
               help="Click me to go to Google"
)

st.link_button("GO TO spotify",
               url="https://open.spotify.com",
               disabled= True,
               help="Click me to go to spotify"
)

# download button
# dataframe下載成csv
data = {
    "col1":[1,2,3,4],
    "col2":[5,6,7,8]
}
my_large_df = pd.DataFrame(data)


# 下載按鈕
@st.cache_data #讓電腦做快取，檔案如果很大，第一次會載比較久，之後就會直接從快取拿，以加快下載及轉檔的速度
def convert_df(df):
    return df.to_csv(index=False).encode("utf-8")

csv = convert_df(my_large_df)

st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name="my_large_df.csv",
    mime="text/csv"
)

# 下載圖片
with open("ai_picture.jpg","rb") as f:
    btn = st.download_button(
        label="Download image as JPG",
        data=f,
        file_name="ai_picture.jpg",
        mime="image/png"
    )
    

### 文字輸入框 ###
# 電影
st.title("電影標題")

# 添加說明文本
st.write("請輸入電影標題")

# 創建一個文本輸入框
movie_title = st.text_input("電影標題","ex: 鋼鐵人",max_chars=20)

# 創建確認按鈕
if st.button("確認"):
    # 顯示用戶輸入的電影標題
    st.write("您輸入的電影標題是: ",movie_title)
    

# 線上歡唱系統
st.title("歡迎使用線上歡唱系統")

# 添加說明文本
st.write("請在下方輸入您的姓名，然後點擊確認")

# 創建一個文本輸入框
name = st.text_input("姓名",max_chars=20)

# 創建確認按鈕
if st.button("確認姓名"):
    # 顯示歡迎詞
    if user_name:
        st.write(f"歡迎您，,{name}!感謝您使用我們的線上歡唱系統") 
    else:
        st.write("請先輸入您的姓名在按下確認鍵")
        
         
# 密碼
st.title("密碼輸入應用") 

# 添加說明文本
st.write("請輸入您的密碼，然後按下確認")

# 創建一個密碼輸入框
password = st.text_input("密碼","",type="password")

# 創建確認按鈕
if st.button("確認密碼"):
    # 檢查用戶輸入的密碼
    if password == "123456":
        st.write("密碼正確")
    else:
        st.write("密碼錯誤")
        
# 計算字符數
st.title("計算字符數")

txt = st.text_area(
    "輸入您的文字",
    "請在這裡輸入文字"
)

# 創建確認按鈕
if st.button("確認輸入完畢"):
    st.write(f"您輸入的字符有: {len(txt)}個")
    
    
# 文字分析應用
st.title("文字分析應用")

# 添加說明文本
st.write("請在下方輸入您的文字，然後點擊確認")

# 創建一個文本輸入框
story_text = st.text_area("輸入您的文字",max_chars=1000)

# 創建確認按鈕
if st.button("確認文字"):
    # 分割故事文本並計算單詞數量
    if story_text:
        words = story_text.split("，")
        words_count = len(words)
        st.write(f"您的故事中有{words_count}個單詞")
    else:
        st.write("請先輸入您的故事在按下確認鍵")
        
        

### 數字輸入 ###
st.title("數字輸入應用")
number = st.number_input("請輸入一個數字",value=None,placeholder="在這裡輸入數字...",min_value=0,max_value=100,step=2)
st.write(f"您輸入的數字是: {number}")


# 生日
# 設定應用標題
st.title("生日輸入應用")

# 添加說明文本
st.write("請在下方輸入您的生日，然後點擊確認")

# 創建一個日期輸入框
birthday = st.date_input("生日",datetime.date(2024,1,1))

# 創建確認按鈕
if st.button("確認生日"):
    # 顯示用戶輸入的生日
    st.write(f'您的生日是:', birthday.strftime("%年-%月-%日"))


# 休假時間
st.title("休假時間輸入應用")
today = datetime.datetime.now()
next_year = today.year + 1
jan_1 = datetime.date(next_year,1,1)
dec_31 = datetime.date(next_year,12,31)

d = st.date_input(
    "選擇明年要休假的時間",
    (jan_1,datetime.date(next_year,1,1)),
    jan_1,
    dec_31,
    format="YYYY-MM-DD"
)
d

# 設定自動發信時間
st.title("自動發信時間設定")

t = st.time_input("設定自動發信時間", value = None,step = 3600,label_visibility="hidden")
st.write(f"您設定的自動發信時間是: {t}")

