import streamlit as st  
from datetime import datetime
import datetime as dt
import numpy as np

st.title("first page")

# radio
st.subheader("radio")
genre = st.radio(
    "What's your favorite movie genre",
    [":rainbow[Comedy]", "***Drama***", "Documentary :movie_camera:"],
    captions=["Laugh out loud.", "Get the popcorn.", "Never stop learning."])

if genre == ":rainbow[Comedy]":
    st.write("You selected comedy.")
else:
    st.write("You didn't select comedy.")
        
# Store the initial value of widgets in session state
st.markdown("Store the initial value of widgets in session state")
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False
    st.session_state.horizontal = False

col1, col2 = st.columns(2)

with col1:
    st.checkbox("Disable radio widget", key="disabled")
    st.checkbox("Orient radio options horizontally", key="horizontal")

with col2:
    st.radio(
        "Set label visibility 👇",
        ["visible", "hidden", "collapsed"],
        key="visibility",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
        horizontal=st.session_state.horizontal,
    )
    
    
# slider
st.subheader("slider")
start_time = st.slider(
    "When do you start?",
    value=datetime(2020, 1, 1, 9, 30),
    format="MM/DD/YY - hh:mm")
st.write("Start time:", start_time)


# color picker
st.subheader("color picker")
color = st.color_picker("Pick A Color", "#00f900")
st.write("The current color is", color)


# date input
st.subheader("date input")
today = dt.datetime.now()
next_year = today.year + 1
jan_1 = dt.date(next_year, 1, 1)
dec_31 = dt.date(next_year, 12, 31)

d = st.date_input(
    "Select your vacation for next year",
    (jan_1, dt.date(next_year, 1, 7)),
    jan_1,
    dec_31,
    format="MM.DD.YYYY",
)
st.write("Selected dates:", d)


# time input
st.subheader("time input")
t = st.time_input("Set an alarm for", value=None)
st.write("Alarm is set for", t)




# popover
st.subheader("popover")
col1, col2 = st.columns([5,5])
with col1:
    with st.popover("Open popover"):
        st.markdown("Hello World 👋")
        name = st.text_input("What's your name?")

    st.write("Your name:", name)

with col2:
    popover = st.popover("Filter items")
    red = popover.checkbox("Show red items.", True)
    blue = popover.checkbox("Show blue items.", True)

    if red:
        st.write(":red[This is a red item.]")
    if blue:
        st.write(":blue[This is a blue item.]")
    

# tabs
st.subheader("tabs")
tab1, tab2,tab3,tab4,tab5 = st.tabs(["📈 Chart", "🗃 Data","Cat", "Dog", "Owl"])
data = np.random.randn(10, 1)

tab1.subheader("A tab with a chart")
tab1.line_chart(data)

tab2.subheader("A tab with the data")
tab2.write(data)

with tab3:
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

with tab4:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab5:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
