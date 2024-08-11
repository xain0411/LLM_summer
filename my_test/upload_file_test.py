import streamlit as st
import time
chat_box = st.chat_input("What do you want to do?")
if "uploader_visible" not in st.session_state:
    st.session_state["uploader_visible"] = False
def show_upload(state:bool):
    st.session_state["uploader_visible"] = state
    
with st.chat_message("system"):
    cols= st.columns((3,1,1))
    cols[0].write("Do you want to upload a file?")
    cols[1].button("yes", use_container_width=True, on_click=show_upload, args=[True])
    cols[2].button("no", use_container_width=True, on_click=show_upload, args=[False])

if st.session_state["uploader_visible"]:
    with st.chat_message("system"):
        file = st.file_uploader("Upload your data")
        if file:
            with st.spinner("Processing your file"):
                 time.sleep(5) #<- dummy wait for demo. 