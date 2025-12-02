import streamlit as st
import sys, os
from dotenv import load_dotenv
import os
from agents.manager import AgentManager
from storage.session_store import load_session

# Load all environment variables from .env file
load_dotenv()


st.title("Rural Education AI Tutor (Multimedia Enhanced)")

session_id = st.text_input("Student ID", "student001")
question = st.text_input("Ask something to learn")

if st.button("Get Answer"):
    manager = AgentManager()
    response = manager.handle_question(session_id, question)

    st.subheader("Answer")
    st.write(response["answer"])

    # Videos
    if response["videos"]:
        st.subheader("Videos for Understanding")
        for vid in response["videos"]:
            st.video(vid["video_url"])
            st.caption(vid["title"])

    # Images
    if response["images"]:
        st.subheader("Illustrations")
        for img in response["images"]:
            st.image(img)

    st.subheader("Session Memory")
    st.json(load_session(session_id))
    
from storage.session_store import save_session

if st.button("Reset Memory"):
    save_session(session_id, {}, {"qa_history": []})
    st.success("Memory cleared!")
