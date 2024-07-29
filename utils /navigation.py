import streamlit as st


async def navigation():
    # Navigation
    st.sidebar.page_link("app.py", label="Home", icon="🤖")
    st.sidebar.page_link("pages/01_🕰️_History.py", label="History", icon="🕰️")

    # Divider
    st.sidebar.divider()
