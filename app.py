import streamlit as st
import requests

st.set_page_config(
    page_title="KrishiSahay",
    page_icon="🌾",
    layout="wide"
)

def check_internet():
    try:
        requests.get("https://www.google.com", timeout=3)
        return True
    except:
        return False


st.title("🌾 KrishiSahay - Smart Agriculture Assistant")

if check_internet():

    st.success("🌐 Internet Detected — Running Online Assistant")

    from online.apponline import run_online
    run_online()

else:

    st.warning("📡 No Internet — Running Offline Assistant")

    from offline.appoffline import run_offline
    run_offline()