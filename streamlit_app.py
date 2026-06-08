import os
from typing import Any

import requests
import streamlit as st


API_URL = os.getenv("MSME_API_URL", "http://api:8000/chat")


def ask_api(query: str) -> dict[str, Any]:
    response = requests.post(
        API_URL,
        json={"query": query},
        timeout=90,
    )
    response.raise_for_status()
    return response.json()


st.set_page_config(
    page_title="MSME Advisory Assistant",
    page_icon="",
    layout="wide",
)

st.title("MSME Advisory Assistant")
st.caption("MSME, startup, entrepreneurship, schemes, registration, funding, compliance, and business support guidance.")

with st.sidebar:
    st.header("Connection")
    api_url = st.text_input("FastAPI endpoint", value=API_URL)
    if api_url != API_URL:
        API_URL = api_url

    st.header("Examples")
    examples = [
        "What is Udyam Registration?",
        "What schemes are available for MSME manufacturing businesses?",
        "What is PMEGP and who is eligible?",
        "Latest MSME policy updates in India",
    ]
    selected_example = st.radio("Try a question", examples, index=None)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

default_query = selected_example or ""
query = st.chat_input("Ask about MSME, startups, schemes, registration, funding, or compliance...")

if selected_example and st.button("Ask selected example"):
    query = default_query

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Searching knowledge base and preparing answer..."):
            try:
                result = ask_api(query)
            except requests.HTTPError as exc:
                detail = exc.response.text if exc.response is not None else str(exc)
                st.error(f"API error: {detail}")
                st.stop()
            except requests.RequestException as exc:
                st.error(f"Could not connect to FastAPI server at {API_URL}. Error: {exc}")
                st.stop()

        answer = result.get("answer", "")
        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Route")
            st.code(result.get("route", "unknown"))

        with col2:
            st.subheader("Metrics")
            st.json(result.get("metrics", {}))

        sources = result.get("sources", [])
        if sources:
            st.subheader("Sources")
            for index, source in enumerate(sources, start=1):
                label = source.get("source_file") or f"Source {index}"
                with st.expander(label):
                    st.json(source)

if st.session_state.messages and st.button("Clear chat"):
    st.session_state.messages = []
    st.rerun()
