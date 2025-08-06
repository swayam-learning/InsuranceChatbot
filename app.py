import streamlit as st
import requests
import os

# Backend API endpoint and token
API_URL = "https://bajajfinservhackatho-production.up.railway.app/hackrx/run"
API_TOKEN = "ssbakscstobcb3609e845e387e9f7ac988ea36090473eefbe6dae9cfe880c35c6b67d87a7757"

st.set_page_config(page_title="PDF QnA Assistant", layout="centered")
st.title(" Ask Questions from Your PDF")

# File uploader
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Multiline query input
query_input = st.text_area("Ask your question(s):", height=100, placeholder="E.g., What is the insurance coverage?")

# Submit button
if st.button("Submit"):

    if not uploaded_file:
        st.error("❌ Please upload a PDF file.")
    elif not query_input.strip():
        st.error("❌ Please enter a question.")
    else:
        try:
            # Prepare request
            files = {
                "pdf": (uploaded_file.name, uploaded_file, uploaded_file.type)
            }
            data = {
                "query": query_input.strip()
            }
            headers = {
                "Authorization": f"Bearer {API_TOKEN}"
            }

            # Send request to API
            with st.spinner("Processing..."):
                response = requests.post(API_URL, files=files, data=data, headers=headers)

            # Handle response
            if response.status_code == 200:
                result = response.json()
                answer = result.get("answer", "")
                if answer:
                    st.success("Answer from PDF:")
                    st.markdown(f"**{answer}**")
                else:
                    st.warning(" No answer returned.")
            else:
                st.error(f" API Error [{response.status_code}]: {response.text}")

        except Exception as e:
            st.error(f"Unexpected error: {e}")
