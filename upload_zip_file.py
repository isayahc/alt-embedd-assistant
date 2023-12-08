import streamlit as st
import zipfile
import os
from typing import Optional
from io import BytesIO

# Function to extract and save files from a zip archive
def save_uploaded_file(uploaded_file: BytesIO, path: str) -> None:
    with zipfile.ZipFile(uploaded_file, "r") as zip_ref:
        zip_ref.extractall(path)

# Streamlit UI
def main() -> None:
    st.title("Upload and Unzip a File")

    uploaded_file = st.file_uploader("Choose a ZIP file", type="zip")
    if uploaded_file is not None:
        # Save the uploaded file to a directory
        save_path = "data_corpus"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save_uploaded_file(uploaded_file, save_path)

        st.success("File Uploaded and Extracted Successfully!")

if __name__ == "__main__":
    main()
