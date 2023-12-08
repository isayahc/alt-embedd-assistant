import streamlit as st
import gdown
import os

# Function to download a file from Google Drive
def download_file_from_google_drive(url: str, output: str) -> None:
    gdown.download(url, output, quiet=False)

# Streamlit interface
def main():
    st.title("Google Drive File Downloader")

    # User inputs for the Google Drive file URL and output filename
    url = st.text_input("Enter the Google Drive file URL")
    output = st.text_input("Enter the output filename")

    if st.button("Download File"):
        # Download the file
        download_file_from_google_drive(url, output)
        st.success(f"File downloaded successfully: {output}")

        # Show the file in the sidebar if it's downloaded
        if os.path.exists(output):
            with open(output, "rb") as file:
                st.sidebar.download_button(
                    label="Download File",
                    data=file,
                    file_name=output,
                    mime="application/octet-stream"
                )

if __name__ == "__main__":
    main()
