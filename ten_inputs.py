import streamlit as st
from typing import List

def main() -> None:
    st.title("String List Input Application")

    default_prompts = [
        "What's the best national park near Honolulu",
        "What are some famous universities in Tucson?",
        "What bodies of water are near Chicago?",
        "What is the name of Chicago's central business district?",
        "What are the two most famous universities in Los Angeles?",
        "What are some famous festivals in Mexico City?",
        "What are some famous festivals in Los Angeles?",
        "What professional sports teams are located in Los Angeles",
        "How do you classify Houston's climate?",
        "What landmarks should I know about in Cincinnati"
    ]

    # Initialize session state for string list if it doesn't exist
    if 'string_list' not in st.session_state:
        st.session_state.string_list = default_prompts.copy()

    max_strings = 10

    st.write("Enter up to 10 strings (default prompts are pre-loaded):")

    # Text input for strings
    new_string: str = st.text_input("Enter a string")

    # Button to add a string to the list
    if st.button("Add String"):
        if new_string:
            # Add new string and maintain list size
            st.session_state.string_list = [new_string] + st.session_state.string_list[:max_strings - 1]
            st.success(f"String added! List updated.")
            st.text_input("Enter a string", value="", key="new_string")  # Reset input field
        else:
            st.error("Please enter a string.")

    # Display the list of strings
    if st.session_state.string_list:
        st.write("Strings entered:")
        for i, string in enumerate(st.session_state.string_list, start=1):
            st.write(f"{i}. {string}")

    # Clear list button
    if st.button("Clear List"):
        st.session_state.string_list.clear()

if __name__ == "__main__":
    main()
