import streamlit as st
from typing import List

def main() -> None:
    st.title("String List Input Application")

    # Initialize session state for string list if it doesn't exist
    if 'string_list' not in st.session_state:
        st.session_state.string_list = []

    max_strings = 10

    st.write("Enter up to 10 strings:")

    # Text input for strings
    new_string: str = st.text_input("Enter a string")

    # Button to add a string to the list
    if st.button("Add String"):
        if new_string and len(st.session_state.string_list) < max_strings:
            st.session_state.string_list.append(new_string)
            st.success(f"String added! {len(st.session_state.string_list)} strings in the list.")
            st.text_input("Enter a string", value="", key="new_string")  # Reset input field
        elif not new_string:
            st.error("Please enter a string.")
        else:
            st.error("Maximum of 10 strings reached.")

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
