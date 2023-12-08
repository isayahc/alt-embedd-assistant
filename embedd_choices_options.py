import streamlit as st
from typing import List

def main():
    st.title("Streamlit UI for Option Selection")

    # Define your options
    distance_metrics = ["L2", "IP"]
    index_types = ["FLAT", "IVF_FLAT", "IVF_SQ8", "IVF_PQ", "HNSW", "IVF_HNSW", "RHNSW_FLAT", "RHNSW_SQ", "RHNSW_PQ", "ANNOY"]
    nprobe_values: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    # Create UI elements
    selected_distance_metric = st.selectbox("Select Distance Metric", distance_metrics)
    selected_index_type = st.selectbox("Select Index Type", index_types)
    selected_nprobe_value = st.selectbox("Select Nprobe Value", nprobe_values)

    # Displaying the selected options
    st.write(f"Selected Distance Metric: {selected_distance_metric}")
    st.write(f"Selected Index Type: {selected_index_type}")
    st.write(f"Selected Nprobe Value: {selected_nprobe_value}")

if __name__ == "__main__":
    main()
