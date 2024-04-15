import streamlit as st

import pandas as pd
class Description():
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def run(self)-> None:
        st.empty()
        st.write("""
        # Description Page
        """)