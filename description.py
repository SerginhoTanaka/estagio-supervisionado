import streamlit as st
import pandas as pd

class Description():
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def run(self) -> None:
        st.empty()
        st.write("""
        # Description Page
        """)
        # Calcula a matriz de correlação
        corr_matrix = self.data.corr()

        # Mostra as colunas com maior correlação com a coluna "situacao"
        correlation_ranking = corr_matrix['situacao'].sort_values(ascending=False)
        st.write("Ranking de Correlação com 'situacao':")
        st.write(correlation_ranking)
