import streamlit as st
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

class Description():
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def run(self) -> None:
        st.empty()
        st.write("""lllll
        # Description Page
        """)
        # Calcula a matriz de correlação
        corr_matrix = self.data.corr()

        #  TODO Mostra as colunas com maior correlação com a coluna "situacao"
        #  TODO pesquisar feature selection para substituar a corr para dados categóricos
        correlation_ranking = corr_matrix['situacao'].sort_values(ascending=False)
        st.write("Ranking de Correlação com 'situacao':")
        st.write(correlation_ranking)
