#streamlit
import streamlit as st
#pandas
import pandas as pd
from preprocessing import Preprocessing
#matplotlib
import matplotlib.pyplot as plt


class Dashboard:
    def __init__(self):
        self.data = pd.read_csv('novo_arquivo.csv')
        self.preprocessor = Preprocessing(self.data)

    def __run(self):
        st.sidebar.title("Pré-processamento de dados")

        if st.sidebar.button("Pré-processamento"):
            if st.sidebar.button("Fechar"):
                st.sidebar.empty()
            else:
                self.preprocessor.select_preprocessing_method()
                self.preprocessor.apply_preprocessing()

        if st.sidebar.button("Descrição"):
            st.empty() 
            st.write("""
            # Description Page
            """)
        
        if st.sidebar.button("Gráfico sem pré-processamento"):
            data_situation = self.data['situacao']
            self.__normalize_situation(data_situation)
            
    def __normalize_situation(self,data_situation):
        data = data_situation.dropna()

        active = data == 'Ativo'
        inactive = data == 'Inativo'
        fig, ax = plt.subplots()
        ax.hist([active['idade'], inactive['idade']], bins=20, label=['Ativos', 'Inativos'])
        ax.set_xlabel('Idade')
        ax.set_ylabel('Número de Pessoas')
        ax.legend()
        st.pyplot(fig)
if __name__ == '__main__':
    dashboard = Dashboard()
    dashboard.__run()