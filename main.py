#streamlit
import streamlit as st
#pandas
import pandas as pd
#matplotlib
import matplotlib.pyplot as plt
#python
from preprocessing import Preprocessing
from description import Description

class Dashboard():
    def __init__(self):
        self.data = pd.read_csv('novo_arquivo.csv', low_memory=False)
        self.preprocessor = Preprocessing(self.data)

    def _run(self)-> None:
        st.sidebar.title("Pré-processamento de dados")

        if st.sidebar.button("Pré-processamento"):
            if st.sidebar.button("Fechar"):
                st.sidebar.empty()
            else:
                self.preprocessor.select_preprocessing_method()

        if st.sidebar.button("Descrição"):
            description = Description(self.data)
            description.run()
        
        if st.sidebar.button("Gráfico sem pré-processamento"):
            data_situation = self.data['situacao']
            self.__normalize_situation(data_situation)
            
    def __normalize_situation(self, data_situation: pd.DataFrame) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))

        data = data_situation.dropna()
        counts = data.value_counts()

        counts.plot(kind='bar', ax=ax)

        ax.set_xlabel('Situação')
        ax.set_ylabel('Quantidade')
        ax.set_title('Quantidade de Ativos e Inativos')

        for i, value in enumerate(counts):
            ax.text(i, value + 0.1, str(value), ha='center')

        st.pyplot(fig)

if __name__ == '__main__':
    dashboard = Dashboard()
    dashboard._run()
