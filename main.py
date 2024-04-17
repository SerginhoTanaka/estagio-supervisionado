#streamlit
import streamlit as st
#pandas
import pandas as pd
#matplotlib
import matplotlib.pyplot as plt
#python
from preprocessing import Preprocessing
from description import Description
from typing import List, Optional

class Dashboard():
    def __init__(self):
        self.data = pd.read_csv('novo_arquivo.csv', low_memory=False)
        self.preprocessor = Preprocessing(self.data)
        self.description = Description(self.data)


    def _run(self)-> None:
        st.sidebar.title("Pré-processamento de dados")

        if st.sidebar.button("Pré-processamento"):
            if st.sidebar.button("Fechar"):
                st.sidebar.empty()
            else:
                self.preprocessor._select_preprocessing_method()

        if st.sidebar.button("Descrição"):
            self.description.run()
        
        if st.sidebar.button("Gráfico sem pré-processamento"):
            data_headers = self.data.columns.tolist()
            print(data_headers)
            columns =  st.sidebar.multiselect(
            'Selecione as colunas:',
            data_headers
        )#pedir ajuda aqui também pois ele não deixa selecionar as colunas
            # print(columns)
            st.write("Selected Columns:", columns)
            if columns:
                clened_data = self.preprocessor._clean_data(self.data, columns)
                self.generate_graph(clened_data,columns)
            
    def generate_graph(self,data, columns: Optional[List[str]] = ['situacao']) -> None:
        for col in columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            col_data = data[col]
            counts = col_data.value_counts()

            counts.plot(kind='bar', ax=ax)

            ax.set_xlabel(col)
            ax.set_ylabel('Quantidade')
            ax.set_title(f'Quantidade de {col}')

            for i, value in enumerate(counts):
                ax.text(i, value + 0.1, str(value), ha='center')

            st.pyplot(fig)
    

if __name__ == '__main__':
    dashboard = Dashboard()
    dashboard._run()
