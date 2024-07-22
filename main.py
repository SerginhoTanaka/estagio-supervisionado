# streamlit
import streamlit as st
# Pandas
import pandas as pd
# Matplotlib
import matplotlib.pyplot as plt
# Custom modules
from preprocessing import Preprocessing
from description import Description
from aiprocessing import AiProcessing
# Typing
from typing import List
from typing import Optional

import numpy as np

import base64


class Dashboard:
    """
    A class for data preprocessing and visualization.
    """

    def __init__(self):
        self.data = pd.read_csv('database_300.csv')
        self.preprocessor = Preprocessing(self.data)
        self.description = Description(self.data)
        self.aiprocessing = AiProcessing(self.data)

    def run(self) -> None:
        """
        Run the dashboard.
        """
        st.sidebar.title("CooperGest")
        selected_option = st.sidebar.radio(
            "Selecione uma opção",
            ["Pré-processamento", "Análise sem pré-processamento", "Descrição", "Processamento com IA", "Chat", "Upload de arquivo", "Mesclar Planilhas"]
        )
        options = {
            "Pré-processamento": self.preprocessor.run,
            "Análise sem pré-processamento": self.__generate_graph,
            "Descrição": lambda: st.write('Em desenvolvimento...'),
            "Processamento com IA": self.aiprocessing.run,
            "Chat": lambda: st.write('Em desenvolvimento...'),
            "Upload de arquivo": self.__upload_file,
            "Mesclar Planilhas": self.__merge_spreadsheets
        }
        options[selected_option]()

    def __upload_file(self) -> None:
        """
        Upload a file.
        """
        file = st.file_uploader("Upload file", type={"csv", "xlsx"})
        if file is not None and file.name.endswith('.csv'):
            self.data = pd.read_csv(file)
        elif file is not None and file.name.endswith('.xlsx'):
            self.data = pd.read_excel(file)
        elif self.data is not None:
            # st.write("Arquivo carregado com sucesso!")
            st.write(self.data.head()) 
    def __merge_spreadsheets(self) -> None:
        """
        Merge two spreadsheets based on user-selected keys.
        """
        st.subheader("Carregar Planilha 1:")
        file1 = st.file_uploader("Upload planilha 1", type={"csv", "xlsx"})

        st.subheader("Carregar Planilha 2:")
        file2 = st.file_uploader("Upload planilha 2", type={"csv", "xlsx"})

        if file1 is not None and file2 is not None:
            if file1.name.endswith('.csv'):
               data1 = pd.read_csv(file1)
            elif file1.name.endswith('.xlsx'):
                data1 = pd.read_excel(file1)

            if file2.name.endswith('.csv'):
                data2 = pd.read_csv(file2)
            elif file2.name.endswith('.xlsx'):
                data2 = pd.read_excel(file2)

            if data1 is not None and data2 is not None:
                st.subheader("Selecionar chave de junção para cada planilha:")
                key_column1 = st.selectbox("Selecionar chave de junção para Planilha 1", data1.columns)
                key_column2 = st.selectbox("Selecionar chave de junção para Planilha 2", data2.columns)

                if st.button("Mesclar Planilhas"):
                    merged_data = pd.merge(data1, data2, how='inner', left_on=key_column1, right_on=key_column2)
                    st.write("Planilhas mescladas com sucesso!")
                    st.write(merged_data.head())
                    csv = merged_data.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="planilhas_mescladas.csv">Download Planilhas Mescladas</a>'
                    st.markdown(href, unsafe_allow_html=True)
    def __generate_graph(self) -> None:
        """
        Generate a graph without preprocessing.
        """
        st.sidebar.write("Análise sem pré-processamento")
        columns = self.description._select_columns()

        self.__plot_graph(self.data, columns)

    def __plot_graph(self, not_cleaned_data: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
        """
        Plot a graph.
        """
        
        if columns is None:
            columns = ['situacao']
        numeric_columns = not_cleaned_data.select_dtypes(include=np.number).columns
        selected_numeric_columns = [col for col in numeric_columns if col in columns]
        non_numeric_columns = [col for col in columns if col not in selected_numeric_columns]
        if selected_numeric_columns:
            for col in selected_numeric_columns:
                st.write(not_cleaned_data[col].describe())
        if non_numeric_columns:
            for col in non_numeric_columns:
                fig_width = 20 / len(columns)
                fig, ax = plt.subplots(figsize=(fig_width, 6))

                col_data = not_cleaned_data[col]
                counts = col_data.value_counts()

                counts.plot(kind='bar', ax=ax)

                ax.set_xlabel(col)
                ax.set_ylabel('Quantidade')
                ax.set_title(f'Quantidade de {col}')

                for i, v in enumerate(counts.values):
                    ax.text(i, v + 0.1, str(v), ha='center')

                st.pyplot(fig)

if __name__ == '__main__':
    dashboard = Dashboard()
    dashboard.run()
