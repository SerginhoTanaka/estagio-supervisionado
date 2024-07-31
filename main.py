import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
# Removed global imports for potential circular dependency issues

class Dashboard:
    """
    A class for data preprocessing and visualization.
    """

    def __init__(self):
        # Initialize with an empty DataFrame if no data is loaded
        self.data = st.session_state.get('data', pd.DataFrame())
        from preprocessing import Preprocessing  # Local import
        self.preprocessor = Preprocessing(self.data)
        from description import Description  # Local import
        self.description = Description(self.data)
        from aiprocessing import AiProcessing  # Local import
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
        file = st.file_uploader("Upload arquivo CSV", type="csv")
        if file is not None:
            try:
                self.data = pd.read_csv(file)
                st.session_state.data = self.data  # Save data to session state
                st.write("Arquivo CSV carregado com sucesso!")
                st.write(self.data.head())
            except Exception as e:
                st.error(f"Erro ao carregar o arquivo: {e}")

    @staticmethod
    @st.cache_data
    def convert_df(_df: pd.DataFrame) -> bytes:
        return _df.to_csv(index=False).encode('utf-8')
    
    def download_spreadsheet(self, df: pd.DataFrame, filename: str) -> None:
        """
        Download a spreadsheet.
        """
        try:
            csv = self.convert_df(df)
            st.download_button(
                label="Baixar Planilha",
                data=csv,
                file_name=filename,
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Erro ao baixar a planilha: {e}")

    def __merge_spreadsheets(self) -> None:
        """
        Merge two spreadsheets based on user-selected keys.
        """
        st.subheader("Carregar Planilha 1 (CSV somente):")
        file1 = st.file_uploader("Upload planilha 1", type="csv", key="file1")

        st.subheader("Carregar Planilha 2 (CSV somente):")
        file2 = st.file_uploader("Upload planilha 2", type="csv", key="file2")

        if file1 is not None and file2 is not None:
            try:
                data1 = pd.read_csv(file1)
                data2 = pd.read_csv(file2)

                st.subheader("Selecionar chave de junção para cada planilha:")
                key_column1 = st.selectbox("Selecionar chave de junção para Planilha 1", data1.columns)
                key_column2 = st.selectbox("Selecionar chave de junção para Planilha 2", data2.columns)

                if st.button("Mesclar Planilhas"):
                    merged_data = pd.merge(data1, data2, how='inner', left_on=key_column1, right_on=key_column2)
                    st.write("Planilhas mescladas com sucesso!")
                    st.write(merged_data.head())
                    st.write(f"len(merged_data): {len(merged_data)}")
                    
                    if not merged_data.empty:
                        self.download_spreadsheet(merged_data, "merged_data.csv")
                    else:
                        st.write("O DataFrame está vazio. Nenhum arquivo CSV será gerado.")
            except Exception as e:
                st.error(f"Erro ao processar as planilhas: {e}")

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
