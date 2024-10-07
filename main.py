import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
from models import TBPrimaryActions
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from report import ReportsDashboard
from file_viewer import FileViewer
engine = create_engine('sqlite:///actions.db')
Session = sessionmaker(bind=engine)
session = Session()
class Dashboard:
    """
    A class for data preprocessing and visualization.
    """

    def __init__(self) -> None:
        self.data: pd.DataFrame = st.session_state.get('data', pd.DataFrame())
        self.processed_data: pd.DataFrame = st.session_state.get('processed_data', None)
        self.dataset_name: str = st.session_state.get('dataset_name', 'Desconhecido')
        from preprocessing import Preprocessing 
        self.preprocessor: Preprocessing = Preprocessing(self.data)
        from description import Description 
        self.description: Description = Description(self.data)
        from aiprocessing import AiProcessing 
        self.aiprocessing: AiProcessing = AiProcessing(self.data,self.processed_data)
        if 'action_saved' not in st.session_state:
            st.session_state['action_saved'] = False
    def run(self) -> None:
        """
        Run the dashboard.
        """
        st.sidebar.title("CooperGest")
        selected_option: str = st.sidebar.radio(
            "Selecione uma opção",
            ["Pré-processamento", "Análise sem pré-processamento", "Descrição", "Processamento com IA", "Upload de arquivo", "Mesclar Bases", "Relatórios", "Visualizar Arquivos", "Updolad do Drive"]
        )
        from drive_upload import GoogleDriveUploader

        options: dict[str, callable] = {
            "Pré-processamento": self.preprocessor.run,
            "Análise sem pré-processamento": self.__generate_graph,
            "Descrição":self.description.run,
            "Processamento com IA": self.__process_with_ai,
            "Upload de arquivo": self.__upload_file,
            "Mesclar Bases": self.__merge_spreadsheets,
            "Relatórios": ReportsDashboard().run,
            "Visualizar Arquivos": FileViewer().visualizar_arquivos,
            "Updolad do Drive": GoogleDriveUploader(self).display
            
        }
        
        result = options[selected_option]()
        last_option = st.session_state.get('last_option', None)
        if selected_option != last_option:
            if selected_option in ["Pré-processamento", "Processamento com IA"]:
                is_ai = (selected_option == "Processamento com IA")
                self.__save_primary_action(selected_option, is_ai)
            st.session_state['last_option'] = selected_option
            
        if selected_option == "Processamento com IA":
            processed_data, method = result
            if method == 'Regressão':
                self.aiprocessing.regression()
            elif method == 'Classificação':
                self.aiprocessing.classification()

    def __save_primary_action(self, action_name: str, is_ai: bool) -> None:
        """
        Save the primary action to the database.
        """
        try:
            dataset_name = st.session_state['dataset_name']

            new_action = TBPrimaryActions(
                action_name=action_name,
                dataset_name=dataset_name,
                is_ai=is_ai
            )
            session.add(new_action)
            session.commit()
            session.close()
            st.write(f"Ação '{action_name}' salva com sucesso!")
        except Exception as e:
            print(f"Erro ao salvar a ação: {e}")
        
    def __process_with_ai(self):
        """
        Get the data for AI processing.
        """
        if 'processed_data' in st.session_state and st.session_state['processed_data'] is not None:
            return st.session_state['processed_data'], st.session_state.get('method', None)

        processed_data, method = self.aiprocessing.run()

        st.session_state['processed_data'] = processed_data
        st.session_state['method'] = method

        return processed_data, method

    def __upload_file(self) -> None:
        """
        Upload a file.
        """
        file: Optional[st.uploaded_file_manager.UploadedFile] = st.file_uploader("Upload arquivo CSV", type="csv")
        if file is not None:
            try:
                data = pd.read_csv(file)
                self.save_df(data,file.name)
            except Exception as e:
                st.error(f"Erro ao carregar o arquivo: {e}")

    def save_df(self, data: pd.DataFrame,df_name) -> None:
        if isinstance(data, pd.DataFrame):
            print('data is pd.DataFrame')
            st.session_state.data = data 
            st.session_state['dataset_name'] = df_name
            st.write("Arquivo CSV carregado com sucesso!")


    @staticmethod
    @st.cache_data
    def convert_df(_df: pd.DataFrame) -> bytes:
        return _df.to_csv(index=False).encode('utf-8')
    
    def download_spreadsheet(self, df: pd.DataFrame, filename: str) -> None:
        """
        Download a spreadsheet.
        """
        try:
            
            csv: bytes = self.convert_df(df)
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
        Merge two spreadsheets based on user-selected keys, remove values above 50000
        from selected columns, and remove duplicated columns.
        """
        st.subheader("Carregar Planilha 1:")
        file1: Optional[st.uploaded_file_manager.UploadedFile] = st.file_uploader("Upload planilha 1", type="csv", key="file1")

        st.subheader("Carregar Planilha 2:")
        file2: Optional[st.uploaded_file_manager.UploadedFile] = st.file_uploader("Upload planilha 2", type="csv", key="file2")

        if file1 is not None and file2 is not None:
            try:
                data1: pd.DataFrame = pd.read_csv(file1)
                data2: pd.DataFrame = pd.read_csv(file2)

                # Selecionar as chaves de junção para cada base
                st.subheader("Selecionar chave de junção para cada planilha:")
                key_column1: str = st.selectbox("Selecionar chave de junção para Planilha 1", data1.columns)
                key_column2: str = st.selectbox("Selecionar chave de junção para Planilha 2", data2.columns)

                if st.button("Mesclar Bases"):
                    # Remover valores maiores que 50000 das chaves selecionadas
                    st.write(f"Removendo valores maiores que 50000 da coluna '{key_column1}' na Planilha 1...")
                    data1 = data1[data1[key_column1] <= 50000]
                    st.write(f"Planilha 1 filtrada: {len(data1)} registros restantes.")

                    st.write(f"Removendo valores maiores que 50000 da coluna '{key_column2}' na Planilha 2...")
                    data2 = data2[data2[key_column2] <= 50000]
                    st.write(f"Planilha 2 filtrada: {len(data2)} registros restantes.")

                    # Realizar a junção dos dados
                    st.write("Mesclando as duas bases...")
                    merged_data: pd.DataFrame = pd.merge(
                        data1,
                        data2,
                        left_on=key_column1,
                        right_on=key_column2,
                        how='inner'  # Realizando junção interna para remover linhas sem correspondência
                    )
                    st.write(f"Bases mescladas com sucesso! {len(merged_data)} registros resultantes.")
                    st.write(merged_data.head())

                    # Remover colunas duplicadas
                    st.write("Removendo colunas duplicadas...")
                    duplicated_columns = merged_data.columns[merged_data.T.duplicated()]
                    merged_data = merged_data.loc[:, ~merged_data.T.duplicated()]
                    st.write(f"\nColunas duplicadas removidas: {list(duplicated_columns)}")
                    st.write(f"Dimensões finais após remover colunas duplicadas: {merged_data.shape}")

                    # Exibir os dados resultantes
                    st.write("Dados finais após junção e remoção de duplicatas:")
                    st.write(merged_data.head())

                    # Baixar o resultado final
                    if not merged_data.empty:
                        self.download_spreadsheet(merged_data, "classif_prod_merged.csv")
                    else:
                        st.write("O DataFrame está vazio. Nenhum arquivo CSV será gerado.")
            
            except Exception as e:
                st.error(f"Erro ao processar as planilhas: {e}")


    def __generate_graph(self) -> None:
        """
        Generate a graph without preprocessing.
        """
        st.sidebar.write("Análise sem pré-processamento")
        columns: List[str] = self.description._select_columns()

        self.__plot_graph(self.data, columns)

    def __plot_graph(self, not_cleaned_data: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
        """
        Plot a graph.
        """
        numeric_columns: pd.Index = not_cleaned_data.select_dtypes(include=np.number).columns
        selected_numeric_columns: List[str] = [col for col in numeric_columns if col in columns]
        non_numeric_columns: List[str] = [col for col in columns if col not in selected_numeric_columns]
        if selected_numeric_columns:
            for col in selected_numeric_columns:
                st.write(not_cleaned_data[col].describe())
        if non_numeric_columns:
            for col in non_numeric_columns:
                fig_width: float = 20 / len(columns)
                fig, ax = plt.subplots(figsize=(fig_width, 6))

                col_data: pd.Series = not_cleaned_data[col]
                counts: pd.Series = col_data.value_counts()

                counts.plot(kind='bar', ax=ax)

                ax.set_xlabel(col)
                ax.set_ylabel('Quantidade')
                ax.set_title(f'Quantidade de {col}')

                for i, v in enumerate(counts.values):
                    ax.text(i, v + 0.1, str(v), ha='center')

                st.pyplot(fig)

if __name__ == '__main__':
    dashboard = Dashboard()
    #uploader = GoogleDriveUploader(dashboard)  
    dashboard.run()
