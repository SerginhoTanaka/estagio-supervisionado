import streamlit as st
import pandas as pd
import gdown

class GoogleDriveUploader:
    def __init__(self):
        self.drive_link = ""

    def display(self):
        st.header("📂 Ler Arquivo do Google Drive")
        st.subheader('Insira o link do arquivo do Google Drive')
        self.drive_link = st.text_input("Link do Google Drive", placeholder="https://drive.google.com/file/d/FILE_ID/view?usp=sharing")

        if self.drive_link:
            self.__process_drive_file()

    def __process_drive_file(self):
        try:
            file_id = self.__extract_file_id(self.drive_link)
            if file_id:
                file_url = f'https://drive.google.com/uc?id={file_id}'
                output_file = 'file_from_drive.csv'
                gdown.download(file_url, output_file, quiet=False)

                df = pd.read_csv(output_file)
                self.__display_data(df)
            else:
                st.error("Link inválido. Por favor, insira um link correto do Google Drive.")
        except Exception as e:
            st.error(f"Erro ao processar o arquivo do Google Drive: {e}")

    @staticmethod
    def __extract_file_id(link):
        if 'id=' in link:
            return link.split('id=')[-1]
        elif '/d/' in link:
            return link.split('/d/')[-1].split('/')[0]
        return None

    def __display_data(self, df):
        st.write("Dados do arquivo:")
        selected_columns = st.multiselect(
            'Selecione as colunas para exibir',
            options=df.columns.tolist(),
            default=df.columns.tolist()
        )

        if selected_columns:
            df_filtered = df[selected_columns]
            self.__paginate_data(df_filtered)

            st.write("Selecione uma coluna para visualizar mais informações:")
            selected_column = st.selectbox("Escolha uma coluna", options=df.columns.tolist())

            if selected_column:
                unique_values = df[selected_column].unique()
                st.write(f"Valores únicos de '{selected_column}':")
                st.write(unique_values)

                total_unique_values = df[selected_column].nunique()
                total_rows = df.shape[0]

                qtd_unicos, qtd_linhas = st.columns(2)
                qtd_unicos.metric(label=f"Quantidade de valores únicos em {selected_column}", value=total_unique_values)
                qtd_linhas.metric(label="Quantidade total de linhas no DataFrame", value=total_rows)

                unique_values_str = ', '.join(map(str, unique_values))
                st.text_area("Copie o texto abaixo:", value=unique_values_str)

    @staticmethod
    def __paginate_data(df, page_size=100000):
        num_pages = len(df) // page_size + 1
        page = st.slider('Selecione a página', 1, num_pages, 1)
        start = (page - 1) * page_size
        end = start + page_size
        st.write(df[start:end])