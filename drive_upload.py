import streamlit as st
import pandas as pd
import gdown
from main import Dashboard
class GoogleDriveUploader:
    def __init__(self, dashboard: Dashboard):
        self.dashboard = dashboard  # Armazenar a instância de Dashboard
        self.drive_link = ""
        self.name = ""
    def display(self):
        st.header("📂 Ler Arquivo do Google Drive")
        st.subheader('Insira o link do arquivo do Google Drive')
        self.drive_link = st.text_input("Link do Google Drive", placeholder="https://drive.google.com/file/d/FILE_ID/view?usp=sharing")
        self.name = st.text_input("Informe o nome do arquivo")

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
                if self.name != "" and df is not None:  
                    self.dashboard.save_df(df, self.name) 
                
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
