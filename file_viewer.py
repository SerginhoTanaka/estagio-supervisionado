import pandas as pd
import streamlit as st

class FileViewer:
    def __init__(self):
        self.page_size = 100000  # Tamanho padrão para paginação



    def visualizar_arquivos(self):
        """Método para visualizar o arquivo CSV ou XLSX."""
        st.header("📊 Visualizar Arquivo")
        st.subheader('Upload de Arquivo')
        st.text("Extensões permitidas: CSV, XLSX")
        
        uploaded_file = st.file_uploader("Escolha um arquivo", type=["csv", "xlsx"])
        if uploaded_file:
            df = self.process_file(uploaded_file)
            if df is not None:
                self.display_file_data(df)

    def process_file(self, uploaded_file):
        """Processa o arquivo CSV ou XLSX e retorna um DataFrame."""
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            else:
                st.error("Formato de arquivo não suportado!")
                return None

            if df.empty:
                st.warning("O arquivo está vazio ou não contém dados válidos.")
                return None

            return df
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")
            return None

    def display_file_data(self, df):
        """Exibe os dados do DataFrame e permite selecionar colunas e visualizar informações detalhadas."""
        st.write("Colunas do arquivo:")
        selected_columns = st.multiselect(
            'Selecione as colunas para exibir',
            options=df.columns.tolist(),
            default=df.columns.tolist()
        )

        if selected_columns:
            df_filtered = df[selected_columns]
            st.write(df_filtered)

            st.write("Selecione uma coluna para visualizar mais informações:")
            coluna_selecionada = st.selectbox(
                "Escolha uma coluna",
                options=df.columns.tolist()
            )

            if coluna_selecionada:
                self.display_column_info(df, coluna_selecionada)

    def display_column_info(self, df, coluna_selecionada):
        """Exibe informações detalhadas sobre uma coluna selecionada."""
        valores_unicos = df[coluna_selecionada].unique()
        st.write(f"Valores únicos de '{coluna_selecionada}':")
        st.write(valores_unicos)

        quantidade_valores_unicos = df[coluna_selecionada].nunique()
        quantidade_linhas = df.shape[0]

        qtd_unicos, qtd_linhas = st.columns(2)
        qtd_unicos.metric(label=f"Quantidade de valores únicos em {coluna_selecionada}", value=quantidade_valores_unicos)
        qtd_linhas.metric(label="Quantidade total de linhas no DataFrame", value=quantidade_linhas)

        valores_unicos_str = ', '.join(map(str, valores_unicos))
        st.text_area("Copie o texto abaixo:", value=valores_unicos_str)

# Executando a aplicação
if __name__ == '__main__':
    viewer = FileViewer()
    viewer.run()
