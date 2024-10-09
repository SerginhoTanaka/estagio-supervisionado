from typing import List, Optional

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import streamlit as st
from sklearn.preprocessing import (
    MinMaxScaler, StandardScaler, RobustScaler, Normalizer, MaxAbsScaler, OneHotEncoder
)
from scipy import stats
from main import Dashboard


class Preprocessing:
    def __init__(self, data: pd.DataFrame):
        self.data: pd.DataFrame = data
        self.scaler: Optional[str] = None
        self.cleaning_methods: Optional[List[str]] = None

    def run(self) -> Optional[pd.DataFrame]:
        """
        Método principal para executar o fluxo de pré-processamento.

        Returns:
            Optional[pd.DataFrame]: DataFrame pré-processado, ou None se nenhuma ação for realizada.
        """

        st.title("Pré processamento")
        st.write("Realize as configurações de Pré processamento")

        self.select_preprocessing_method()
        self.select_cleaning_method()
        if st.button('Aplicar'):
            new_data = self.__apply_preprocessing()
            #show = st.checkbox('Mostrar dados após pré-processamento', value=True)
            if new_data is not None: #and show:
                self.__show(new_data)
            return new_data
        return None

    def select_cleaning_method(self) -> None:
        """
        Exibe uma caixa de seleção no sidebar para selecionar métodos de limpeza.
        """
        cleaning_method = st.multiselect(
            'Selecione um método de limpeza:',
            ('nenhum','Remover valores nulos', 'Imputar valores médios', 'Remover linhas duplicadas', 'Remover ruídos'),
            default='nenhum'
        )
        self.cleaning_methods = cleaning_method

    def select_preprocessing_method(self) -> None:
        """
        Exibe uma caixa de seleção no sidebar para selecionar método de normalização.
        """
        preprocessing_method = st.selectbox(
            'Selecione um método de normalização:',
            ('nenhum', 'MinMaxScaler', 'StandardScaler', 'RobustScaler', 'MaxAbsScaler'),
            index=0
        )

        if preprocessing_method == 'MinMaxScaler':
            with st.expander("Informações sobre o MinMaxScaler"):
                st.write("Você selecionou MinMaxScaler. Este método escala os dados para um intervalo específico, geralmente entre 0 e 1.")
                st.write("Dica: O MinMaxScaler é útil quando você deseja preservar a forma geral da distribuição dos dados, mas normalizá-los para um intervalo específico.")

        elif preprocessing_method == 'StandardScaler':
            with st.expander("Informações sobre o StandardScaler"):
                st.write("Você selecionou StandardScaler. Este método padroniza os dados, subtraindo a média e dividindo pelo desvio padrão.")
                st.write("Dica: O StandardScaler é útil quando você deseja transformar seus dados para que eles tenham média zero e desvio padrão igual a 1.")

        elif preprocessing_method == 'RobustScaler':
            with st.expander("Informações sobre o RobustScaler"):
                st.write("Você selecionou RobustScaler. Este método escala os dados usando estatísticas robustas para lidar com outliers.")
                st.write("Dica: O RobustScaler é útil quando seus dados contêm outliers e você deseja escalá-los usando estatísticas resistentes a outliers.")

        elif preprocessing_method == 'MaxAbsScaler':
            with st.expander("Informações sobre o MaxAbsScaler"):
                st.write("Você selecionou MaxAbsScaler. Este método escala os dados para o intervalo [-1, 1] dividindo pelo valor máximo absoluto.")
                st.write("Dica: O MaxAbsScaler é útil quando você deseja preservar a relação de ordem dos seus dados, mas normalizá-los para o intervalo [-1, 1].")
        else:
            st.write("Nenhum método de normalização selecionado.")
        self.scaler = preprocessing_method

    def __apply_preprocessing(self) -> pd.DataFrame:
        # Dicionário de scalers
        scaler_map = {
            'MinMaxScaler': MinMaxScaler(),
            'StandardScaler': StandardScaler(),
            'RobustScaler': RobustScaler(),
            'Normalizer': Normalizer(),
            'MaxAbsScaler': MaxAbsScaler()
        }

        scaler = scaler_map.get(self.scaler)
        df = self.data.copy()

        # Identificar colunas de data e converter para dias desde 1970-01-01
        date_cols = df.select_dtypes(include=['datetime64', 'datetime']).columns
        for col in date_cols:
            # Converter para o número de dias desde 01-01-1970
            df[col] = (df[col] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1D')  # Diferença em dias

        # Aplicar limpeza
        if 'nenhum' not in self.cleaning_methods:
            for method in self.cleaning_methods:
                if method ==   'Remover valores nulos':
                    df = self.__clean_null(df)
                elif method == 'Remover linhas duplicadas':
                    df = self.__clean_duplicates(df)
                elif method == 'Remover ruídos':
                    df = self.__clean_noise(df)
                elif method == 'Imputar valores médios':
                    df = self.__impute_mean(df)  # Chama a função de imputação de médias

        # Separar as colunas numéricas
        numerical_df = df.select_dtypes(exclude=['object'])

        # Aplicar normalização às colunas numéricas
        if self.scaler != 'nenhum' and scaler:
            if not numerical_df.empty:
                for col in numerical_df.columns:
                    scaled_data = scaler.fit_transform(numerical_df[[col]])
                    numerical_df[col] = scaled_data
                    st.write(f"Coluna {col} normalizado pelo método {self.scaler}:")
                    with st.expander("Informações sobre o MaxAbsScaler"):
                        st.write(numerical_df[[col]])
            else:
                st.write("Não há colunas numéricas para normalização.")
        else:
            st.write("Nenhum método de normalização selecionado.")

        # Passar os dados normalizados para o processamento categórico
        st.write("Dados normalizados após a execução")
        final_df = self.preprocess_categorical_data(df, numerical_df)

        st.write("Dados após pré-processamento (apenas 3 amostras):")
        st.write(final_df.head(3))  # Exibe apenas 3 linhas

        return final_df

    def preprocess_categorical_data(self, df: pd.DataFrame, numerical_df: pd.DataFrame) -> pd.DataFrame:
        """
        Método para pré-processar dados categóricos e concatenar com os dados numéricos.

        Args:
            df (pd.DataFrame): DataFrame original contendo colunas categóricas.
            numerical_df (pd.DataFrame): DataFrame com colunas numéricas pré-processadas.

        Returns:
            pd.DataFrame: DataFrame após pré-processamento e concatenação.
        """
        categorical_df = df.select_dtypes(include=['object'])
        
        # Se houver colunas categóricas, aplicamos Label Encoding
        if not categorical_df.empty:
            le = LabelEncoder()
            for col in categorical_df.columns:
                df[col] = le.fit_transform(df[col])

            # Concatenar os dados numéricos e categóricos
            final_df = pd.concat([numerical_df.reset_index(drop=True), df[categorical_df.columns]], axis=1)
        else:
            final_df = numerical_df

        st.write(final_df.head(3))  # Mostrar os primeiros 3 registros para visualização
        return final_df

   

    def __show(self, new_data: pd.DataFrame) -> None:
        """
        Exibe o DataFrame antes e depois da aplicação dos métodos de limpeza e normalização.

        Args:
            new_data (pd.DataFrame): DataFrame após aplicação dos métodos.
        """
        st.write("Tabela original:")
        st.write(self.data.head(3))  # Exibe apenas 3 amostras
        st.write("Tabela após pré-processamento (apenas 3 amostras):")
        st.write(new_data.head(3))  # Exibe apenas 3 amostras
        print(new_data)
        file_name = st.text_input("Digite o nome do arquivo CSV:")

# Garante que o nome do arquivo termine com '.csv'
        if not file_name.endswith('.csv'):
            file_name += '.csv'        
        if st.button('Usar dataset pre-processado'):
            Dashboard().save_df(new_data, file_name)
        Dashboard().download_spreadsheet(new_data, "preprocessed_data.csv")

    def __clean_null(self, df: pd.DataFrame) -> pd.DataFrame: 
        """
        Remove linhas com valores nulos do DataFrame.

        Args:
            df (pd.DataFrame): DataFrame a ser limpo.

        Returns:
            pd.DataFrame: DataFrame sem valores nulos.
        """
        st.write('Valores nulos antes da limpeza:', df.isnull().sum())
        return df.dropna()

    def __clean_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove linhas duplicadas do DataFrame.

        Args:
            df (pd.DataFrame): DataFrame a ser limpo.

        Returns:
            pd.DataFrame: DataFrame sem linhas duplicadas.
        """
        st.write('Valores duplicados antes da limpeza:', df.duplicated().sum())
        return df.drop_duplicates()

    def __clean_noise(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove ruídos dos dados utilizando o método do Z-Score.

        Args:
            df (pd.DataFrame): DataFrame a ser limpo.

        Returns:
            pd.DataFrame: DataFrame sem ruídos.
        """
        categorical_values = df.select_dtypes(include=['object'])
        numerical_df = df.drop(columns=categorical_values.columns.tolist())
        bool_columns = numerical_df.select_dtypes(include=["bool"]).columns.tolist()
        numerical_df[bool_columns] = numerical_df[bool_columns].astype(int)

        st.write('Valores ruidosos antes da limpeza:', (np.abs(stats.zscore(numerical_df)) > 3).sum())
        cleaned_df = numerical_df[(np.abs(stats.zscore(numerical_df)) < 3).all(axis=1)]

        cleaned_df = pd.concat([cleaned_df, categorical_values], axis=1)
        return cleaned_df
    def __impute_mean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Substitui valores NaN pela média das colunas numéricas.

        Args:
            df (pd.DataFrame): DataFrame a ser imputado.

        Returns:
            pd.DataFrame: DataFrame com valores NaN substituídos pelas médias.
        """
        st.write('Valores nulos antes da imputação de média:', df.isnull().sum())
        
        # Substitui valores NaN pela média nas colunas numéricas
        numerical_columns = df.select_dtypes(include=[np.number]).columns
        for col in numerical_columns:
            mean_value = df[col].mean()
            df[col].fillna(mean_value, inplace=True)  # Substitui NaN pela média da coluna
        
        st.write('Valores nulos após a imputação de média:', df.isnull().sum())
        return df
