# sklearn methods
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder

# streamlit
import streamlit as st

# pandas
import pandas as pd

from typing import List
from scipy.signal import savgol_filter
from scipy import stats
import numpy as np  


class Preprocessing:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.scaler = None
        self.cleaning_methods = None

    def run(self):
        """
        Método principal para executar o fluxo de pré-processamento.
        """
        self.select_preprocessing_method()
        self.select_cleaning_method()
        if st.sidebar.button('Aplicar'):
            new_data = self.__apply_preprocessing()
            show = st.sidebar.checkbox('Mostrar dados após pré-processamento', value=True)
            if new_data is not None and show:
                self.__show(new_data)
            return new_data

    def select_cleaning_method(self) -> None:
        """
        Exibe uma caixa de seleção no sidebar para selecionar métodos de limpeza.
        """
        cleaning_method = st.sidebar.multiselect(
            'Selecione um método de limpeza:',
            ('nenhum', 'Remover linhas com valores nulos', 'Remover linhas duplicadas', 'Remover ruídos'),
            default='nenhum'
        )
        self.cleaning_methods = cleaning_method

    def select_preprocessing_method(self) -> None:
        """
        Exibe uma caixa de seleção no sidebar para selecionar método de normalização.
        """
        preprocessing_method = st.sidebar.selectbox(
            'Selecione um método de normalização:',
            ('nenhum', 'MinMaxScaler', 'StandardScaler', 'RobustScaler', 'Normalizer', 'MaxAbsScaler'),
            index=0
        ) 
        self.scaler = preprocessing_method

    def __apply_preprocessing(self) -> pd.DataFrame:
        """
        Aplica os métodos de limpeza e normalização selecionados aos dados.

        Returns:
            pd.DataFrame: DataFrame após aplicação dos métodos.
        """
        scaler_map = {
            'MinMaxScaler': MinMaxScaler(),
            'StandardScaler': StandardScaler(),
            'RobustScaler': RobustScaler(),
            'Normalizer': Normalizer(),
            'MaxAbsScaler': MaxAbsScaler()
        }
        
        scaler = scaler_map.get(self.scaler)
        df = self.data.copy()
        
        # Aplicar limpeza
        if 'nenhum' not in self.cleaning_methods:
            for method in self.cleaning_methods:
                if method == 'Remover linhas com valores nulos':
                    df = self.__clean_null(df)
                elif method == 'Remover linhas duplicadas':
                    df = self.__clean_duplicates(df)
                elif method == 'Remover ruídos':
                    df = self.__clean_noise(df)

        # Aplicar normalização
        if self.scaler != 'nenhum' and scaler:
            columns_to_normalize = df.columns[df.dtypes != object]  # Seleciona apenas colunas numéricas

            if len(columns_to_normalize) > 0:
                for col in columns_to_normalize:
                    scaled_data = scaler.fit_transform(df[[col]])
                    df[col] = scaled_data
                    st.write(f"{col} normalizado pelo método {self.scaler}:")
                    st.write(df[[col]])
            else:
                st.write("Não há colunas numéricas para normalização.")
        else:
            st.write("Nenhum método de normalização selecionado.")

        return df


    def __show(self, new_data: pd.DataFrame) -> None:
        """
        Exibe o DataFrame antes e depois da aplicação dos métodos de limpeza e normalização.

        Args:
            new_data (pd.DataFrame): DataFrame após aplicação dos métodos.
        """
        st.write("Tabela original:")
        st.write(self.data.count())
        st.write("Tabela após pré-processamento:")
        st.write(new_data.count())

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