#sklearn methods
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import OneHotEncoder
#streamlit
import streamlit as st
#pandas
import pandas as pd

from typing import List


class Preprocessing():
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.scaler = None
        
    def _clean_data(self, data: pd.DataFrame, columns:List[str]) -> pd.DataFrame:
        for column in columns:
            data[column] = data[column].dropna()
        return data
    
    def _select_columns(self) -> List[str]:
        columns = st.sidebar.multiselect(
            "Selecione as colunas:",
            self.data.columns.tolist()
        )
        return columns
        
    def _select_preprocessing_method(self) -> None:
        preprocessing_method = st.sidebar.selectbox(
            'Selecione um método de normalização:',
            ('MinMaxScaler', 'StandardScaler', 'RobustScaler', 'Normalizer', 'MaxAbsScaler')
        ) 
        self.__apply_preprocessing(preprocessing_method)

    def __apply_preprocessing(self,preprocessing_method: str) -> None:
        scaler_map = {
            'MinMaxScaler': MinMaxScaler(),
            'StandardScaler': StandardScaler(),
            'RobustScaler': RobustScaler(),
            'Normalizer': Normalizer(),
            'MaxAbsScaler': MaxAbsScaler()
        }
        
        self.scaler = scaler_map.get(preprocessing_method)
        
        cleaned_data = self._clean_data(self.data, self.data.columns.tolist())
        selected_columns = self._select_columns()
        
        numerical_df = cleaned_data[selected_columns]

        categorical_columns = cleaned_data.select_dtypes(include=['object']).columns
        if len(categorical_columns) > 0:
            one_hot_encoder = OneHotEncoder()
            encoded_categorical_data = one_hot_encoder.fit_transform(cleaned_data[categorical_columns])
            st.write("Dados categóricos após One-Hot Encoding:")
            st.write(encoded_categorical_data.toarray())
            
            if len(numerical_df.columns) > 0:
                normalized_data = self.scaler.fit_transform(pd.concat([numerical_df, pd.DataFrame(encoded_categorical_data.toarray())], axis=1))
                st.write("Dados após normalização:")
                st.write(normalized_data)
        else:
            if len(numerical_df.columns) > 0:
                normalized_data = self.scaler.fit_transform(numerical_df)
                st.sidebar.write(f"Método selecionado: {preprocessing_method}")
                st.write("Dados após normalização:")
                st.write(normalized_data)
