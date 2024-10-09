import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from typing import List
from preprocessing import Preprocessing

class Description:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.target_column = None

    def run(self) -> None:
        try:
            st.title("Descrição dos dados")

            # Seletor de colunas para o alvo
            self.target_column = st.selectbox("Selecione a coluna com dados contínuos para serem descritos os dados:", self.data.columns.tolist())

            if self.target_column:
                # Exibe o tipo de dado de cada coluna
                st.write("### Tipos de Dados")
                st.write(self.data.dtypes)
                st.write("### Estatísticas Descritivas")
                st.write(self.data.describe())

                # Calcula e mostra a importância das features
                feature_importance = self.__calculate_feature_importance()
                if feature_importance is not None:
                    st.write("### Importância das Features (Colunas)")
                    st.write(feature_importance)
        except Exception as e:
            st.error(f"Ocorreu um erro ao tentar exibir as informações: {e}")

    def _select_columns(self) -> List[str]:
        try:
            columns = st.multiselect(
                "Selecione as colunas:",
                self.data.columns.tolist()
            )
            return columns
        except Exception as e:
            st.error(f"Erro ao selecionar colunas: {e}")
            return []

    def __calculate_feature_importance(self):
        try:
            if self.target_column is None:
                st.error("Coluna alvo não foi selecionada!")
                return None

            X = self.data.drop(columns=[self.target_column])
            y = self.data[self.target_column]
            numerical_df = X.select_dtypes(exclude=['object'])

            preprocessing = Preprocessing(self.data)

            X_processed = preprocessing.preprocess_categorical_data(X, numerical_df)

            # Codifica a coluna alvo se for categórica
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
                st.write(f"Coluna alvo '{self.target_column}' codificada:", pd.DataFrame({self.target_column: y}))

            # Treina um modelo de RandomForest para calcular a importância das features
            model = RandomForestClassifier()
            model.fit(X_processed, y)
            importance = model.feature_importances_

            # Criar um DataFrame para mostrar a importância das features
            feature_importance_df = pd.DataFrame({
                'Feature': X_processed.columns,
                'Importance': importance
            }).sort_values(by='Importance', ascending=False).reset_index(drop=True)

            return feature_importance_df

        except ValueError as ve:
            st.error(f"Erro de valor: {ve}")
        except KeyError as ke:
            st.error(f"Erro de chave: {ke}")
        except Exception as e:
            st.error(f"Ocorreu um erro ao calcular a importância das features: {e}")
            return None
