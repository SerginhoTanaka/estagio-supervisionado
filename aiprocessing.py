from typing import List, Optional, Tuple, Union
import pandas as pd
import numpy as np
import streamlit as st
from preprocessing import Preprocessing
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import TBPrimaryActions, TBAiActions  # Importar as classes do banco de dados

# Classificação
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# Regressão
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score

engine = create_engine('sqlite:///actions.db')
Session = sessionmaker(bind=engine)
session = Session()

class AiProcessing:
    def __init__(self, data: pd.DataFrame, processed_data) -> None:
        self.data: pd.DataFrame = data
        self.ai: Optional[str] = None
        self.normalized_data: Optional[pd.DataFrame] = processed_data
        self.target_column: Optional[str] = None

    def run(self) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Método principal para executar o fluxo de pré-processamento e seleção do modelo de IA.
        """
        try:
            method: str = st.selectbox('Selecione um modelo de IA:', ('', 'Regressão', 'Classificação'))
            preprocessing_option: str = st.selectbox('Deseja pré-processar os dados?', ('', 'Sim', 'Não'))
            
            if method and preprocessing_option:
                if preprocessing_option == 'Sim':
                    norm = self.__preprocessing()
                    return norm, method
                elif preprocessing_option == 'Não':
                    numerical_df: pd.DataFrame = self.data.select_dtypes(exclude=['object'])
                    preprocessing = Preprocessing(self.data)
                    final_data: pd.DataFrame = preprocessing.preprocess_categorical_data(self.data, numerical_df)
                    self.data = final_data
                    return final_data, method
            return None, None

        except Exception as e:
            st.error(f"Ocorreu um erro durante a execução: {e}")
            return None, None

    def __preprocessing(self) -> Optional[pd.DataFrame]:
        """
        Método para executar o pré-processamento dos dados.
        :return: DataFrame com os dados pré-processados.
        """
        try:
            preprocessing = Preprocessing(self.data)
            processed_data: pd.DataFrame = preprocessing.run()
            if processed_data is not None:
                st.write('Dados pré-processados!')
            return processed_data
        except Exception as e:
            st.error(f"Erro no pré-processamento: {e}")
            return None

    def regression(self) -> None:
        """
        Método genérico para executar a regressão.
        """
        try:
            data_to_use: pd.DataFrame = self.normalized_data if self.normalized_data is not None else self.data
            self.ai = st.selectbox('Selecione um modelo de regressão:', 
                                   ('', 'Linear Regression', 'SVR', 'Random Forest'))
            self.target_column = st.selectbox('Selecione a coluna alvo para a regressão:', data_to_use.columns)
            
            if self.ai:
                model = self.__get_model()
                self.__train_and_evaluate(model, is_regression=True)
        except Exception as e:
            st.error(f"Erro na regressão: {e}")

    def classification(self) -> None:
        """
        Método genérico para executar a classificação.
        """
        try:
            st.subheader('Modelos de Classificação Disponíveis:')
            data_to_use: pd.DataFrame = self.normalized_data if self.normalized_data is not None else self.data

            self.target_column = st.selectbox('Selecione a coluna alvo para a classificação:', data_to_use.columns)
            self.ai = st.selectbox('Selecione um modelo de classificação:', 
                                   ('', 'Logistic Regression', 'KNN', 'Random Forest', 'Decision Tree'))

            if self.ai:
                model = self.__get_model()
                self.__train_and_evaluate(model, is_regression=False)
        except Exception as e:
            st.error(f"Erro na classificação: {e}")

    def __get_model(self) -> Union[LinearRegression, SVR, RandomForestRegressor, LogisticRegression, KNeighborsClassifier, RandomForestClassifier, DecisionTreeClassifier]:
        """
        Retorna o modelo de IA baseado na escolha do usuário.
        :return: Instância do modelo de IA selecionado.
        """
        try:
            models = {
                'Linear Regression': LinearRegression(),
                'SVR': SVR(),
                'Random Forest': RandomForestRegressor(),
                'Logistic Regression': LogisticRegression(),
                'KNN': KNeighborsClassifier(),
                'Random Forest': RandomForestClassifier(),
                'Decision Tree': DecisionTreeClassifier()
            }
            return models[self.ai]
        except KeyError as e:
            st.error(f"Modelo não encontrado: {e}")
            raise e

    def __split_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Método auxiliar para separar as colunas X e y do DataFrame.
        :return: Tuple contendo as features (X) e a target (y).
        """
        try:
            data_to_use: pd.DataFrame = self.normalized_data if self.normalized_data is not None else self.data
            X: pd.DataFrame = data_to_use.drop(columns=[self.target_column])
            y: pd.Series = data_to_use[self.target_column]
            return X, y
        except KeyError as e:
            st.error(f"Erro ao dividir os dados: coluna alvo não encontrada. {e}")
            raise e
        except Exception as e:
            st.error(f"Erro ao dividir os dados: {e}")
            raise e

    def __train_and_evaluate(self, model: Union[LinearRegression, SVR, RandomForestRegressor, LogisticRegression, KNeighborsClassifier, RandomForestClassifier, DecisionTreeClassifier], is_regression: bool) -> None:
        """
        Método para treinar e avaliar o modelo selecionado.
        :param model: Instância do modelo de IA selecionado.
        :param is_regression: Booleano indicando se o modelo é de regressão (True) ou classificação (False).
        """
        try:
            X, y = self.__split_data()
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            model.fit(X_train, y_train)
            predictions: np.ndarray = model.predict(X_test)

            if is_regression:
                mse: float = mean_squared_error(y_test, predictions)
                st.write(f'MSE: {mse}')
                metrics = {'mse': mse}
            else:
                accuracy: float = accuracy_score(y_test, predictions)
                st.write(f'Acurácia: {accuracy}')
                metrics = {'accuracy': accuracy}

            # Adicionar resultados e predições aos dados das métricas
            metrics['real_values'] = y_test.tolist()
            metrics['predictions'] = predictions.tolist()

            # Exibir os resultados
            results_df: pd.DataFrame = pd.DataFrame({'Real': y_test, 'Predição': predictions})
            results_df.reset_index(drop=True, inplace=True)
            st.write('Resultados:')
            st.dataframe(results_df, use_container_width=True)
            
            if st.button('Salvar Dados'):
                self.__save_metrics_to_db(metrics)

        except Exception as e:
            st.error(f"Erro durante o treinamento e avaliação: {e}")
    def __save_metrics_to_db(self, metrics: dict) -> None:
        """
        Salva as métricas no banco de dados.
        :param metrics: Dicionário contendo as métricas do modelo.
        """
        try:
            print(metrics)
            primary_action = session.query(TBPrimaryActions).order_by(TBPrimaryActions.id.desc()).first()

            if primary_action:
                ai_action = TBAiActions(
                    paradigm='Regression' if 'mse' in metrics else 'Classification',
                    model=self.ai,
                    target_column=self.target_column,
                    metrics=metrics,
                    primary_action_id=primary_action.id
                )
                session.add(ai_action)
                session.commit()
                st.write("Métricas salvas com sucesso!")
            else:
                st.error("Nenhuma ação primária encontrada para associar.")
        except Exception as e:
            st.error(f"Erro ao salvar métricas no banco de dados: {e}")


