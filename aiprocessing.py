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
        self.selected_method = None 
    def run(self) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """
        Main method to run preprocessing and AI model selection flow.
        """
        try:
            # Toggle para escolher entre Regressão e Classificação
            self.selected_method = st.radio("Selecione o tipo de processamento:", 
                                            options=["Classificação", "Regressão"], 
                                            index=0, 
                                            key="process_type")

            st.write(f"Método selecionado: {self.selected_method}")
            
            categorical_columns: List[str] = self.data.select_dtypes(include=['object']).columns.tolist()

            if categorical_columns:
                numerical_df: pd.DataFrame = self.data.select_dtypes(exclude=['object'])
                preprocessing = Preprocessing(self.data)
                final_data: pd.DataFrame = preprocessing.preprocess_categorical_data(self.data, numerical_df)
                self.data = final_data
                return final_data, self.selected_method
            else:
                return self.data, self.selected_method

        except Exception as e:
            st.error(f"Ocorreu um erro durante a execução: {e}")
            return None, None
        
    def remove_coluna(self) -> None:
        colunas = st.multiselect("Remova colunas", self.data.columns)
        if st.button("Remover"):
            # Atualiza a base principal com a nova versão dos dados após remover as colunas
            self.data = self.data.drop(columns=colunas)
            st.session_state['data'] = self.data  # Atualiza a base principal no session_state
            st.write(f"Colunas removidas: {colunas}")
            st.dataframe(self.data)  # Exibe a base atualizada
    def regression(self) -> None:
        """
        Exibe as opções de regressão e executa o modelo selecionado.
        """
        try:
            regression_models = ['Decision Tree (available)', 'Linear Regression (disabled)', 'SVR (disabled)', 'Random Forest (disabled)']
            self.ai = st.selectbox('Selecione um modelo de regressão:', options=regression_models, key="regression_model")
            
            # Exibe a seleção da coluna alvo
            self.target_column = st.selectbox('Selecione a coluna alvo para a regressão:', self.data.columns, key="regression_target")

            if st.button('Executar Regressão'):
                if 'Decision Tree' in self.ai:
                    model = DecisionTreeClassifier()  # Usa Decision Tree como exemplo
                    self.__train_and_evaluate(model, is_regression=True)
                else:
                    st.error("Por favor, selecione o modelo Decision Tree.")
        except Exception as e:
            st.error(f"Erro na regressão: {e}")

    def classification(self) -> None:
        """
        Exibe as opções de classificação e executa o modelo selecionado.
        """
        try:
            classification_models = ['Decision Tree (available)', 'Logistic Regression (disabled)', 'KNN (disabled)', 'Random Forest (disabled)']
            self.ai = st.selectbox('Selecione um modelo de classificação:', options=classification_models, key="classification_model")
            
            # Exibe a seleção da coluna alvo
            self.target_column = st.selectbox('Selecione a coluna alvo para a classificação:', self.data.columns, key="classification_target")

            if st.button('Executar Classificação'):
                if 'Decision Tree' in self.ai:
                    model = DecisionTreeClassifier()  # Usa Decision Tree como exemplo
                    self.__train_and_evaluate(model, is_regression=False)
                else:
                    st.error("Por favor, selecione o modelo Decision Tree.")
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
            data_to_use: pd.DataFrame = self.data
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
                st.write(f'Métrica de avaliação de erro (MSE): {mse}')
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
    

