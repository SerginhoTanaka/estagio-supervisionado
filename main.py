# streamlit
import streamlit as st
# Pandas
import pandas as pd
# Matplotlib
import matplotlib.pyplot as plt
# Custom modules
from preprocessing import Preprocessing
from description import Description
# Typing
from typing import List
from typing import Optional

import numpy as np
"""
This module provides a Dashboard class for data preprocessing and visualization.
"""

class Dashboard():
    """
    A class for data preprocessing and visualization.
    """

    def __init__(self):
        self.data = pd.read_csv('database_300.csv')
        self.preprocessor = Preprocessing(self.data)
        self.description = Description(self.data)

    def run(self) -> None:
        """
        Run the dashboard.
        """
        st.sidebar.title("Pré-processamento de dados")
        selected_option = st.sidebar.radio(
            "Selecione uma opção",
            ["Pré-processamento", "Descrição", "Gráfico sem pré-processamento"]
        )

        if selected_option == "Pré-processamento":
            self.preprocessor._select_preprocessing_method()

        elif selected_option == "Descrição":
            self.description.run()

        elif selected_option == "Gráfico sem pré-processamento":
            self.__generate_graph()

    def __generate_graph(self) -> None:
        """
        Generate a graph without preprocessing.
        """
        st.sidebar.write("Gráfico sem pré-processamento")
        columns = self.description._select_columns()

        self.__plot_graph(self.data, columns)

    def __plot_graph(self, not_cleaned_data: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
        """
        Plot a graph.
        """
        
        if columns is None:
            columns = ['situacao']
        numeric_columns = not_cleaned_data.select_dtypes(include=np.number).columns
        selected_numeric_columns = [col for col in numeric_columns if col in columns]
        non_numeric_columns = [col for col in columns if col not in selected_numeric_columns]
        if selected_numeric_columns:
            for col in selected_numeric_columns:
                st.write(not_cleaned_data[col].describe())
        if non_numeric_columns:
            for col in non_numeric_columns:
                fig_width = 20 / len(columns)
                fig, ax = plt.subplots(figsize=(fig_width, 6))

                col_data = not_cleaned_data[col]
                counts = col_data.value_counts()

                counts.plot(kind='bar', ax=ax)

                ax.set_xlabel(col)
                ax.set_ylabel('Quantidade')
                ax.set_title(f'Quantidade de {col}')

                for i, v in enumerate(counts.values):
                    ax.text(i, v + 0.1, str(v), ha='center')

                st.pyplot(fig)

if __name__ == '__main__':
    dashboard = Dashboard()
    dashboard.run()
