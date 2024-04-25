# Estágio

## Installation

To set up the project, follow these steps:

1. Create a virtual environment:
    ```bash
    python -m venv env
    ```

2. Activate the virtual environment:
    - **Windows**:
        ```bash
        env\Scripts\activate
        ```
    - **Unix or MacOS**:
        ```bash
        source env/bin/activate
        ```

3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the project, execute the following command:
```bash
streamlit run main.py
```

## Diagram

### Claas Diagram

```mermaid
classDiagram
    class Preprocessing {
        - data: pd.DataFrame
        - scaler: Any
        # clean_data(data: pd.DataFrame, columns: List[str]): pd.DataFrame
        # select_preprocessing_method(): void
        - apply_preprocessing(preprocessing_method: str): void
    }
    class Dashboard {
        - data: pd.DataFrame
        - preprocessor: Preprocessing
        - description: Description
        + run(): void
        - generate_graph(): void
        - plot_graph(data, columns: Optional[List[str]]): void
    }
    
    class Description {
        - data: pd.DataFrame
        + run(): void
    }
    
Dashboard "1" -- "1" Preprocessing: Seleciona
Dashboard "1" -- "1" Description: Escolhe
