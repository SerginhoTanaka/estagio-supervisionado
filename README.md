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

### DER Diagram

```mermaid
erDiagram
    TBPrimaryActions {
        int id PK
        string action_name
        string dataset_name
        boolean is_ai
        datetime timestamp
    }

    TBAiActions {
        int id PK
        string paradigm
        string model
        string target_column
        json metrics
        int primary_action_id FK
    }

    TBPrimaryActions ||--o{ TBAiActions : "1:N"


```
### Sequence Diagram

```mermaid

sequenceDiagram
    participant Pesquisador
    participant Tela steamlit
    participant Preprocessing

    Pesquisador->> Tela steamlit: select_preprocessing_method()
    Tela steamlit->> Preprocessing: apply_preprocessing()
    Preprocessing->> Preprocessing: clean_data()
    Preprocessing ->>Tela steamlit: Dados descrito
    Tela steamlit->> Pesquisador : mostrar informação
    
```
