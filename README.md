# ISIS IA (v5): Previsão Inteligente de Custos Logísticos - Pacote para Implantação Manual

Este pacote contém os arquivos necessários e as instruções para implantar manualmente a aplicação Streamlit ISIS IA (v5).

## Descrição

A aplicação ISIS IA (v5) é um Modelo de Prova de Conceito (MVP) que estima custos logísticos para a exportação de polpas de frutas da América do Sul (Brasil, Peru, Colômbia) para a Europa. A versão 5 inclui:

*   Seleção de portos de origem na América do Sul e destino na Europa.
*   Seleção de tipo (Reefer/Dry) e tamanho (20ft/40ft) de container.
*   Integração com API para taxas de câmbio em tempo real (BRL, USD, EUR).
*   Seleção de moeda para exibição dos resultados.
*   Cálculo do custo total e custo por tonelada.
*   Uso de modelos de Machine Learning (Regressão Linear e Random Forest) treinados com dados simulados.

## Arquivos Incluídos

*   `app_v5.py`: O código do frontend da aplicação Streamlit.
*   `backend_structure_v5.py`: O código do backend contendo a lógica de pré-processamento, treinamento de modelo, previsão e busca de câmbio.
*   `requirements.txt`: Lista de dependências Python necessárias.
*   `dados_logisticos_simulados_v5.csv`: Arquivo CSV com dados simulados usados para treinar os modelos (pode ser regenerado pelo `backend_structure_v5.py`).
*   `modelo_custo_v5_linear_regression.joblib`: Arquivo do modelo de Regressão Linear treinado.
*   `modelo_custo_v5_random_forest.joblib`: Arquivo do modelo Random Forest treinado.
*   `README.md`: Este arquivo de instruções.

## Pré-requisitos

*   **Python:** Versão 3.10 ou superior recomendada.
*   **pip:** Gerenciador de pacotes Python (geralmente incluído com Python).
*   **Acesso à Internet:** Necessário para instalar dependências e buscar taxas de câmbio em tempo real.

## Instruções de Implantação

1.  **Descompacte o Pacote:** Extraia todos os arquivos deste pacote para um diretório de sua escolha no seu servidor ou máquina local.

2.  **Navegue até o Diretório:** Abra um terminal ou prompt de comando e navegue até o diretório onde você extraiu os arquivos.
    ```bash
    cd /caminho/para/o/diretorio/isis_ia_v5
    ```

3.  **Crie um Ambiente Virtual (Recomendado):** É uma boa prática isolar as dependências do projeto.
    ```bash
    python -m venv venv
    ```

4.  **Ative o Ambiente Virtual:**
    *   No Linux/macOS:
        ```bash
        source venv/bin/activate
        ```
    *   No Windows:
        ```bash
        .\venv\Scripts\activate
        ```
    Você verá `(venv)` no início do prompt do seu terminal se a ativação for bem-sucedida.

5.  **Instale as Dependências:** Use o pip para instalar todas as bibliotecas listadas no `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```
    Aguarde a conclusão da instalação.

6.  **Execute a Aplicação Streamlit:** Inicie a aplicação usando o comando `streamlit run`.
    ```bash
    streamlit run app_v5.py
    ```

7.  **Acesse a Aplicação:** O Streamlit geralmente informa o URL local (ex: `http://localhost:8501`) e o URL de rede para acessar a aplicação no seu navegador.

## Notas Adicionais

*   **Geração de Modelos/Dados:** Os arquivos `.joblib` (modelos) e `.csv` (dados) estão incluídos. No entanto, se eles forem excluídos ou corrompidos, o script `backend_structure_v5.py` tentará regerá-los automaticamente na primeira execução que precisar deles (o que pode levar alguns minutos).
*   **API de Câmbio:** A aplicação usa a API pública `economia.awesomeapi.com.br` para taxas de câmbio. Se esta API ficar indisponível, a aplicação usará valores de fallback, mas os custos convertidos não serão em tempo real.
*   **Porta:** Por padrão, o Streamlit tenta usar a porta 8501. Se essa porta estiver ocupada, você pode especificar outra porta usando a flag `--server.port`:
    ```bash
    streamlit run app_v5.py --server.port 8505
    ```
*   **Implantação em Servidor:** Para uma implantação mais robusta em um servidor, considere usar ferramentas como Nginx como proxy reverso, Gunicorn (embora não seja o ideal para Streamlit) ou soluções de conteinerização como Docker.

---
*Fim das Instruções*

