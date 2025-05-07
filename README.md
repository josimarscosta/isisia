# ISIS IA - Previsão Inteligente de Custos Logísticos (v5 com SHAP e Novos Modelos)

Este pacote contém o aplicativo ISIS IA atualizado, com foco em maior interpretabilidade e exploração de novos algoritmos de Machine Learning para a previsão de custos logísticos no setor de polpas de frutas (América do Sul ⇄ Europa).

## Novas Funcionalidades e Melhorias:

1.  **Seleção Ampliada de Modelos Preditivos:**
    *   Agora você pode escolher entre quatro algoritmos de Machine Learning para realizar as previsões de custo:
        *   **Random Forest Regressor** (Default, agora com interpretabilidade SHAP)
        *   **Linear Regression**
        *   **Gradient Boosting Regressor** (Novo)
        *   **MLP Regressor (Rede Neural)** (Novo)
    *   A seleção do modelo é feita na barra lateral, permitindo comparar diferentes abordagens.

2.  **Interpretabilidade com SHAP Values (para Random Forest):**
    *   Ao selecionar o modelo **Random Forest**, o aplicativo agora exibe uma análise de interpretabilidade utilizando SHAP (SHapley Additive exPlanations) values.
    *   **O que são SHAP Values?** Eles mostram a contribuição individual de cada fator (ex: volume da carga, porto de origem, tipo de container) para a previsão de custo específica gerada.
    *   **Como interpretar:**
        *   Uma tabela exibirá os **principais fatores impactantes**, ordenados pela magnitude de sua contribuição.
        *   Um gráfico de barras mostrará visualmente o impacto desses fatores. Valores positivos indicam que o fator aumentou o custo previsto, enquanto valores negativos indicam que o fator diminuiu o custo previsto (sempre em relação à previsão base do modelo, em BRL).
    *   Esta funcionalidade visa aumentar a transparência e a confiança nas previsões do modelo Random Forest.

3.  **Treinamento e Avaliação de Novos Modelos:**
    *   Os modelos Gradient Boosting e MLP Regressor foram treinados e avaliados com os mesmos dados simulados dos modelos anteriores.
    *   **Comparativo de Desempenho (Métricas em BRL - dados de teste):**
        *   **Linear Regression:** RMSE ~5186, MAE ~4059, R² ~0.99
        *   **Random Forest:** RMSE ~6534, MAE ~5212, R² ~0.98 (Nota: SHAP pode ser mais útil aqui do que a precisão absoluta em alguns casos)
        *   **Gradient Boosting:** RMSE ~4627, MAE ~3664, R² ~0.99 (Apresentou bom desempenho nos testes)
        *   **MLP Regressor:** RMSE ~8624, MAE ~6675, R² ~0.97 (Modelo mais complexo, pode necessitar de mais dados/ajustes finos de hiperparâmetros para otimizar)
    *   As métricas (RMSE, MAE, R²) de cada modelo selecionado (referentes ao seu treinamento em BRL) são exibidas na interface.

4.  **Exibição Destacada da Taxa de Câmbio:**
    *   A taxa de câmbio utilizada para converter o custo final para a moeda selecionada (USD, EUR, BRL) é agora exibida de forma mais proeminente na interface, logo abaixo das métricas de custo total, indicando a fonte (API ou fallback).

## Arquivos Inclusos no Pacote:

*   `app_v5.py`: Frontend do Streamlit atualizado.
*   `backend_structure_v5.py`: Backend com a lógica de previsão, treinamento dos 4 modelos e integração SHAP.
*   `requirements.txt`: Dependências Python (incluindo `shap`).
*   `dados_logisticos_simulados_v5.csv`: Dados simulados utilizados.
*   `modelo_custo_v5_linear_regression.joblib`: Modelo treinado.
*   `modelo_custo_v5_random_forest.joblib`: Modelo treinado.
*   `modelo_custo_v5_gradient_boosting.joblib`: Modelo treinado.
*   `modelo_custo_v5_mlp_regressor.joblib`: Modelo treinado.
*   `README.md`: Este arquivo.

## Instruções para Implantação (Streamlit Cloud):

1.  **Prepare seu Repositório Git:**
    *   Descompacte este arquivo .zip.
    *   Crie um novo repositório no GitHub (ou similar).
    *   Adicione todos os arquivos descompactados a este repositório e faça o commit/push.
2.  **Acesse o Streamlit Community Cloud ([share.streamlit.io](https://share.streamlit.io)).**
3.  **Implante um Novo Aplicativo:**
    *   Clique em "New app".
    *   Cole a URL do seu repositório Git.
    *   Selecione a branch (geralmente `main` ou `master`).
    *   Indique `app_v5.py` como o arquivo principal.
    *   Clique em "Deploy!". O Streamlit Cloud instalará as dependências do `requirements.txt` e executará o aplicativo.

## Próximos Passos Sugeridos (Evolução Contínua):

*   Ajuste fino de hiperparâmetros para os modelos Gradient Boosting e MLP Regressor.
*   Implementação de SHAP para outros modelos (se aplicável e desejado).
*   Desenvolvimento da funcionalidade de "Cenários Comparativos".
*   Expansão da base de dados com dados reais ou mais complexos.

Esperamos que estas evoluções tornem o ISIS IA uma ferramenta ainda mais poderosa e transparente para suas análises de custos logísticos!

