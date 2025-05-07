# Relatório Comparativo de Desempenho dos Modelos de ML - ISIS IA v5

Data da Análise: 07 de Maio de 2025

## Introdução
Este relatório apresenta uma comparação do desempenho de quatro algoritmos de Machine Learning implementados no aplicativo ISIS IA para previsão de custos logísticos. Os modelos foram treinados e avaliados utilizando o mesmo conjunto de dados simulados (`dados_logisticos_simulados_v5.csv`) e as métricas foram calculadas sobre um conjunto de teste (20% dos dados), com as previsões de custo base em BRL.

Os modelos avaliados são:
1.  Regressão Linear (Linear Regression)
2.  Random Forest Regressor
3.  Gradient Boosting Regressor
4.  MLP Regressor (Rede Neural Multicamadas)

## Métricas de Avaliação
As seguintes métricas foram utilizadas para comparar o desempenho dos modelos:
*   **RMSE (Root Mean Squared Error):** Raiz do Erro Quadrático Médio. Indica a magnitude média dos erros de previsão, na mesma unidade da variável alvo (BRL). Valores menores são melhores.
*   **MAE (Mean Absolute Error):** Erro Absoluto Médio. Indica a média da diferença absoluta entre os valores previstos e os reais. Valores menores são melhores.
*   **R² (R-squared):** Coeficiente de Determinação. Indica a proporção da variância na variável dependente que é previsível a partir das variáveis independentes. Varia de 0 a 1, com valores mais próximos de 1 indicando melhor ajuste do modelo aos dados.

## Resultados Comparativos

Abaixo estão as métricas de desempenho obtidas para cada modelo durante a fase de treinamento e teste:

| Modelo                    | RMSE (BRL) | MAE (BRL) | R²    |
| ------------------------- | ---------- | --------- | ----- |
| Linear Regression         | 5186.84    | 4059.26   | 0.99  |
| Random Forest Regressor   | 6534.67    | 5212.64   | 0.98  |
| Gradient Boosting Regressor | 4627.39    | 3664.32   | 0.99  |
| MLP Regressor             | 8624.86    | 6675.29   | 0.97  |

*Nota: Os valores exatos podem variar ligeiramente a cada execução do script de treinamento devido a inicializações aleatórias em alguns algoritmos, mesmo com `random_state` definido, ou pequenas variações no ambiente de execução. Os valores apresentados foram coletados da última execução do script `backend_structure_v5.py`.*

## Análise e Conclusões

*   **Gradient Boosting Regressor** apresentou o melhor desempenho geral em termos de RMSE e MAE, sugerindo ser o modelo mais preciso entre os avaliados para este conjunto de dados simulados e configuração de hiperparâmetros.
*   **Linear Regression** também demonstrou um bom desempenho, com métricas competitivas, e tem a vantagem de ser um modelo mais simples e altamente interpretável.
*   **Random Forest Regressor**, embora com RMSE e MAE ligeiramente superiores ao Gradient Boosting e Linear Regression neste teste, oferece a vantagem da interpretabilidade através dos SHAP values, que foi implementada nesta versão do ISIS IA. A capacidade de entender *quais* fatores mais influenciam a previsão pode ser crucial para o usuário, compensando uma pequena diferença na precisão bruta.
*   **MLP Regressor (Rede Neural)** apresentou o maior RMSE e MAE. Modelos de redes neurais são frequentemente mais sensíveis à arquitetura e aos hiperparâmetros, e podem requerer um volume maior de dados e/ou um processo de otimização de hiperparâmetros mais extenso (ex: Grid Search, Random Search) para atingir seu potencial máximo. A configuração atual utilizou hiperparâmetros básicos e `early_stopping` para evitar overfitting e longos tempos de treinamento.

## Recomendações

*   Para usuários que priorizam a **máxima precisão** nos valores previstos, o **Gradient Boosting Regressor** parece ser a melhor escolha inicial com base nestes resultados.
*   Para usuários que valorizam a **interpretabilidade** e desejam entender os fatores que impulsionam as previsões, o **Random Forest Regressor** com a análise SHAP é altamente recomendado.
*   A **Linear Regression** permanece uma opção sólida e rápida, especialmente para obter uma estimativa rápida e compreensível.
*   O **MLP Regressor** pode ser explorado futuramente com otimização de hiperparâmetros e, possivelmente, com um conjunto de dados maior ou mais diversificado para verificar se seu desempenho pode ser aprimorado.

Este relatório serve como um guia para a seleção do modelo mais adequado dentro do aplicativo ISIS IA, considerando as prioridades do usuário em termos de precisão e interpretabilidade.

