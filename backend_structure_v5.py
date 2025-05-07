# backend_structure_v5.py

"""
Backend (v5) com mais portos SA, tipos/tamanhos de container, API de câmbio, polpa de goiaba e custo por tonelada.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # Adicionado GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor # Adicionado MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import datetime
import requests
import json
import shap # Adicionado para SHAP

# --- Configurações (v5) ---
MODEL_PATH_PREFIX = "modelo_custo_v5"
DATA_FILEPATH = "dados_logisticos_simulados_v5.csv"

# Portos (v5 - Expandido)
PORTOS_ORIGEM_SA = {
    "Brasil": ["Santos", "Paranaguá", "Pecém", "Suape"],
    "Peru": ["Callao", "Chancay", "Salaverry", "Paita"], # Paita é comum para frutas
    "Colômbia": ["Cartagena", "Buenaventura", "Santa Marta"]
}
ALL_PORTOS_ORIGEM = [p for country_ports in PORTOS_ORIGEM_SA.values() for p in country_ports]
PORTOS_DESTINO_EU = ["Rotterdam", "Hamburg", "Le Havre", "Valencia", "Antwerp"]

# Containers (v5)
CONTAINER_TYPES = ["Reefer", "Dry"] # Reefer é essencial para polpa congelada/resfriada
CONTAINER_SIZES = ["20ft", "40ft"]

# Moedas Suportadas
MOEDAS_SUPORTADAS = ["BRL", "USD", "EUR"]

# --- Estrutura de Dados de Entrada (v5) ---
input_data_example = {
    "porto_origem": "Santos",
    "porto_destino": "Rotterdam",
    "modal_logistico": "marítimo",
    "volume_carga_ton": 20.0,
    "tipo_produto": "polpa de goiaba",
    "tipo_embalagem": "containerizado",
    "container_type": "Reefer", # Novo
    "container_size": "40ft", # Novo
    "preco_combustivel_indice": 1.5,
    "taxas_impostos_percent": 15.0,
    "cambio_brl_eur_estimado": 5.5,
    "prazo_entrega_dias": 30,
    "data_previsao_inicio": (datetime.date.today() + datetime.timedelta(days=30)).strftime("%Y-%m-%d"),
    "data_previsao_fim": (datetime.date.today() + datetime.timedelta(days=60)).strftime("%Y-%m-%d"),
    "target_currency": "EUR"
}

# --- Estrutura de Dados de Saída (v5) ---
# Similar à v4, mas o custo será influenciado pelos novos parâmetros
output_data_example = {
    "custo_total_estimado": 19500.00, # Exemplo, valor será calculado
    "custo_por_tonelada": 975.00, # Exemplo, valor será calculado
    "moeda": "EUR",
    "composicao_custos": { # Composição também na moeda de saída
        "frete": 11700.00,
        "armazenagem": 1950.00,
        "seguro": 975.00,
        "taxas_burocracia": 4875.00
    },
    "modelo_utilizado": "Random Forest",
    "metricas_modelo": {
        "r2_score": 0.92,
        "mae": 420.0
    },
    "periodo_previsao": "30/05/2025 - 29/07/2025",
    "taxa_cambio_aplicada": {"BRL_para_EUR": 0.1818} # Exemplo
}

# --- Funções Auxiliares (v5 - Câmbio igual à v4) ---
def get_exchange_rates():
    """Busca taxas de câmbio BRL-USD e BRL-EUR da AwesomeAPI."""
    try:
        response = requests.get("https://economia.awesomeapi.com.br/json/last/USD-BRL,EUR-BRL")
        response.raise_for_status()
        rates_data = response.json()
        rates = {
            "USD_BRL": float(rates_data["USDBRL"]["bid"]),
            "EUR_BRL": float(rates_data["EURBRL"]["bid"])
        }
        return rates
    except requests.exceptions.RequestException as e:
        print(f"Erro ao buscar taxas de câmbio da API: {e}")
        return {"USD_BRL": 5.0, "EUR_BRL": 5.5} # Fallback
    except (KeyError, json.JSONDecodeError) as e:
        print(f"Erro ao processar resposta da API de câmbio: {e}")
        return {"USD_BRL": 5.0, "EUR_BRL": 5.5} # Fallback

# --- Lógica do Backend (v5) ---
def gerar_dados_simulados(num_samples=3000, filepath=DATA_FILEPATH):
    """Gera dados simulados (v5) com mais portos, containers, salva em CSV."""
    np.random.seed(42)
    start_date_range = datetime.date(2022, 1, 1)
    dates = [start_date_range + datetime.timedelta(days=np.random.randint(0, 1095)) for _ in range(num_samples)]

    data = {
        "data_evento": dates,
        "porto_origem": np.random.choice(ALL_PORTOS_ORIGEM, num_samples),
        "porto_destino": np.random.choice(PORTOS_DESTINO_EU, num_samples),
        "modal_logistico": np.random.choice(["marítimo", "aéreo", "multimodal"], num_samples, p=[0.80, 0.10, 0.10]), # Mais marítimo
        "volume_carga_ton": np.random.uniform(5, 50, num_samples),
        "tipo_produto": np.random.choice(["polpa de acerola", "polpa de caju", "polpa de manga", "polpa de maracujá", "polpa de goiaba"], num_samples),
        "tipo_embalagem": np.random.choice(["containerizado", "granel"], num_samples, p=[0.95, 0.05]), # Mais container
        "container_type": np.random.choice(CONTAINER_TYPES, num_samples, p=[0.85, 0.15]), # Maioria Reefer para polpa
        "container_size": np.random.choice(CONTAINER_SIZES, num_samples, p=[0.4, 0.6]), # Mais 40ft
        "preco_combustivel_indice": np.random.uniform(1.0, 2.5, num_samples),
        "taxas_impostos_percent": np.random.uniform(5, 25, num_samples),
        "cambio_brl_eur_estimado": np.random.uniform(5.0, 6.5, num_samples),
        "prazo_entrega_dias": np.random.randint(10, 60, num_samples),
        "custo_total_logistico_brl": 0
    }
    df = pd.DataFrame(data)
    df["data_evento"] = pd.to_datetime(df["data_evento"])

    # Features de data
    df["ano"] = df["data_evento"].dt.year
    df["mes"] = df["data_evento"].dt.month
    df["dia_do_ano"] = df["data_evento"].dt.dayofyear
    df["dia_da_semana"] = df["data_evento"].dt.dayofweek
    df["semana_do_ano"] = df["data_evento"].dt.isocalendar().week.astype(int)

    # Simular custo base em BRL (v5 com containers)
    df["custo_base"] = df["volume_carga_ton"] * 2600
    df["custo_modal"] = df["modal_logistico"].apply(lambda x: 5500 if x == "marítimo" else (27000 if x == "aéreo" else 16000))
    # Custo adicional significativo para Reefer
    df["custo_container_tipo"] = df["container_type"].apply(lambda x: 6000 if x == "Reefer" else 500)
    # Custo adicional para 40ft
    df["custo_container_tamanho"] = df["container_size"].apply(lambda x: 3000 if x == "40ft" else 1500)
    df["custo_combustivel"] = df["preco_combustivel_indice"] * df["volume_carga_ton"] * 260
    df["custo_taxas"] = (df["taxas_impostos_percent"] / 100) * df["custo_base"]
    df["fator_cambio_risco"] = df["custo_base"] * abs(df["cambio_brl_eur_estimado"] / 5.5 - 1) * 0.3
    df["custo_prazo"] = (30 / df["prazo_entrega_dias"].clip(lower=5)) * 5500 # Evitar divisão por zero ou prazo muito baixo
    df["tendencia_temporal"] = (df["data_evento"] - df["data_evento"].min()).dt.days * 11
    df["fator_sazonal"] = np.sin((df["mes"] - 3) * np.pi / 6) * 3800 + 3800

    df["custo_total_logistico_brl"] = (df["custo_base"] + df["custo_modal"] +
                                      df["custo_container_tipo"] + df["custo_container_tamanho"] +
                                      df["custo_combustivel"] + df["custo_taxas"] +
                                      df["fator_cambio_risco"] + df["custo_prazo"] +
                                      df["tendencia_temporal"] + df["fator_sazonal"] +
                                      np.random.normal(0, 3500, num_samples))

    df["custo_total_logistico_brl"] = df["custo_total_logistico_brl"].clip(lower=6000)

    # Remover colunas intermediárias
    cols_to_drop = ["custo_base", "custo_modal", "custo_container_tipo", "custo_container_tamanho",
                    "custo_combustivel", "custo_taxas", "fator_cambio_risco", "custo_prazo",
                    "tendencia_temporal", "fator_sazonal", "data_evento"]
    df = df.drop(columns=cols_to_drop)

    df.to_csv(filepath, index=False)
    print(f"Dados simulados (v5) gerados e salvos em {filepath}")
    return df

def carregar_dados(filepath=DATA_FILEPATH):
    """Carrega os dados (v5) de um arquivo CSV."""
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        print(f"Arquivo {filepath} não encontrado. Gerando dados simulados (v5).")
        return gerar_dados_simulados(filepath=filepath)

def preprocessar_dados(df):
    """Realiza o pré-processamento (v5), incluindo portos e containers."""
    categorical_cols = ["porto_origem", "porto_destino", "modal_logistico", "tipo_produto",
                        "tipo_embalagem", "container_type", "container_size"]
    # Garantir que todas as colunas categóricas existem
    for col in categorical_cols:
        if col not in df.columns:
            raise ValueError(f"Coluna categórica esperada não encontrada no DataFrame: {col}")
    df_processed = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    return df_processed

def treinar_modelos(df_processed, model_types=["linear_regression", "random_forest", "gradient_boosting", "mlp_regressor"]):
    """Treina múltiplos modelos de ML (v5), incluindo Gradient Boosting e MLP, avalia e salva cada um."""
    if "custo_total_logistico_brl" not in df_processed.columns:
        raise ValueError("Coluna alvo 'custo_total_logistico_brl' não encontrada no DataFrame processado.")
    X = df_processed.drop("custo_total_logistico_brl", axis=1)
    y = df_processed["custo_total_logistico_brl"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    trained_models_info = {}

    for model_type in model_types:
        print(f"\n--- Treinando Modelo (v5): {model_type} ---")
        if model_type == "linear_regression":
            model = LinearRegression()
        elif model_type == "random_forest":
            # Hiperparâmetros podem precisar de ajuste com mais features
            model = RandomForestRegressor(n_estimators=180, random_state=42, n_jobs=-1, max_depth=22, min_samples_split=6)
        elif model_type == "gradient_boosting": # Adicionado Gradient Boosting
            model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42)
        elif model_type == "mlp_regressor": # Adicionado MLP Regressor
            # Hiperparâmetros básicos, podem precisar de ajuste e mais tempo de treino
            model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42, early_stopping=True, n_iter_no_change=10)
        else:
            continue

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = {
            "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
            "mae": round(mean_absolute_error(y_test, y_pred), 2),
            "r2": round(r2_score(y_test, y_pred), 2)
        }
        print(f"Metricas (BRL): {metrics}")

        save_path = f"{MODEL_PATH_PREFIX}_{model_type}.joblib"
        model_data = {"model": model, "columns": X_train.columns.tolist(), "metrics": metrics}
        joblib.dump(model_data, save_path)
        print(f"Modelo treinado (v5) e salvo em {save_path}")
        trained_models_info[model_type] = model_data

    return trained_models_info

def prever_custo(input_data, model_type="random_forest"):
    """Carrega um modelo específico (v5), faz a previsão e converte para a moeda alvo."""
    model_path = f"{MODEL_PATH_PREFIX}_{model_type}.joblib"

    # Carregar modelo
    try:
        saved_model_data = joblib.load(model_path)
        model = saved_model_data["model"]
        model_columns = saved_model_data["columns"]
        model_metrics = saved_model_data["metrics"]
    except FileNotFoundError:
        print(f"Arquivo de modelo {model_path} não encontrado. Treinando modelos (v5)...")
        df_simulado = carregar_dados()
        df_proc = preprocessar_dados(df_simulado)
        treinar_modelos(df_proc)
        try:
            saved_model_data = joblib.load(model_path)
            model = saved_model_data["model"]
            model_columns = saved_model_data["columns"]
            model_metrics = saved_model_data["metrics"]
        except FileNotFoundError:
             raise FileNotFoundError(f"Falha ao carregar o modelo {model_path} mesmo após tentar treinar.")

    # Preparar Input
    target_currency = input_data.pop("target_currency", "BRL")
    if target_currency not in MOEDAS_SUPORTADAS:
        raise ValueError(f"Moeda alvo não suportada.")

    input_df = pd.DataFrame([input_data])

    # Processar datas
    try:
        data_inicio = pd.to_datetime(input_df["data_previsao_inicio"].iloc[0])
        data_fim = pd.to_datetime(input_df["data_previsao_fim"].iloc[0])
        data_media = data_inicio + (data_fim - data_inicio) / 2
    except Exception as e:
        raise ValueError(f"Erro ao processar datas de previsão: {e}.")

    input_df["ano"] = data_media.year
    input_df["mes"] = data_media.month
    input_df["dia_do_ano"] = data_media.dayofyear
    input_df["dia_da_semana"] = data_media.dayofweek
    input_df["semana_do_ano"] = data_media.isocalendar().week

    input_df_processed = input_df.drop(columns=["data_previsao_inicio", "data_previsao_fim"])

    # Pré-processar categóricas (v5)
    categorical_cols = ["porto_origem", "porto_destino", "modal_logistico", "tipo_produto",
                        "tipo_embalagem", "container_type", "container_size"]
    input_df_processed = pd.get_dummies(input_df_processed, columns=categorical_cols, drop_first=True)

    # Alinhar colunas com o modelo
    missing_cols = set(model_columns) - set(input_df_processed.columns)
    for c in missing_cols:
        input_df_processed[c] = 0
    # Garantir ordem e remover extras
    input_df_processed = input_df_processed.reindex(columns=model_columns, fill_value=0)

    # Previsão (em BRL)
    custo_predito_brl = model.predict(input_df_processed)[0]

    # Calcular SHAP values se o modelo for RandomForestRegressor
    shap_values_dict = None
    if isinstance(model, RandomForestRegressor):
        try:
            explainer = shap.TreeExplainer(model)
            shap_values_instance = explainer.shap_values(input_df_processed)
            # Para uma única instância, shap_values_instance pode ser um array 1D
            # Mapear para nomes de colunas
            shap_values_dict = dict(zip(model_columns, shap_values_instance[0]))
        except Exception as e_shap:
            print(f"Erro ao calcular SHAP values: {e_shap}")
            # Continuar sem SHAP values em caso de erro
            shap_values_dict = None

    # Obter taxas de câmbio atuais
    live_rates = get_exchange_rates()
    usd_brl_rate = live_rates["USD_BRL"]
    eur_brl_rate = live_rates["EUR_BRL"]

    # Converter para moeda alvo
    conversion_rate = 1.0
    applied_rate_info = {}
    if target_currency == "USD":
        conversion_rate = 1 / usd_brl_rate
        applied_rate_info = {"BRL_para_USD": round(conversion_rate, 4)}
    elif target_currency == "EUR":
        conversion_rate = 1 / eur_brl_rate
        applied_rate_info = {"BRL_para_EUR": round(conversion_rate, 4)}
    else: # BRL
        applied_rate_info = {"BRL_para_BRL": 1.0}

    custo_predito_final = custo_predito_brl * conversion_rate

    # Calcular custo por tonelada
    volume = input_data.get("volume_carga_ton", 1) # Evitar divisão por zero
    custo_por_tonelada = custo_predito_final / volume if volume > 0 else 0

    # Simular composição (convertida) - Manter percentuais fixos por simplicidade no MVP
    composicao_brl = {
        "frete": custo_predito_brl * 0.6,
        "armazenagem": custo_predito_brl * 0.1,
        "seguro": custo_predito_brl * 0.05,
        "taxas_burocracia": custo_predito_brl * 0.25
    }
    composicao_final = {k: v * conversion_rate for k, v in composicao_brl.items()}

    # Formatar período
    periodo_str = f"{data_inicio.strftime('%d/%m/%Y')} - {data_fim.strftime('%d/%m/%Y')}"

    # Construir output (v5)
    output = {
        "custo_total_estimado": round(custo_predito_final, 2),
        "custo_por_tonelada": round(custo_por_tonelada, 2),
        "moeda": target_currency,
        "composicao_custos": {k: round(v, 2) for k, v in composicao_final.items()},
        "modelo_utilizado": model_type.replace("_", " ").title(),
        "metricas_modelo": model_metrics,
        "periodo_previsao": periodo_str,
        "taxa_cambio_aplicada": applied_rate_info,
        "shap_values": shap_values_dict  # Adicionado SHAP values
    }

    return output

# --- Exemplo de Uso (v5) ---
if __name__ == "__main__":
    # 1. Gerar/Carregar Dados (v5)
    df_dados = carregar_dados()

    # 2. Pré-processar (v5)
    df_processado = preprocessar_dados(df_dados.copy())

    # 3. Treinar Modelos (v5)
    trained_models = treinar_modelos(df_processado)

    # 4. Fazer previsões com dados de exemplo
    print("\n--- Previsões Exemplo (v5) ---")
    for currency in MOEDAS_SUPORTADAS:
        print(f"\n==> Previsão para Moeda: {currency} ==")
        input_data_test = input_data_example.copy()
        input_data_test["target_currency"] = currency
        for model_name in trained_models.keys():
            print(f"  Usando Modelo: {model_name}")
            try:
                # Passa cópia para não modificar o original no loop
                previsao = prever_custo(input_data_test.copy(), model_type=model_name)
                print(f"    Resultado: {previsao}")
            except Exception as e:
                print(f"    Erro na previsão com {model_name}: {e}")

