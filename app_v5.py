# app_v5.py

"""
Frontend (v5) com mais portos SA, tipos/tamanhos de container, API de câmbio, polpa de goiaba e custo por tonelada.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import datetime

# Importar funções do backend (v5)
try:
    from backend_structure_v5 import (
        prever_custo,
        gerar_dados_simulados,
        carregar_dados,
        treinar_modelos,
        preprocessar_dados,
        ALL_PORTOS_ORIGEM, # Lista completa de portos de origem SA
        PORTOS_DESTINO_EU,
        CONTAINER_TYPES,
        CONTAINER_SIZES,
        MOEDAS_SUPORTADAS
    )
    BACKEND_VERSION = 5
except ImportError as e:
    st.error(f"Erro ao importar backend_structure_v5.py: {e}. Certifique-se de que o arquivo está no diretório correto.")
    # Tentar executar o backend v5 para gerar os modelos
    try:
        import subprocess
        st.info("Tentando executar script backend_structure_v5.py para gerar modelos...")
        # Usar python3.11 explicitamente se necessário
        process = subprocess.run(["python3.11", "backend_structure_v5.py"], capture_output=True, text=True, check=True, timeout=240) # Aumentar timeout
        st.code(process.stdout)
        # Tentar importar v5 novamente
        from backend_structure_v5 import (
            prever_custo, ALL_PORTOS_ORIGEM, PORTOS_DESTINO_EU,
            CONTAINER_TYPES, CONTAINER_SIZES, MOEDAS_SUPORTADAS
        )
        BACKEND_VERSION = 5
        st.success("Backend v5 executado e importado com sucesso.")
    except Exception as run_e:
        st.error(f"Falha ao executar backend_structure_v5.py: {run_e}")
        st.stop()

# --- Configuração da Página ---
st.set_page_config(layout="wide", page_title="ISIS IA v5 - Previsão de Custos Logísticos")

# --- Título ---
st.title("🚢✈️ ISIS IA (v5): Previsão Inteligente de Custos Logísticos")
st.markdown("**MVP para o setor de Polpas de Frutas (América do Sul ⇄ Europa) - v5 com Portos SA, Containers, Câmbio Real e Custo/Ton**")

# --- Barra Lateral (Inputs do Usuário v5) ---
st.sidebar.header("Parâmetros da Simulação")

# Inputs v5
porto_origem = st.sidebar.selectbox("Porto de Origem (América do Sul)", ALL_PORTOS_ORIGEM, index=0)
porto_destino = st.sidebar.selectbox("Porto de Destino (Europa)", PORTOS_DESTINO_EU, index=0)
modal_logistico = st.sidebar.selectbox("Modal Logístico", ["marítimo", "aéreo", "multimodal"], index=0)
volume_carga_ton = st.sidebar.number_input("Volume da Carga (toneladas)", min_value=1.0, max_value=1000.0, value=20.0, step=1.0)
tipo_produto = st.sidebar.selectbox("Tipo de Produto (Polpa)", ["polpa de acerola", "polpa de caju", "polpa de manga", "polpa de maracujá", "polpa de goiaba"], index=4)
tipo_embalagem = st.sidebar.selectbox("Tipo de Embalagem", ["containerizado", "granel"], index=0)

# Novos Inputs de Container (v5)
st.sidebar.divider()
st.sidebar.subheader("Detalhes do Container")
container_type = st.sidebar.selectbox("Tipo de Container", CONTAINER_TYPES, index=0, help="Reefer é recomendado para polpas de frutas.")
container_size = st.sidebar.selectbox("Tamanho do Container", CONTAINER_SIZES, index=1)

st.sidebar.divider()
st.sidebar.subheader("Fatores Econômicos e Operacionais")
preco_combustivel_indice = st.sidebar.slider("Preço Combustível / Índice BDI (Valor relativo)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
taxas_impostos_percent = st.sidebar.slider("Taxas e Impostos (% estimado)", min_value=0.0, max_value=50.0, value=15.0, step=1.0)
cambio_brl_eur_estimado = st.sidebar.slider("Câmbio BRL/EUR (Estimativa para Modelo)", min_value=4.0, max_value=8.0, value=5.5, step=0.05, help="Esta é uma estimativa usada como input para o modelo. A conversão final usará taxas de câmbio atuais.")
prazo_entrega_dias = st.sidebar.slider("Prazo Estimado de Entrega (dias)", min_value=5, max_value=90, value=30, step=1)

# Datas (mantido)
st.sidebar.divider()
st.sidebar.subheader("Período da Previsão")
today = datetime.date.today()
data_previsao_inicio = st.sidebar.date_input("Data de Início da Previsão", value=today + datetime.timedelta(days=30))
data_previsao_fim = st.sidebar.date_input("Data de Fim da Previsão", value=today + datetime.timedelta(days=60))

# Validação simples das datas
if data_previsao_inicio >= data_previsao_fim:
    st.sidebar.error("A data de fim deve ser posterior à data de início.")
    valid_dates = False
else:
    valid_dates = True

# Seleção de Modelo e Moeda (mantido)
st.sidebar.divider()
st.sidebar.subheader("Configuração do Modelo e Saída")
modelo_selecionado = st.sidebar.selectbox(
    "Modelo Preditivo",
    ["random_forest", "linear_regression"],
    index=0,
    help="Escolha o modelo de Machine Learning para a previsão (base em BRL)."
)
target_currency = st.sidebar.selectbox(
    "Moeda de Saída",
    MOEDAS_SUPORTADAS,
    index=2, # Default EUR
    help="Escolha a moeda para exibir o custo final."
)

# Botão para executar a previsão
executar_previsao = st.sidebar.button("Calcular Custo Logístico Estimado", disabled=not valid_dates)

# --- Área Principal (Outputs v5) ---
st.header("Resultados da Previsão")

if executar_previsao and valid_dates:
    # Coletar dados do input (v5)
    input_data = {
        "porto_origem": porto_origem,
        "porto_destino": porto_destino,
        "modal_logistico": modal_logistico,
        "volume_carga_ton": volume_carga_ton,
        "tipo_produto": tipo_produto,
        "tipo_embalagem": tipo_embalagem,
        "container_type": container_type, # Adicionado
        "container_size": container_size, # Adicionado
        "preco_combustivel_indice": preco_combustivel_indice,
        "taxas_impostos_percent": taxas_impostos_percent,
        "cambio_brl_eur_estimado": cambio_brl_eur_estimado,
        "prazo_entrega_dias": prazo_entrega_dias,
        "data_previsao_inicio": data_previsao_inicio.strftime("%Y-%m-%d"),
        "data_previsao_fim": data_previsao_fim.strftime("%Y-%m-%d"),
        "target_currency": target_currency
    }

    # Chamar a função de previsão do backend (v5)
    with st.spinner(f"Calculando previsão usando {modelo_selecionado.replace('_', ' ').title()} e convertendo para {target_currency}..."):
        try:
            resultado_previsao = prever_custo(input_data, model_type=modelo_selecionado)

            custo_total = resultado_previsao.get("custo_total_estimado", 0)
            custo_ton = resultado_previsao.get("custo_por_tonelada", 0)
            moeda = resultado_previsao.get("moeda", "N/A")
            composicao = resultado_previsao.get("composicao_custos", {})
            modelo_usado = resultado_previsao.get("modelo_utilizado", "N/A")
            metricas = resultado_previsao.get("metricas_modelo", {})
            periodo = resultado_previsao.get("periodo_previsao", "N/A")
            cambio_aplicado = resultado_previsao.get("taxa_cambio_aplicada", {})

            st.subheader(f"Estimativa de Custo Total para o Período: {periodo}")
            col1_met, col2_met = st.columns(2)
            with col1_met:
                st.metric(label=f"Custo Logístico Total ({moeda}) - Modelo: {modelo_usado}", value=f"{custo_total:,.2f}")
            with col2_met:
                st.metric(label=f"Custo por Tonelada ({moeda})", value=f"{custo_ton:,.2f}")

            # Exibir taxa de câmbio aplicada
            if cambio_aplicado:
                cambio_key = list(cambio_aplicado.keys())[0]
                cambio_value = list(cambio_aplicado.values())[0]
                st.caption(f"Taxa de câmbio aplicada (tempo real): 1 {cambio_key.split('_')[0]} = {cambio_value:.4f} {cambio_key.split('_')[-1]}")

            # Exibir métricas do modelo (ainda são do modelo base em BRL)
            if metricas and isinstance(metricas, dict):
                 st.markdown("**Métricas do Modelo Base (em BRL):**")
                 cols_metrics = st.columns(len(metricas))
                 i = 0
                 for key, value in metricas.items():
                     with cols_metrics[i]:
                         try:
                             st.metric(label=f"{key.upper()}", value=f"{float(value):.2f}")
                         except (ValueError, TypeError):
                             st.metric(label=f"{key.upper()}", value=str(value))
                     i += 1

            st.subheader(f"Composição Estimada dos Custos ({moeda})")
            if composicao:
                df_composicao = pd.DataFrame(list(composicao.items()), columns=["Componente", f"Custo ({moeda})"])
                fig_pie = px.pie(df_composicao, values=f"Custo ({moeda})", names="Componente",
                                 title="Distribuição Percentual dos Custos", hole=0.3)
                fig_pie.update_traces(textposition="inside", textinfo="percent+label")
                fig_bar = px.bar(df_composicao, x="Componente", y=f"Custo ({moeda})",
                                 title="Valor Absoluto por Componente de Custo", text_auto=".2s")
                fig_bar.update_traces(texttemplate=f"%{{value:,.0f}} {moeda}", textposition="outside")

                col1_chart, col2_chart = st.columns(2)
                with col1_chart:
                    st.plotly_chart(fig_pie, use_container_width=True)
                with col2_chart:
                    st.plotly_chart(fig_bar, use_container_width=True)

                st.dataframe(df_composicao.style.format({f"Custo ({moeda})": "{:,.2f}"}))

                @st.cache_data
                def convert_df_to_csv(df):
                    return df.to_csv(index=False).encode("utf-8")

                csv_composicao = convert_df_to_csv(df_composicao)
                periodo_cleaned = periodo.replace(" ", "").replace("/", "-")
                modelo_cleaned = modelo_usado.replace(" ", "_")
                nome_arquivo = f"composicao_custos_{porto_origem}_{porto_destino}_{periodo_cleaned}_{modelo_cleaned}_{moeda}_v5.csv"
                st.download_button(
                    label=f"Download Composição de Custos ({moeda}) (.csv)",
                    data=csv_composicao,
                    file_name=nome_arquivo,
                    mime="text/csv",
                )
            else:
                st.warning("Não foi possível obter a composição dos custos.")

            st.subheader("Cenários Comparativos (Em Desenvolvimento)")
            st.info("Funcionalidade para comparar diferentes modais, rotas ou prazos será adicionada futuramente.")

        except FileNotFoundError as fnf_error:
            st.error(f"Erro: Arquivo do modelo não encontrado ({fnf_error}). Execute o script `backend_structure_v5.py` primeiro para treinar e salvar os modelos.")
        except ValueError as val_error:
            st.error(f"Erro nos dados de entrada ou configuração: {val_error}")
        except Exception as e:
            st.error(f"Ocorreu um erro inesperado durante a previsão: {e}")
            st.exception(e)

elif not executar_previsao and valid_dates:
    st.info("Ajuste os parâmetros na barra lateral e clique em 'Calcular Custo Logístico Estimado' para ver a previsão.")
elif not valid_dates:
    st.warning("Ajuste as datas de início e fim na barra lateral.")

st.divider()
st.markdown("Desenvolvido como MVP (v5) para o projeto ISIS IA.")

