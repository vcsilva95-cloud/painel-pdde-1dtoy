import json
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="1dtoy | PDDE UFJF e Cobre",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { color-scheme: light; }
    .stApp, [data-testid="stAppViewContainer"],
    [data-testid="stHeader"], [data-testid="stSidebar"],
    [data-testid="stSidebarContent"], [data-testid="stToolbar"],
    [data-testid="stFileUploaderDropzone"], [data-testid="stAlert"],
    [data-testid="stExpander"] details, [data-testid="stExpander"] summary,
    [data-testid="stCode"], pre, code, input, textarea, select,
    [data-baseweb="input"], [data-baseweb="base-input"],
    [data-baseweb="select"] > div, [data-baseweb="popover"] > div,
    [role="listbox"], [role="option"], [role="menu"],
    .stApp button {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3,
    .stApp li, [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"], [data-testid="stCaptionContainer"] {
        color: #000000 !important;
    }
    .stApp button { border-color: #c4cbd2; }
    .stApp button:hover, [role="option"]:hover {
        background-color: #eef5fc !important;
    }
    .stApp button:focus-visible, .stApp input:focus-visible {
        outline: 2px solid #1662a5 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

RAIZ = Path(__file__).resolve().parent
COBRE = shutil.which("cobre")
CORES = ["#1662a5", "#c65f20", "#358353", "#a54277"]

# Arquivos do 1dtoy usados pela linha de comando do Cobre.
# O modelo Python continua lendo seus CSVs normalmente.
ARQUIVOS_COBRE = [
    "config.json",
    "initial_conditions.json",
    "penalties.json",
    "stages.json",
    "system/buses.json",
    "system/hydros.json",
    "system/thermals.json",
    "system/lines.json",
    "system/hydro_production_models.json",
    "scenarios/inflow_seasonal_stats.parquet",
    "scenarios/load_seasonal_stats.parquet",
]


def pasta_caso():
    for pasta in (RAIZ, RAIZ / "1dtoy"):
        if (pasta / "stages.json").is_file():
            return pasta
    return RAIZ


def localizar(pasta, relativo):
    caminho = pasta / relativo
    if caminho.is_file():
        return caminho
    caminho = pasta / Path(relativo).name
    return caminho if caminho.is_file() else None


def ler_json(origem):
    dados = json.loads(origem.read().decode("utf-8-sig"))
    if not isinstance(dados, dict):
        raise ValueError("O arquivo JSON deve conter um objeto.")
    return dados


def json_local(caminho):
    with caminho.open("rb") as arquivo:
        return ler_json(arquivo)


def exibir_grafico(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#000000"),
        hoverlabel=dict(bgcolor="#ffffff", font_color="#000000"),
        legend_title_text="",
        margin=dict(l=25, r=25, t=55, b=35),
    )
    fig.update_xaxes(tickfont=dict(color="#000000"))
    fig.update_yaxes(tickfont=dict(color="#000000"))
    st.plotly_chart(fig, theme=None, use_container_width=True)


def imprimir_valor(valor):
    if valor is None:
        return "Não disponível"
    if isinstance(valor, float):
        return f"{valor:,.3f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return str(valor)


def resumo_cobre(treino, simulacao=None):
    if treino.get("software") != "cobre":
        raise ValueError("Selecione o training/metadata.json de uma execução do Cobre.")
    if (
        treino.get("status") != "complete"
        or treino.get("convergence", {}).get("termination_reason") == "error"
    ):
        raise ValueError("O treinamento do Cobre não foi concluído com sucesso.")

    simulacao = simulacao or {}
    if simulacao and "scenarios" not in simulacao:
        raise ValueError("O segundo JSON deve ser o simulation/metadata.json.")

    iteracoes = treino.get("iterations", {}).get("completed")
    forward = treino.get("configuration", {}).get("forward_passes")
    cenarios = simulacao.get("scenarios", {})
    return {
        "versao": treino.get("software_version"),
        "tempo_treino_s": treino.get("duration_seconds"),
        "n_iter": iteracoes,
        "n_forward": forward,
        "n_trajetorias_treino": (
            iteracoes * forward
            if isinstance(iteracoes, int) and isinstance(forward, int)
            else None
        ),
        "n_sim": cenarios.get("completed"),
        "n_sim_solicitadas": cenarios.get("requested"),
        "n_sim_falhas": cenarios.get("failed"),
        "n_pl_treino": treino.get("solve_stats", {}).get("total_lp_solves"),
    }


def rodar_cobre(n_iter, n_forward, n_sim, seed, n_aberturas):
    """Executa o binário oficial em uma cópia temporária do caso."""
    with TemporaryDirectory(prefix="1dtoy_") as pasta_tmp:
        base = Path(pasta_tmp)
        destino = base / "caso"
        saida = base / "resultados"
        origem = pasta_caso()

        for relativo in ARQUIVOS_COBRE:
            arquivo = localizar(origem, relativo)
            if arquivo is None:
                raise FileNotFoundError(f"Arquivo necessário para o Cobre: {relativo}")
            alvo = destino / relativo
            alvo.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(arquivo, alvo)

        cfg = json_local(destino / "config.json")
        treino = cfg["training"]
        if treino.get("selection", {}).get("method") != "sampled":
            raise ValueError("Para esta comparação, configure o treinamento como sampled.")
        treino["selection"]["forward_passes"] = n_forward
        treino["stopping_rules"] = [{"type": "iteration_limit", "limit": n_iter}]
        treino.setdefault("scenario_source", {})["seed"] = seed

        sim = cfg.setdefault("simulation", {})
        sim["enabled"] = True
        sim["selection"] = {"method": "sampled", "num_scenarios": n_sim}

        caminho_cfg = destino / "config.json"
        caminho_cfg.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

        if n_aberturas is not None:
            caminho_etapas = destino / "stages.json"
            etapas = json_local(caminho_etapas)
            for etapa in etapas["stages"]:
                etapa["num_openings"] = n_aberturas
            caminho_etapas.write_text(json.dumps(etapas, indent=2), encoding="utf-8")

        processo = subprocess.run(
            [
                COBRE, "run", str(destino), "--output", str(saida),
                "--threads", "1", "--comm-backend", "local",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=240,
            check=False,
        )
        if processo.returncode != 0:
            detalhe = (processo.stderr or processo.stdout).strip()
            raise RuntimeError(detalhe[-2500:] or "O Cobre terminou com erro.")

        meta_t = json_local(saida / "training" / "metadata.json")
        meta_s = json_local(saida / "simulation" / "metadata.json")
        resultado = resumo_cobre(meta_t, meta_s)
        for chave, relativo in {
            "convergencia": "training/convergence.parquet",
            "custos": "simulation/scenario_summary.parquet",
        }.items():
            caminho = saida / relativo
            if caminho.is_file():
                resultado[chave] = pd.read_parquet(caminho)
        resultado["origem"] = "Executado no servidor deste painel"
        return resultado


st.title("1dtoy | Comparação de PDDE")
st.caption("Despacho hidrotérmico com uma UHE, duas UTEs e uma barra.")

# O import fica protegido para a aba Cobre continuar disponível
# mesmo se houver erro de sintaxe ou dependência no modelo.py.
try:
    from modelo import carregar_1dtoy, executar_pdde

    caso = carregar_1dtoy(pasta_caso())
    erro_ufjf = None
except Exception as erro:
    caso = {}
    erro_ufjf = str(erro)

with st.sidebar:
    st.header("Parâmetros do estudo")
    with st.form("parametros"):
        n_iter = int(st.number_input(
            "Iterações de treinamento", min_value=1,
            value=int(caso.get("n_iter", 128)), step=1,
        ))
        n_forward = int(st.number_input(
            "Trajetórias forward por iteração", min_value=1,
            value=int(caso.get("n_forward", 1)), step=1,
        ))
        n_sim = int(st.number_input(
            "Simulações finais", min_value=1,
            value=int(caso.get("n_sim", 100)), step=1,
        ))
        seed = int(st.number_input(
            "Semente", min_value=0, value=int(caso.get("seed", 42)), step=1,
        ))
        manter_aberturas = st.checkbox("Usar aberturas do stages.json", value=True)
        aberturas = int(st.number_input(
            "Aberturas por estágio, se substituir", min_value=1, value=10, step=1
        ))
        executar_ufjf = st.form_submit_button("Executar PDDE UFJF")
        executar_oficial = st.form_submit_button(
            "Executar Cobre oficial", disabled=COBRE is None
        )

    st.caption("As entradas só são aplicadas ao clicar em um botão de execução.")

n_aberturas = None if manter_aberturas else aberturas

if executar_ufjf:
    if erro_ufjf is None:
        try:
            with st.spinner("Treinando a política e simulando o 1dtoy..."):
                st.session_state["ufjf_resultado"] = executar_pdde(
                    pasta_caso(), n_iter=n_iter, n_forward=n_forward, n_sim=n_sim,
                    seed=seed, n_aberturas=n_aberturas,
                )
        except Exception as erro:
            st.error(f"Não foi possível executar a PDDE UFJF: {erro}")

if executar_oficial:
    try:
        with st.spinner("Executando o Cobre oficial..."):
            st.session_state["cobre_resultado"] = rodar_cobre(
                n_iter, n_forward, n_sim, seed, n_aberturas
            )
    except subprocess.TimeoutExpired:
        st.error("O Cobre excedeu o tempo limite. Reduza o tamanho do estudo.")
    except Exception as erro:
        st.error(f"Não foi possível executar o Cobre: {erro}")

aba_ufjf, aba_cobre, aba_comparacao = st.tabs(
    ["1dtoy | PDDE UFJF", "1dtoy | PDDE Cobre", "Comparação"]
)

with aba_ufjf:
    st.subheader("PDDE UFJF")
    st.caption("Aberturas fixas no treinamento, forward amostrado e cortes no backward.")

    if erro_ufjf is not None:
        st.error(f"Não foi possível carregar o modelo.py ou os dados do 1dtoy: {erro_ufjf}")

    ufjf = st.session_state.get("ufjf_resultado")
    if ufjf is None:
        st.info("Selecione os parâmetros e clique em Executar PDDE UFJF.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Treinamento (s)", imprimir_valor(ufjf["tempo_treino_s"]))
        b.metric("Simulação (s)", imprimir_valor(ufjf["tempo_simulacao_s"]))
        c.metric("Trajetórias de treinamento", ufjf["n_trajetorias_treino"])
        d.metric("Simulações finais", ufjf["n_sim"])

        conv = ufjf["convergencia"].rename(columns={
            "limite_inferior": "Limite inferior",
            "custo_forward_medio": "Custo médio forward",
        })
        exibir_grafico(px.line(
            conv, x="iteracao", y=["Limite inferior", "Custo médio forward"],
            title="Evolução do treinamento",
            labels={"iteracao": "Iteração", "value": "Custo descontado"},
            color_discrete_sequence=CORES,
        ))
        st.caption("O custo médio forward é uma estimativa amostral, não um limite superior exato.")

        sim_df = ufjf["simulacao"]
        campos = {
            "geracao_hidro_mw": "UHE",
            "ute1_mw": "UTE 1",
            "ute2_mw": "UTE 2",
            "deficit_mw": "Déficit",
        }
        medias = sim_df.groupby("data", as_index=False)[
            ["volume_final_hm3", *campos]
        ].mean()
        esquerda, direita = st.columns(2)
        with esquerda:
            exibir_grafico(px.line(
                medias, x="data", y="volume_final_hm3", markers=True,
                title="Volume final médio por estágio",
                labels={"data": "Estágio", "volume_final_hm3": "Volume (hm³)"},
                color_discrete_sequence=CORES,
            ))
        with direita:
            exibir_grafico(px.bar(
                medias.rename(columns=campos), x="data", y=list(campos.values()),
                title="Despacho médio e déficit", barmode="stack",
                labels={"data": "Estágio", "value": "Potência (MW)"},
                color_discrete_sequence=CORES,
            ))

        with st.expander("Aberturas utilizadas no treinamento"):
            st.dataframe(ufjf["aberturas"], hide_index=True, use_container_width=True)
        with st.expander("Resultados médios por estágio"):
            st.dataframe(medias.round(3), hide_index=True, use_container_width=True)
        st.download_button(
            "Baixar simulações UFJF em CSV",
            data=sim_df.to_csv(index=False).encode("utf-8-sig"),
            file_name="1dtoy_simulacoes_ufjf.csv",
            mime="text/csv",
        )

with aba_cobre:
    st.subheader("Cobre | Execução oficial")
    if COBRE is None:
        st.info(
            "O executável cobre não está instalado neste servidor. "
            "Você pode importar os arquivos de uma execução real abaixo."
        )
    else:
        st.caption(
            "Para executar aqui, coloque os Parquet originais de entrada no GitHub "
            "em scenarios/. Os CSVs continuam disponíveis para o modelo UFJF."
        )

    with st.expander("Importar resultados de uma execução oficial"):
        with st.form("importar_cobre"):
            arq_t = st.file_uploader("Do Cobre: training/metadata.json", type=["json"])
            arq_s = st.file_uploader("Do Cobre: simulation/metadata.json", type=["json"])
            arq_c = st.file_uploader(
                "Opcional: training/convergence.parquet", type=["parquet"]
            )
            arq_r = st.file_uploader(
                "Opcional: simulation/scenario_summary.parquet", type=["parquet"]
            )
            importar = st.form_submit_button("Carregar resultados do Cobre")

        if importar:
            try:
                if arq_t is None or arq_s is None:
                    raise ValueError("Selecione os dois arquivos metadata.json.")
                r = resumo_cobre(ler_json(arq_t), ler_json(arq_s))
                if arq_c is not None:
                    r["convergencia"] = pd.read_parquet(arq_c)
                if arq_r is not None:
                    r["custos"] = pd.read_parquet(arq_r)
                r["origem"] = "Arquivos importados de uma execução externa"
                st.session_state["cobre_resultado"] = r
            except Exception as erro:
                st.error(f"Não foi possível importar os resultados: {erro}")

    cobre = st.session_state.get("cobre_resultado")
    if cobre is not None:
        st.caption(f"{cobre['origem']} | Versão: {cobre['versao'] or 'não informada'}")
        a, b, c = st.columns(3)
        a.metric("Treinamento (s)", imprimir_valor(cobre["tempo_treino_s"]))
        b.metric("Iterações concluídas", imprimir_valor(cobre["n_iter"]))
        c.metric("Simulações concluídas", imprimir_valor(cobre["n_sim"]))
        if cobre["n_sim_falhas"]:
            st.warning(f"Simulações com falha: {cobre['n_sim_falhas']}.")

        convergencia = cobre.get("convergencia", pd.DataFrame())
        if {"iteration", "lower_bound"}.issubset(convergencia.columns):
            exibir_grafico(px.line(
                convergencia, x="iteration", y="lower_bound",
                title="Limite inferior do Cobre por iteração",
                labels={"iteration": "Iteração", "lower_bound": "Limite inferior"},
                color_discrete_sequence=CORES,
            ))

        custos = cobre.get("custos", pd.DataFrame())
        if {"scenario_id", "discounted_immediate_cost"}.issubset(custos.columns):
            exibir_grafico(px.scatter(
                custos, x="scenario_id", y="discounted_immediate_cost",
                title="Custo descontado por cenário simulado",
                labels={
                    "scenario_id": "Cenário",
                    "discounted_immediate_cost": "Custo descontado (USD)",
                },
                color_discrete_sequence=CORES,
            ))

with aba_comparacao:
    st.subheader("Comparação das últimas execuções")
    ufjf = st.session_state.get("ufjf_resultado")
    cobre = st.session_state.get("cobre_resultado")

    if ufjf is None or cobre is None:
        st.info("Execute a PDDE UFJF e execute ou importe os resultados do Cobre.")
    else:
        linhas = []
        for rotulo, chave in [
            ("Treinamento (s)", "tempo_treino_s"),
            ("Iterações concluídas", "n_iter"),
            ("Trajetórias forward por iteração", "n_forward"),
            ("Trajetórias de treinamento", "n_trajetorias_treino"),
            ("Simulações finais concluídas", "n_sim"),
            ("PLs resolvidos no treinamento", "n_pl_treino"),
        ]:
            u, c = ufjf.get(chave), cobre.get(chave)
            delta = c - u if c is not None and u is not None else None
            linhas.append({
                "Indicador": rotulo,
                "PDDE UFJF": imprimir_valor(u),
                "Cobre oficial": imprimir_valor(c),
                    "Diferença (Cobre − UFJF)": imprimir_valor(delta),
            })

        st.dataframe(pd.DataFrame(linhas), hide_index=True, use_container_width=True)
        st.caption(f"Origem do resultado Cobre: {cobre['origem']}.")

    st.markdown(
        """
        **PDDE UFJF:** implementação didática em Python/SciPy. Sorteia aberturas
        independentes por estágio, fixa essas aberturas durante o treinamento
        e constrói cortes da função de custo futuro no backward.

        **Cobre:** execução oficial em Rust. Usa a configuração de cenários,
        treinamento e simulação do caso. A mesma semente não garante as
        mesmas trajetórias nos dois programas.

        **Como ler a diferença de tempo:** a tabela compara apenas o tempo
        registrado para treinamento, não o tempo total da interface.
        Se as execuções vieram de máquinas diferentes, ou se o esforço
        computacional e os cenários não foram equivalentes, a diferença
        é descritiva e não demonstra ganho de desempenho.
        """
    )
