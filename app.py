import io
import json
import math
import shutil
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="1dtoy | PDDE UFJF e COBRE",
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
    [data-testid="stForm"], [data-testid="stMetric"],
    [data-testid="stFileUploader"], [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploaderFile"], [data-testid="stFileUploaderFileIcon"],
    [data-testid="stAlert"], [data-testid="stAlert"] *,
    [data-testid="stNotification"], [data-testid="stNotification"] *,
    [data-testid="stExpander"] details, [data-testid="stExpander"] summary,
    input, textarea, select, [data-baseweb="input"],
    [data-baseweb="base-input"], [data-baseweb="select"] > div,
    [data-baseweb="popover"] > div, [role="option"], .stApp button {
        background-color: #ffffff !important;
        color: #000000 !important;
    }
    .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3,
    .stApp small, .stApp li, [data-testid="stMetricValue"],
    [data-testid="stMetricLabel"], [data-testid="stFileUploaderFile"] * {
        color: #000000 !important;
    }
    [data-testid="stAlert"], [data-testid="stNotification"] {
        border: 1px solid #b8c2cc !important;
    }
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

ENTRADAS_COBRE = (
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
)


def pasta_caso():
    if (RAIZ / "stages.json").is_file():
        return RAIZ
    if (RAIZ / "1dtoy" / "stages.json").is_file():
        return RAIZ / "1dtoy"
    return RAIZ


def objeto(valor):
    return valor if isinstance(valor, dict) else {}


def ler_json(conteudo, rotulo):
    try:
        dados = json.loads(conteudo.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as erro:
        raise ValueError(f"{rotulo}: o arquivo não é um JSON válido.") from erro
    if not isinstance(dados, dict):
        raise ValueError(f"{rotulo}: era esperado um objeto JSON.")
    return dados


def inteiro(valor):
    return valor if type(valor) is int and valor >= 0 else None


def segundos(valor):
    if type(valor) in (int, float) and math.isfinite(valor) and valor >= 0:
        return float(valor)
    return None


def formatar(valor):
    if valor is None:
        return "Não informado"
    if isinstance(valor, float):
        return f"{valor:,.3f}".replace(",", "_").replace(".", ",").replace("_", ".")
    return str(valor)


def grafico(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#000000"),
        hoverlabel=dict(bgcolor="#ffffff", font_color="#000000"),
        legend_title_text="",
        margin=dict(l=20, r=20, t=55, b=35),
    )
    st.plotly_chart(fig, theme=None, use_container_width=True)


def ler_parquet(opcional, nome, colunas, avisos):
    if opcional is None:
        return pd.DataFrame()
    try:
        dados = pd.read_parquet(io.BytesIO(opcional), engine="pyarrow")
    except Exception as erro:
        avisos.append(f"{nome} não pôde ser lido: {erro}")
        return pd.DataFrame()

    faltam = set(colunas) - set(dados.columns)
    if faltam:
        avisos.append(f"{nome} não tem as colunas: {', '.join(sorted(faltam))}.")
        return pd.DataFrame()

    return dados


def resumo_cobre(treino, simulacao, conv, custos, origem, avisos):
    convergencia = objeto(treino.get("convergence"))
    status_treino = str(treino.get("status", "")).strip().lower()
    status_sim = str(simulacao.get("status", "")).strip().lower()
    motivo = str(convergencia.get("termination_reason", "")).strip().lower()

    if motivo == "error" or status_treino in {"error", "failed"}:
        raise ValueError("O JSON informa que o treinamento terminou com erro.")
    if status_treino == "partial":
        avisos.append("O treinamento foi marcado como parcial.")
    if status_sim in {"partial", "error", "failed"}:
        avisos.append(f"A simulação foi marcada como '{status_sim}'.")

    # Identifica o treinamento pelos dados, sem exigir software == "cobre".
    n_iter = inteiro(objeto(treino.get("iterations")).get("completed"))
    n_iter_parquet = None
    if not conv.empty:
        n_iter_parquet = int(conv["iteration"].dropna().nunique())

    if n_iter is None:
        tem_indicio_treino = any(
            chave in treino
            for chave in ("iterations", "convergence", "bounds", "solve_stats", "configuration")
        )
        if not tem_indicio_treino or not n_iter_parquet:
            campos = ", ".join(sorted(treino)) or "(nenhum)"
            raise ValueError(
                "O primeiro JSON não pôde ser identificado como resultado de "
                "treinamento. Confira se ele veio de training/metadata.json. "
                f"Campos encontrados: {campos}."
            )
        n_iter = n_iter_parquet
        avisos.append("Iterações contadas no arquivo convergence.parquet.")
    elif n_iter_parquet is not None and n_iter != n_iter_parquet:
        avisos.append(
            "A contagem de iterações do JSON difere da do Parquet. "
            "Confira se ambos vieram da mesma rodada."
        )

    cenarios = objeto(simulacao.get("scenarios"))
    if not cenarios and custos.empty:
        campos = ", ".join(sorted(simulacao)) or "(nenhum)"
        raise ValueError(
            "O segundo JSON não contém o bloco scenarios e não há um "
            "scenario_summary.parquet válido para complementar a leitura. "
            "Confira se o JSON veio de simulation/metadata.json. "
            f"Campos encontrados: {campos}."
        )

    n_sim = inteiro(cenarios.get("completed"))
    n_sim_parquet = None
    if not custos.empty:
        n_sim_parquet = int(custos["scenario_id"].dropna().nunique())

    fonte_sim = "simulation/metadata.json"
    if n_sim is None and n_sim_parquet is not None:
        n_sim = n_sim_parquet
        fonte_sim = "cenários presentes em scenario_summary.parquet"
        avisos.append(
            "A quantidade de simulações mostra os cenários presentes no Parquet, "
            "pois o JSON não informa a contagem de concluídas."
        )
    elif n_sim is not None and n_sim_parquet is not None and n_sim != n_sim_parquet:
        avisos.append(
            "O número de simulações no JSON difere do número de cenários no Parquet. "
            "Confira se ambos vieram da mesma rodada."
        )

    forward = inteiro(objeto(treino.get("configuration")).get("forward_passes"))

    return {
        "origem": origem,
        "software": treino.get("software"),
        "versao": treino.get("software_version"),
        "status_treino": status_treino,
        "status_sim": status_sim,
        "tempo_treino_s": segundos(treino.get("duration_seconds")),
        "tempo_simulacao_s": segundos(simulacao.get("duration_seconds")),
        "n_iter": n_iter,
        "n_forward": forward,
        "n_trajetorias_treino": n_iter * forward if forward is not None else None,
        "n_sim": n_sim,
        "fonte_sim": fonte_sim if n_sim is not None else None,
        "n_sim_falhas": inteiro(cenarios.get("failed")),
        "n_pl_treino": inteiro(objeto(treino.get("solve_stats")).get("total_lp_solves")),
        "convergencia": conv,
        "custos": custos,
        "avisos": avisos,
    }


def importar_cobre(treino_bytes, sim_bytes, conv_bytes, custos_bytes, origem):
    treino = ler_json(treino_bytes, "training/metadata.json")
    simulacao = ler_json(sim_bytes, "simulation/metadata.json")
    avisos = []
    conv = ler_parquet(
        conv_bytes, "convergence.parquet", ("iteration", "lower_bound"), avisos
    )
    custos = ler_parquet(
        custos_bytes,
        "scenario_summary.parquet",
        ("scenario_id", "discounted_immediate_cost"),
        avisos,
    )
    return resumo_cobre(treino, simulacao, conv, custos, origem, avisos)


def rodar_cobre(n_iter, n_forward, n_sim, seed, n_aberturas):
    if COBRE is None:
        raise RuntimeError("O executável cobre não está disponível neste servidor.")

    with TemporaryDirectory(prefix="cobre_1dtoy_") as tmp:
        caso_tmp = Path(tmp) / "caso"
        saida = Path(tmp) / "resultados"
        origem = pasta_caso()

        for relativo in ENTRADAS_COBRE:
            fonte = origem / relativo
            if not fonte.is_file():
                fonte = origem / Path(relativo).name
            if not fonte.is_file():
                raise FileNotFoundError(
                    f"Para executar o COBRE neste servidor, falta {relativo}. "
                    "Para apenas importar resultados, esse arquivo não é necessário."
                )

            destino = caso_tmp / relativo
            destino.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fonte, destino)

        config_path = caso_tmp / "config.json"
        config = ler_json(config_path.read_bytes(), "config.json")
        treino = config.setdefault("training", {})
        selecao = treino.setdefault("selection", {})
        if selecao.get("method") != "sampled":
            raise ValueError("O caso deve estar configurado com treinamento sampled.")
        selecao["forward_passes"] = n_forward
        treino["stopping_rules"] = [{"type": "iteration_limit", "limit": n_iter}]
        treino.setdefault("scenario_source", {})["seed"] = seed
        sim = config.setdefault("simulation", {})
        sim["enabled"] = True
        sim["selection"] = {"method": "sampled", "num_scenarios": n_sim}
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        if n_aberturas is not None:
            stages_path = caso_tmp / "stages.json"
            stages = ler_json(stages_path.read_bytes(), "stages.json")
            for etapa in stages["stages"]:
                etapa["num_openings"] = n_aberturas
            stages_path.write_text(json.dumps(stages, indent=2), encoding="utf-8")

        processo = subprocess.run(
            [COBRE, "run", str(caso_tmp), "--output", str(saida)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=240,
            check=False,
        )
        if processo.returncode != 0:
            detalhe = (processo.stderr or processo.stdout).strip()
            raise RuntimeError(detalhe[-2500:] or "O COBRE terminou com erro.")

        def opcional(nome):
            arquivo = saida / nome
            return arquivo.read_bytes() if arquivo.is_file() else None

        return importar_cobre(
            (saida / "training/metadata.json").read_bytes(),
            (saida / "simulation/metadata.json").read_bytes(),
            opcional("training/convergence.parquet"),
            opcional("simulation/scenario_summary.parquet"),
            "Executado no servidor do painel",
        )


st.title("1dtoy | Comparação de PDDE")
st.caption("Despacho hidrotérmico com uma UHE, duas UTEs e uma barra.")

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
        n_iter = st.number_input(
            "Iterações de treinamento", min_value=1,
            value=int(caso.get("n_iter", 128)), step=1,
        )
        n_forward = st.number_input(
            "Trajetórias forward por iteração", min_value=1,
            value=int(caso.get("n_forward", 1)), step=1,
        )
        n_sim = st.number_input(
            "Simulações finais", min_value=1,
            value=int(caso.get("n_sim", 100)), step=1,
        )
        seed = st.number_input(
            "Semente", min_value=0, value=int(caso.get("seed", 42)), step=1,
        )
        manter_aberturas = st.checkbox("Usar aberturas do stages.json", value=True)
        aberturas = st.number_input(
            "Aberturas por estágio, se substituir", min_value=1, value=10, step=1
        )
        executar_ufjf = st.form_submit_button("Executar PDDE UFJF")
        executar_oficial = st.form_submit_button(
            "Executar COBRE oficial", disabled=COBRE is None
        )

    st.caption("Esses parâmetros não alteram os resultados importados do COBRE.")

n_aberturas = None if manter_aberturas else int(aberturas)

if executar_ufjf and erro_ufjf is None:
    try:
        with st.spinner("Treinando a política e simulando o 1dtoy..."):
            st.session_state["ufjf_resultado"] = executar_pdde(
                pasta_caso(),
                n_iter=int(n_iter),
                n_forward=int(n_forward),
                n_sim=int(n_sim),
                seed=int(seed),
                n_aberturas=n_aberturas,
            )
    except Exception as erro:
        st.session_state.pop("ufjf_resultado", None)
        st.error(f"Não foi possível executar a PDDE UFJF: {erro}")

if executar_oficial:
    try:
        with st.spinner("Executando o COBRE oficial..."):
            st.session_state["cobre_resultado"] = rodar_cobre(
                int(n_iter), int(n_forward), int(n_sim), int(seed), n_aberturas
            )
    except subprocess.TimeoutExpired:
        st.session_state.pop("cobre_resultado", None)
        st.error("O COBRE excedeu o tempo limite desta execução.")
    except Exception as erro:
        st.session_state.pop("cobre_resultado", None)
        st.error(f"Não foi possível executar o COBRE: {erro}")

aba_ufjf, aba_cobre, aba_comp = st.tabs(
    ["1dtoy | PDDE UFJF", "1dtoy | PDDE COBRE", "Comparação"]
)

with aba_ufjf:
    st.subheader("PDDE UFJF")
    st.caption("Aberturas fixas no treinamento, forward amostrado e cortes no backward.")
    ufjf = st.session_state.get("ufjf_resultado")

    if erro_ufjf is not None:
        st.error(f"Não foi possível carregar o modelo.py ou os dados do 1dtoy: {erro_ufjf}")
    elif ufjf is None:
        st.info("Clique em Executar PDDE UFJF na barra lateral.")
    else:
        a, b, c, d = st.columns(4)
        a.metric("Treinamento (s)", formatar(ufjf["tempo_treino_s"]))
        b.metric("Simulação (s)", formatar(ufjf["tempo_simulacao_s"]))
        c.metric("Trajetórias no treino", ufjf["n_trajetorias_treino"])
        d.metric("Simulações finais", ufjf["n_sim"])

        conv = ufjf["convergencia"].rename(columns={
            "limite_inferior": "Limite inferior",
            "custo_forward_medio": "Custo médio forward",
        })
        grafico(px.line(
            conv, x="iteracao", y=["Limite inferior", "Custo médio forward"],
            title="Evolução do treinamento",
            labels={"iteracao": "Iteração", "value": "Custo descontado"},
            color_discrete_sequence=CORES,
        ))
        st.caption("O custo médio forward é amostral, não um limite superior exato.")

        dados = ufjf["simulacao"]
        fontes = {
            "geracao_hidro_mw": "UHE",
            "ute1_mw": "UTE 1",
            "ute2_mw": "UTE 2",
            "deficit_mw": "Déficit",
        }
        medias = dados.groupby("data", as_index=False)[
            ["volume_final_hm3", *fontes]
        ].mean()

        esquerda, direita = st.columns(2)
        with esquerda:
            grafico(px.line(
                medias, x="data", y="volume_final_hm3", markers=True,
                title="Volume final médio por estágio",
                labels={"data": "Estágio", "volume_final_hm3": "Volume (hm³)"},
                color_discrete_sequence=CORES,
            ))
        with direita:
            grafico(px.bar(
                medias.rename(columns=fontes), x="data", y=list(fontes.values()),
                title="Despacho médio e déficit", barmode="stack",
                labels={"data": "Estágio", "value": "Potência (MW)"},
                color_discrete_sequence=CORES,
            ))

        with st.expander("Aberturas de treinamento"):
            st.dataframe(ufjf["aberturas"], hide_index=True, use_container_width=True)
        with st.expander("Resultados médios por estágio"):
            st.dataframe(medias.round(3), hide_index=True, use_container_width=True)

        st.download_button(
            "Baixar simulações UFJF em CSV",
            data=dados.to_csv(index=False).encode("utf-8-sig"),
            file_name="1dtoy_simulacoes_ufjf.csv",
            mime="text/csv",
        )

with aba_cobre:
    st.subheader("Resultados do COBRE")
    st.caption(
        "Use arquivos da mesma rodada. A importação não depende do executável COBRE."
    )

    with st.expander("Importar resultados de uma execução", expanded=True):
        with st.form("importar_cobre"):
            arq_t = st.file_uploader(
                "training/metadata.json", type=["json"], key="meta_treino"
            )
            arq_s = st.file_uploader(
                "simulation/metadata.json", type=["json"], key="meta_sim"
            )
            arq_c = st.file_uploader(
                "Opcional: training/convergence.parquet",
                type=["parquet"], key="convergencia_cobre",
            )
            arq_r = st.file_uploader(
                "Opcional: simulation/scenario_summary.parquet",
                type=["parquet"], key="resumo_cenarios",
            )
            importar = st.form_submit_button("Carregar resultados do COBRE")

        if importar:
            try:
                if arq_t is None or arq_s is None:
                    raise ValueError(
                        "Selecione os dois metadata.json, um da pasta training "
                        "e outro da pasta simulation."
                    )
                st.session_state["cobre_resultado"] = importar_cobre(
                    arq_t.getvalue(),
                    arq_s.getvalue(),
                    arq_c.getvalue() if arq_c is not None else None,
                    arq_r.getvalue() if arq_r is not None else None,
                    "Arquivos importados de uma execução externa",
                )
                st.success("Resultados do COBRE carregados.")
            except Exception as erro:
                st.session_state.pop("cobre_resultado", None)
                st.error(f"Não foi possível importar os resultados: {erro}")

    cobre = st.session_state.get("cobre_resultado")
    if cobre is not None:
        st.caption(f"{cobre['origem']} | Versão: {cobre['versao'] or 'não informada'}")
        if cobre["software"]:
            st.caption(f"Software informado no arquivo: {cobre['software']}.")
        for aviso in cobre["avisos"]:
            st.warning(aviso)
        if cobre["n_sim_falhas"]:
            st.warning(f"Simulações com falha: {cobre['n_sim_falhas']}.")

        a, b, c, d = st.columns(4)
        a.metric("Treinamento (s)", formatar(cobre["tempo_treino_s"]))
        b.metric("Simulação (s)", formatar(cobre["tempo_simulacao_s"]))
        c.metric("Iterações registradas", formatar(cobre["n_iter"]))
        d.metric("Simulações registradas", formatar(cobre["n_sim"]))
        if cobre["fonte_sim"]:
            st.caption(f"Fonte da contagem de simulações: {cobre['fonte_sim']}.")

        conv = cobre["convergencia"]
        if not conv.empty:
            grafico(px.line(
                conv.sort_values("iteration"), x="iteration", y="lower_bound",
                title="Limite inferior do COBRE por iteração",
                labels={"iteration": "Iteração", "lower_bound": "Limite inferior"},
                color_discrete_sequence=CORES,
            ))

        custos = cobre["custos"]
        if not custos.empty:
            grafico(px.scatter(
                custos, x="scenario_id", y="discounted_immediate_cost",
                title="Custo descontado por cenário",
                labels={
                    "scenario_id": "Cenário",
                    "discounted_immediate_cost": "Custo descontado (USD)",
                },
                color_discrete_sequence=CORES,
            ))

with aba_comp:
    st.subheader("Comparação das últimas execuções")
    ufjf = st.session_state.get("ufjf_resultado")
    cobre = st.session_state.get("cobre_resultado")

    if ufjf is None or cobre is None:
        st.info("Execute a PDDE UFJF e carregue os resultados do COBRE.")
    else:
        linhas = []
        for rotulo, chave in [
            ("Treinamento (s)", "tempo_treino_s"),
            ("Simulação (s)", "tempo_simulacao_s"),
            ("Iterações registradas", "n_iter"),
            ("Trajetórias forward por iteração", "n_forward"),
            ("Trajetórias no treinamento", "n_trajetorias_treino"),
            ("Simulações finais registradas", "n_sim"),
            ("PLs resolvidos no treinamento", "n_pl_treino"),
        ]:
            u = ufjf.get(chave)
            c = cobre.get(chave)
            diff = c - u if c is not None and u is not None else None
            linhas.append({
                "Indicador": rotulo,
                "PDDE UFJF": formatar(u),
                "COBRE": formatar(c),
                "Diferença (COBRE menos UFJF)": formatar(diff),
            })

        st.dataframe(pd.DataFrame(linhas), hide_index=True, use_container_width=True)
        st.caption(f"Origem do resultado COBRE: {cobre['origem']}.")
        if cobre["status_treino"] == "partial" or cobre["status_sim"] in {
            "partial", "error", "failed"
        }:
            st.warning("A rodada do COBRE não está completa. Leia as diferenças com cautela.")

    st.markdown(
        """
        **PDDE UFJF:** implementação didática em Python/SciPy, com aberturas
        fixas no treinamento e cortes de custo futuro no backward.

        **COBRE:** usa os resultados da execução real. A mesma semente não
        garante que os dois programas usem as mesmas trajetórias.

        **Diferença de tempo:** valores de ambientes, configurações ou amostras
        diferentes são descritivos. A diferença não comprova ganho de desempenho.
        Um tempo ausente não é tratado como zero.
        """
    )
