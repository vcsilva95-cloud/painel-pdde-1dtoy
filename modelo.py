"""PDDE didatica para o caso 1dtoy e o app Streamlit existente."""

import json
from datetime import date
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import linprog

# Parametros de rede da versao anterior do painel, embutidos sem novos JSONs.
VOLUME = (0.0, 1000.0)  # hm3
Q_MAX = 50.0            # m3/s
G_H_MAX = 50.0          # MW
DEFLUENCIA_MAX = 50.0   # m3/s
RHO = 1.0             # MW por (m3/s)
UTE_MAX = (15.0, 15.0)  # MW
UTE_CUSTO = (5.0, 10.0) # por MWh


def carregar_1dtoy(pasta=None):
    raiz = Path(pasta) if pasta is not None else Path(__file__).resolve().parent
    if not (raiz / "stages.json").is_file() and (raiz / "1dtoy" / "stages.json").is_file():
        raiz = raiz / "1dtoy"

    def ler_json(nome):
        return json.loads((raiz / nome).read_text(encoding="utf-8-sig"))

    def ler_stats(nome, entidade, media, desvio, ids, entidade_id=None):
        caminho = raiz / nome
        if not caminho.is_file():
            caminho = raiz / "scenarios" / nome
        with caminho.open(encoding="utf-8-sig") as arquivo:
            separador = ";" if ";" in arquivo.readline() else ","
        df = pd.read_csv(caminho, sep=separador, encoding="utf-8-sig")
        df.columns = df.columns.str.strip()
        colunas = [entidade, "stage_id", media, desvio]
        faltantes = [c for c in colunas if c not in df.columns]
        if faltantes:
            raise ValueError(f"{nome}: colunas ausentes: {', '.join(faltantes)}.")
        for c in colunas:
            df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="raise")
        if entidade_id is not None:
            df = df.loc[df[entidade] == entidade_id]
        if df[entidade].nunique() != 1:
            raise ValueError(f"{nome}: deve haver dados de uma unica entidade do 1dtoy.")
        df = df.set_index("stage_id", verify_integrity=True).reindex(ids)
        valores = df[[media, desvio]].to_numpy(dtype=float)
        if not np.isfinite(valores).all() or (valores[:, 1] < 0).any():
            raise ValueError(f"{nome}: confira os estagios e os desvios no CSV.")
        return valores

    temporal = ler_json("stages.json")
    estagios = sorted(temporal["stages"], key=lambda e: e["start_date"])
    if not estagios or any(len(e["blocks"]) != 1 for e in estagios):
        raise ValueError("O 1dtoy precisa de um bloco de carga por estagio.")
    ids = [e["id"] for e in estagios]
    if len(set(ids)) != len(ids):
        raise ValueError("Ha IDs de estagio duplicados em stages.json.")
    inicial, = ler_json("initial_conditions.json")["storage"]
    pen = ler_json("penalties.json")
    cfg = ler_json("config.json")
    tr = cfg.get("training", {})
    aflu = ler_stats("inflow_seasonal_stats.csv", "hydro_id", "mean_m3s", "std_m3s", ids, inicial["hydro_id"])
    carga = ler_stats("load_seasonal_stats.csv", "bus_id", "mean_mw", "std_mw", ids)
    base = date.fromisoformat(estagios[0]["start_date"])
    taxa = float(temporal.get("policy_graph", {}).get("annual_discount_rate", 0))
    etapas = []
    for i, e in enumerate(estagios):
        dias = (date.fromisoformat(e["start_date"]) - base).days
        etapas.append(dict(id=e["id"], data=e["start_date"], horas=float(e["blocks"][0]["hours"]),
                           aberturas=int(e["num_openings"]), desconto=(1 + taxa) ** (-dias / 365),
                           aflu=aflu[i], carga=carga[i]))
    seg = pen["bus"]["deficit_segments"]
    if not seg or seg[-1].get("depth_mw") is not None:
        raise ValueError("A ultima faixa de deficit em penalties.json deve ser ilimitada (null).")
    return dict(etapas=etapas, volume_inicial=float(inicial["value_hm3"]), pen=pen,
                n_iter=next((int(r["limit"]) for r in tr.get("stopping_rules", [])
                             if r.get("type") == "iteration_limit"), 128),
                n_forward=int(tr.get("selection", {}).get("forward_passes", 1)),
                n_sim=int(cfg.get("simulation", {}).get("selection", {}).get("num_scenarios", 100)),
                seed=int(tr.get("scenario_source", {}).get("seed") or 0)
                if "seed" in tr.get("scenario_source", {}) else 42)


def _resolver(caso, etapa, volume, abertura, cortes):
    aflu, carga = abertura
    k = 0.0036 * etapa["horas"]  # m3/s -> hm3 no estagio
    ph, pb = caso["pen"]["hydro"], caso["pen"]["bus"]
    seg = pb["deficit_segments"]

    # x: q, spill, UTE1, UTE2, Vfinal, FCF, excesso, folga defluencia, deficits.
    custos = np.array([ph["turbined_cost"], ph["spillage_cost"], *UTE_CUSTO, 0, 0,
                       pb["excess_cost"], ph["outflow_violation_above_cost"],
                       *[s["cost"] for s in seg]], dtype=float)
    custos *= etapa["horas"] * etapa["desconto"]
    objetivo = custos.copy()
    objetivo[5] = 1.0

    n = len(custos)
    eq = np.zeros((2, n))
    eq[0, [0, 2, 3, 6]] = [RHO, 1, 1, -1]
    eq[0, 8:] = 1
    eq[1, [0, 1, 4]] = [k, k, 1]

    ub = np.zeros((1 + len(cortes), n))
    ub[0, [0, 1, 7]] = [1, 1, -1]
    rhs = [DEFLUENCIA_MAX]
    for i, (beta, alfa) in enumerate(cortes, start=1):
        ub[i, [4, 5]] = [beta, -1]
        rhs.append(-alfa)

    limites = [(0, min(Q_MAX, G_H_MAX / RHO)), (0, None), (0, UTE_MAX[0]),
               (0, UTE_MAX[1]), VOLUME, (0, None), (0, None), (0, None)]
    limites += [(0, s.get("depth_mw")) for s in seg]

    sol = linprog(objetivo, A_ub=ub, b_ub=rhs, A_eq=eq,
                  b_eq=[carga, volume + k * aflu], bounds=limites, method="highs")
    if not sol.success:
        raise RuntimeError(f"Estagio {etapa['id']}: {sol.message}")

    x = sol.x
    return dict(objetivo=float(sol.fun), dual_volume=float(sol.eqlin.marginals[1]),
                custo_estagio=float(custos @ x), custo_futuro_estimado=float(x[5]),
                volume_inicial_hm3=volume, volume_final_hm3=float(x[4]),
                afluencia_m3s=aflu, carga_mw=carga, turbinamento_m3s=float(x[0]),
                vertimento_m3s=float(x[1]), geracao_hidro_mw=float(RHO * x[0]),
                ute1_mw=float(x[2]), ute2_mw=float(x[3]), deficit_mw=float(x[8:].sum()))


def executar_pdde(pasta=None, *, n_iter=None, n_forward=None, n_sim=None,
                  seed=None, n_aberturas=None):
    caso = carregar_1dtoy(pasta)
    etapas = caso["etapas"]
    T = len(etapas)
    n_iter = int(caso["n_iter"] if n_iter is None else n_iter)
    n_forward = int(caso["n_forward"] if n_forward is None else n_forward)
    n_sim = int(caso["n_sim"] if n_sim is None else n_sim)
    seed = int(caso["seed"] if seed is None else seed)
    quantidades = [e["aberturas"] if n_aberturas is None else int(n_aberturas) for e in etapas]
    if min(n_iter, n_forward, n_sim, *quantidades) < 1 or seed < 0:
        raise ValueError("Use contagens positivas e uma semente nao negativa.")
    rng_ab, rng_fw, rng_sim = [
        np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(3)
    ]

    def sortear(e, n, rng):
        # Amostras normais independentes; negativos sao ajustados para zero.
        a = np.maximum(0, rng.normal(*e["aflu"], size=n))
        d = np.maximum(0, rng.normal(*e["carga"], size=n))
        return list(zip(a.tolist(), d.tolist()))

    inicio = perf_counter()
    aberturas = [sortear(e, n, rng_ab) for e, n in zip(etapas, quantidades)]
    cortes = [[] for _ in range(T + 1)]
    historico = []
    n_pl_treino = 0

    def resolver(t, volume, abertura):
        return _resolver(caso, etapas[t], volume, abertura, cortes[t + 1])

    for it in range(1, n_iter + 1):
        estados = [[] for _ in etapas]
        custos_forward = []

        # FORWARD: sorteia trajetorias e guarda os volumes visitados.
        for _ in range(n_forward):
            volume, custo = caso["volume_inicial"], 0.0
            for t, opcoes in enumerate(aberturas):
                estados[t].append(volume)
                r = resolver(t, volume, opcoes[rng_fw.integers(len(opcoes))])
                n_pl_treino += 1
                volume = r["volume_final_hm3"]
                custo += r["custo_estagio"]
            custos_forward.append(custo)

        # BACKWARD: resolve todas as aberturas para cada volume visitado.
        for t in range(T - 1, 0, -1):
            for volume in estados[t]:
                respostas = [resolver(t, volume, a) for a in aberturas[t]]
                n_pl_treino += len(respostas)
                valor = float(np.mean([r["objetivo"] for r in respostas]))
                beta = float(np.mean([r["dual_volume"] for r in respostas]))
                cortes[t].append((beta, valor - beta * volume))

        raiz = [resolver(0, caso["volume_inicial"], a) for a in aberturas[0]]
        n_pl_treino += len(raiz)
        historico.append(dict(iteracao=it,
                              limite_inferior=float(np.mean([r["objetivo"] for r in raiz])),
                              custo_forward_medio=float(np.mean(custos_forward))))

    tempo_treino = perf_counter() - inicio
    inicio = perf_counter()
    amostras = [sortear(e, n_sim, rng_sim) for e in etapas]
    resultados = []
    for s in range(n_sim):
        volume = caso["volume_inicial"]
        for t, e in enumerate(etapas):
            r = resolver(t, volume, amostras[t][s])
            resultados.append(dict(simulacao=s+1, estagio=e["id"], data=e["data"], **r))
            volume = r["volume_final_hm3"]
    tempo_sim = perf_counter() - inicio

    simulacao = pd.DataFrame(resultados)
    tabela_ab = pd.DataFrame([
        dict(estagio=e["id"], abertura=j, afluencia_m3s=a, carga_mw=d)
        for e, opcoes in zip(etapas, aberturas)
        for j, (a, d) in enumerate(opcoes, start=1)
    ])
    return dict(nome="1dtoy | PDDE UFJF", convergencia=pd.DataFrame(historico),
                simulacao=simulacao, aberturas=tabela_ab, cortes=cortes, n_iter=n_iter,
                n_forward=n_forward, n_trajetorias_treino=n_iter*n_forward, n_sim=n_sim,
                n_pl_treino=n_pl_treino, n_pl_simulacao=len(resultados),
                tempo_treino_s=tempo_treino, tempo_simulacao_s=tempo_sim,
                tempo_s=tempo_treino+tempo_sim, custo_medio_total=float(
                    simulacao.groupby("simulacao")["custo_estagio"].sum().mean()))
