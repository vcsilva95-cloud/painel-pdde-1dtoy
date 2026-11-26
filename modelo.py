"""PDDE didática da UFJF aplicada ao caso 1dtoy."""

import json
from datetime import date
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import linprog


def carregar_1dtoy(pasta=None):
    """Lê os JSONs e CSVs do caso 1dtoy."""
    raiz = Path(pasta) if pasta is not None else Path(__file__).resolve().parent

    def localizar(nome, subpasta=""):
        for base in (raiz, raiz / "1dtoy"):
            for caminho in (base / nome, base / subpasta / nome):
                if caminho.is_file():
                    return caminho
        raise FileNotFoundError(f"Arquivo não encontrado no caso 1dtoy: {nome}")

    def ler_json(nome, subpasta=""):
        caminho = localizar(nome, subpasta)
        return json.loads(caminho.read_text(encoding="utf-8-sig"))

    def ler_csv(nome):
        caminho = localizar(nome, "scenarios")
        primeira_linha = caminho.read_text(encoding="utf-8-sig").splitlines()[0]
        if ";" in primeira_linha:
            return pd.read_csv(caminho, sep=";", decimal=",", encoding="utf-8-sig")
        return pd.read_csv(caminho, encoding="utf-8-sig")

    def por_estagio(tabela, entidade, identificador):
        dados = tabela.loc[tabela[entidade] == identificador]
        return dados.set_index("stage_id", verify_integrity=True)

    def faixa(item, minimo, maximo):
        return float(item.get(minimo, 0.0)), item.get(maximo)

    temporal = ler_json("stages.json")
    estagios = sorted(temporal["stages"], key=lambda e: e["start_date"])
    if not estagios:
        raise ValueError("O arquivo stages.json não contém estágios.")

    hidros = ler_json("hydros.json", "system")["hydros"]
    termicas = ler_json("thermals.json", "system")["thermals"]
    barras = ler_json("buses.json", "system")["buses"]

    if (len(hidros), len(termicas), len(barras)) != (1, 2, 1):
        raise ValueError("Este modelo é para o 1dtoy: uma UHE, duas UTEs e uma barra.")
    if any(len(e["blocks"]) != 1 for e in estagios):
        raise ValueError("Este modelo aceita um único bloco de carga por estágio.")

    hidro, barra = hidros[0], barras[0]
    geracao = hidro["generation"]
    reservatorio = hidro["reservoir"]
    defluencia = hidro["outflow"]

    penalidades = ler_json("penalties.json")
    pen_h = penalidades["hydro"]
    pen_b = penalidades["bus"]
    inicial = ler_json("initial_conditions.json")

    volume_inicial = next(
        float(v["value_hm3"])
        for v in inicial["storage"]
        if v["hydro_id"] == hidro["id"]
    )

    producao = ler_json("hydro_production_models.json", "system")
    modelo_h = next(
        m for m in producao["production_models"] if m["hydro_id"] == hidro["id"]
    )

    afluencias = por_estagio(
        ler_csv("inflow_seasonal_stats.csv"), "hydro_id", hidro["id"]
    )
    cargas = por_estagio(
        ler_csv("load_seasonal_stats.csv"), "bus_id", barra["id"]
    )

    config = ler_json("config.json")
    treino = config.get("training", {})
    regras = treino.get("stopping_rules", [])
    n_iter = next(
        (int(r["limit"]) for r in regras if r.get("type") == "iteration_limit"),
        128,
    )

    data_base = date.fromisoformat(estagios[0]["start_date"])
    taxa = float(temporal.get("policy_graph", {}).get("annual_discount_rate", 0))
    etapas = []

    for e in estagios:
        i = e["id"]
        faixas = [
            f for f in modelo_h["stage_ranges"]
            if f["start_stage_id"] <= i
            and (f.get("end_stage_id") is None or i <= f["end_stage_id"])
        ]
        if len(faixas) != 1:
            raise ValueError(f"O estágio {i} deve ter uma faixa de produtividade.")

        rho = float(faixas[0]["productivity_mw_per_m3s"])
        if rho <= 0:
            raise ValueError(f"Produtividade inválida no estágio {i}: {rho}")

        dias = (date.fromisoformat(e["start_date"]) - data_base).days
        etapas.append({
            "id": i,
            "data": e["start_date"],
            "horas": float(e["blocks"][0]["hours"]),
            "aberturas": int(e["num_openings"]),
            "rho": rho,
            "desconto": (1 + taxa) ** (-dias / 365),
            "afl_media": float(afluencias.at[i, "mean_m3s"]),
            "afl_desvio": float(afluencias.at[i, "std_m3s"]),
            "carga_media": float(cargas.at[i, "mean_mw"]),
            "carga_desvio": float(cargas.at[i, "std_mw"]),
        })

    return {
        "etapas": etapas,
        "volume_inicial": volume_inicial,
        "volume": faixa(reservatorio, "min_storage_hm3", "max_storage_hm3"),
        "turbinamento": faixa(geracao, "min_turbined_m3s", "max_turbined_m3s"),
        "geracao_hidro": faixa(geracao, "min_generation_mw", "max_generation_mw"),
        "defluencia": faixa(defluencia, "min_outflow_m3s", "max_outflow_m3s"),
        "termicas": termicas,
        "deficit": barra.get("deficit_segments") or pen_b["deficit_segments"],
        "custo_excesso": float(barra.get("excess_cost", pen_b["excess_cost"])),
        "pen_h": pen_h,
        "n_iter": n_iter,
        "n_forward": int(treino.get("selection", {}).get("forward_passes", 1)),
        "n_sim": int(
            config.get("simulation", {}).get("selection", {}).get("num_scenarios", 100)
        ),
        "seed": int(treino.get("scenario_source", {}).get("seed", 42)),
    }


def _resolver_estagio(caso, etapa, volume, afluencia, carga, cortes):
    """Resolve o PL do estágio e retorna o dual do balanço hídrico."""
    horas = etapa["horas"]
    k = 0.0036 * horas  # Converte vazão média em m³/s para volume em hm³.
    rho = etapa["rho"]
    pen = caso["pen_h"]
    ute1, ute2 = caso["termicas"]
    segmentos = caso["deficit"]
    n = 9 + len(segmentos)

    # x = [turbinamento, vertimento, UTE1, UTE2, volume final, FCF,
    #      excesso, folga defluência mínima, folga defluência máxima,
    #      déficit por faixa]
    custos = np.zeros(n)
    custos[:9] = [
        pen["turbined_cost"],
        pen["spillage_cost"],
        ute1["cost_per_mwh"],
        ute2["cost_per_mwh"],
        0,
        0,
        caso["custo_excesso"],
        pen["outflow_violation_below_cost"],
        pen["outflow_violation_above_cost"],
    ]
    custos[9:] = [s["cost"] for s in segmentos]
    custos *= horas * etapa["desconto"]

    objetivo = custos.copy()
    objetivo[5] = 1.0

    qmin, qmax = caso["turbinamento"]
    gmin, gmax = caso["geracao_hidro"]
    qmin = max(qmin, gmin / rho)
    if gmax is not None:
        qmax = gmax / rho if qmax is None else min(qmax, gmax / rho)

    limites = [
        (qmin, qmax),
        (0, None),
        (ute1["generation"].get("min_mw", 0), ute1["generation"]["max_mw"]),
        (ute2["generation"].get("min_mw", 0), ute2["generation"]["max_mw"]),
        caso["volume"],
        (0, None),
        (0, None),
        (0, None),
        (0, None),
    ] + [(0, s.get("depth_mw")) for s in segmentos]

    igualdades = np.zeros((2, n))

    # Balanço elétrico: geração + déficit - excesso = carga.
    igualdades[0, [0, 2, 3, 6]] = [rho, 1, 1, -1]
    igualdades[0, 9:] = 1

    # Balanço hídrico: volume final + k × (turbinamento + vertimento)
    #                  = volume inicial + k × afluência.
    igualdades[1, [0, 1, 4]] = [k, k, 1]

    a_ub, b_ub = [], []

    def restricao(coeficientes, rhs):
        linha = np.zeros(n)
        for posicao, valor in coeficientes.items():
            linha[posicao] = valor
        a_ub.append(linha)
        b_ub.append(rhs)

    def_min, def_max = caso["defluencia"]
    restricao({0: -1, 1: -1, 7: -1}, -def_min)
    if def_max is not None:
        restricao({0: 1, 1: 1, 8: -1}, def_max)

    # Cortes de Benders: FCF >= intercepto + inclinação × volume final.
    for inclinacao, intercepto in cortes:
        restricao({4: inclinacao, 5: -1}, -intercepto)

    sol = linprog(
        objetivo,
        A_ub=np.asarray(a_ub),
        b_ub=np.asarray(b_ub),
        A_eq=igualdades,
        b_eq=[carga, volume + k * afluencia],
        bounds=limites,
        method="highs",
    )
    if not sol.success:
        raise RuntimeError(f"Falha no PL do estágio {etapa['id']}: {sol.message}")

    x = sol.x
    return {
        "objetivo": float(sol.fun),
        "dual_volume": float(sol.eqlin.marginals[1]),
        "custo_estagio": float(custos @ x),
        "custo_futuro_estimado": float(x[5]),
        "volume_inicial_hm3": volume,
        "volume_final_hm3": float(x[4]),
        "afluencia_m3s": afluencia,
        "turbinamento_m3s": float(x[0]),
        "vertimento_m3s": float(x[1]),
        "geracao_hidro_mw": float(rho * x[0]),
        "ute1_mw": float(x[2]),
        "ute2_mw": float(x[3]),
        "deficit_mw": float(x[9:].sum()),
        "carga_mw": carga,
    }


def executar_pdde(
    pasta=None, *, n_iter=None, n_forward=None, n_sim=None,
    seed=None, n_aberturas=None,
):
    """Treina a política por PDDE e simula com uma amostra separada."""
    caso = carregar_1dtoy(pasta)
    etapas = caso["etapas"]
    T = len(etapas)

    n_iter = int(caso["n_iter"] if n_iter is None else n_iter)
    n_forward = int(caso["n_forward"] if n_forward is None else n_forward)
    n_sim = int(caso["n_sim"] if n_sim is None else n_sim)
    seed = int(caso["seed"] if seed is None else seed)

    if min(n_iter, n_forward, n_sim) < 1:
        raise ValueError("Iterações, trajetórias forward e simulações devem ser positivas.")

    sementes = np.random.SeedSequence(seed).spawn(3)
    rng_ab, rng_fw, rng_sim = [np.random.default_rng(s) for s in sementes]

    def sortear(etapa, n, rng):
        # Hipótese didática: sorteios independentes por estágio.
        # Valores negativos são ajustados para zero. Não se estima PAR(p).
        afl = rng.normal(etapa["afl_media"], etapa["afl_desvio"], n)
        carga = rng.normal(etapa["carga_media"], etapa["carga_desvio"], n)
        return np.maximum(afl, 0), np.maximum(carga, 0)

    inicio = perf_counter()
    aberturas = []

    for e in etapas:
        n = e["aberturas"] if n_aberturas is None else int(n_aberturas)
        if n < 1:
            raise ValueError("O número de aberturas deve ser positivo.")
        afls, cargas = sortear(e, n, rng_ab)
        aberturas.append(list(zip(afls.tolist(), cargas.tolist())))

    # cortes[t] aproxima o custo esperado a partir do início do estágio t.
    # cortes[T] fica vazio porque não há custo futuro após o último estágio.
    cortes = [[] for _ in range(T + 1)]
    historico = []
    n_pl_treino = 0

    def resolver(t, volume, abertura):
        afl, carga = abertura
        return _resolver_estagio(caso, etapas[t], volume, afl, carga, cortes[t + 1])

    for it in range(1, n_iter + 1):
        estados = [[] for _ in etapas]
        custos_forward = []

        # FORWARD: sorteia caminhos e guarda os volumes visitados.
        for _ in range(n_forward):
            volume = caso["volume_inicial"]
            custo = 0.0

            for t, opcoes in enumerate(aberturas):
                estados[t].append(volume)
                r = resolver(t, volume, opcoes[rng_fw.integers(len(opcoes))])
                n_pl_treino += 1
                custo += r["custo_estagio"]
                volume = r["volume_final_hm3"]

            custos_forward.append(custo)

        # BACKWARD: para cada volume visitado, resolve todas as aberturas.
        for t in range(T - 1, 0, -1):
            for volume in estados[t]:
                respostas = [resolver(t, volume, a) for a in aberturas[t]]
                n_pl_treino += len(respostas)

                valor = float(np.mean([r["objetivo"] for r in respostas]))
                beta = float(np.mean([r["dual_volume"] for r in respostas]))
                cortes[t].append((beta, valor - beta * volume))

        # ZINF para a distribuição discreta usada no treinamento.
        raiz = [resolver(0, caso["volume_inicial"], a) for a in aberturas[0]]
        n_pl_treino += len(raiz)
        historico.append({
            "iteracao": it,
            "limite_inferior": float(np.mean([r["objetivo"] for r in raiz])),
            "custo_forward_medio": float(np.mean(custos_forward)),
        })

    tempo_treino = perf_counter() - inicio
    inicio_sim = perf_counter()
    resultados = []
    aberturas_sim = [sortear(e, n_sim, rng_sim) for e in etapas]

    # SIMULAÇÃO: aplica a política treinada a trajetórias novas.
    for s in range(n_sim):
        volume = caso["volume_inicial"]

        for t, e in enumerate(etapas):
            afls, cargas = aberturas_sim[t]
            r = resolver(t, volume, (float(afls[s]), float(cargas[s])))
            volume = r["volume_final_hm3"]
            resultados.append({
                "simulacao": s + 1,
                "estagio": e["id"],
                "data": e["data"],
                **r,
            })

    tempo_sim = perf_counter() - inicio_sim
    simulacao = pd.DataFrame(resultados)

    return {
        "nome": "1dtoy | PDDE UFJF",
        "convergencia": pd.DataFrame(historico),
        "simulacao": simulacao,
        "aberturas": pd.DataFrame([
            {
                "estagio": etapas[t]["id"],
                "abertura": j,
                "afluencia_m3s": afl,
                "carga_mw": carga,
            }
            for t, opcoes in enumerate(aberturas)
            for j, (afl, carga) in enumerate(opcoes, start=1)
        ]),
        "cortes": cortes,
        "n_iter": n_iter,
        "n_forward": n_forward,
        "n_trajetorias_treino": n_iter * n_forward,
        "n_sim": n_sim,
        "n_pl_treino": n_pl_treino,
        "n_pl_simulacao": len(resultados),
        "tempo_treino_s": tempo_treino,
        "tempo_simulacao_s": tempo_sim,
        "custo_medio_total": float(
            simulacao.groupby("simulacao")["custo_estagio"].sum().mean()
        ),
    }
