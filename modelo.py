"""PDDE didática da UFJF aplicada ao caso 1dtoy."""

import json
from datetime import date
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import linprog


def carregar_1dtoy(pasta=None):
    """Lê os dados do caso sem depender do diretório de execução do app."""
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
        # Aceita CSV com vírgula e CSV brasileiro com ponto e vírgula.
        primeira_linha = caminho.read_text(encoding="utf-8-sig").splitlines()[0]
        if ";" in primeira_linha:
            return pd.read_csv(caminho, sep=";", decimal=",", encoding="utf-8-sig")
        return pd.read_csv(caminho, encoding="utf-8-sig")

    def por_estagio(tabela, entidade, identificador):
        dados = tabela.loc[tabela[entidade] == identificador]
        return dados.set_index("stage_id"
