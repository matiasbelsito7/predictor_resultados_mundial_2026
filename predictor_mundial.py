"""
#predictor_mundial.py
====================
Módulo de predicción para el Mundial 2026.
Simula la fase de grupos y el bracket de eliminación completo.

Uso:
    #python predictor_mundial.py --features features_mundial.csv
"""

import argparse
import random
import warnings
import numpy as np
import pandas as pd
from itertools import combinations
from collections import defaultdict

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURE OFICIAL MUNDIAL 2026 — FASE DE GRUPOS
# ─────────────────────────────────────────────────────────────────────────────

GRUPOS = {
    "A": ["mexico",       "south korea",  "south africa", "czech republic"],
    "B": ["canada",       "bosnia and herzegovina", "qatar", "switzerland"],
    "C": ["brazil",       "morocco",      "haiti",        "scotland"],
    "D": ["united states","paraguay",     "australia",    "turkey"],
    "E": ["germany",      "curacao",      "ivory coast",  "ecuador"],
    "F": ["netherlands",  "japan",        "sweden",       "tunisia"],
    "G": ["belgium",      "egypt",        "iran",         "new zealand"],
    "H": ["spain",        "cape verde",   "saudi arabia", "uruguay"],
    "I": ["france",       "senegal",      "iraq",         "norway"],
    "J": ["argentina",    "algeria",      "austria",      "jordan"],
    "K": ["portugal",     "dr congo",     "uzbekistan",   "colombia"],
    "L": ["england",      "croatia",      "ghana",        "panama"],
}

# Fixture de fase de grupos (todos contra todos dentro del grupo)
# Se genera dinámicamente desde GRUPOS

# Ranking FIFA junio 2026 (aproximado, para peso_rival)
RANKING_FIFA = {
    "francia": 1, "france": 1,
    "españa": 2, "spain": 2,
    "argentina": 3,
    "inglaterra": 4, "england": 4,
    "portugal": 5,
    "brasil": 6, "brazil": 6,
    "países bajos": 7, "netherlands": 7,
    "marruecos": 8, "morocco": 8,
    "bélgica": 9, "belgium": 9,
    "alemania": 10, "germany": 10,
    "croacia": 11, "croatia": 11,
    "colombia": 13,
    "senegal": 14,
    "méxico": 15, "mexico": 15,
    "eeuu": 16, "usa": 16, "united states": 16,
    "uruguay": 17,
    "japón": 18, "japan": 18,
    "suiza": 19, "switzerland": 19,
    "dinamarca": 20, "denmark": 20,
    "irán": 21, "iran": 21,
    "turquía": 22, "turkey": 22,
    "ecuador": 23,
    "austria": 24,
    "república de corea": 25, "south korea": 25,
    "nigeria": 26,
    "australia": 27,
    "argelia": 28, "algeria": 28,
    "egipto": 29, "egypt": 29,
    "canadá": 30, "canada": 30,
    "noruega": 31, "norway": 31,
    "panamá": 33, "panama": 33,
    "costa de marfil": 34, "ivory coast": 34,
    "polonia": 35, "poland": 35,
    "gales": 37, "wales": 37,
    "suecia": 38, "sweden": 38,
    "paraguay": 40,
    "república checa": 41, "czechia": 41, "czech republic": 41,
    "escocia": 43, "scotland": 43,
    "túnez": 44, "tunisia": 44,
    "camerún": 45, "cameroon": 45,
    "rd del congo": 46, "dr congo": 46, "democratic republic of congo": 46,
    "eslovaquia": 48, "slovakia": 48,
    "uzbekistán": 50, "uzbekistan": 50,
    "qatar": 55,
    "sudáfrica": 60, "south africa": 60,
    "jordania": 63, "jordan": 63,
    "bosnia y herzegovina": 65, "bosnia and herzegovina": 65,
    "cabo verde": 69, "cape verde": 69,
    "ghana": 74,
    "curazao": 82, "curacao": 82,
    "haití": 83, "haiti": 83,
    "nueva zelanda": 85, "new zealand": 85
}

CONFEDERACION = {
    "france": "UEFA", "spain": "UEFA", "england": "UEFA", "portugal": "UEFA",
    "germany": "UEFA", "netherlands": "UEFA", "belgium": "UEFA", "croatia": "UEFA",
    "norway": "UEFA", "switzerland": "UEFA", "sweden": "UEFA", "turkey": "UEFA",
    "scotland": "UEFA", "austria": "UEFA", "bosnia and herzegovina": "UEFA",
    "czech republic": "UEFA", "iran": "AFC", "south korea": "AFC", "japan": "AFC",
    "saudi arabia": "AFC", "australia": "AFC", "iraq": "AFC", "uzbekistan": "AFC",
    "jordan": "AFC", "qatar": "AFC", "argentina": "CONMEBOL", "brazil": "CONMEBOL",
    "colombia": "CONMEBOL", "ecuador": "CONMEBOL", "paraguay": "CONMEBOL",
    "uruguay": "CONMEBOL", "united states": "CONCACAF", "mexico": "CONCACAF",
    "canada": "CONCACAF", "panama": "CONCACAF", "haiti": "CONCACAF",
    "curacao": "CONCACAF", "senegal": "CAF", "morocco": "CAF", "ivory coast": "CAF",
    "ghana": "CAF", "south africa": "CAF", "algeria": "CAF", "egypt": "CAF",
    "tunisia": "CAF", "dr congo": "CAF", "cape verde": "CAF",
    "new zealand": "OFC",
}


# ─────────────────────────────────────────────────────────────────────────────
# NORMALIZACIÓN Y CARGA DE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

# Mapeo de nombres en los CSV → nombres canónicos en GRUPOS
NOMBRE_CANONICAL = {
    # variantes con guión bajo o tilde
    "united_states": "united states",
    "south_korea": "south korea",
    "south_africa": "south africa",
    "czech_republic": "czech republic",
    "saudi_arabia": "saudi arabia",
    "ivory_coast": "ivory coast",
    "cape_verde": "cape verde",
    "new_zealand": "new zealand",
    "dr_congo": "dr congo",
    "bosnia_and_herzegovina": "bosnia and herzegovina",
    "bosnia_herzegovina": "bosnia and herzegovina",
    "united states": "united states",
    "cote divoire": "ivory coast",
    "cote_d_ivoire": "ivory coast",
    "türkiye": "turkey",
    "turkiye": "turkey",
    "czechia": "czech republic",
}

FEATURE_COLS_OFENSIVOS = [
    "xg_neto_ponderado", "xg_ofensivo_pond", "goles_xg_ratio",
    "big_chances_diff", "toques_area_rival_pond", "pases_3ro_tercio_pond",
    "xg_pond_delantero", "xg_pond_mediocampista",
    "toques_area_delanteros", "shots_pond_delanteros",
    "dribbles_pond_atac", "ratio_crecimiento_2t",
]

FEATURE_COLS_DEFENSIVOS = [
    "solidez_arquero", "goals_prevented_pond",
    "interceptions_pond", "clearances_pond", "tackles_pct_pond",
    "xg_pond_defensor", "pases_pct_defensores",
]

FEATURE_COLS_CONTROL = [
    "pts_ponderados", "posesion_promedio", "precision_pases",
    "indice_contraataque", "rating_prom_top11", "rating_std_top11",
    "concentracion_xg_top3", "disciplina_score", "consistencia_xg",
]

ALL_FEATURES = FEATURE_COLS_OFENSIVOS + FEATURE_COLS_DEFENSIVOS + FEATURE_COLS_CONTROL


def canonicalizar(nombre: str) -> str:
    s = str(nombre).lower().strip().replace("_", " ")
    return NOMBRE_CANONICAL.get(s, NOMBRE_CANONICAL.get(nombre.lower().replace(" ", "_"), s))


def cargar_features(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["seleccion"] = df["seleccion"].apply(canonicalizar)
    df = df.set_index("seleccion")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# FUERZA DE EQUIPO — ÍNDICE COMPUESTO
# ─────────────────────────────────────────────────────────────────────────────

def calcular_fuerza(df: pd.DataFrame) -> pd.Series:
    """
    Calcula un índice de fuerza [0,1] para cada selección
    combinando features ofensivos, defensivos y de control,
    con pesos diferenciados y penalización por NaN.
    """
    from sklearn.preprocessing import MinMaxScaler

    cols_disponibles = [c for c in ALL_FEATURES if c in df.columns]
    sub = df[cols_disponibles].copy()

    # ─────────────────────────────────────────────
    # 1. INVERSION DE MÉTRICAS (dirección correcta)
    # ─────────────────────────────────────────────
    if "disciplina_score" in sub.columns:
        sub["disciplina_score"] = -sub["disciplina_score"]

    if "consistencia_xg" in sub.columns:
        sub["consistencia_xg"] = -sub["consistencia_xg"]

    if "rating_std_top11" in sub.columns:
        sub["rating_std_top11"] = -sub["rating_std_top11"]

    # ─────────────────────────────────────────────
    # 2. IMPUTACIÓN BASE
    # ─────────────────────────────────────────────
    median = sub.median(numeric_only=True)
    imputed = sub.fillna(median)

    # máscara de valores existentes (para penalización)
    valid_mask = sub.notna().astype(float)

    # ─────────────────────────────────────────────
    # 3. ESCALADO
    # ─────────────────────────────────────────────
    scaler = MinMaxScaler()
    scaled = pd.DataFrame(
        scaler.fit_transform(imputed),
        index=sub.index,
        columns=sub.columns,
    )

    # ─────────────────────────────────────────────
    # 4. PESOS DIFERENCIADOS (feature-level)
    # ─────────────────────────────────────────────
    PESOS_FEATURES = {
        # OFENSIVOS
        "xg_neto_ponderado": 1.6,
        "xg_ofensivo_pond": 1.5,
        "goles_xg_ratio": 1.3,
        "big_chances_diff": 1.4,
        "toques_area_rival_pond": 1.2,
        "pases_3ro_tercio_pond": 1.0,
        "xg_pond_delantero": 1.4,
        "xg_pond_mediocampista": 1.1,
        "toques_area_delanteros": 1.2,
        "shots_pond_delanteros": 1.3,
        "dribbles_pond_atac": 1.1,
        "ratio_crecimiento_2t": 1.0,

        # DEFENSIVOS
        "solidez_arquero": 1.3,
        "goals_prevented_pond": 1.5,
        "interceptions_pond": 1.2,
        "clearances_pond": 1.1,
        "tackles_pct_pond": 1.1,
        "xg_pond_defensor": 1.3,
        "pases_pct_defensores": 1.0,

        # CONTROL
        "pts_ponderados": 1.4,
        "posesion_promedio": 1.1,
        "precision_pases": 1.2,
        "indice_contraataque": 1.2,
        "rating_prom_top11": 1.3,
        "rating_std_top11": 1.2,
        "concentracion_xg_top3": 1.3,
        "disciplina_score": 1.1,
        "consistencia_xg": 1.3,
    }

    pesos = {}
    for c in scaled.columns:
        pesos[c] = PESOS_FEATURES.get(c, 1.0)

    # normalizar pesos
    total_w = sum(pesos.values())
    for c in pesos:
        pesos[c] /= total_w

    # ─────────────────────────────────────────────
    # 5. SCORE BASE
    # ─────────────────────────────────────────────
    fuerza = sum(scaled[c] * pesos[c] for c in scaled.columns)

    # ─────────────────────────────────────────────
    # 6. PENALIZACIÓN POR NA (progresiva)
    # ─────────────────────────────────────────────
    missing_ratio = 1 - valid_mask.mean(axis=1)

    # penalización suave pero creciente
    penalty = np.exp(-1.5 * missing_ratio)

    fuerza = fuerza * penalty

    return fuerza.rename("fuerza")


# ─────────────────────────────────────────────────────────────────────────────
# PREDICCIÓN DE UN PARTIDO
# ─────────────────────────────────────────────────────────────────────────────

def predecir_partido(
    equipo_a: str,
    equipo_b: str,
    fuerza: pd.Series,
    ranking: dict = None,
    stochastic: bool = True,
    temperatura: float = 1.5,
) -> dict:
    """
    Predice el resultado de un partido entre equipo_a y equipo_b.

    Retorna:
        {
          "ganador": str | None,  # None = empate
          "prob_a": float,
          "prob_empate": float,
          "prob_b": float,
          "goles_a": int,
          "goles_b": int,
        }
    """
    def get_fuerza(equipo):
        canon = canonicalizar(equipo)
        if canon in fuerza.index:
            return fuerza[canon]
        # Fallback por ranking FIFA
        if ranking and canon in ranking:
            return max(0.0, 1.0 - ranking[canon] / 200)
        return 0.40  # valor neutro

    fa = get_fuerza(equipo_a)
    fb = get_fuerza(equipo_b)

    # ── Probabilidades base via softmax con temperatura ───────────────────────
    # Modelamos un sistema 3-resultado: W_a, D, W_b
    # La "energía" del empate decrece cuando hay diferencia grande
    diff = fa - fb
    logit_a = diff * 4.0          # escalado para rango razonable de prob
    logit_draw = -abs(diff) * 2.5  # empate menos probable si hay diferencia
    logit_b = -diff * 4.0

    def softmax3(a, d, b, T=1.0):
        exps = np.exp(np.array([a, d, b]) / T)
        return exps / exps.sum()

    prob_a, prob_d, prob_b = softmax3(logit_a, logit_draw, logit_b, temperatura)

    # Ajuste leve por ranking FIFA (si disponible)
    if ranking:
        ra = ranking.get(canonicalizar(equipo_a), 50)
        rb = ranking.get(canonicalizar(equipo_b), 50)
        rank_boost = (rb - ra) / 400  # pequeño boost para el mejor rankeado
        prob_a = np.clip(prob_a + rank_boost, 0.05, 0.85)
        prob_b = np.clip(prob_b - rank_boost, 0.05, 0.85)
        total = prob_a + prob_b + prob_d
        prob_a /= total; prob_b /= total; prob_d /= total

    # ── Sorteo de resultado (o determinístico) ────────────────────────────────
    if stochastic:
        r = random.random()
        if r < prob_a:
            resultado = "A"
        elif r < prob_a + prob_d:
            resultado = "D"
        else:
            resultado = "B"
    else:
        resultado = "A" if prob_a > prob_b and prob_a > prob_d else \
                    "B" if prob_b > prob_a and prob_b > prob_d else "D"

    # ── Goles esperados (modelo de Poisson simplificado) ─────────────────────
    xg_a = max(0.3, fa * 3.0 - fb * 0.8 + 0.5)
    xg_b = max(0.3, fb * 3.0 - fa * 0.8 + 0.5)

    if stochastic:
        goles_a = np.random.poisson(xg_a)
        goles_b = np.random.poisson(xg_b)
        # Forzar coherencia con resultado
        if resultado == "A" and goles_a <= goles_b:
            goles_a = goles_b + 1
        elif resultado == "B" and goles_b <= goles_a:
            goles_b = goles_a + 1
        elif resultado == "D":
            min_g = min(goles_a, goles_b)
            goles_a = goles_b = min_g
    else:
        goles_a = round(xg_a)
        goles_b = round(xg_b)
        if resultado == "A" and goles_a <= goles_b:
            goles_a = goles_b + 1
        elif resultado == "B" and goles_b <= goles_a:
            goles_b = goles_a + 1
        elif resultado == "D":
            g = round((xg_a + xg_b) / 2)
            goles_a = goles_b = g

    ganador = equipo_a if resultado == "A" else \
              equipo_b if resultado == "B" else None

    return {
        "ganador": ganador,
        "prob_a": round(prob_a, 3),
        "prob_empate": round(prob_d, 3),
        "prob_b": round(prob_b, 3),
        "goles_a": int(goles_a),
        "goles_b": int(goles_b),
        "resultado": resultado,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIMULACIÓN DE FASE DE GRUPOS 
# ─────────────────────────────────────────────────────────────────────────────

def simular_grupo(grupo: str, equipos: list, fuerza: pd.Series,
                   stochastic: bool = True) -> pd.DataFrame:
    """
    Simula los 6 partidos del grupo y devuelve la tabla de posiciones.
    """
    tabla = {e: {"pts": 0, "gf": 0, "gc": 0, "gd": 0, "pj": 0} for e in equipos}
    partidos = []

    for a, b in combinations(equipos, 2):
        res = predecir_partido(a, b, fuerza, RANKING_FIFA, stochastic)
        partidos.append({"local": a, "visitante": b, **res})

        ga, gb = res["goles_a"], res["goles_b"]

        tabla[a]["gf"] += ga
        tabla[a]["gc"] += gb
        tabla[b]["gf"] += gb
        tabla[b]["gc"] += ga

        tabla[a]["gd"] += ga - gb
        tabla[b]["gd"] += gb - ga

        tabla[a]["pj"] += 1
        tabla[b]["pj"] += 1

        if res["resultado"] == "A":
            tabla[a]["pts"] += 3
        elif res["resultado"] == "B":
            tabla[b]["pts"] += 3
        else:
            tabla[a]["pts"] += 1
            tabla[b]["pts"] += 1

    df_tabla = pd.DataFrame(tabla).T

    # 🔥 desempates más robustos
    df_tabla = df_tabla.sort_values(
        ["pts", "gd", "gf", "pj"],
        ascending=False
    )

    df_tabla["grupo"] = grupo

    return df_tabla, partidos

def simular_fase_grupos(fuerza: pd.Series, stochastic: bool = True):
    """
    Simula todos los grupos. Devuelve:
    - tablas
    - partidos
    - clasificados (top 2 reales)
    - mejores terceros (top 8)
    """

    todas_tablas = []
    todos_partidos = []

    # 🔥 ahora sí consistente por grupo
    tablas_por_grupo = {}
    orden_grupos = list(GRUPOS.keys())

    for grupo, equipos in GRUPOS.items():
        tabla, partidos = simular_grupo(grupo, equipos, fuerza, stochastic)

        todas_tablas.append(tabla)
        todos_partidos.extend(partidos)
        tablas_por_grupo[grupo] = tabla

    # ─────────────────────────────────────────────
    # CLASIFICADOS REALES (TOP 2 POR GRUPO)
    # ─────────────────────────────────────────────
    clasificados = {}

    for grupo, tabla in tablas_por_grupo.items():
        clasificados[grupo] = tabla.index[:2].tolist()

    # ─────────────────────────────────────────────
    # MEJORES TERCEROS (TOP 8 GLOBAL)
    # ─────────────────────────────────────────────
    terceros = []

    for grupo, tabla in tablas_por_grupo.items():
        if len(tabla) >= 3:
            equipo = tabla.index[2]
            fila = tabla.loc[equipo].copy()
            fila["equipo"] = equipo
            fila["grupo_origen"] = grupo
            terceros.append(fila)

    df_terceros = pd.DataFrame(terceros)

    df_terceros = df_terceros.sort_values(
        ["pts", "gd", "gf"],
        ascending=False
    )

    mejores_terceros = df_terceros["equipo"].head(8).tolist()

    return pd.concat(todas_tablas), todos_partidos, clasificados, mejores_terceros

# ─────────────────────────────────────────────────────────────────────────────
# ELIMINATORIAS — ROUND OF 32 → FINAL
# ─────────────────────────────────────────────────────────────────────────────

def partido_eliminatorio(equipo_a: str, equipo_b: str, fuerza: pd.Series,
                          stochastic: bool = True) -> str:
    """Partido eliminatorio: si empate, decide penales (50/50 + pequeña ventaja al mejor)."""
    res = predecir_partido(equipo_a, equipo_b, fuerza, RANKING_FIFA, stochastic)
    if res["ganador"]:
        return res["ganador"]
    # Penales: probabilidad ligeramente sesgada por fuerza
    fa = fuerza.get(canonicalizar(equipo_a), 0.5)
    fb = fuerza.get(canonicalizar(equipo_b), 0.5)
    p_a_penales = 0.45 + 0.1 * (fa - fb)  # rango ~[0.40, 0.55]
    p_a_penales = np.clip(p_a_penales, 0.35, 0.65)
    return equipo_a if (random.random() < p_a_penales) else equipo_b


def armar_bracket_r32(clasificados: dict, mejores_terceros: list) -> list:
    """
    Construye los partidos 73-88 del Mundial 2026.
    NOTA:
    Los terceros todavía se asignan por ranking (t[0]..t[7]).
    Más adelante debería reemplazarse por la tabla oficial FIFA.
    """

    g = clasificados

    def p1(grp):
        return g[grp][0] if len(g[grp]) > 0 else "TBD"

    def p2(grp):
        return g[grp][1] if len(g[grp]) > 1 else "TBD"

    t = mejores_terceros

    cruces = [

        # Partido 73
        (p2("A"), p2("B")),

        # Partido 74
        (p1("E"), t[0] if len(t) > 0 else "TBD"),

        # Partido 75
        (p1("F"), p2("C")),

        # Partido 76
        (p1("C"), p2("F")),

        # Partido 77
        (p1("I"), t[1] if len(t) > 1 else "TBD"),

        # Partido 78
        (p2("E"), p2("I")),

        # Partido 79
        (p1("A"), t[2] if len(t) > 2 else "TBD"),

        # Partido 80
        (p1("L"), t[3] if len(t) > 3 else "TBD"),

        # Partido 81
        (p1("D"), t[4] if len(t) > 4 else "TBD"),

        # Partido 82
        (p1("G"), t[5] if len(t) > 5 else "TBD"),

        # Partido 83
        (p2("K"), p2("L")),

        # Partido 84
        (p1("H"), p2("J")),

        # Partido 85
        (p1("B"), t[6] if len(t) > 6 else "TBD"),

        # Partido 86
        (p1("J"), p2("H")),

        # Partido 87
        (p1("K"), t[7] if len(t) > 7 else "TBD"),

        # Partido 88
        (p2("D"), p2("G")),
    ]

    # Chequeo útil para debug
    equipos = [e for partido in cruces for e in partido]

    if len(equipos) != len(set(equipos)):
        repetidos = [e for e in set(equipos) if equipos.count(e) > 1]
        print("\n[WARNING] Equipos repetidos en Round of 32:")
        print(repetidos)

    return cruces


def simular_eliminatorias(
    clasificados: dict,
    mejores_terceros: list,
    fuerza: pd.Series,
    stochastic: bool = True
) -> dict:
    """
    Simula desde Round of 32 hasta la Final usando el bracket FIFA 2026.
    """

    cruces_r32 = armar_bracket_r32(clasificados, mejores_terceros)
    bracket = {}

    def simular_ronda(cruces, nombre):
        ganadores = []
        bracket[nombre] = []

        for a, b in cruces:
            ganador = partido_eliminatorio(
                a, b, fuerza, stochastic
            )

            bracket[nombre].append({
                "a": a,
                "b": b,
                "ganador": ganador
            })

            ganadores.append(ganador)

        return ganadores

    # ------------------------------------------------------------------
    # PARTIDOS 73-88
    # ------------------------------------------------------------------

    g_r32 = simular_ronda(cruces_r32, "Round of 32")

    # índice 0 = partido 73
    # índice 1 = partido 74
    # ...
    # índice 15 = partido 88

    # ------------------------------------------------------------------
    # PARTIDOS 89-96
    # ------------------------------------------------------------------

    cruces_r16 = [
        (g_r32[1],  g_r32[4]),   # 74 vs 77
        (g_r32[0],  g_r32[2]),   # 73 vs 75
        (g_r32[3],  g_r32[5]),   # 76 vs 78
        (g_r32[6],  g_r32[7]),   # 79 vs 80
        (g_r32[10], g_r32[11]),  # 83 vs 84
        (g_r32[8],  g_r32[9]),   # 81 vs 82
        (g_r32[13], g_r32[15]),  # 86 vs 88
        (g_r32[12], g_r32[14]),  # 85 vs 87
    ]

    g_r16 = simular_ronda(cruces_r16, "Round of 16")

    # ------------------------------------------------------------------
    # PARTIDOS 97-100
    # ------------------------------------------------------------------

    cruces_qf = [
        (g_r16[0], g_r16[1]),   # 89 vs 90
        (g_r16[4], g_r16[5]),   # 93 vs 94
        (g_r16[2], g_r16[3]),   # 91 vs 92
        (g_r16[6], g_r16[7]),   # 95 vs 96
    ]

    g_qf = simular_ronda(cruces_qf, "Cuartos de Final")

    # ------------------------------------------------------------------
    # PARTIDOS 101-102
    # ------------------------------------------------------------------

    cruces_sf = [
        (g_qf[0], g_qf[1]),   # 97 vs 98
        (g_qf[2], g_qf[3]),   # 99 vs 100
    ]

    g_sf = simular_ronda(cruces_sf, "Semifinales")

    # ------------------------------------------------------------------
    # PARTIDO 104
    # ------------------------------------------------------------------

    g_final = simular_ronda(
        [(g_sf[0], g_sf[1])],
        "Final"
    )

    bracket["Campeon"] = g_final[0]

    return bracket


# ─────────────────────────────────────────────────────────────────────────────
# SIMULACIÓN MONTE CARLO
# ─────────────────────────────────────────────────────────────────────────────

def simular_torneo(fuerza: pd.Series, stochastic: bool = True) -> dict:
    """Simula un torneo completo. Retorna el campeón."""
    _, _, clasificados, mejores_terceros = simular_fase_grupos(fuerza, stochastic)
    bracket = simular_eliminatorias(clasificados, mejores_terceros, fuerza, stochastic)
    return bracket


def monte_carlo(fuerza: pd.Series, n: int = 5000) -> pd.DataFrame:
    """
    Probabilidades de alcanzar cada ronda.
    """

    equipos_todos = [e for grupo in GRUPOS.values() for e in grupo]
    conteos = defaultdict(lambda: defaultdict(int))

    for _ in range(n):

        _, _, clasificados, terceros = simular_fase_grupos(
            fuerza,
            stochastic=True
        )

        bracket = simular_eliminatorias(
            clasificados,
            terceros,
            fuerza,
            stochastic=True
        )

        # ----------------------------------------------------
        # Equipos que alcanzaron cada ronda
        # ----------------------------------------------------

        r32 = set()
        r16 = set()
        qf = set()
        sf = set()
        final = set()

        for p in bracket.get("Round of 32", []):
            r32.add(p["a"])
            r32.add(p["b"])

        for p in bracket.get("Round of 16", []):
            r16.add(p["a"])
            r16.add(p["b"])

        for p in bracket.get("Cuartos de Final", []):
            qf.add(p["a"])
            qf.add(p["b"])

        for p in bracket.get("Semifinales", []):
            sf.add(p["a"])
            sf.add(p["b"])

        for p in bracket.get("Final", []):
            final.add(p["a"])
            final.add(p["b"])

        # ----------------------------------------------------
        # Acumular
        # ----------------------------------------------------

        for e in r32:
            conteos[e]["r32"] += 1

        for e in r16:
            conteos[e]["r16"] += 1

        for e in qf:
            conteos[e]["qf"] += 1

        for e in sf:
            conteos[e]["sf"] += 1

        for e in final:
            conteos[e]["final"] += 1

        campeon = bracket.get("Campeon")
        if campeon:
            conteos[campeon]["campeon"] += 1

    rows = []

    for equipo in equipos_todos:

        c = conteos[equipo]

        rows.append({
            "equipo": equipo,
            "grupo": next(g for g, es in GRUPOS.items() if equipo in es),

            "prob_r32": round(c["r32"] / n * 100, 1),
            "prob_r16": round(c["r16"] / n * 100, 1),
            "prob_qf": round(c["qf"] / n * 100, 1),
            "prob_sf": round(c["sf"] / n * 100, 1),
            "prob_final": round(c["final"] / n * 100, 1),
            "prob_campeon": round(c["campeon"] / n * 100, 1),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("prob_campeon", ascending=False)
        .reset_index(drop=True)
    )


# ─────────────────────────────────────────────────────────────────────────────
# PREDICCIÓN DETERMINÍSTICA (MEJOR ESTIMACIÓN ÚNICA)
# ─────────────────────────────────────────────────────────────────────────────

def predecir_torneo_deterministico(fuerza: pd.Series) -> dict:
    """
    Usa la predicción determinística (sin aleatoriedad) para dar
    el resultado más probable de cada partido.
    """
    bracket = simular_torneo(fuerza, stochastic=False)
    return bracket


# ─────────────────────────────────────────────────────────────────────────────
# IMPRIMIR RESULTADOS
# ─────────────────────────────────────────────────────────────────────────────

def imprimir_bracket(bracket: dict):
    separador = "─" * 55
    rondas = ["Round of 32", "Round of 16", "Cuartos de Final", "Semifinales", "Final"]
    for ronda in rondas:
        if ronda not in bracket:
            continue
        print(f"\n{'':>2}{'═'*53}")
        print(f"{'':>2}  {ronda.upper()}")
        print(f"{'':>2}{'═'*53}")
        for partido in bracket[ronda]:
            a, b, g = partido["a"].upper(), partido["b"].upper(), partido["ganador"].upper()
            marcador = "→" if partido["ganador"] == partido["a"] else "←" if partido["ganador"] == partido["b"] else "?"
            print(f"  {a:<25} {marcador}  {b:<25}   AVANZA: {g}")
    print(f"\n{'':>2}{'★'*53}")
    print(f"  🏆  CAMPEÓN: {bracket.get('Campeon','?').upper()}")
    print(f"{'':>2}{'★'*53}\n")


def imprimir_monte_carlo(df_mc: pd.DataFrame, top_n: int = 20):
    print(f"\n{'═'*75}")
    print(f"  PROBABILIDADES MONTE CARLO — TOP {top_n}")
    print(f"{'═'*75}")
    header = f"  {'EQUIPO':<22} {'GRP':^4} {'R32':>5} {'R16':>5} {'QF':>5} {'SF':>5} {'FINAL':>6} {'CAMPEÓN':>8}"
    print(header)
    print(f"  {'─'*70}")
    for _, row in df_mc.head(top_n).iterrows():
        print(
            f"  {row['equipo'].upper():<22} {row['grupo']:^4} "
            f"{row['prob_r32']:>4.1f}% {row['prob_r16']:>4.1f}% "
            f"{row['prob_qf']:>4.1f}% {row['prob_sf']:>4.1f}% "
            f"{row['prob_final']:>5.1f}% {row['prob_campeon']:>7.1f}%"
        )
    print(f"{'═'*75}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predictor del Mundial 2026")
    parser.add_argument("--features",     default="features_mundial.csv")
    parser.add_argument("--simulations",  type=int, default=5000,
                        help="Iteraciones Monte Carlo")
    parser.add_argument("--output_mc",    default="probabilidades_mundial.csv")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 60)
    print("  PREDICTOR — FIFA WORLD CUP 2026")
    print("=" * 60)

    # 1. Cargar features
    print(f"\n[1] Cargando features desde: {args.features}")
    df_features = cargar_features(args.features)
    print(f"    Selecciones con datos: {len(df_features)}")

    # 2. Calcular fuerza
    print("[2] Calculando índice de fuerza...")
    try:
        from sklearn.preprocessing import MinMaxScaler
        fuerza = calcular_fuerza(df_features)
    except ImportError:
        print("    [!] sklearn no disponible. Usando pts_ponderados como proxy.")
        col = "pts_ponderados" if "pts_ponderados" in df_features.columns else df_features.columns[0]
        fuerza = df_features[col].fillna(df_features[col].median())
        fuerza = (fuerza - fuerza.min()) / (fuerza.max() - fuerza.min())
        fuerza.name = "fuerza"

    print(f"\n  Top 10 selecciones por índice de fuerza:")
    for equipo, f in fuerza.sort_values(ascending=False).head(10).items():
        print(f"    {equipo:<25} {f:.3f}")

    # 3. Predicción determinística
    print("\n[3] Predicción determinística (resultado más probable)...")
    bracket_det = predecir_torneo_deterministico(fuerza)
    imprimir_bracket(bracket_det)

    # 4. Monte Carlo
    print(f"[4] Simulando {args.simulations:,} torneos (Monte Carlo)...")
    df_mc = monte_carlo(fuerza, n=args.simulations)
    imprimir_monte_carlo(df_mc, top_n=20)

    df_mc.to_csv(args.output_mc, index=False)
    print(f"\n✓ Probabilidades guardadas en: {args.output_mc}")
