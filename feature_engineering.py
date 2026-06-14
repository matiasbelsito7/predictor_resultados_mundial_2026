"""
feature_engineering.py
=======================
Pipeline de ingeniería de atributos para el predictor del Mundial.
Procesa archivos {seleccion}_partidos.csv y {seleccion}_jugadores.csv.

Uso:
    python feature_engineering.py --data_dir /ruta/a/csvs
    python feature_engineering.py --data_dir /ruta/a/csvs --seleccion argentina
"""

import os
import re
import glob
import argparse
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PESOS
# ─────────────────────────────────────────────────────────────────────────────

# Peso por tipo de competición (extraído del string de la columna 'competicion')
PESO_COMPETICION = {
    "world_cup":           1.5,   # Partido de Copa del Mundo
    "copa_continental":    1.2,   # Copa América, Eurocopa, Copa Africa, etc.
    "qualification":       1.0,   # Eliminatorias (cualquier confederación)
    "friendly":            0.5,   # Amistosos internacionales
    "other":               0.75,  # Cualquier otra competición oficial
}

# Peso por confederación del rival
PESO_CONFEDERACION = {
    "UEFA":      1.0,
    "CONMEBOL":  1.0,
    "CONCACAF":  0.75,
    "CAF":       0.70,
    "AFC":       0.55,
    "OFC":       0.45,
    "unknown":   0.65,
}

# ── Ranking FIFA oficial (junio 2026) ─────────────────────────────────────────
# Claves en español (notebook) + inglés (como aparecen en los CSVs).
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
    "nueva zelanda": 85, "new zealand": 85,

    # SELECCIONES QUE NO ESTÁN EN EL MUNDIAL (ES + EN)

    # Europa
    "italia": 12, "italy": 12,
    "ucrania": 32, "ukraine": 32,
    "polonia": 35, "poland": 35,
    "rusia": 36, "russia": 36,
    "serbia": 39, "serbia": 39,
    "hungria": 42, "hungary": 42,
    "grecia": 47, "greece": 47,
    "eslovaquia": 48, "slovakia": 48,
    "rumania": 56, "romania": 56,
    "eslovenia": 58, "slovenia": 58,
    "irlanda": 59, "ireland": 59,
    "albania": 64, "albania": 64,
    "macedonia_del_norte": 67, "north macedonia": 67,
    "irlanda_del_norte": 70, "northern ireland": 70,
    "georgia": 72, "georgia": 72,
    "finlandia": 73, "finland": 73,
    "islandia": 75, "iceland": 75,
    "israel": 77, "israel": 77,
    "kosovo": 78, "kosovo": 78,
    "montenegro": 81, "montenegro": 81,
    "bulgaria": 86, "bulgaria": 86,
    "bielorrusia": 97, "belarus": 97,
    "luxemburgo": 98, "luxembourg": 98,
    "estonia": 129, "estonia": 129,
    "letonia": 137, "latvia": 137,
    "lituania": 148, "lithuania": 148,
    "chipre": 126, "cyprus": 126,
    "azerbaiyan": 124, "azerbaijan": 124,
    "kazajstan": 110, "kazakhstan": 110,
    "islas_feroe": 123, "faroe islands": 123,
    "gibraltar": 203, "gibraltar": 203,
    "andorra": 173, "andorra": 173,
    "san_marino": 211, "san marino": 211,
    "liechtenstein": 206, "liechtenstein": 206,

    # África
    "burkina_faso": 62, "burkina faso": 62,
    "guinea": 80, "guinea": 80,
    "gabon": 87, "gabon": 87,
    "uganda": 88, "uganda": 88,
    "benin": 90, "benin": 90,
    "mozambique": 101, "mozambique": 101,
    "guinea_ecuatorial": 105, "equatorial guinea": 105,
    "kenia": 111, "kenya": 111,
    "tanzania": 113, "tanzania": 113,
    "gambia": 116, "gambia": 116,
    "sudan": 117, "sudan": 117,
    "sierra_leona": 119, "sierra leone": 119,
    "namibia": 120, "namibia": 120,
    "togo": 121, "togo": 121,
    "ruanda": 128, "rwanda": 128,
    "liberia": 140, "liberia": 140,
    "etiopia": 144, "ethiopia": 144,

    # Asia
    "oman": 79, "oman": 79,
    "siria": 84, "syria": 84,
    "barein": 91, "bahrain": 91,
    "china": 94, "china pr": 94,
    "palestina": 95, "palestine": 95,
    "tayikistan": 103, "tajikistan": 103,
    "kirguistan": 107, "kyrgyzstan": 107,
    "libano": 108, "lebanon": 108,
    "kuwait": 134, "kuwait": 134,
    "india": 136, "india": 136,
    "malasia": 138, "malaysia": 138,
    "vietnam": 99, "vietnam": 99,
    "filipinas": 135, "philippines": 135,
    "tailandia": 93, "thailand": 93,
    "indonesia": 122, "indonesia": 122,
    "singapur": 147, "singapore": 147,
    "hong_kong": 155, "hong kong": 155,
    "myanmar": 158, "myanmar": 158,
    "afganistan": 169, "afghanistan": 169,

    # CONCACAF
    "jamaica": 71, "jamaica": 71,
    "trinidad_y_tobago": 102, "trinidad and tobago": 102,
    "nicaragua": 131, "nicaragua": 131,
    "republica_dominicana": 143, "dominican republic": 143,
    "cuba": 164, "cuba": 164,
    "belice": 180, "belize": 180,
    "guyana": 150, "guyana": 150,
    "surinam": 125, "suriname": 125,
    "puerto_rico": 156, "puerto rico": 156,
    "bermudas": 166, "bermuda": 166,

    # Oceanía
    "fiyi": 154, "fiji": 154,
    "islas_salomon": 153, "solomon islands": 153,
    "papua_nueva_guinea": 168, "papua new guinea": 168,
    "tahiti": 157, "tahiti": 157,
    "vanuatu": 160, "vanuatu": 160,
    "nueva_caledonia": 151, "new caledonia": 151,
}

# Confederación de cada rival (para w_confederacion automático)
CONFEDERACION_RIVALES = {
    "argentina": "CONMEBOL", "brazil": "CONMEBOL", "brasil": "CONMEBOL",
    "colombia": "CONMEBOL", "uruguay": "CONMEBOL", "ecuador": "CONMEBOL",
    "paraguay": "CONMEBOL", "peru": "CONMEBOL", "chile": "CONMEBOL",
    "bolivia": "CONMEBOL", "venezuela": "CONMEBOL",
    "spain": "UEFA", "france": "UEFA", "england": "UEFA", "portugal": "UEFA",
    "netherlands": "UEFA", "germany": "UEFA", "belgium": "UEFA", "croatia": "UEFA",
    "austria": "UEFA", "switzerland": "UEFA", "norway": "UEFA", "sweden": "UEFA",
    "scotland": "UEFA", "turkey": "UEFA", "czech republic": "UEFA",
    "czechia": "UEFA", "iceland": "UEFA",
    "bosnia and herzegovina": "UEFA", "bosnia herzegovina": "UEFA",
    "united states": "CONCACAF", "usa": "CONCACAF", "mexico": "CONCACAF",
    "canada": "CONCACAF", "panama": "CONCACAF", "curacao": "CONCACAF",
    "haiti": "CONCACAF", "costa rica": "CONCACAF", "el salvador": "CONCACAF",
    "honduras": "CONCACAF", "guatemala": "CONCACAF", "puerto rico": "CONCACAF",
    "japan": "AFC", "south korea": "AFC", "iran": "AFC", "australia": "AFC",
    "saudi arabia": "AFC", "uzbekistan": "AFC", "iraq": "AFC", "jordan": "AFC",
    "qatar": "AFC", "indonesia": "AFC",
    "morocco": "CAF", "senegal": "CAF", "egypt": "CAF", "algeria": "CAF",
    "ivory coast": "CAF", "ghana": "CAF", "dr congo": "CAF",
    "south africa": "CAF", "tunisia": "CAF", "cape verde": "CAF",
    "angola": "CAF", "zambia": "CAF", "mauritania": "CAF",
    "new zealand": "OFC",
}


def normalizar_rival(nombre: str) -> str:
    """Convierte el nombre del rival al formato canónico (minúsculas)."""
    s = str(nombre).lower().strip()
    reemplazos = {
        "korea republic": "south korea",
        "republic of korea": "south korea",
        "côte d'ivoire": "ivory coast",
        "cote d'ivoire": "ivory coast",
        "türkiye": "turkey",
        "usa": "united states",
        "united states of america": "united states",
        "curaçao": "curacao",
        "panamá": "panama",
        "canadá": "canada",
        "perú": "peru",
        "españa": "spain",
        "países bajos": "netherlands",
        "paises bajos": "netherlands",
        "bélgica": "belgium",
        "belgica": "belgium",
    }
    return reemplazos.get(s, s)


# Decaimiento temporal: semivida en días (más viejo → menos peso)
# 730 días ≈ 2 años de semivida. Ajustable.
LAMBDA_TEMPORAL = np.log(2) / 730


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE PARSEO
# ─────────────────────────────────────────────────────────────────────────────

def parse_pct(val):
    """'62%' → 62.0  |  NaN/'-' → NaN"""
    if pd.isna(val) or str(val).strip() in ("-", "", "nan"):
        return np.nan
    m = re.match(r"^([\d.]+)%", str(val).strip())
    return float(m.group(1)) if m else np.nan


def parse_ratio(val):
    """
    '33%(1/3)' → (33.0, 1, 3)
    '92%(334/362)' → (92.0, 334, 362)
    '62%' → (62.0, NaN, NaN)
    '-' / NaN → (NaN, NaN, NaN)
    """
    if pd.isna(val) or str(val).strip() in ("-", "", "nan"):
        return np.nan, np.nan, np.nan
    s = str(val).strip()
    m_full = re.match(r"^([\d.]+)%\((\d+)/(\d+)\)", s)
    if m_full:
        return float(m_full.group(1)), int(m_full.group(2)), int(m_full.group(3))
    m_pct = re.match(r"^([\d.]+)%", s)
    if m_pct:
        return float(m_pct.group(1)), np.nan, np.nan
    return np.nan, np.nan, np.nan


def parse_fraction(val):
    """'24/26 (92%)' → (24, 26, 92.0)  |  '-' → (NaN, NaN, NaN)"""
    if pd.isna(val) or str(val).strip() in ("-", "", "nan"):
        return np.nan, np.nan, np.nan
    s = str(val).strip()
    m = re.match(r"(\d+)/(\d+)\s*\(([\d.]+)%\)", s)
    if m:
        return int(m.group(1)), int(m.group(2)), float(m.group(3))
    m2 = re.match(r"(\d+)/(\d+)", s)
    if m2:
        a, b = int(m2.group(1)), int(m2.group(2))
        return a, b, (100 * a / b if b > 0 else np.nan)
    return np.nan, np.nan, np.nan


def parse_float(val):
    """'0.21' → 0.21  |  '-'/NaN → NaN"""
    if pd.isna(val) or str(val).strip() in ("-", "", "nan"):
        return np.nan
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


# ─────────────────────────────────────────────────────────────────────────────
# PESO POR COMPETICIÓN
# ─────────────────────────────────────────────────────────────────────────────

def clasificar_competicion(comp_str):
    """Devuelve la clave de PESO_COMPETICION dada la cadena de competición."""
    s = str(comp_str).lower()
    if "world cup" in s and "qualification" not in s:
        return "world_cup"
    if any(k in s for k in ["copa america", "euro", "copa africa",
                              "asian cup", "gold cup", "nations cup",
                              "copa oro", "afcon"]):
        return "copa_continental"
    if "qualification" in s or "qualifying" in s or "world championship" in s:
        return "qualification"
    if "friendly" in s or "amistoso" in s:
        return "friendly"
    return "other"


def peso_competicion(comp_str):
    clave = clasificar_competicion(comp_str)
    return PESO_COMPETICION[clave]


# ─────────────────────────────────────────────────────────────────────────────
# CARGA Y NORMALIZACIÓN DE PARTIDOS
# ─────────────────────────────────────────────────────────────────────────────

def cargar_partidos(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["fecha"] = pd.to_datetime(df["fecha"], format="%d.%m.%Y %H:%M", errors="coerce")

    # ── Determinar si la selección jugó de local o visitante ──────────────────
    sel = df["seleccion"].iloc[0].lower().strip()
    df["es_local"] = df["equipo_local"].str.lower().str.strip() == sel

    # ── Resultado desde perspectiva de la selección ───────────────────────────
    df["goles_favor"] = np.where(df["es_local"], df["goles_local"], df["goles_visitante"])
    df["goles_contra"] = np.where(df["es_local"], df["goles_visitante"], df["goles_local"])
    df["resultado"] = np.where(
        df["goles_favor"] > df["goles_contra"], "W",
        np.where(df["goles_favor"] < df["goles_contra"], "L", "D")
    )
    df["puntos"] = df["resultado"].map({"W": 3, "D": 1, "L": 0})

    # ── Prefijos de columnas para la selección y el rival ────────────────────
    df["prefijo_sel"] = np.where(df["es_local"], "local", "visitante")
    df["prefijo_rival"] = np.where(df["es_local"], "visitante", "local")

    return df


def extraer_stats_partido(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada partido construye un DataFrame con las estadísticas
    de la selección y del rival (1er + 2do tiempo sumados).
    """
    rows = []
    for _, row in df.iterrows():
        p_sel = row["prefijo_sel"]
        p_riv = row["prefijo_rival"]

        def get(prefix, stat, parser=parse_float):
            c1 = f"{prefix}__1st_half__{stat}"
            c2 = f"{prefix}__2nd_half__{stat}"
            v1 = parser(row.get(c1, np.nan))
            v2 = parser(row.get(c2, np.nan))
            if callable(parser) and parser == parse_pct:
                # promedio de los dos tiempos
                vals = [v for v in [v1, v2] if not np.isnan(v)]
                return np.mean(vals) if vals else np.nan
            # para numeric, suma de los dos tiempos
            vals = [v for v in [v1, v2] if not np.isnan(v) and not isinstance(v, tuple)]
            return sum(vals) if vals else np.nan

        def get_ratio(prefix, stat):
            """Retorna el valor numérico total (suma de numeradores / suma denominadores)."""
            c1 = f"{prefix}__1st_half__{stat}"
            c2 = f"{prefix}__2nd_half__{stat}"
            _, n1, d1 = parse_ratio(row.get(c1, np.nan))
            _, n2, d2 = parse_ratio(row.get(c2, np.nan))
            n_total = sum(x for x in [n1, n2] if not (isinstance(x, float) and np.isnan(x)))
            d_total = sum(x for x in [d1, d2] if not (isinstance(x, float) and np.isnan(x)))
            pct = (100 * n_total / d_total) if d_total > 0 else np.nan
            return pct, n_total, d_total

        pases_pct, pases_ok, pases_tot = get_ratio(p_sel, "passes")
        cruces_pct, _, _ = get_ratio(p_sel, "crosses")
        tackles_pct, _, _ = get_ratio(p_sel, "tackles")
        dribles_pct, _, _ = get_ratio(p_sel, "accurate_through_passes")

        def get_ratio_count(prefix, stat):
            """Para columnas tipo '81%(30/37)' devuelve la suma de numeradores (1t+2t)."""
            c1 = f"{prefix}__1st_half__{stat}"
            c2 = f"{prefix}__2nd_half__{stat}"
            _, n1, _ = parse_ratio(row.get(c1, np.nan))
            _, n2, _ = parse_ratio(row.get(c2, np.nan))
            vals = [v for v in [n1, n2] if not (isinstance(v, float) and np.isnan(v))]
            return sum(vals) if vals else np.nan

        pases_3t = get_ratio_count(p_sel, "passes_in_final_third")

        r = {
            "match_id":          row["match_id"],
            "seleccion":         row["seleccion"],
            "fecha":             row["fecha"],
            "competicion":       row["competicion"],
            "rival":             row["equipo_visitante"] if row["es_local"] else row["equipo_local"],
            "es_local":          row["es_local"],
            "resultado":         row["resultado"],
            "puntos":            row["puntos"],
            "goles_favor":       row["goles_favor"],
            "goles_contra":      row["goles_contra"],

            # xG y xGOT
            "xg_sel":            get(p_sel, "expected_goals_xg"),
            "xg_riv":            get(p_riv, "expected_goals_xg"),
            "xgot_sel":          get(p_sel, "xg_on_target_xgot"),
            "xgot_faced":        get(p_sel, "xgot_faced"),      # xGOT recibido por el arq.

            # Tiros
            "shots_total_sel":   get(p_sel, "total_shots"),
            "shots_on_tgt_sel":  get(p_sel, "shots_on_target"),
            "shots_inside_sel":  get(p_sel, "shots_inside_the_box"),
            "big_chances_sel":   get(p_sel, "big_chances"),
            "big_chances_riv":   get(p_riv, "big_chances"),
            # Diferencia de big chances (más estable que ratio cuando rival=0)
            "big_chances_diff":  get(p_sel, "big_chances") - get(p_riv, "big_chances"),

            # Pases
            "posesion":          (parse_pct(row.get(f"{p_sel}__1st_half__ball_possession", np.nan)) +
                                  parse_pct(row.get(f"{p_sel}__2nd_half__ball_possession", np.nan))) / 2,
            "pases_pct":         pases_pct,
            "pases_totales":     pases_tot,
            "pases_3ro_tercio":  pases_3t,
            "toques_area_rival": get(p_sel, "touches_in_opposition_box"),

            # Defensa
            "interceptions":     get(p_sel, "interceptions"),
            "clearances":        get(p_sel, "clearances"),
            "tackles_pct":       tackles_pct,
            "goals_prevented":   get(p_sel, "goals_prevented"),

            # Disciplina
            "amarillas":         get(p_sel, "yellow_cards"),
            "rojas":             get(p_sel, "red_cards"),

            # 2do tiempo
            "xg_sel_1t":         parse_float(row.get(f"{p_sel}__1st_half__expected_goals_xg", np.nan)),
            "xg_sel_2t":         parse_float(row.get(f"{p_sel}__2nd_half__expected_goals_xg", np.nan)),
        }
        rows.append(r)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PESOS COMBINADOS POR PARTIDO
# ─────────────────────────────────────────────────────────────────────────────

def calcular_pesos(stats: pd.DataFrame,
                   ranking_rivales: dict = None,
                   confederacion_rivales: dict = None) -> pd.DataFrame:
    """
    Agrega columnas de peso al DataFrame de stats de partidos.
    Si no se pasan dicts externos, usa RANKING_FIFA y CONFEDERACION_RIVALES
    definidos globalmente (con el ranking oficial FIFA junio 2026).
    """
    # Usar los globales como fallback
    ranking_rivales      = ranking_rivales      or RANKING_FIFA
    confederacion_rivales = confederacion_rivales or CONFEDERACION_RIVALES

    max_rank = max(ranking_rivales.values()) if ranking_rivales else 100

    hoy = pd.Timestamp.now()

    # 1. Peso competición
    stats["w_competicion"] = stats["competicion"].apply(peso_competicion)

    # 2. Peso temporal (decaimiento exponencial)
    stats["dias_atras"] = (hoy - stats["fecha"]).dt.days.clip(lower=0)
    stats["w_temporal"] = np.exp(-LAMBDA_TEMPORAL * stats["dias_atras"])

    # 3. Peso confederación del rival (normalizado al nombre canónico)
    def get_confederacion(nombre_rival):
        canon = normalizar_rival(nombre_rival)
        conf = confederacion_rivales.get(canon, confederacion_rivales.get(nombre_rival.lower(), "unknown"))
        return PESO_CONFEDERACION.get(conf, PESO_CONFEDERACION["unknown"])

    stats["w_confederacion"] = stats["rival"].apply(get_confederacion)

    # 4. Peso ranking FIFA del rival (normalizado 0.30–1.0)
    #    Equipo rank 1  → peso 1.00
    #    Equipo rank 85 → peso ~0.30
    #    Equipo sin ranking → peso 0.50 (neutro-bajo)
    def get_w_ranking(nombre_rival):
        canon = normalizar_rival(nombre_rival)
        rank = ranking_rivales.get(canon, ranking_rivales.get(nombre_rival.lower(), None))
        if rank is None:
            return 0.50
        return 0.30 + 0.70 * (1 - (rank - 1) / (max_rank - 1))

    stats["w_ranking"] = stats["rival"].apply(get_w_ranking)

    # Peso final combinado
    stats["peso_total"] = (
        stats["w_competicion"] *
        stats["w_temporal"] *
        stats["w_confederacion"] *
        stats["w_ranking"]
    )

    return stats


# ─────────────────────────────────────────────────────────────────────────────
# FEATURES DE PARTIDOS
# ─────────────────────────────────────────────────────────────────────────────

def wavg(series, weights):
    """Promedio ponderado ignorando NaN."""
    mask = series.notna() & weights.notna()
    if mask.sum() == 0:
        return np.nan
    return np.average(series[mask], weights=weights[mask])


def calcular_features_partidos(stats: pd.DataFrame) -> dict:
    w = stats["peso_total"]

    # ── Rendimiento general ───────────────────────────────────────────────────
    pts_pond = wavg(stats["puntos"], w)
    xg_neto = wavg(stats["xg_sel"] - stats["xg_riv"], w)
    goles_xg_ratio = (
        stats["goles_favor"].sum() / stats["xg_sel"].sum()
        if stats["xg_sel"].sum() > 0 else np.nan
    )
    # Overperformance defensiva: cuántos goles evitó el arq. sobre xGOT recibido
    solidez_arq = wavg(stats["xgot_faced"] - stats["goles_contra"], w)

    # ── Creación de juego ────────────────────────────────────────────────────
    xg_pond = wavg(stats["xg_sel"], w)
    big_chances_diff = wavg(stats["big_chances_diff"], w)
    toques_area = wavg(stats["toques_area_rival"], w)
    pases_3ro = wavg(stats["pases_3ro_tercio"], w)

    # ── Estilo ───────────────────────────────────────────────────────────────
    posesion_pond = wavg(stats["posesion"], w)
    precision_pases = wavg(stats["pases_pct"], w)

    # Índice de contragolpe: xG generado en partidos con posesión < 45%
    mask_contra = stats["posesion"] < 45
    xg_contra = wavg(stats.loc[mask_contra, "xg_sel"], w[mask_contra]) if mask_contra.sum() > 0 else 0.0

    # Crecimiento en 2do tiempo
    ratio_2t = wavg(
        stats["xg_sel_2t"] / (stats["xg_sel_1t"] + 1e-6), w
    )

    # ── Defensa ──────────────────────────────────────────────────────────────
    interceptions_pond = wavg(stats["interceptions"], w)
    clearances_pond = wavg(stats["clearances"], w)
    tackles_pond = wavg(stats["tackles_pct"], w)
    goals_prevented = wavg(stats["goals_prevented"].fillna(0), w)

    # ── Disciplina ───────────────────────────────────────────────────────────
    disciplina = wavg(stats["amarillas"] + stats["rojas"] * 3, w)

    # ── Consistencia (std del xG en partidos de alto peso) ───────────────────
    top_partidos = stats[w >= w.quantile(0.5)]
    consistencia_xg = top_partidos["xg_sel"].std() if len(top_partidos) > 1 else np.nan

    return {
        # Rendimiento
        "pts_ponderados":         pts_pond,
        "xg_neto_ponderado":      xg_neto,
        "xg_ofensivo_pond":       xg_pond,
        "goles_xg_ratio":         goles_xg_ratio,
        "solidez_arquero":        solidez_arq,
        "goals_prevented_pond":   goals_prevented,
        # Creación
        "big_chances_diff":       big_chances_diff,  # big chances propias - rivales
        "toques_area_rival_pond": toques_area,
        "pases_3ro_tercio_pond":  pases_3ro,
        # Estilo
        "posesion_promedio":      posesion_pond,
        "precision_pases":        precision_pases,
        "indice_contraataque":    xg_contra,
        "ratio_crecimiento_2t":   ratio_2t,
        # Defensa
        "interceptions_pond":     interceptions_pond,
        "clearances_pond":        clearances_pond,
        "tackles_pct_pond":       tackles_pond,
        # Disciplina
        "disciplina_score":       disciplina,
        # Consistencia
        "consistencia_xg":        consistencia_xg,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FEATURES DE JUGADORES
# ─────────────────────────────────────────────────────────────────────────────

GRUPO_POSICION = {
    "Goalkeeper":          "portero",
    "Centre back":         "defensor",
    "Defender":            "defensor",
    "Fullback":            "defensor",
    "Wingback":            "defensor",
    "Midfielder":          "mediocampista",
    "Attacking midfielder":"mediocampista",
    "Forward":             "delantero",
    "Winger":              "delantero",
    "Striker":             "delantero",
}


def cargar_jugadores(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Extraer nombre y posición de la columna 'all'
    df["nombre"] = df["all"].str.extract(r"^(.+?)\.")
    df["posicion_raw"] = df["all"].str.extract(r"\.(.+)$")
    df["grupo_pos"] = df["posicion_raw"].map(GRUPO_POSICION).fillna("otro")

    # Parsear columnas numéricas
    df["rating"] = df["ratingrating"].apply(parse_float)
    df["xg"] = df["expected_goals_xgexpected_goals_xg"].apply(parse_float)
    df["touches_area"] = pd.to_numeric(
        df["touches_in_opposition_boxtouches_in_opposition_box"], errors="coerce"
    )
    df["touches_total"] = pd.to_numeric(df["touchestouches"], errors="coerce")
    df["duels"] = pd.to_numeric(df["duelsduels"], errors="coerce")

    def parse_suc_drib(val):
        """'1/1 (100%)' → 1.0 (exitosos) | '-' → 0"""
        if pd.isna(val) or str(val).strip() in ("-", "", "nan"):
            return 0.0
        n, _, _ = parse_fraction(val)
        return float(n) if not (isinstance(n, float) and np.isnan(n)) else 0.0

    df["dribbles_ok"] = df["successful_dribblessuccessful_dribbles"].apply(parse_suc_drib)

    def parse_pases_jug(val):
        """'24/26 (92%)' → 92.0"""
        if pd.isna(val) or str(val).strip() in ("-", "", "nan"):
            return np.nan
        _, _, pct = parse_fraction(val)
        return pct

    df["pases_pct_jug"] = df["accurate_passesaccurate_passes"].apply(parse_pases_jug)

    def parse_shots_jug(val):
        if pd.isna(val) or str(val).strip() in ("-", "", "nan"):
            return 0.0
        try:
            return float(val)
        except (ValueError, TypeError):
            return 0.0

    df["shots"] = df["total_shotstotal_shots"].apply(parse_shots_jug)

    return df


def calcular_features_jugadores(df_jug: pd.DataFrame,
                                df_stats_partidos: pd.DataFrame) -> dict:
    """
    df_stats_partidos se usa para heredar el peso_total de cada partido,
    de forma que las métricas de jugadores también queden ponderadas
    por tipo de competición y calidad del rival.
    """

    if len(df_jug) == 0:
        return {
            "rating_prom_top11": np.nan,
            "rating_std_top11": np.nan,
            "xg_pond_delantero": np.nan,
            "xg_pond_mediocampista": np.nan,
            "xg_pond_defensor": np.nan,
            "xg_pond_portero": np.nan,
            "concentracion_xg_top3": np.nan,
            "dribbles_pond_atac": np.nan,
            "toques_area_delanteros": np.nan,
            "pases_pct_defensores": np.nan,
            "duelos_pond_defensor": np.nan,
            "duelos_pond_mediocampista": np.nan,
            "duelos_pond_delantero": np.nan,
            "shots_pond_delanteros": np.nan,
            "rating_std_partidos_clave": np.nan,
        }

    # Unir peso del partido a jugadores
    df = df_jug.merge(
        df_stats_partidos[["match_id", "peso_total", "competicion"]],
        on="match_id", how="left"
    )
    df["peso_total"] = df["peso_total"].fillna(1.0)

    feats = {}

    # ── Rating global ─────────────────────────────────────────────────────────
    # Top 11 por promedio ponderado de rating (proxy del XI titular)
    rating_por_jugador = (
        df.groupby("nombre")
        .apply(lambda g: wavg(g["rating"], g["peso_total"]))
        .dropna()
        .sort_values(ascending=False)
    )
    top11 = rating_por_jugador.head(11)
    feats["rating_prom_top11"]  = top11.mean() if len(top11) > 0 else np.nan
    feats["rating_std_top11"]   = top11.std()  if len(top11) > 1 else 0.0

    # ── xG por grupo de posición ──────────────────────────────────────────────
    for grupo in ["delantero", "mediocampista", "defensor", "portero"]:
        sub = df[df["grupo_pos"] == grupo]
        feats[f"xg_pond_{grupo}"] = wavg(sub["xg"], sub["peso_total"]) if len(sub) > 0 else 0.0

    # ── Concentración de xG en top 3 jugadores (riesgo de dependencia) ───────
    xg_por_jug = (
        df.groupby("nombre")
        .apply(lambda g: wavg(g["xg"].fillna(0), g["peso_total"]))
        .sort_values(ascending=False)
    )
    xg_total = xg_por_jug.sum()
    xg_top3 = xg_por_jug.head(3).sum()
    feats["concentracion_xg_top3"] = (xg_top3 / xg_total) if xg_total > 0 else np.nan

    # ── Dribbles (mediocampistas y delanteros) ────────────────────────────────
    sub_atac = df[df["grupo_pos"].isin(["mediocampista", "delantero"])]
    feats["dribbles_pond_atac"] = wavg(sub_atac["dribbles_ok"], sub_atac["peso_total"])

    # ── Toques en área rival (delanteros) ────────────────────────────────────
    sub_del = df[df["grupo_pos"] == "delantero"]
    feats["toques_area_delanteros"] = wavg(sub_del["touches_area"], sub_del["peso_total"])

    # ── Pases % defensores (construcción desde atrás) ────────────────────────
    sub_def = df[df["grupo_pos"] == "defensor"]
    feats["pases_pct_defensores"] = wavg(sub_def["pases_pct_jug"], sub_def["peso_total"])

    # ── Duelos ganados por grupo ──────────────────────────────────────────────
    for grupo in ["defensor", "mediocampista", "delantero"]:
        sub = df[df["grupo_pos"] == grupo]
        feats[f"duelos_pond_{grupo}"] = wavg(sub["duels"], sub["peso_total"]) if len(sub) > 0 else np.nan

    # ── Disparos de delanteros ───────────────────────────────────────────────
    feats["shots_pond_delanteros"] = wavg(sub_del["shots"], sub_del["peso_total"])

    # ── Consistencia del rating en partidos de alto peso ─────────────────────
    partidos_top = df_stats_partidos[
        df_stats_partidos["peso_total"] >= df_stats_partidos["peso_total"].quantile(0.5)
    ]["match_id"]
    df_top = df[df["match_id"].isin(partidos_top)]
    feats["rating_std_partidos_clave"] = (
        df_top.groupby("match_id")["rating"].mean().std()
        if len(df_top) > 1 else np.nan
    )

    return feats


# ─────────────────────────────────────────────────────────────────────────────
# PIPELINE COMPLETO POR SELECCIÓN
# ─────────────────────────────────────────────────────────────────────────────

def procesar_seleccion(nombre: str,
                        path_partidos: str,
                        path_jugadores: str,
                        ranking_rivales: dict = None,
                        confederacion_rivales: dict = None) -> dict:
    """
    Retorna un dict con todos los features de una selección.
    """
    # Partidos
    df_raw = cargar_partidos(path_partidos)
    stats  = extraer_stats_partido(df_raw)
    stats  = calcular_pesos(stats, ranking_rivales, confederacion_rivales)
    feats_partidos = calcular_features_partidos(stats)

    # Jugadores
    df_jug = cargar_jugadores(path_jugadores)
    feats_jugadores = calcular_features_jugadores(df_jug, stats)

    return {"seleccion": nombre, **feats_partidos, **feats_jugadores}


# ─────────────────────────────────────────────────────────────────────────────
# PROCESAMIENTO MASIVO DE LAS 48 SELECCIONES
# ─────────────────────────────────────────────────────────────────────────────

def procesar_todas(data_dir: str,
                   ranking_rivales: dict = None,
                   confederacion_rivales: dict = None) -> pd.DataFrame:
    """
    Busca todos los archivos *_partidos*.csv en data_dir,
    infiere el nombre de la selección y procesa cada una.
    """
    patron_partidos  = os.path.join(data_dir, "*_partidos*.csv")
    archivos_partidos = glob.glob(patron_partidos)

    if not archivos_partidos:
        raise FileNotFoundError(
            f"No se encontraron archivos *_partidos*.csv en {data_dir}"
        )

    resultados = []
    errores    = []

    for path_p in sorted(archivos_partidos):
        nombre_archivo = os.path.basename(path_p)

        # Extraer nombre de selección: todo lo anterior a "_partidos"
        m = re.match(r"^(.+?)_partidos", nombre_archivo)
        if not m:
            print(f"  [SKIP] No se pudo inferir selección de: {nombre_archivo}")
            continue

        seleccion = m.group(1).replace("_", " ")

        # Buscar archivo de jugadores
        path_j = os.path.join(data_dir, f"{m.group(1)}_jugadores.csv")
        if not os.path.exists(path_j):
            print(f"  [WARN] Sin archivo de jugadores para {seleccion}, se omiten features de jugadores.")
            path_j = None

        print(f"  Procesando: {seleccion} ...", end=" ")
        try:
            if path_j:
                feats = procesar_seleccion(
                    seleccion, path_p, path_j,
                    ranking_rivales, confederacion_rivales
                )
            else:
                # Solo features de partidos
                df_raw = cargar_partidos(path_p)
                stats  = extraer_stats_partido(df_raw)
                stats  = calcular_pesos(stats, ranking_rivales, confederacion_rivales)
                feats  = {"seleccion": seleccion, **calcular_features_partidos(stats)}

            resultados.append(feats)
            print("OK")
        except Exception as e:
            errores.append((seleccion, str(e)))
            print(f"ERROR: {e}")

    df_features = pd.DataFrame(resultados)

    if errores:
        print(f"\n[!] Errores en {len(errores)} selecciones:")
        for sel, err in errores:
            print(f"    {sel}: {err}")

    return df_features


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature engineering para predictor del Mundial")
    parser.add_argument("--data_dir",  default=".",     help="Directorio con los CSVs")
    parser.add_argument("--output",    default="features_mundial.csv", help="CSV de salida")
    parser.add_argument("--seleccion", default=None,
                        help="Procesar solo una selección (ej: argentina)")
    args = parser.parse_args()

    print("=" * 60)
    print("  PIPELINE DE FEATURES — PREDICTOR DEL MUNDIAL")
    print("=" * 60)

    if args.seleccion:
        sel = args.seleccion.lower().replace(" ", "_")
        path_p = os.path.join(args.data_dir, f"{sel}_partidos.csv")
        path_j = os.path.join(args.data_dir, f"{sel}_jugadores.csv")

        # Tolerar variantes de nombre (ej: argentina_partidos__1_.csv)
        if not os.path.exists(path_p):
            candidates = glob.glob(os.path.join(args.data_dir, f"{sel}_partidos*.csv"))
            path_p = candidates[0] if candidates else path_p

        feats = procesar_seleccion(
            sel, path_p, path_j,
            ranking_rivales=RANKING_FIFA,
            confederacion_rivales=CONFEDERACION_RIVALES,
        )
        df_out = pd.DataFrame([feats])
    else:
        df_out = procesar_todas(
            args.data_dir,
            ranking_rivales=RANKING_FIFA,
            confederacion_rivales=CONFEDERACION_RIVALES,
        )

    df_out.to_csv(args.output, index=False, float_format="%.4f")
    print(f"\n✓ Features guardados en: {args.output}")
    print(f"  Selecciones procesadas: {len(df_out)}")
    print(f"  Features por selección: {df_out.shape[1] - 1}")
    print()
    print(df_out.to_string(index=False))
