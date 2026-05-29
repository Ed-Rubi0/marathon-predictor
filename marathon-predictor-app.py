"""
═══════════════════════════════════════════════════════════════════════
  PREDICTOR DE RENDIMIENTO EN MARATÓN
  Kilómetros de Entrenamiento × Sport Science
  
  Modelos implementados:
    1. Riegel (1977)          - Modelo empírico de equivalencia
    2. Daniels VDOT (2005)   - Modelo fisiológico VO2max
    3. Velocidad Crítica      - Jones & Vanhatalo (2017)
    4. Modelo Poblacional     - Vickers & Vertosick (2016)

  Despliegue: streamlit run app.py
═══════════════════════════════════════════════════════════════════════
"""

import math
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN DE PÁGINA
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Predictor Maratón | Kilómetros de Entrenamiento",
    page_icon="🏃",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# CONTROL DE ACCESO — Contraseña
# ─────────────────────────────────────────────────────────────────────

def check_password() -> bool:
    """
    Muestra una pantalla de acceso con contraseña.
    Devuelve True si la contraseña es correcta, False en caso contrario.
    La contraseña se define en st.secrets["password"] o como variable de
    entorno. Si no se ha configurado, usa el valor por defecto de abajo.
    """
    import os

    # ── Contraseña (cámbiala aquí o en Streamlit Cloud > Secrets) ──
    CORRECT_PASSWORD = st.secrets.get("password", "KMentrenamiento2024")

    # Si ya autenticado en esta sesión, no volver a pedir
    if st.session_state.get("authenticated"):
        return True

    # ── Pantalla de login ─────────────────────────────────────────
    st.markdown(f"""
    <style>
    .login-wrap {{
        max-width: 420px;
        margin: 80px auto 0 auto;
        background: white;
        border-radius: 20px;
        padding: 44px 40px 36px;
        box-shadow: 0 8px 40px rgba(0,0,0,0.10);
        text-align: center;
    }}
    .login-logo {{
        font-size: 2.4em;
        margin-bottom: 10px;
    }}
    .login-brand {{
        font-weight: 900;
        font-size: 1.05em;
        color: #1A1A2E;
        letter-spacing: 0.01em;
    }}
    .login-sub {{
        color: #888;
        font-size: 0.85em;
        margin: 4px 0 28px 0;
    }}
    </style>
    <div class="login-wrap">
        <div class="login-logo">🏃</div>
        <div class="login-brand">Kilómetros de Entrenamiento</div>
        <div class="login-sub">Laboratorio de Alto Rendimiento Humano</div>
    </div>
    """, unsafe_allow_html=True)

    # Columnas para centrar el formulario
    _, col, _ = st.columns([1, 2, 1])
    with col:
        pwd = st.text_input(
            "Contraseña de acceso",
            type="password",
            placeholder="Introduce la contraseña...",
        )
        if st.button("🔓  Acceder", use_container_width=True):
            if pwd == CORRECT_PASSWORD:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("❌ Contraseña incorrecta. Inténtalo de nuevo.")
    return False


if not check_password():
    st.stop()


# ─────────────────────────────────────────────────────────────────────
# ESTILOS CSS — Identidad visual de Kilómetros de Entrenamiento
# ─────────────────────────────────────────────────────────────────────
BRAND_ORANGE = "#E8521A"
BRAND_DARK   = "#1A1A2E"
BRAND_NAVY   = "#16213E"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

html, body, [class*="css"] {{
    font-family: 'Inter', sans-serif;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(170deg, {BRAND_DARK} 0%, {BRAND_NAVY} 100%);
    border-right: 3px solid {BRAND_ORANGE};
}}
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {{
    color: #FFFFFF !important;
}}
[data-testid="stSidebar"] .stSlider > div > div > div {{
    background: {BRAND_ORANGE};
}}
[data-testid="stSidebar"] .stSelectbox > div > div {{
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.2);
    color: white;
}}

/* ── Main area ── */
.main {{ background-color: #F4F6FA; }}

/* ── Metric cards ── */
.hero-card {{
    background: linear-gradient(135deg, {BRAND_DARK} 0%, {BRAND_NAVY} 100%);
    border-radius: 20px;
    padding: 32px;
    text-align: center;
    border: 2px solid {BRAND_ORANGE};
    margin-bottom: 24px;
}}
.model-card {{
    background: white;
    border-radius: 14px;
    padding: 18px 12px;
    text-align: center;
    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    height: 180px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: transform 0.2s;
}}
.model-card:hover {{ transform: translateY(-3px); }}
.model-card-disabled {{
    background: #f0f0f0;
    border-radius: 14px;
    padding: 18px 12px;
    text-align: center;
    height: 180px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
}}

/* ── Citation box ── */
.citation-box {{
    background: #F0F4FF;
    border-left: 4px solid {BRAND_DARK};
    padding: 14px 18px;
    border-radius: 0 8px 8px 0;
    margin: 10px 0 18px 20px;
    border: 1px solid #E0E7FF;
}}

/* ── Section title ── */
.section-title {{
    color: {BRAND_DARK};
    font-weight: 800;
    font-size: 1.2em;
    border-bottom: 3px solid {BRAND_ORANGE};
    padding-bottom: 6px;
    margin: 20px 0 12px 0;
    display: inline-block;
}}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {{
    background: white;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}}
.stTabs [data-baseweb="tab"] {{
    border-radius: 8px;
    font-weight: 600;
    font-size: 0.9em;
    color: #555;
}}
.stTabs [aria-selected="true"] {{
    background: {BRAND_ORANGE} !important;
    color: white !important;
}}

/* ── Streamlit button ── */
div.stButton > button {{
    background: {BRAND_ORANGE};
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1em;
    padding: 14px 0;
    width: 100%;
    letter-spacing: 0.5px;
    transition: background 0.2s;
}}
div.stButton > button:hover {{
    background: #c94415;
    color: white;
}}

/* ── Zone bars ── */
.zone-bar {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: white;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}}

/* ── Footer ── */
.app-footer {{
    text-align: center;
    padding: 24px 0 8px 0;
    color: #888;
    font-size: 0.82em;
    border-top: 2px solid {BRAND_ORANGE};
    margin-top: 40px;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# UTILIDADES DE FORMATO
# ─────────────────────────────────────────────────────────────────────

def fmt_time(total_sec: float) -> str:
    """Convierte segundos totales a HH:MM:SS."""
    if total_sec is None or total_sec <= 0:
        return "—"
    h = int(total_sec // 3600)
    m = int((total_sec % 3600) // 60)
    s = int(total_sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def fmt_pace(sec_per_km: float) -> str:
    """Convierte segundos/km a MM:SS /km."""
    if sec_per_km is None or sec_per_km <= 0:
        return "—"
    m = int(sec_per_km // 60)
    s = int(sec_per_km % 60)
    return f"{m:02d}:{s:02d}"


def hms_to_sec(h: int, m: int, s: int) -> float:
    return float(h * 3600 + m * 60 + s)


# ─────────────────────────────────────────────────────────────────────
# MODELOS DE PREDICCIÓN
# ─────────────────────────────────────────────────────────────────────

# ── Modelo 1: Riegel (1977) ──────────────────────────────────────────
def model_riegel(ref_sec: float, ref_km: float, target_km: float = 42.195) -> float:
    """
    T2 = T1 × (D2 / D1)^1.06
    Riegel, P.S. (1977). Athletic records and human endurance.
    American Scientist, 65(2), 214–220.
    """
    if ref_sec <= 0 or ref_km <= 0:
        return None
    return ref_sec * (target_km / ref_km) ** 1.06


# ── Modelo 2: Daniels VDOT (2005) ───────────────────────────────────
def _vo2_at_velocity(v_m_min: float) -> float:
    """VO2 (ml/kg/min) en función de la velocidad (m/min). Daniels & Gilbert (1979)."""
    return -4.60 + 0.182258 * v_m_min + 0.000104 * v_m_min ** 2


def _pct_vo2max(t_min: float) -> float:
    """Fracción de VO2max utilizada según la duración (min). Daniels (2005)."""
    return (0.8
            + 0.1894393 * math.exp(-0.012778 * t_min)
            + 0.2989558 * math.exp(-0.1932605 * t_min))


def calc_vdot(race_sec: float, race_km: float) -> float:
    """Calcula VDOT a partir de una marca de referencia."""
    t_min  = race_sec / 60.0
    v      = (race_km * 1000) / t_min          # m/min
    vo2    = _vo2_at_velocity(v)
    pct    = _pct_vo2max(t_min)
    return (vo2 / pct) if pct > 0 else None


def vdot_to_marathon(vdot: float, target_km: float = 42.195) -> float:
    """Predicción de tiempo de maratón desde VDOT (iteración numérica)."""
    if vdot is None or vdot <= 0:
        return None
    dist_m = target_km * 1000
    # Estimación inicial
    t_min = dist_m / (vdot * 0.60)
    for _ in range(2000):
        pct    = _pct_vo2max(t_min)
        vo2req = vdot * pct
        # Resolver cuadrática: 0.000104v² + 0.182258v - (vo2req+4.60) = 0
        a, b, c = 0.000104, 0.182258, -(vo2req + 4.60)
        disc = b ** 2 - 4 * a * c
        if disc < 0:
            return None
        v = (-b + math.sqrt(disc)) / (2 * a)   # m/min
        if v <= 0:
            return None
        new_t = dist_m / v
        if abs(new_t - t_min) < 0.005:
            break
        t_min = new_t
    return t_min * 60.0    # → segundos


def model_daniels(ref_sec: float, ref_km: float):
    """Devuelve (tiempo_maratón_seg, VDOT)."""
    vdot = calc_vdot(ref_sec, ref_km)
    if vdot is None:
        return None, None
    return vdot_to_marathon(vdot), vdot


# ── Modelo 3: Velocidad Crítica ──────────────────────────────────────
def model_cs_two_trials(d1_m, t1_s, d2_m, t2_s, target_m: float = 42195):
    """
    CS = (D2-D1)/(t2-t1)   [m/s]
    D' = D1 - CS × t1      [m]
    T_marathon = (target_m - D') / CS

    Jones, A.M. & Vanhatalo, A. (2017). Sports Medicine, 47(S1), 65–78.
    Galbraith, A. et al. (2014). J Sports Sciences, 32(10), 961–966.
    """
    # Ordenar por duración
    if t1_s > t2_s:
        d1_m, d2_m = d2_m, d1_m
        t1_s, t2_s = t2_s, t1_s
    dt = t2_s - t1_s
    if dt == 0:
        return None, None, None
    cs     = (d2_m - d1_m) / dt
    d_prime = d1_m - cs * t1_s
    if cs <= 0 or (target_m - d_prime) <= 0:
        return None, cs, d_prime
    return (target_m - d_prime) / cs, cs, d_prime


def model_cs_from_vvo2max(dist_15min_m: float, target_m: float = 42195):
    """
    vVO2max = D_15min / 900   [m/s]
    CS ≈ 0.77 × vVO2max   (Burnley & Jones, 2018)
    D' ≈ 200 m  (valor poblacional típico)
    """
    vvo2max = dist_15min_m / 900.0
    cs      = vvo2max * 0.77
    d_prime = 200.0
    if cs <= 0:
        return None, cs, d_prime
    return (target_m - d_prime) / cs, cs, d_prime


# ── Modelo 4: Modelo Poblacional ─────────────────────────────────────
def model_population(ref_sec: float, ref_km: float, age: int, sex: str,
                     weekly_km: float = 50.0):
    """
    Predicción de maratón basada en datos poblacionales.

    Vickers, A.J. & Vertosick, E.A. (2016).
    BMC Sports Science, Medicine and Rehabilitation, 8(1), 26.

    Jokl et al. (2004). British Journal of Sports Medicine, 38(5), 612–617.
    """
    if ref_sec is None or ref_km is None:
        return None

    # Convertir a media maratón equivalente (Riegel) para usar ratio validado
    half_sec = model_riegel(ref_sec, ref_km, 21.0975)
    if half_sec is None:
        return None
    half_min = half_sec / 60.0

    # Ratio base hombre/mujer (Vickers & Vertosick, 2016)
    base_ratio = 2.09 if sex == "M" else 2.14

    # Corrección por edad (Jokl et al., 2004; Leyk et al., 2007)
    if age <= 35:
        age_f = 1.000
    elif age <= 50:
        age_f = 1.000 + (age - 35) * 0.008    # +0.8 % /año
    else:
        age_f = 1.120 + (age - 50) * 0.014    # +1.4 % /año tras 50

    # Corrección por volumen semanal (Vickers & Vertosick, 2016)
    if   weekly_km >= 70: vol_f = 0.96
    elif weekly_km >= 50: vol_f = 0.98
    elif weekly_km >= 35: vol_f = 1.00
    elif weekly_km >= 20: vol_f = 1.03
    else:                 vol_f = 1.07

    marathon_min = half_min * base_ratio * age_f * vol_f
    return marathon_min * 60.0   # → segundos


# ─────────────────────────────────────────────────────────────────────
# ESTRATEGIA DE RITMO
# ─────────────────────────────────────────────────────────────────────

def pacing_strategy(marathon_sec: float, strategy: str = "even"):
    """
    Devuelve (km_points, elapsed_sec, pace_sec_per_km).

    Tucker, R., Lambert, M.I. & Noakes, T.D. (2006).
    IJSPP, 1(3), 233–245.
    """
    kms    = [float(k) for k in range(1, 43)] + [42.195]
    paces  = []
    times  = []

    if strategy == "even":
        base_pace = marathon_sec / 42.195
        for km in kms:
            paces.append(base_pace)
            times.append(base_pace * km)

    elif strategy == "negative":
        # 51 % primera mitad, 49 % segunda
        p1 = (marathon_sec * 0.51) / 21.0975
        p2 = (marathon_sec * 0.49) / 21.0975
        for km in kms:
            if km <= 21.0975:
                paces.append(p1)
                times.append(p1 * km)
            else:
                paces.append(p2)
                times.append(p1 * 21.0975 + p2 * (km - 21.0975))

    elif strategy == "positive":
        p1 = (marathon_sec / 42.195) * 0.97
        p2 = (marathon_sec - p1 * 21.0975) / 21.0975
        for km in kms:
            if km <= 21.0975:
                paces.append(p1)
                times.append(p1 * km)
            else:
                paces.append(p2)
                times.append(p1 * 21.0975 + p2 * (km - 21.0975))

    return kms, times, paces


# ─────────────────────────────────────────────────────────────────────
# ZONAS DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────

ZONES = [
    ("Z1 — Recuperación",     1.28, "#93C5FD", "Carrera muy suave. Conversacional. Regeneración activa."),
    ("Z2 — Base Aeróbica",    1.15, "#6EE7B7", "Fondo largo y rodajes fáciles. El 70-80 % del volumen total."),
    ("Z3 — Ritmo Maratón",    1.00, BRAND_ORANGE, "Ritmo objetivo del día de carrera."),
    ("Z4 — Umbral Láctico",   0.92, "#FCD34D", "Tempo y ritmo de media maratón. Series de 20-40 min."),
    ("Z5 — VO₂máx",           0.85, "#FCA5A5", "Intervalos cortos. 5K / 10K. Series de 3-8 min."),
]


def training_zones(marathon_sec: float) -> list[dict]:
    base = marathon_sec / 42.195    # seg/km en ritmo de maratón
    result = []
    for name, factor, color, desc in ZONES:
        pace_s = base * factor
        result.append({"zone": name, "pace_s": pace_s,
                        "pace": fmt_pace(pace_s), "color": color, "desc": desc})
    return result


# ─────────────────────────────────────────────────────────────────────
# SIDEBAR — INPUTS
# ─────────────────────────────────────────────────────────────────────

with st.sidebar:
    # Logo — coloca logo.png en la misma carpeta que app.py
    import os
    if os.path.exists("logo.png"):
        st.image("logo.png", width=200)
    else:
        st.markdown(f"""
        <div style="padding: 10px 0 6px 0;">
            <div style="font-size:1.15em; font-weight:900; color:white;
                        letter-spacing:0.01em; line-height:1.25;">
                🏃 Kilómetros de<br>Entrenamiento
            </div>
            <div style="font-size:0.72em; color:{BRAND_ORANGE}; font-weight:600;
                        letter-spacing:0.06em; text-transform:uppercase; margin-top:5px;">
                Laboratorio de Alto<br>Rendimiento Humano
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 👤 Datos del Atleta")
    name      = st.text_input("Nombre", placeholder="Ej. Carlos García")
    age       = st.slider("Edad", 18, 75, 35)
    sex       = st.radio("Sexo", ["M", "F"], horizontal=True,
                         format_func=lambda x: "Hombre" if x == "M" else "Mujer")

    st.markdown("---")
    st.markdown("#### 🔬 Test de Campo (Velocidad Crítica)")
    field_type = st.selectbox(
        "Tipo de test disponible",
        ["test_15min", "two_trials", "none"],
        format_func=lambda x: {
            "test_15min": "⏱ Test 15 min (vVO₂máx)",
            "two_trials": "📏 Dos pruebas cronometradas",
            "none":       "❌ Sin test de campo",
        }[x],
    )

    vvo2max_ms = cs_t  = cs_d  = None
    cs2_t = cs2_d = None

    if field_type == "test_15min":
        d15 = st.number_input("Distancia recorrida en 15 min (m)",
                              min_value=1500, max_value=6000, value=3600, step=50)
        vvo2max_ms  = d15 / 900.0
        vvo2max_kmh = vvo2max_ms * 3.6
        vo2max_est  = 0.178 * (vvo2max_ms * 60) + 3.6   # Léger & Mercier, 1984
        st.success(
            f"**vVO₂máx ≈ {vvo2max_kmh:.1f} km/h**\n\n"
            f"VO₂máx estimado ≈ **{vo2max_est:.1f} ml/kg/min**"
        )

    elif field_type == "two_trials":
        dist_opts = {1.5: "1.500 m", 3.0: "3K", 5.0: "5K", 10.0: "10K", 21.0975: "Media"}

        st.markdown("**Prueba 1 — distancia corta**")
        d1_km  = st.selectbox("Distancia 1", list(dist_opts.keys()), index=2,
                               format_func=lambda x: dist_opts[x], key="d1")
        c1a, c1b = st.columns(2)
        t1m = c1a.number_input("min", 0, 180, 12, key="t1m")
        t1s = c1b.number_input("seg", 0, 59,  30, key="t1s")
        cs_d = d1_km * 1000
        cs_t = float(t1m * 60 + t1s)

        st.markdown("**Prueba 2 — distancia larga**")
        d2_km  = st.selectbox("Distancia 2", list(dist_opts.keys()), index=3,
                               format_func=lambda x: dist_opts[x], key="d2")
        c2a, c2b = st.columns(2)
        t2m = c2a.number_input("min", 0, 240, 46, key="t2m")
        t2s = c2b.number_input("seg", 0, 59,  0,  key="t2s")
        cs2_d = d2_km * 1000
        cs2_t = float(t2m * 60 + t2s)

    st.markdown("---")
    st.markdown("#### 🏅 Marca de Referencia en Competición")
    ref_dist_km = st.selectbox(
        "Distancia",
        [5.0, 10.0, 21.0975],
        index=1,
        format_func=lambda x: {5.0: "5K", 10.0: "10K", 21.0975: "Media Maratón"}[x],
    )
    rh, rm, rs = st.columns(3)
    ref_h = rh.number_input("h",  0, 5,  0,  key="rh")
    ref_m = rm.number_input("min",0, 59, 45, key="rm")
    ref_s = rs.number_input("seg",0, 59, 0,  key="rs")
    ref_sec = hms_to_sec(ref_h, ref_m, ref_s)

    st.markdown("---")
    st.markdown("#### 📊 Entrenamiento")
    weekly_km   = st.slider("Volumen semanal (km)", 10, 150, 55)
    pacing_strat = st.selectbox(
        "Estrategia de carrera",
        ["even", "negative", "positive"],
        format_func=lambda x: {
            "even":     "⚖️ Ritmo uniforme (recomendado)",
            "negative": "📈 Split negativo",
            "positive": "📉 Split positivo",
        }[x],
    )

    st.markdown("---")
    run_btn = st.button("🔬  ANALIZAR RENDIMIENTO")

# ─────────────────────────────────────────────────────────────────────
# CABECERA PRINCIPAL
# ─────────────────────────────────────────────────────────────────────

st.markdown(f"""
<div style="text-align:center; padding:18px 0 6px 0;">
  <h1 style="color:{BRAND_DARK}; font-size:2em; font-weight:900; margin:0;">
    🏃 Predictor de Rendimiento en Maratón
  </h1>
  <p style="color:#555; font-size:1.05em; margin:4px 0;">
    Modelos científicos validados · Estrategia de carrera · Zonas de entrenamiento
  </p>
  <span style="background:{BRAND_ORANGE}; color:white; font-size:0.78em;
               font-weight:700; padding:3px 14px; border-radius:30px; letter-spacing:0.5px;">
    KILÓMETROS DE ENTRENAMIENTO × Laboratorio de Alto Rendimiento Humano
  </span>
</div>
<hr style="border:none; border-top:2px solid {BRAND_ORANGE}; margin:18px 0 0 0;">
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# CÓMPUTO DE PREDICCIONES
# ─────────────────────────────────────────────────────────────────────

# Siempre calcula (Streamlit recalcula en cada interacción)
r_time  = model_riegel(ref_sec, ref_dist_km)
d_time, vdot_val = model_daniels(ref_sec, ref_dist_km)

# Velocidad crítica
if field_type == "test_15min" and vvo2max_ms:
    cs_time, cs_val, dprime = model_cs_from_vvo2max(d15)
elif field_type == "two_trials" and cs_t and cs2_t:
    cs_time, cs_val, dprime = model_cs_two_trials(cs_d, cs_t, cs2_d, cs2_t)
else:
    cs_time, cs_val, dprime = None, None, None

ml_time = model_population(ref_sec, ref_dist_km, age, sex, weekly_km)

# Ensemble (media de modelos disponibles)
valid = [t for t in [r_time, d_time, cs_time, ml_time] if t is not None]
ensemble = float(np.mean(valid))  if valid else None
ens_std  = float(np.std(valid))   if len(valid) > 1 else 0.0

# ─────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────

tab_pred, tab_pace, tab_zones, tab_refs = st.tabs([
    "🎯  Predicciones",
    "📈  Estrategia de Carrera",
    "💪  Zonas de Entrenamiento",
    "📚  Referencias Científicas",
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — PREDICCIONES
# ═══════════════════════════════════════════════════════════════════
with tab_pred:

    if ensemble:
        athlete_label = name if name else "Atleta"
        sex_icon = "♂" if sex == "M" else "♀"
        lo = fmt_time(ensemble - ens_std)
        hi = fmt_time(ensemble + ens_std)

        st.markdown(f"""
        <div class="hero-card">
          <p style="color:#aaa; margin:0 0 4px 0; font-size:0.95em;">
            {athlete_label} &nbsp;·&nbsp; {age} años &nbsp;·&nbsp; {sex_icon} &nbsp;·&nbsp;
            {weekly_km} km/sem
          </p>
          <div style="color:{BRAND_ORANGE}; font-size:4em; font-weight:900; line-height:1.1;">
            {fmt_time(ensemble)}
          </div>
          <p style="color:#ccc; margin:4px 0; font-size:1em; font-weight:500;">
            Predicción ensemble · media de {len(valid)} modelo{"s" if len(valid)>1 else ""}
          </p>
          <p style="color:{BRAND_ORANGE}; font-size:0.92em; margin:0;">
            Ritmo: <strong>{fmt_pace(ensemble/42.195)} /km</strong>
            &nbsp;|&nbsp;
            Rango: {lo} – {hi}
          </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Introduce los datos del atleta en el panel izquierdo para ver las predicciones.")

    # ── Tarjetas de los 4 modelos ──────────────────────────────────
    st.markdown(f'<div class="section-title">Comparativa de Modelos</div>', unsafe_allow_html=True)

    models_info = [
        ("📐", "Riegel (1977)",           r_time,  "#3B82F6",
         f"Exponente 1.06 sobre {ref_dist_km:.4g} km"),
        ("🫁", "Daniels VDOT (2005)",    d_time,  "#10B981",
         f"VDOT = {vdot_val:.1f}" if vdot_val else "Modelo VO₂máx efectivo"),
        ("⚡", "Velocidad Crítica",       cs_time, "#8B5CF6",
         f"CS = {cs_val*3.6:.2f} km/h | D' = {dprime:.0f} m"
         if cs_val else "Requiere test de campo"),
        ("🤖", "Modelo Poblacional",      ml_time, "#F59E0B",
         f"Ratio × edad × volumen × sexo"),
    ]

    cols = st.columns(4)
    for col, (icon, mname, mtime, color, mdesc) in zip(cols, models_info):
        with col:
            if mtime:
                diff_s   = mtime - ensemble if ensemble else 0
                sign     = "+" if diff_s >= 0 else "−"
                diff_abs = abs(diff_s)
                diff_str = f"{sign}{int(diff_abs//60)}:{int(diff_abs%60):02d}"
                st.markdown(f"""
                <div class="model-card" style="border-top: 5px solid {color};">
                  <div style="font-size:1.8em;">{icon}</div>
                  <div style="font-weight:700; font-size:0.82em; color:#333; line-height:1.2;">
                    {mname}
                  </div>
                  <div style="font-size:1.75em; font-weight:900; color:{color}; line-height:1.1;">
                    {fmt_time(mtime)}
                  </div>
                  <div style="font-size:0.78em; color:#666;">{fmt_pace(mtime/42.195)} /km</div>
                  <div style="font-size:0.72em; color:#999; margin-top:2px;">
                    {diff_str} vs ensemble
                  </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="model-card-disabled">
                  <div style="font-size:1.8em; opacity:0.3;">{icon}</div>
                  <div style="font-weight:700; font-size:0.82em; color:#aaa;">{mname}</div>
                  <div style="color:#ccc; font-size:0.82em; margin-top:8px;">
                    Sin datos suficientes
                  </div>
                </div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gráfico de barras comparativo ─────────────────────────────
    valid_models = [(n, t, c) for (_, n, t, c, _) in models_info if t is not None]
    if len(valid_models) >= 2:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            x=[m[0] for m in valid_models],
            y=[m[1] / 60 for m in valid_models],
            marker_color=[m[2] for m in valid_models],
            text=[fmt_time(m[1]) for m in valid_models],
            textposition="outside",
            textfont=dict(size=13, family="Inter", color="#333"),
            width=0.5,
        ))
        if ensemble:
            fig_bar.add_hline(
                y=ensemble / 60,
                line_dash="dash", line_color=BRAND_ORANGE, line_width=2.5,
                annotation_text=f"  Ensemble: {fmt_time(ensemble)}",
                annotation_position="top right",
                annotation_font=dict(color=BRAND_ORANGE, size=12, family="Inter"),
            )
        # Custom y-axis: mostrar en formato HH:MM
        tick_vals  = list(range(150, 400, 15))
        tick_texts = [fmt_time(v * 60) for v in tick_vals]
        fig_bar.update_layout(
            title=dict(
                text="Comparativa de predicciones por modelo",
                font=dict(family="Inter", size=15, color=BRAND_DARK),
            ),
            yaxis=dict(
                title="Tiempo",
                gridcolor="#f0f0f0",
                tickvals=tick_vals,
                ticktext=tick_texts,
            ),
            xaxis=dict(title=""),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter"),
            showlegend=False,
            height=360,
            margin=dict(t=60, b=20, l=90, r=140),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # ── Splits clave ───────────────────────────────────────────────
    if ensemble:
        st.markdown(f'<div class="section-title">Splits Clave Estimados</div>', unsafe_allow_html=True)
        split_defs = [
            ("10K",         10.0   / 42.195),
            ("Media (21K)", 21.0975/ 42.195),
            ("30K",         30.0   / 42.195),
            ("35K",         35.0   / 42.195),
            ("Meta",        1.0),
        ]
        split_cols = st.columns(5)
        for col, (label, frac) in zip(split_cols, split_defs):
            t = ensemble * frac
            col.metric(label, fmt_time(t))


# ═══════════════════════════════════════════════════════════════════
# TAB 2 — ESTRATEGIA DE CARRERA
# ═══════════════════════════════════════════════════════════════════
with tab_pace:

    if ensemble:
        kms, t_even, p_even = pacing_strategy(ensemble, "even")
        _,   t_neg,  p_neg  = pacing_strategy(ensemble, "negative")
        _,   t_pos,  p_pos  = pacing_strategy(ensemble, "positive")

        strat_colors = {"even": "#3B82F6", "negative": "#10B981", "positive": "#EF4444"}
        strat_labels = {
            "even":     "Uniforme",
            "negative": "Split negativo",
            "positive": "Split positivo",
        }
        all_paces = {"even": p_even, "negative": p_neg, "positive": p_pos}
        all_times = {"even": t_even, "negative": t_neg, "positive": t_pos}
        sel_paces = all_paces[pacing_strat]
        sel_times = all_times[pacing_strat]

        # Gráfico de ritmo + tiempo acumulado
        fig_pace = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=("Ritmo por kilómetro (min/km)", "Tiempo acumulado"),
            vertical_spacing=0.14,
        )

        strategy_pairs = [
            ("even",     p_even, t_even),
            ("negative", p_neg,  t_neg),
            ("positive", p_pos,  t_pos),
        ]
        dash_map = {"even": "solid", "negative": "dash", "positive": "dot"}

        for strat, paces_s, times_s in strategy_pairs:
            color = strat_colors[strat]
            label = strat_labels[strat]
            show  = strat == pacing_strat

            fig_pace.add_trace(go.Scatter(
                x=kms, y=[p / 60 for p in paces_s],
                name=label,
                line=dict(color=color, width=3 if show else 1.5, dash=dash_map[strat]),
                opacity=1.0 if show else 0.35,
                mode="lines",
            ), row=1, col=1)

            fig_pace.add_trace(go.Scatter(
                x=kms, y=[t / 3600 for t in times_s],
                name=label,
                line=dict(color=color, width=3 if show else 1.5, dash=dash_map[strat]),
                opacity=1.0 if show else 0.35,
                mode="lines",
                showlegend=False,
            ), row=2, col=1)

        # Líneas verticales en puntos clave
        for km_mark, km_label in [(10, "10K"), (21.0975, "21K"), (30, "30K"), (35, "35K")]:
            fig_pace.add_vline(
                x=km_mark, line_dash="dot",
                line_color=BRAND_ORANGE, line_width=1.2,
                annotation_text=km_label,
                annotation_position="top",
                annotation_font=dict(size=10, color=BRAND_ORANGE),
            )

        # Eje Y del ritmo invertido (menos seg = más rápido)
        min_pace = min(p_even) / 60 - 0.1
        max_pace = max(p_even) / 60 + 0.1
        fig_pace.update_yaxes(
            range=[max_pace, min_pace],
            title_text="min/km",
            tickformat=".2f",
            row=1, col=1,
        )
        fig_pace.update_yaxes(
            title_text="horas",
            tickformat=".2f",
            row=2, col=1,
        )
        fig_pace.update_xaxes(title_text="Kilómetro", row=2, col=1)
        fig_pace.update_layout(
            height=540,
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter"),
            legend=dict(
                orientation="h", yanchor="bottom",
                y=1.04, xanchor="right", x=1,
            ),
            margin=dict(t=60, b=20, l=70, r=30),
        )
        st.plotly_chart(fig_pace, use_container_width=True)

        # ── Tabla de ritmos por tramos ────────────────────────────
        st.markdown(f'<div class="section-title">Tabla de Ritmos — {strat_labels[pacing_strat]}</div>',
                    unsafe_allow_html=True)

        checkpoints = [
            ("5K",          5.0),
            ("10K",         10.0),
            ("15K",         15.0),
            ("Media (21K)", 21.0975),
            ("25K",         25.0),
            ("30K",         30.0),
            ("35K",         35.0),
            ("40K",         40.0),
            ("Meta",        42.195),
        ]
        rows = []
        for cp_label, cp_km in checkpoints:
            idx = min(range(len(kms)), key=lambda j: abs(kms[j] - cp_km))
            pace_s    = sel_paces[idx]
            elapsed_s = sel_times[idx]
            rows.append({
                "Punto":            cp_label,
                "Ritmo objetivo":   fmt_pace(pace_s) + " /km",
                "Tiempo acumulado": fmt_time(elapsed_s),
            })

        st.dataframe(
            pd.DataFrame(rows).set_index("Punto"),
            use_container_width=True,
        )

        # ── Recomendación científica ──────────────────────────────
        st.markdown(f'<div class="section-title">Recomendación Científica</div>', unsafe_allow_html=True)
        st.info(
            "**Split negativo o uniforme** son las estrategias más efectivas para corredores de fondo, "
            "tanto en rendimiento como en reducción del riesgo de 'muro' (km 30–35). "
            "Comenzar al 100–101 % del ritmo objetivo los primeros 21 km y acelerar progresivamente "
            "en la segunda mitad maximiza el aprovechamiento energético.\n\n"
            "📖 *Tucker, R., Lambert, M.I. & Noakes, T.D. (2006). IJSPP, 1(3), 233–245. "
            "DOI: 10.1123/ijspp.1.3.233*"
        )

    else:
        st.info("Introduce los datos del atleta para ver la estrategia de carrera.")


# ═══════════════════════════════════════════════════════════════════
# TAB 3 — ZONAS DE ENTRENAMIENTO
# ═══════════════════════════════════════════════════════════════════
with tab_zones:

    if ensemble:
        st.markdown(
            f'<div class="section-title">Zonas de Entrenamiento Personalizadas</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f"Calculadas a partir del tiempo predicho de maratón: "
            f"**{fmt_time(ensemble)}** ({fmt_pace(ensemble/42.195)} /km)"
        )

        zones = training_zones(ensemble)
        for z in zones:
            st.markdown(f"""
            <div class="zone-bar" style="border-left: 5px solid {z['color']};">
              <div>
                <span style="font-weight:700; color:{BRAND_DARK};">{z['zone']}</span><br>
                <small style="color:#666;">{z['desc']}</small>
              </div>
              <div style="text-align:right;">
                <span style="font-size:1.5em; font-weight:900; color:{z['color']};">
                  {z['pace']} /km
                </span>
              </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Gráfico de zonas ──────────────────────────────────────
        fig_zones = go.Figure()
        zone_names = [z["zone"] for z in zones]
        zone_paces = [z["pace_s"] / 60 for z in zones]
        zone_colors = [z["color"] for z in zones]

        fig_zones.add_trace(go.Bar(
            x=zone_names,
            y=zone_paces,
            marker_color=zone_colors,
            text=[f"{z['pace']} /km" for z in zones],
            textposition="outside",
            textfont=dict(size=13, family="Inter"),
            width=0.55,
        ))
        tick_vals  = list(range(3, 12))
        tick_texts = [fmt_pace(v * 60) + " /km" for v in tick_vals]
        fig_zones.update_layout(
            title=dict(
                text="Ritmos objetivo por zona de entrenamiento",
                font=dict(family="Inter", size=14, color=BRAND_DARK),
            ),
            yaxis=dict(
                title="Ritmo (min/km)",
                tickvals=tick_vals,
                ticktext=tick_texts,
                range=[max(zone_paces) + 0.5, min(zone_paces) - 0.5],
                gridcolor="#f0f0f0",
            ),
            xaxis=dict(title=""),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(family="Inter"),
            showlegend=False,
            height=340,
            margin=dict(t=50, b=20, l=90, r=40),
        )
        st.plotly_chart(fig_zones, use_container_width=True)

        # ── Datos fisiológicos complementarios ───────────────────
        if field_type == "test_15min" and vvo2max_ms:
            st.markdown(f'<div class="section-title">Perfil Fisiológico Estimado</div>',
                        unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("vVO₂máx",     f"{vvo2max_ms*3.6:.1f} km/h",
                      help="Velocidad aeróbica máxima — test 15 min")
            c2.metric("VO₂máx est.", f"{0.178*(vvo2max_ms*60)+3.6:.1f} ml/kg/min",
                      help="Léger & Mercier (1984)")
            if cs_val:
                c3.metric("Vel. Crítica",
                          f"{cs_val*3.6:.2f} km/h",
                          help="77 % de vVO₂máx (Burnley & Jones, 2018)")
                c4.metric("D' (reserva)", f"{dprime:.0f} m",
                          help="Reserva de capacidad anaerobia estimada")
    else:
        st.info("Introduce los datos del atleta para ver las zonas de entrenamiento.")


# ═══════════════════════════════════════════════════════════════════
# TAB 4 — REFERENCIAS CIENTÍFICAS
# ═══════════════════════════════════════════════════════════════════
with tab_refs:

    st.markdown(f'<div class="section-title">Base Científica de los Modelos</div>',
                unsafe_allow_html=True)

    refs = [
        {
            "model":  "📐 Modelo de Riegel",
            "color":  "#3B82F6",
            "papers": [
                {
                    "title":   "Athletic records and human endurance",
                    "authors": "Riegel, P.S. (1977)",
                    "journal": "American Scientist, 65(2), 214–220.",
                    "doi":     "Artículo seminal — acceso libre",
                    "detail":  (
                        "Establece la fórmula T₂ = T₁ × (D₂/D₁)^1.06 derivada empíricamente "
                        "de miles de marcas atléticas. El exponente 1.06 captura la penalización "
                        "energética al aumentar la distancia. Modelo sencillo, ampliamente validado, "
                        "especialmente preciso entre 1.5K y 50K."
                    ),
                },
            ],
        },
        {
            "model":  "🫁 Daniels VDOT",
            "color":  "#10B981",
            "papers": [
                {
                    "title":   "Daniels' Running Formula (3rd ed.)",
                    "authors": "Daniels, J. (2005)",
                    "journal": "Human Kinetics, Champaign, IL. ISBN: 978-0-7360-6218-0.",
                    "doi":     "",
                    "detail":  (
                        "El VDOT es un VO₂máx 'efectivo' que incorpora implícitamente la economía "
                        "de carrera. Permite comparar y proyectar rendimientos entre distancias "
                        "usando las tablas de equivalencia de Daniels, con planificación de zonas "
                        "de entrenamiento (E, M, T, I, R) individualizadas."
                    ),
                },
                {
                    "title":   "Oxygen power: Performance tables for distance runners",
                    "authors": "Daniels, J. & Gilbert, J. (1979)",
                    "journal": "Contempo, Tempe, AZ.",
                    "doi":     "Publicación original de las fórmulas matemáticas",
                    "detail":  (
                        "Fundamento matemático del cálculo VDOT: el consumo de O₂ en función de la "
                        "velocidad y el porcentaje de VO₂máx utilizable según la duración de la prueba."
                    ),
                },
            ],
        },
        {
            "model":  "⚡ Velocidad Crítica",
            "color":  "#8B5CF6",
            "papers": [
                {
                    "title":   "The 'Critical Power' Concept: Applications to Sports Performance",
                    "authors": "Jones, A.M. & Vanhatalo, A. (2017)",
                    "journal": "Sports Medicine, 47(Suppl 1), 65–78.",
                    "doi":     "DOI: 10.1007/s40279-017-0688-0",
                    "detail":  (
                        "Revisión exhaustiva del modelo bifactorial: la Velocidad Crítica (CS) es "
                        "la mayor intensidad sostenible sin fatiga progresiva, y D' es la reserva "
                        "de trabajo finita por encima de CS. El tiempo de maratón se predice como "
                        "T = (42 195 − D') / CS."
                    ),
                },
                {
                    "title":   "Predicting marathon race time in recreational marathon runners",
                    "authors": "Galbraith, A. et al. (2014)",
                    "journal": "Journal of Sports Sciences, 32(10), 961–966.",
                    "doi":     "DOI: 10.1080/02640414.2013.876128",
                    "detail":  (
                        "Validación directa del modelo de velocidad crítica para la predicción del "
                        "tiempo de maratón en corredores recreacionales. La velocidad crítica derivada "
                        "de dos pruebas cronometradas predice el ritmo de maratón con alta precisión "
                        "(r² > 0.90)."
                    ),
                },
                {
                    "title":   "Power-duration relationship of human muscular performance",
                    "authors": "Burnley, M. & Jones, A.M. (2018)",
                    "journal": "Journal of Applied Physiology, 125(5), 1743–1750.",
                    "doi":     "DOI: 10.1152/japplphysiol.00310.2018",
                    "detail":  (
                        "Establece que CS ≈ 77 % de vVO₂máx en corredores entrenados, permitiendo "
                        "estimar la velocidad crítica directamente desde el test de 15 minutos sin "
                        "necesidad de dos pruebas cronometradas."
                    ),
                },
            ],
        },
        {
            "model":  "🤖 Modelo Poblacional",
            "color":  "#F59E0B",
            "papers": [
                {
                    "title":   "An empirical study of race times in recreational endurance runners",
                    "authors": "Vickers, A.J. & Vertosick, E.A. (2016)",
                    "journal": "BMC Sports Science, Medicine and Rehabilitation, 8(1), 26.",
                    "doi":     "DOI: 10.1186/s13102-016-0052-y",
                    "detail":  (
                        "Análisis de > 2 600 corredores recreacionales. El tiempo de media maratón "
                        "es el predictor más robusto del maratón (r² = 0.96). El estudio cuantifica "
                        "los ratios diferenciados por sexo y la influencia del volumen de entrenamiento "
                        "en la conversión de rendimiento entre distancias."
                    ),
                },
                {
                    "title":   "Age and sex interactions in marathon performance — 1,610,627 runners",
                    "authors": "Jokl, P., Sethi, P.M. & Cooper, A.J. (2004)",
                    "journal": "British Journal of Sports Medicine, 38(5), 612–617.",
                    "doi":     "DOI: 10.1136/bjsm.2003.011718",
                    "detail":  (
                        "Cuantifica el declive del rendimiento en maratón por efecto del envejecimiento: "
                        "+0.8 %/año entre 35–50 años y +1.4 %/año a partir de los 50 en ambos sexos. "
                        "Base de las correcciones demográficas del modelo poblacional."
                    ),
                },
            ],
        },
        {
            "model":  "📈 Estrategia de Carrera (Pacing)",
            "color":  BRAND_ORANGE,
            "papers": [
                {
                    "title":   "An analysis of pacing strategies during men's world-record performances in track athletics",
                    "authors": "Tucker, R., Lambert, M.I. & Noakes, T.D. (2006)",
                    "journal": "International Journal of Sports Physiology and Performance, 1(3), 233–245.",
                    "doi":     "DOI: 10.1123/ijspp.1.3.233",
                    "detail":  (
                        "El split negativo o uniforme es la estrategia óptima tanto en élite como en "
                        "corredores recreacionales. Comenzar al 98–102 % del ritmo objetivo minimiza "
                        "el vaciado glucolítico prematuro y reduce el riesgo del 'muro' del kilómetro 30."
                    ),
                },
            ],
        },
        {
            "model":  "🏃 Test de 15 min / vVO₂máx",
            "color":  "#6B7280",
            "papers": [
                {
                    "title":   "Maximal aerobic speed: an important criterion for physical fitness evaluation",
                    "authors": "Léger, L. & Mercier, D. (1984)",
                    "journal": "Canadian Journal of Applied Sport Sciences, 9(2), 64–74.",
                    "doi":     "PMID: 6375061",
                    "detail":  (
                        "Establece la fórmula VO₂máx ≈ 0.178 × vVO₂máx + 3.6 (vVO₂máx en m/min), "
                        "que permite estimar el VO₂máx absoluto a partir de la distancia recorrida "
                        "en 15 minutos a velocidad máxima sostenida."
                    ),
                },
            ],
        },
    ]

    for group in refs:
        st.markdown(f"""
        <div style="border-left:5px solid {group['color']};
                    padding-left:14px; margin:20px 0 8px 0;">
          <span style="font-weight:800; color:{group['color']}; font-size:1.05em;">
            {group['model']}
          </span>
        </div>
        """, unsafe_allow_html=True)

        for p in group["papers"]:
            st.markdown(f"""
            <div class="citation-box">
              <p style="margin:0 0 2px 0; font-weight:700; color:{BRAND_DARK};">{p['title']}</p>
              <p style="margin:0; color:{BRAND_ORANGE}; font-size:0.85em;">{p['authors']}</p>
              <p style="margin:2px 0; color:#555; font-size:0.84em; font-style:italic;">{p['journal']}</p>
              <p style="margin:8px 0 4px 0; color:#333; font-size:0.9em;">{p['detail']}</p>
              {"<code style='font-size:0.8em;color:#666;'>" + p['doi'] + "</code>" if p['doi'] else ""}
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="app-footer">
  <strong>Kilómetros de Entrenamiento × Laboratorio de Alto Rendimiento Humano</strong><br>
  Herramienta de predicción basada en modelos científicos validados.<br>
  <em>Los resultados son orientativos. Deben interpretarse por un profesional del deporte.</em>
</div>
""", unsafe_allow_html=True)
