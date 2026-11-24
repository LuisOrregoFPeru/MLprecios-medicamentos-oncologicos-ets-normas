"""
Observatorio de Precios de Medicamentos Oncológicos de Alto Costo: Impacto Normativo y ETS
==========================================================================================
Misma arquitectura que el Observatorio del Financiamiento Oncológico INEN (inen-sisfissal-dashboard),
aplicada a los precios de venta en el mercado privado (DIGEMID, OPPF) de los medicamentos de alto costo
del listado INEN. Evalúa:
  - el impacto de la normativa (ITS multi-quiebre, Chow, quiebre automático, diferencia en diferencias),
  - el precio antes y después de cada informe de ETS del INS, según su dictamen y conclusión,
  - todo por medicamento o por grupos de afinidad terapéutica.

Ejecutar localmente:
    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import data_processing as dp
from scipy import stats as _st
from stats_analysis import (
    alpha_stars,
    chow_test_event,
    detect_breakpoint_auto,
    diff_in_diff,
    mann_kendall_trend,
    segmented_its,
    significance_label,
    smooth_series,
)

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------
_ASSETS = Path(__file__).parent / "assets"
st.set_page_config(
    page_title="Observatorio de Precios de Medicamentos Oncológicos | Impacto normativo y ETS",
    page_icon="💊", layout="wide", initial_sidebar_state="expanded",
)

PRIMARY_COLOR = "#2C5F8A"
SECONDARY_COLOR = "#D9542F"
ACCENT_COLOR = "#1F4E78"
OK_COLOR = "#2F9E44"
WARN_COLOR = "#E8A33D"
BAD_COLOR = "#C94141"
GROUP_PALETTE = ["#2C5F8A", "#D9542F", "#2F9E44", "#8850C4", "#C9A13B", "#4AA8C9", "#C2477E", "#5C6AC4", "#7D7D7D"]
COL_DICT = {"A favor": "#2a78d6", "En contra": "#eb6834", "Mixta": "#6250d6", "Sin recomendación formal": "#8a8985"}
SIMB_DICT = {"A favor": "triangle-up", "En contra": "triangle-down", "Mixta": "diamond", "Sin recomendación formal": "circle"}
COL_EJE = {"Financiamiento y cobertura oncológica": "#2C5F8A", "Listas y petitorio": "#5C6AC4", "Compras públicas": "#4AA8C9",
           "Registro sanitario y entrada al mercado": "#2F9E44", "Genéricos y transparencia de precios": "#D9542F",
           "Evaluación de tecnologías sanitarias": "#8850C4"}

CUSTOM_CSS = f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@700;800;900&display=swap');
    .main .block-container {{ padding-top: 1.3rem; max-width: 1400px; }}
    h1, h2, h3 {{ color: {ACCENT_COLOR}; }}
    .app-header {{
        background: linear-gradient(90deg, {ACCENT_COLOR} 0%, {PRIMARY_COLOR} 100%);
        padding: 1.1rem 1.6rem; border-radius: 10px; color: white; margin-bottom: 1.2rem;
        display: flex; align-items: center; gap: 1rem;
    }}
    .app-header .opm-logo {{ flex-shrink: 0; width: 64px; height: 64px; }}
    .app-header .opm-logo * {{ fill: white; stroke: white; }}
    .app-header .opm-logo .opm-ring, .app-header .opm-logo polyline {{ fill: none; }}
    .app-header .opm-logo rect:first-child {{ fill: none; }}
    .app-header h1 {{
        color: white; margin: 0; font-size: 1.6rem; font-family: 'Poppins', sans-serif;
        font-weight: 800; text-transform: uppercase; letter-spacing: 0.01em; line-height: 1.15;
    }}
    .app-header p {{ color: #e8eef5; margin: 0.2rem 0 0 0; font-size: 0.95rem; }}
    div[data-testid="stMetric"] {{ background: #f7f8fb; border: 1px solid #e2e6ee; border-radius: 10px; padding: 0.6rem 0.9rem; }}
    .source-note {{ font-size: 0.8rem; color: #666; margin-top: 0.4rem; }}
    .interp-box {{ background: #f4f7fb; border-left: 4px solid {PRIMARY_COLOR}; border-radius: 6px;
                   padding: 0.7rem 1rem; margin: 0.6rem 0 1rem 0; font-size: 0.92rem; color: #333; }}
    .warn-box {{ background: #fff8ec; border-left: 4px solid {WARN_COLOR}; border-radius: 6px;
                 padding: 0.7rem 1rem; margin: 0.6rem 0 1rem 0; font-size: 0.9rem; color: #333; }}
    .mini-title {{ font-size: 0.95rem; font-weight: 600; color: {ACCENT_COLOR}; margin-bottom: 0.2rem; }}
    .citation-box {{ background: #fafbfc; border: 1px solid #e2e6ee; border-radius: 8px;
                      padding: 1rem 1.2rem; margin-top: 1.2rem; font-size: 0.85rem; color: #333; }}
    .citation-box h4 {{ margin: 0 0 0.5rem 0; font-size: 0.8rem; letter-spacing: 0.04em; color: {ACCENT_COLOR}; }}
    .citation-box p {{ margin: 0 0 0.6rem 0; line-height: 1.5; }}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Plotly: coma decimal y espacio para miles en ejes, etiquetas y tooltips.
pio.templates["opm"] = go.layout.Template(pio.templates["plotly_white"])
pio.templates["opm"].layout.separators = ", "
pio.templates["opm"].layout.font = dict(family="Arial")
pio.templates["opm"].layout.yaxis.tickformat = ",.2~f"  # 29 000 en vez de 29k
pio.templates.default = "opm"


# Ancho completo compatible con cualquier versión de Streamlit.
def _ancho(fn, *args, **kw):
    try:
        return fn(*args, width="stretch", **kw)
    except (TypeError, ValueError):
        return fn(*args, use_container_width=True, **kw)


SEP_MILES = "\u00a0"  # espacio (no separable) para miles; coma para decimales


def fnum(x, d=2, signo=False):
    """Número con miles separados por espacio y decimales con coma: 12 345,67."""
    if x is None or pd.isna(x):
        return "—"
    s = f"{x:{'+' if signo else ''},.{d}f}"
    return s.replace(",", "§").replace(".", ",").replace("§", SEP_MILES)


def pct(x, d=1, signo=True):
    if x is None or pd.isna(x):
        return "—"
    return fnum(x * 100, d, signo) + "%"


def soles(x):
    return "—" if pd.isna(x) else "S/ " + fnum(x, 2)


def pval(p):
    return "—" if p is None or pd.isna(p) else ("< 0,001" if p < 0.001 else fnum(p, 3))


def _fmt_por_formato(fmt):
    """Traduce el formato de st.column_config.NumberColumn a una función de formato en español."""
    if fmt == "percent":
        return lambda v: pct(v, 1, signo=False) if pd.notna(v) else ""
    import re as _re
    m = _re.match(r"%\.?(\d*)([fd])", str(fmt or ""))
    d = 0 if (m and m.group(2) == "d") else (int(m.group(1)) if m and m.group(1) else 2)
    return lambda v: fnum(v, d) if pd.notna(v) else ""


def tabla(data=None, *args, column_config=None, **kw):
    """st.dataframe con números en formato español (miles con espacio, decimales con coma).
    Los formatos de NumberColumn se aplican con un Styler, que conserva el valor numérico para ordenar."""
    if isinstance(data, pd.DataFrame):
        cc = dict(column_config or {})
        fmts = {}
        for col in data.columns:
            conf = cc.get(col)
            fmt = None
            if isinstance(conf, dict) and isinstance(conf.get("type_config"), dict) and conf["type_config"].get("format"):
                fmt = conf["type_config"]["format"]
                conf = dict(conf)
                conf["type_config"] = {k: v for k, v in conf["type_config"].items() if k != "format"}
                cc[col] = conf
            if fmt is not None:
                fmts[col] = _fmt_por_formato(fmt)
            elif pd.api.types.is_bool_dtype(data[col]):
                continue
            elif pd.api.types.is_integer_dtype(data[col]):
                fmts[col] = lambda v: fnum(v, 0) if pd.notna(v) else ""
            elif pd.api.types.is_float_dtype(data[col]):
                nombre = str(col).lower()
                dec = 4 if (nombre.startswith("p ") or nombre.startswith("p (") or nombre == "p" or nombre.endswith(" p") or "p mk" in nombre) else 2
                fmts[col] = (lambda dd: (lambda v: fnum(v, dd) if pd.notna(v) else ""))(dec)
        datos = data.style.format(fmts) if fmts else data
        return _ancho(st.dataframe, datos, *args, column_config=cc or None, **kw)
    return _ancho(st.dataframe, data, *args, column_config=column_config, **kw)


def grafico(*args, **kw):
    return _ancho(st.plotly_chart, *args, **kw)


def caja(texto: str, tipo: str = "interp") -> None:
    st.markdown(f'<div class="{"warn-box" if tipo == "warn" else "interp-box"}">{texto}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
@st.cache_data
def cargar():
    precios = dp.load_precios()
    cat = dp.load_catalogo()
    reg = dp.load_registro()
    norms = dp.load_norms()
    ets, vin = dp.load_ets()
    return precios, cat, reg, norms, ets, vin


@st.cache_data
def cargar_entradas():
    return dp.load_entradas()


@st.cache_data
def eventos_entrada(k: int, tipo: str):
    precios = cargar()[0]
    e = cargar_entradas()
    if e.empty:
        return e
    return dp.estudio_entradas(precios, e[e.tipo_entrada == tipo], k)


@st.cache_data
def eventos_ets(k: int):
    precios, _, _, _, _, vin = cargar()
    ev = dp.estudio_eventos(precios, vin, k)
    return ev


precios, catalogo, registro, norms_all, ets, vinc = cargar()
MIN_M, MAX_M = precios.fecha.min(), precios.fecha.max()
_logo = (_ASSETS / "opm_logo.svg").read_text() if (_ASSETS / "opm_logo.svg").exists() else ""
_logo = _logo.replace("<svg ", '<svg class="opm-logo" ', 1)

st.markdown(
    f"""<div class="app-header">{_logo}<div>
<h1>Observatorio de Precios de Medicamentos Oncológicos de Alto Costo: Impacto Normativo y ETS</h1>
<p style="font-style: italic;">High-Cost Oncology Drug Price Observatory: Regulatory and HTA Impact</p>
<p>Precios de venta en farmacias privadas (DIGEMID/OPPF) de {catalogo.con_serie.sum()} medicamentos del listado INEN ·
serie mensual {MIN_M:%b-%Y} a {MAX_M:%b-%Y} · {len(ets)} informes ETS del INS</p>
</div></div>""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Panel de control")
    medida = st.radio("Medida de precio", list(dp.MEDIDAS), format_func=dp.MEDIDAS.get,
                      help="Índice por producto: cambio de precio de cada producto consigo mismo. Precio mediano o mínimo de "
                           "mercado: incluyen la entrada de genéricos y biosimilares más baratos.")
    dim = st.selectbox("Agrupar medicamentos por", list(dp.DIMENSIONES), format_func=dp.DIMENSIONES.get)
    valores_dim = sorted(precios[dim].dropna().unique())
    grupos_sel = st.multiselect("Grupos a analizar", valores_dim, default=valores_dim)
    rango = st.slider("Rango de meses", min_value=MIN_M.to_pydatetime(), max_value=MAX_M.to_pydatetime(),
                      value=(MIN_M.to_pydatetime(), MAX_M.to_pydatetime()), format="MMM-YYYY")
    desde, hasta = pd.Timestamp(rango[0]).to_period("M").to_timestamp(), pd.Timestamp(rango[1]).to_period("M").to_timestamp()

    st.divider()
    st.markdown("**Normativa**")
    agrupar_normas = st.checkbox("Agrupar normas cercanas (< 3 meses)", value=True,
                                 help="Combina normas del mismo grupo en un solo punto de corte con la vigencia más tardía.")
    norms_serie = norms_all[norms_all.fecha_efectiva > MIN_M]
    vista_all = dp.breakpoint_view(norms_serie, agrupar_normas)
    if agrupar_normas:
        default_ids = norms_serie[norms_serie.por_defecto == 1].grupo_id.unique().tolist()
    else:
        default_ids = norms_serie[norms_serie.por_defecto == 1].id.tolist()
    ids_sel = st.multiselect("Puntos de corte para el ITS", vista_all.id.tolist(), default=default_ids,
                             format_func=lambda i: vista_all.set_index("id").label[i],
                             help="Por defecto: normas de relevancia alta con suficientes meses antes y después.")
    VIEW = vista_all[vista_all.id.isin(ids_sel)].sort_values("fecha_efectiva")
    BREAKS = dp.breaks_dict(VIEW)
    LABELS = VIEW.set_index("id").label.to_dict()
    with st.expander("Normas de contexto (antes de la serie)"):
        for r in norms_all[norms_all.fecha_efectiva <= MIN_M].itertuples():
            st.caption(f"**{r.label}**: {r.norma}. {r.mecanismo}")

    st.divider()
    st.markdown("**ETS y estadística**")
    k_ets = st.select_slider("Ventana ETS (meses a cada lado)", options=[3, 6, 9, 12], value=6)
    ver_ets = st.checkbox("Marcar informes ETS en los gráficos", value=True)
    alpha = st.select_slider("Nivel de significancia (α)", options=[0.01, 0.05, 0.10], value=0.05)
    suavizado = st.checkbox("Suavizado LOWESS", value=False)

    _ids = sorted(BREAKS, key=BREAKS.get)
    for a_, b_ in zip(_ids, _ids[1:]):
        gap = (BREAKS[b_].year - BREAKS[a_].year) * 12 + BREAKS[b_].month - BREAKS[a_].month
        if gap < 3:
            st.warning(f"{LABELS[a_]} y {LABELS[b_]} están a {gap} mes(es): el segmento intermedio es muy corto.")

P = precios[(precios.fecha >= desde) & (precios.fecha <= hasta)]
ns_por_grupo = {g: sorted(P.loc[P[dim] == g, "n"].unique()) for g in grupos_sel}
COLOR_GRUPO = {g: GROUP_PALETTE[i % len(GROUP_PALETTE)] for i, g in enumerate(valores_dim)}
ETIQ = catalogo.set_index("n").medicamento.to_dict()


def serie_grupo(g: str) -> pd.Series:
    d = dp.indice_grupo(precios, ns_por_grupo.get(g, []), medida, desde, hasta)
    return pd.Series(d.indice.values, index=pd.DatetimeIndex(d.fecha)) if not d.empty else pd.Series(dtype=float)


def add_breaks(fig: go.Figure, y_paper: float = 1.04) -> None:
    for k, b in BREAKS.items():
        if desde <= b <= hasta:
            fig.add_vline(x=b, line_dash="dot", line_color="gray", opacity=0.7)
            fig.add_annotation(x=b, y=y_paper, yref="paper", showarrow=False, text=f"<b>{k}</b>",
                               font=dict(size=11, color="#555555"))


def add_ets_markers(fig: go.Figure, ns: list[int], y_val: float) -> None:
    d = vinc[vinc.n.isin(ns) & (vinc.fecha_pub >= desde) & (vinc.fecha_pub <= hasta + pd.offsets.MonthEnd(0))]
    d = d.drop_duplicates(["id", "molecula"])
    for g, dd in d.groupby("dictamen_grupo"):
        fig.add_trace(go.Scatter(
            x=dd.fecha_pub, y=[y_val] * len(dd), mode="markers", name=f"ETS {g.lower()}",
            marker=dict(color=COL_DICT[g], size=11, symbol=SIMB_DICT[g], line=dict(color="white", width=1)),
            customdata=np.stack([dd.codigo, dd.molecula, dd.dictamen], axis=-1),
            hovertemplate="%{customdata[0]}<br>%{customdata[1]}: %{customdata[2]}<extra></extra>"))


def style_fig(fig: go.Figure, height: int = 380, title: str | None = None, ytitle: str | None = None) -> None:
    fig.update_layout(height=height, hovermode="x unified", legend=dict(orientation="h", y=-0.22),
                      margin=dict(t=50 if title else 30, l=10, r=10, b=10), title=title, template="opm",
                      yaxis_title=ytitle or ("Índice (inicio = 100)"), font=dict(family="Arial", size=12))


st.caption(f"Medida: **{dp.MEDIDAS[medida]}** · agrupación: **{dp.DIMENSIONES[dim]}** · "
           f"{len(grupos_sel)} grupo(s) · {desde:%b-%Y} a {hasta:%b-%Y} · {len(BREAKS)} punto(s) de corte normativo")

tabs = st.tabs(["📈 Serie temporal", "🏁 Entrada de competidores", "🏥 Compra pública vs mercado", "⚖️ Impacto normativo (ITS)",
                "🔍 Quiebres", "🧪 Diferencia en diferencias", "🩺 ETS: antes y después", "📝 ETS: conclusiones",
                "📊 Variación y tendencia", "💊 Por medicamento", "📚 Normativa y metodología"])
(tab_serie, tab_ent, tab_pub, tab_its, tab_quiebre, tab_did, tab_ets, tab_concl, tab_var, tab_med, tab_met) = tabs

# ============================== TAB 1: Serie temporal ========================
with tab_serie:
    st.subheader("Índice de precios por grupo")
    st.caption("Cada grupo es un índice encadenado de Jevons de sus medicamentos (inicio del rango = 100): cada mes se "
               "promedian geométricamente los cambios de precio de los medicamentos con dato en ambos meses.")
    modo = st.radio("Presentación", ["Un gráfico por grupo", "Todos en un gráfico"], horizontal=True)
    if not grupos_sel:
        st.info("Selecciona al menos un grupo en el panel lateral.")
    elif modo == "Todos en un gráfico":
        fig = go.Figure()
        for g in grupos_sel:
            s = serie_grupo(g)
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines", name=g, line=dict(width=2, color=COLOR_GRUPO[g])))
        fig.add_hline(y=100, line_color="#c3c2b7", line_width=1)
        add_breaks(fig)
        style_fig(fig, 520)
        grafico(fig)
    else:
        cols = st.columns(2)
        for i, g in enumerate(grupos_sel):
            s = serie_grupo(g)
            if s.empty:
                continue
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines+markers", name="Índice del grupo",
                                     line=dict(width=2, color=COLOR_GRUPO[g]), marker=dict(size=4)))
            if len(BREAKS) and len(s) >= 12:
                res = segmented_its(s, BREAKS, LABELS, alpha=alpha, unit_divisor=1)
                if res is not None:
                    fig.add_trace(go.Scatter(x=res.fitted.index, y=res.fitted.values, mode="lines", name="Ajuste ITS",
                                             line=dict(width=2, color=SECONDARY_COLOR, dash="dash")))
            if suavizado:
                sm_y = smooth_series(s.values)
                if sm_y is not None:
                    fig.add_trace(go.Scatter(x=s.index, y=sm_y, mode="lines", name="LOWESS", line=dict(color=OK_COLOR)))
            fig.add_hline(y=100, line_color="#c3c2b7", line_width=1)
            if ver_ets:
                add_ets_markers(fig, ns_por_grupo[g], float(np.nanmax(s.values)) * 1.04)
            add_breaks(fig)
            nmeds = ", ".join(sorted(catalogo[catalogo.n.isin(ns_por_grupo[g])].molecula.unique()))
            style_fig(fig, 380, f"{g}")
            with cols[i % 2]:
                grafico(fig)
                st.caption(f"Incluye: {nmeds}.")
    caja("Los meses de ene a mar-2023 no tienen datos en el tablero de DIGEMID y se interpolan en escala logarítmica para "
         "mantener la secuencia mensual del modelo. Los triángulos marcan informes ETS de medicamentos del grupo "
         "(azul a favor, naranja en contra).")

# ============================== TAB 2: ITS ===================================
with tab_its:
    st.subheader("Regresión segmentada de interrupciones múltiples (ITS)")
    st.markdown("Modelo: `Índice_t = β0 + β1·t + Σk (δk·Dk,t + γk·Sk,t)` con errores Newey-West (3 rezagos). "
                "δk es el **cambio de nivel** (puntos de índice ≈ % respecto del inicio) y γk el **cambio de pendiente** mensual.")
    if not BREAKS:
        st.info("Selecciona al menos un punto de corte en el panel lateral.")
    else:
        gsel = st.selectbox("Grupo", grupos_sel, key="its_g") if grupos_sel else None
        if gsel:
            s = serie_grupo(gsel)
            res = segmented_its(s, BREAKS, LABELS, alpha=alpha, unit_divisor=1)
            if res is None:
                st.warning("Serie demasiado corta para el modelo.")
            else:
                c1, c2, c3 = st.columns(3)
                c1.metric("Observaciones (meses)", res.n_obs)
                c2.metric("R²", fnum(res.r2, 2))
                c3.metric("Pendiente previa (pts/mes)", fnum(res.pre_slope, 2, True), significance_label(res.pre_slope_p, alpha),
                          delta_color="off")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines+markers", name="Observado",
                                         line=dict(color=PRIMARY_COLOR, width=2), marker=dict(size=4)))
                fig.add_trace(go.Scatter(x=res.fitted.index, y=res.fitted.values, mode="lines", name="Ajuste ITS",
                                         line=dict(color=SECONDARY_COLOR, width=2.5, dash="dash")))
                add_breaks(fig)
                style_fig(fig, 420, gsel)
                grafico(fig)
                rows = []
                for sg in res.segments:
                    rows.append({"Corte": sg.break_id, "Norma(s)": sg.label, "Desde": sg.date.strftime("%b-%Y"), "Meses post": sg.n_post,
                                 "Cambio de nivel": f"{fnum(sg.level_change, 2, True)} {alpha_stars(sg.level_p)}", "p nivel": pval(sg.level_p),
                                 "Cambio de pendiente": f"{fnum(sg.slope_change, 3, True)} {alpha_stars(sg.slope_p)}", "p pendiente": pval(sg.slope_p),
                                 "Pendiente después": round(sg.slope_after, 3),
                                 "Lectura": ("Baja de nivel" if sg.level_change < 0 else "Alza de nivel") if sg.level_significant else
                                            ("Cambio de tendencia" if sg.slope_significant else "Sin cambio detectable")})
                tabla(pd.DataFrame(rows), hide_index=True)
                cortos = [sg.label for sg in res.segments if sg.n_post < 6]
                if cortos:
                    caja("Menos de 6 meses posteriores en: " + "; ".join(cortos) + ". Los coeficientes de esos cortes son poco precisos.", "warn")

        st.markdown("#### Comparación entre grupos")
        st.caption("Cambio de nivel (δ) de cada punto de corte en cada grupo. *** p<0,001 · ** p<0,01 · * p<0,05 · † p<0,10.")
        comp, comp_num = {}, {}
        for g in grupos_sel:
            r = segmented_its(serie_grupo(g), BREAKS, LABELS, alpha=alpha, unit_divisor=1)
            if r is None:
                continue
            comp[g] = {sg.break_id: f"{fnum(sg.level_change, 1, True)}{alpha_stars(sg.level_p)}" for sg in r.segments}
            comp_num[g] = {sg.break_id: (sg.level_change if sg.level_p < 0.10 else np.nan) for sg in r.segments}
        if comp:
            cdf = pd.DataFrame(comp).T
            cdf.columns = [f"{c}: {LABELS.get(c, c)[:28]}" for c in cdf.columns]
            tabla(cdf)
            num_df = pd.DataFrame(comp_num).T
            fig = go.Figure(go.Heatmap(z=num_df.values, x=[LABELS.get(c, c)[:30] for c in num_df.columns], y=num_df.index,
                                       colorscale=[[0, "#2a78d6"], [0.5, "#f2f2f0"], [1, "#eb6834"]], zmid=0,
                                       colorbar=dict(title="δ (pts)"), hovertemplate="%{y}<br>%{x}<br>δ = %{z:.1f}<extra></extra>"))
            fig.update_layout(height=60 + 42 * len(num_df), margin=dict(l=10, r=10, t=10, b=10), template="opm")
            grafico(fig)
            caja("Solo se colorean los cambios de nivel con p < 0,10: azul indica caída del índice tras la norma, naranja alza. "
                 "Un efecto atribuible a una norma debería aparecer en los grupos que esa norma alcanza (por ejemplo, la Ley 32033 "
                 "en grupos multifuente) y no en los demás.")

# ============================== TAB 3: Quiebres ==============================
with tab_quiebre:
    st.subheader("Detección de quiebres estructurales")
    gq = st.selectbox("Grupo", grupos_sel, key="q_g") if grupos_sel else None
    if gq:
        s = serie_grupo(gq)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="mini-title">Quiebre automático (sin fecha impuesta)</div>', unsafe_allow_html=True)
            ab = detect_breakpoint_auto(s)
            if ab is None:
                st.info("Serie insuficiente.")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines+markers", name="Observado", line=dict(color=PRIMARY_COLOR), marker=dict(size=4)))
                fig.add_trace(go.Scatter(x=ab.x_before, y=ab.y_pred_before, mode="lines", name="Tramo 1", line=dict(color=OK_COLOR, width=3)))
                fig.add_trace(go.Scatter(x=ab.x_after, y=ab.y_pred_after, mode="lines", name="Tramo 2", line=dict(color=SECONDARY_COLOR, width=3)))
                fig.add_vline(x=ab.date, line_dash="dash", line_color=BAD_COLOR)
                add_breaks(fig)
                style_fig(fig, 380)
                grafico(fig)
                cerca = VIEW[(VIEW.fecha_efectiva - pd.Timestamp(ab.date)).abs() <= pd.Timedelta(days=95)]
                st.markdown(f"Mes de quiebre: **{ab.date:%b-%Y}** · F = {fnum(ab.f_stat, 2)} · p = {pval(ab.p_value)} "
                            f"({significance_label(ab.p_value, alpha)}). Pendiente {fnum(ab.slope_before, 2, True)} → {fnum(ab.slope_after, 2, True)} pts/mes.")
                if not cerca.empty:
                    caja("El quiebre detectado cae a 3 meses o menos de: " + "; ".join(cerca.label) + ".")
                else:
                    caja("El quiebre detectado no coincide (±3 meses) con ningún punto de corte normativo seleccionado.")
        with c2:
            st.markdown('<div class="mini-title">Test de Chow en una fecha elegida</div>', unsafe_allow_html=True)
            opciones = {f"{r.label} ({r.fecha_efectiva:%b-%Y})": r.fecha_efectiva for r in vista_all.itertuples()}
            ets_g = vinc[vinc.n.isin(ns_por_grupo[gq])].drop_duplicates("id")
            for r in ets_g.itertuples():
                opciones[f"ETS {r.codigo} · {r.molecula} ({r.fecha_pub:%b-%Y})"] = r.fecha_pub.to_period("M").to_timestamp()
            ev_lbl = st.selectbox("Evento", list(opciones), key="chow_ev")
            lag = st.slider("Meses de implementación a excluir", 0, 6, 0, key="chow_lag")
            ch = chow_test_event(s, opciones[ev_lbl], implementation_lag_months=lag)
            if ch is None:
                st.info("No hay suficientes meses a ambos lados del evento.")
            else:
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines+markers", name="Observado", line=dict(color=PRIMARY_COLOR), marker=dict(size=4)))
                fig.add_trace(go.Scatter(x=ch.x_before, y=ch.y_pred_before, mode="lines", name="Antes", line=dict(color=OK_COLOR, width=3)))
                fig.add_trace(go.Scatter(x=ch.x_after, y=ch.y_pred_after, mode="lines", name="Después", line=dict(color=SECONDARY_COLOR, width=3)))
                fig.add_vline(x=opciones[ev_lbl], line_dash="dash", line_color=BAD_COLOR)
                style_fig(fig, 380)
                grafico(fig)
                st.markdown(f"F = {fnum(ch.f_stat, 2)} · p = {pval(ch.p_value)} ({significance_label(ch.p_value, alpha)}) · "
                            f"pendiente {fnum(ch.slope_before, 2, True)} → {fnum(ch.slope_after, 2, True)} pts/mes · n = {ch.n_before} / {ch.n_after} meses.")

# ============================== TAB 4: DiD ===================================
with tab_did:
    st.subheader("Diferencia en diferencias (DiD) entre grupos")
    st.markdown("Compara un grupo **tratamiento** (el que la norma debería afectar) con un **comparador** antes y después del corte: "
                "`log(índice) ~ t + tratamiento + post + tratamiento×t + tratamiento×post`. El coeficiente de tratamiento×post es el "
                "efecto porcentual neto de la tendencia común; tratamiento×t verifica tendencias paralelas previas.")
    caja("Sugerencia técnica: para las normas de genéricos (DU 005-2024, Ley 32033, DS 020-2024-SA) y de registro acelerado "
         "(Ley 31738, Ley 32319) use <b>Estructura de mercado</b>: Multifuente como tratamiento y Monofuente como comparador, porque "
         "solo donde hay o puede haber competidores cabe esperar un efecto sobre el precio.")
    c1, c2, c3 = st.columns(3)
    dim_did = c1.selectbox("Dimensión", list(dp.DIMENSIONES), index=list(dp.DIMENSIONES).index("fuente"), format_func=dp.DIMENSIONES.get, key="did_dim")
    vals = sorted(precios[dim_did].dropna().unique())
    trat = c2.selectbox("Tratamiento", vals, index=vals.index("Multifuente") if "Multifuente" in vals else 0, key="did_t")
    ctrl_opts = [v for v in vals if v != trat] + ["Resto del listado"]
    ctrl = c3.selectbox("Comparador", ctrl_opts, key="did_c")

    def _serie_conj(ns):
        d = dp.indice_grupo(precios, ns, medida, desde, hasta)
        return pd.Series(d.indice.values, index=pd.DatetimeIndex(d.fecha))
    ns_t = precios.loc[precios[dim_did] == trat, "n"].unique().tolist()
    ns_c = (precios.loc[precios[dim_did] != trat, "n"].unique().tolist() if ctrl == "Resto del listado"
            else precios.loc[precios[dim_did] == ctrl, "n"].unique().tolist())
    st_, sc_ = _serie_conj(ns_t), _serie_conj(ns_c)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=st_.index, y=st_.values, name=f"Tratamiento: {trat}", line=dict(color=SECONDARY_COLOR, width=2.5)))
    fig.add_trace(go.Scatter(x=sc_.index, y=sc_.values, name=f"Comparador: {ctrl}", line=dict(color=PRIMARY_COLOR, width=2.5)))
    add_breaks(fig)
    style_fig(fig, 400)
    grafico(fig)
    rows = []
    for k, b in BREAKS.items():
        r = diff_in_diff(st_, sc_, b, alpha=alpha)
        if r is None:
            continue
        rows.append({"Corte": k, "Norma(s)": LABELS[k], "Desde": b.strftime("%b-%Y"), "Meses pre/post": f"{r.n_pre}/{r.n_post}",
                     "Efecto DiD (%)": round(r.pct_effect, 2), "p": round(r.did_p, 4), "Significancia": significance_label(r.did_p, alpha),
                     "Pre-tendencia p": round(r.pretrend_p, 3), "Tendencias paralelas": "Sí" if r.parallel_trends_ok else "Cuestionable"})
    if rows:
        tabla(pd.DataFrame(rows), hide_index=True)
        caja("Cada fila es un DiD independiente de dos períodos alrededor de un corte (no un modelo conjunto). Si las tendencias "
             "paralelas son cuestionables, el efecto mezcla la diferencia de tendencias previas con el de la norma.")
    else:
        st.info("No hay cortes con suficientes meses a ambos lados en el rango seleccionado.")

# ============================== TAB 5: ETS antes/después =====================
with tab_ets:
    st.subheader("Precio antes y después de cada informe ETS")
    st.markdown(f"Promedio de los **{k_ets} meses previos** frente a los **{k_ets} posteriores** al mes de publicación de cada informe que "
                "evalúa un medicamento del listado. Se contrasta con el mismo cambio en los medicamentos sin informe cercano "
                "(todo el listado y solo su grupo de afinidad).")
    ev = eventos_ets(k_ets)
    if ev.empty:
        st.warning("No hay eventos evaluables.")
    else:
        c1, c2, c3 = st.columns([2, 2, 1])
        dsel = c1.multiselect("Dictamen", dp.DICTAMEN_ORDEN[:3], default=dp.DICTAMEN_ORDEN[:3])
        gsel_e = c2.multiselect("Grupo de afinidad", sorted(ev.grupo.unique()), default=sorted(ev.grupo.unique()))
        comb = c3.checkbox("Incluir combinaciones", value=True)
        evf = ev[ev.dictamen_grupo.isin(dsel) & ev.grupo.isin(gsel_e)]
        if not comb:
            evf = evf[~evf.combinacion]
        evf = evf.drop_duplicates(["id", "n"])
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Pares informe–medicamento", len(evf))
        m2.metric("Mediana var. índice", pct(evf.var_indice.median()))
        m3.metric("Mediana var. precio mediano", pct(evf.var_mediana.median()))
        m4.metric("Mediana diferencia vs. resto", pct(evf.dif_vs_resto.median()))

        tr = dp.trayectorias(precios, evf, k_ets)
        fig = go.Figure()
        for g in dsel:
            d = tr[tr.dictamen_grupo == g] if not tr.empty else tr
            if d.empty:
                continue
            for nm, de in d.groupby("evento"):
                fig.add_trace(go.Scatter(x=de.mes_relativo, y=de.valor_rel, mode="lines", line=dict(color=COL_DICT[g], width=0.8),
                                         opacity=0.25, showlegend=False, hovertemplate=f"{nm}<br>Mes %{{x}}: %{{y:.1f}}<extra></extra>"))
            med = d.groupby("mes_relativo").valor_rel.median().reset_index()
            fig.add_trace(go.Scatter(x=med.mes_relativo, y=med.valor_rel, mode="lines+markers", line=dict(color=COL_DICT[g], width=3),
                                     name=f"{g} (mediana, {d.evento.nunique()} eventos)"))
        fig.add_vline(x=0, line_dash="dash", line_color="#52514e", annotation_text="Publicación")
        fig.add_hline(y=100, line_color="#c3c2b7")
        style_fig(fig, 430, "Trayectoria del índice por producto alrededor de cada informe", "Índice (mes previo = 100)")
        fig.update_layout(hovermode="closest", xaxis_title="Meses respecto de la publicación")
        grafico(fig)

        resumen = (evf.groupby(["grupo_corto", "dictamen_grupo"])
                   .agg(eventos=("id", "size"), sube=("var_indice", lambda s: int((s > 0.01).sum())),
                        estable=("var_indice", lambda s: int((s.abs() <= 0.01).sum())), baja=("var_indice", lambda s: int((s < -0.01).sum())),
                        mediana_indice=("var_indice", "median"), mediana_precio=("var_mediana", "median"),
                        mediana_grupo_sin_ets=("var_indice_grupo", "median"))
                   .reset_index())
        st.markdown("#### Resumen por grupo de afinidad y dictamen")
        tabla(resumen, hide_index=True, column_config={
            "grupo_corto": "Grupo", "dictamen_grupo": "Dictamen", "eventos": "Eventos", "sube": "Suben (>1%)", "estable": "Estables (±1%)",
            "baja": "Bajan (<−1%)", "mediana_indice": st.column_config.NumberColumn("Mediana var. índice", format="percent"),
            "mediana_precio": st.column_config.NumberColumn("Mediana var. precio mediano", format="percent"),
            "mediana_grupo_sin_ets": st.column_config.NumberColumn("Mismo grupo sin ETS", format="percent")})

        det = evf[["fecha_pub", "codigo", "molecula", "grupo_corto", "dictamen", "certeza", "combinacion", "var_indice", "var_mediana",
                   "var_minimo", "var_registros", "var_indice_grupo", "var_indice_resto", "dif_vs_resto", "pdf"]].sort_values("fecha_pub", ascending=False)
        st.markdown("#### Detalle por informe")
        tabla(det, hide_index=True, column_config={
            "fecha_pub": st.column_config.DateColumn("Publicación", format="DD/MM/YYYY"), "codigo": "Informe", "molecula": "Molécula",
            "grupo_corto": "Grupo", "dictamen": "Dictamen", "certeza": "Certeza", "combinacion": "Combinación",
            "var_indice": st.column_config.NumberColumn("Var. índice", format="percent"),
            "var_mediana": st.column_config.NumberColumn("Var. precio mediano", format="percent"),
            "var_minimo": st.column_config.NumberColumn("Var. precio mínimo", format="percent"),
            "var_registros": st.column_config.NumberColumn("Δ registros", format="%.1f"),
            "var_indice_grupo": st.column_config.NumberColumn("Grupo sin ETS", format="percent"),
            "var_indice_resto": st.column_config.NumberColumn("Resto del listado", format="percent"),
            "dif_vs_resto": st.column_config.NumberColumn("Diferencia", format="percent"),
            "pdf": st.column_config.LinkColumn("PDF", display_text="Abrir")})
        st.download_button("⬇️ Descargar eventos ETS (CSV)", det.to_csv(index=False).encode("utf-8-sig"), f"eventos_ets_k{k_ets}.csv", "text/csv")
        caja("Las ETS del INS orientan la cobertura pública. Su efecto sobre el precio minorista privado sería indirecto (señal sobre "
             "la demanda institucional). Con pocos eventos por grupo y ventanas superpuestas, esta comparación es descriptiva.", "warn")

# ============================== TAB 6: ETS conclusiones ======================
with tab_concl:
    st.subheader("Conclusiones de los informes ETS sobre medicamentos del listado")
    ev_ids = vinc.drop_duplicates(["id", "molecula"])
    c1, c2, c3, c4 = st.columns(4)
    unic = vinc.drop_duplicates("id")
    c1.metric("Informes que evalúan el listado", unic.id.nunique())
    c2.metric("A favor", int((unic.dictamen_grupo == "A favor").sum()))
    c3.metric("En contra", int((unic.dictamen_grupo == "En contra").sum()))
    c4.metric("Certeza muy baja o sin evidencia", int(unic.certeza.isin(["Muy baja", "Ausencia de evidencia"]).sum()))

    st.markdown("#### Cronología de informes por molécula")
    ev_ids = ev_ids.sort_values(["grupo_id", "molecula"])
    fig = go.Figure()
    for g, d in ev_ids.groupby("dictamen_grupo"):
        fig.add_trace(go.Scatter(x=d.fecha_pub, y=d.molecula, mode="markers", name=g,
                                 marker=dict(color=COL_DICT[g], symbol=SIMB_DICT[g], size=12, line=dict(color="white", width=1)),
                                 customdata=np.stack([d.codigo, d.dictamen, d.tecnologia.str.slice(0, 100)], axis=-1),
                                 hovertemplate="%{y}<br>%{customdata[0]}<br>%{customdata[1]}<br>%{customdata[2]}…<extra></extra>"))
    fig.add_vrect(x0=MIN_M, x1=MAX_M, fillcolor="#2C5F8A", opacity=0.05, line_width=0, annotation_text="Periodo con precios",
                  annotation_position="top left")
    fig.update_layout(height=max(380, 26 * ev_ids.molecula.nunique()), template="opm", margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=-0.12), yaxis=dict(categoryorder="array", categoryarray=ev_ids.molecula.unique()[::-1]))
    grafico(fig)

    st.markdown("#### Dictamen y certeza por grupo de afinidad")
    tab_dc = (ev_ids[ev_ids.dictamen_grupo != "Sin recomendación formal"].groupby(["grupo_corto", "dictamen_grupo", "certeza"])
              .size().rename("informes").reset_index().pivot_table(index="grupo_corto", columns=["dictamen_grupo", "certeza"], values="informes", fill_value=0))
    tabla(tab_dc)
    caja("La mayoría de las recomendaciones (a favor o en contra) se basan en certeza muy baja o en ausencia de evidencia para la "
         "población solicitada. Un dictamen en contra por ausencia de evidencia no informa sobre el valor del medicamento en sus "
         "indicaciones aprobadas, lo que limita su capacidad de mover el precio de mercado.")

    st.markdown("#### Texto de las conclusiones")
    f1, f2, f3 = st.columns(3)
    gq_c = f1.multiselect("Grupo", sorted(ev_ids.grupo.unique()), key="cg")
    dq_c = f2.multiselect("Dictamen", dp.DICTAMEN_ORDEN, key="cd")
    q = f3.text_input("Buscar en tecnología o conclusión", key="cq")
    lista = ev_ids.copy()
    if gq_c:
        lista = lista[lista.grupo.isin(gq_c)]
    if dq_c:
        lista = lista[lista.dictamen_grupo.isin(dq_c)]
    if q:
        ql = q.lower()
        lista = lista[lista.tecnologia.str.lower().str.contains(ql, regex=False) | lista.conclusion.str.lower().str.contains(ql, regex=False)]
    for r in lista.sort_values("fecha_pub", ascending=False).itertuples():
        with st.expander(f"{r.fecha_pub:%d/%m/%Y} · {r.molecula} · {r.codigo} · {r.dictamen}"):
            st.markdown(f"**Tecnología evaluada:** {r.tecnologia}")
            st.markdown(f"**Conclusión:** {r.conclusion}")
            st.markdown(f"**Grupo:** {r.grupo} · **Certeza:** {r.certeza} · **Solicitante:** {r.solicitante}"
                        + (" · Evalúa una combinación" if r.combinacion else ""))
            st.markdown(f"[Abrir PDF del informe]({r.pdf})")

# ============================== TAB 7: Variación y tendencia =================
with tab_var:
    st.subheader("Variación acumulada y tendencia")
    v = dp.variacion_resumen(precios, desde, hasta)
    v = v[v[dim].isin(grupos_sel)]
    col_var = {"indice": "var_indice", "p_mediana": "var_mediana", "p_min": "var_minimo"}[medida]
    v = v.sort_values([dim, col_var])
    fig = go.Figure()
    for g in grupos_sel:
        d = v[v[dim] == g]
        if d.empty:
            continue
        fig.add_trace(go.Bar(x=d[col_var] * 100, y=d.medicamento.str.slice(0, 40), orientation="h", name=g,
                             marker_color=COLOR_GRUPO[g], text=[pct(x) for x in d[col_var]], textposition="outside",
                             hovertemplate="%{y}<br>%{x:.1f}%<extra>" + g[:40] + "</extra>"))
    fig.add_vline(x=0, line_color="#52514e", line_width=1)
    fig.update_layout(height=max(500, 22 * len(v)), template="opm", barmode="overlay", margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title=f"Variación % ({dp.MEDIDAS[medida].lower()}), {desde:%b-%Y} a {hasta:%b-%Y}",
                      legend=dict(orientation="h", y=-0.06), yaxis=dict(autorange="reversed"))
    grafico(fig)

    st.markdown("#### Resumen por grupo")
    filas = []
    for g in grupos_sel:
        s = serie_grupo(g)
        d = v[v[dim] == g]
        mk = mann_kendall_trend(s.values, alpha=alpha) if len(s) >= 8 else None
        filas.append({"Grupo": g, "Medicamentos": len(d), "Multifuente": int((d.fuente == "Multifuente").sum()),
                      "Índice final": round(float(s.iloc[-1]), 1) if len(s) else np.nan,
                      "Mediana var. índice": d.var_indice.median(), "Mediana var. precio mediano": d.var_mediana.median(),
                      "Mediana var. precio mínimo": d.var_minimo.median(),
                      "Tendencia (Mann-Kendall)": mk.trend if mk else "—", "p MK": round(mk.p_value, 4) if mk else np.nan,
                      "Pendiente de Sen (pts/mes)": round(mk.sen_slope, 3) if mk else np.nan})
    tabla(pd.DataFrame(filas), hide_index=True, column_config={
        "Mediana var. índice": st.column_config.NumberColumn(format="percent"),
        "Mediana var. precio mediano": st.column_config.NumberColumn(format="percent"),
        "Mediana var. precio mínimo": st.column_config.NumberColumn(format="percent")})
    caja("Mann-Kendall con corrección de Hamed-Rao (respaldo Yue-Wang) y pendiente de Sen sobre el índice mensual del grupo. "
         "Cuando el precio mediano cae mucho más que el índice por producto, la baja proviene de la entrada de competidores y no "
         "de rebajas de los productos existentes.")

# ============================== TAB 8: Por medicamento =======================
with tab_med:
    st.subheader("Detalle por medicamento")
    con = catalogo[catalogo.con_serie].sort_values(["grupo_id", "molecula"])
    sel = st.selectbox("Medicamento", con.n.tolist(), format_func=lambda n: f"{ETIQ[n]} · {con.set_index('n').grupo_corto[n]}")
    info = catalogo.set_index("n").loc[sel]
    sn = precios[(precios.n == sel) & (precios.fecha >= desde) & (precios.fecha <= hasta)].sort_values("fecha")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Precio mediano actual", soles(sn.p_mediana.iloc[-1]), pct(sn.p_mediana.iloc[-1] / sn.p_mediana.iloc[0] - 1), delta_color="inverse")
    c2.metric("Precio mínimo actual", soles(sn.p_min.iloc[-1]), pct(sn.p_min.iloc[-1] / sn.p_min.iloc[0] - 1), delta_color="inverse")
    c3.metric("Registros que reportan", int(sn.n_registros.iloc[-1]), int(sn.n_registros.iloc[-1] - sn.n_registros.iloc[0]))
    c4.metric("Precio operativo INEN", soles(info.precio_inen))
    st.caption(f"{info.grupo} · {info.mecanismo} · {info.tipo_molecula} · {info.via} · {info.fuente}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sn.fecha, y=sn.p_max, line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=sn.fecha, y=sn.p_min, fill="tonexty", fillcolor="rgba(138,137,133,0.18)", line=dict(width=0),
                             name="Rango entre registros", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=sn.fecha, y=sn.p_mediana, name="Precio mediano", line=dict(color=ACCENT_COLOR, width=2.5),
                             hovertemplate="S/ %{y:,.2f}<extra>Mediano</extra>"))
    fig.add_trace(go.Scatter(x=sn.fecha, y=sn.p_min, name="Precio mínimo", line=dict(color=OK_COLOR, width=1.8, dash="dot"),
                             hovertemplate="S/ %{y:,.2f}<extra>Mínimo</extra>"))
    if not pd.isna(info.precio_inen):
        fig.add_hline(y=info.precio_inen, line=dict(color="#52514e", dash="dash", width=1), annotation_text="Precio operativo INEN",
                      annotation_position="top left")
    add_ets_markers(fig, [sel], float(sn.p_max.max()) * 1.06)
    add_breaks(fig)
    style_fig(fig, 460, ETIQ[sel], "S/ por unidad")
    grafico(fig)
    vi = vinc[vinc.n == sel].sort_values("fecha_pub", ascending=False)
    if not vi.empty:
        st.markdown("#### Informes ETS")
        ev_m = eventos_ets(k_ets)
        for r in vi.itertuples():
            w = ev_m[(ev_m.id == r.id) & (ev_m.n == sel)] if not ev_m.empty else ev_m
            with st.expander(f"{r.fecha_pub:%d/%m/%Y} · {r.codigo} · {r.dictamen}"):
                st.markdown(f"**Tecnología evaluada:** {r.tecnologia}")
                st.markdown(f"**Conclusión:** {r.conclusion}")
                if not w.empty:
                    w = w.iloc[0]
                    st.markdown(f"**Precio ±{k_ets} meses:** índice {pct(w.var_indice)}, precio mediano {pct(w.var_mediana)}, "
                                f"mismo grupo sin ETS {pct(w.var_indice_grupo)}.")
                st.markdown(f"[Abrir PDF]({r.pdf})")
    st.markdown("#### Precios por registro sanitario")
    rn = registro[(registro.n == sel) & (registro.exclusion == "")]
    if not rn.empty:
        ult = rn.fecha.max()
        t = (rn.sort_values("fecha").groupby("unidad")
             .agg(registro=("registro", "first"), nombre=("nombre", "last"), clasificacion=("clasificacion", "last"), titular=("titular", "last"),
                  desde=("fecha", "min"), hasta=("fecha", "max"), primera=("mediana", "first"), ultima=("mediana", "last")).reset_index())
        t["variacion"] = t.ultima / t.primera - 1
        t["vigente"] = np.where(t.hasta == ult, "Sí", "No")
        tabla(t.drop(columns="unidad").sort_values("ultima"), hide_index=True, column_config={
            "desde": st.column_config.DateColumn("Desde", format="MM/YYYY"), "hasta": st.column_config.DateColumn("Hasta", format="MM/YYYY"),
            "primera": st.column_config.NumberColumn("Primer precio (S/)", format="%.2f"), "ultima": st.column_config.NumberColumn("Último precio (S/)", format="%.2f"),
            "variacion": st.column_config.NumberColumn("Variación", format="percent"), "registro": "Registro sanitario", "nombre": "Nombre comercial",
            "clasificacion": "Clasificación", "titular": "Titular", "vigente": "Reporta en el último mes"})

# ============================== TAB: Entrada de competidores ==================
with tab_ent:
    st.subheader("Entrada de competidores y precio de mercado")
    st.markdown("Una **entrada** es el primer mes en que un registro sanitario nuevo reporta precio para un medicamento que ya se "
                "vendía. Es **nuevo competidor** si su titular no tenía registros previos en ese medicamento. Se compara el promedio "
                "de los k meses previos con el de los k posteriores, frente a los medicamentos sin entradas en ±k meses.")
    ENT = cargar_entradas()
    if ENT.empty:
        st.info("No se encontró data/entradas.csv. Ejecute `python prep.py`.")
    else:
        c1, c2 = st.columns([1, 2])
        tipo_e = c1.radio("Tipo de entrada", ["Nuevo competidor", "Extensión del mismo titular"], key="tipo_e")
        k_e = c2.select_slider("Meses a cada lado", options=[3, 6, 9, 12], value=6, key="k_e")
        evE = eventos_entrada(k_e, tipo_e)
        if evE.empty:
            st.warning("Sin eventos evaluables.")
        else:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Entradas evaluables", len(evE))
            m2.metric("Mediana var. precio mediano", pct(evE.var_mediana.median()), delta_color="off")
            m3.metric("Mediana var. precio mínimo", pct(evE.var_minimo.median()), delta_color="off")
            m4.metric("Mediana var. índice por producto", pct(evE.var_indice.median()), delta_color="off")
            difs = evE.dif_mediana.dropna()
            if len(difs) >= 6:
                w = _st.wilcoxon(difs)
                wm = _st.wilcoxon(evE.dif_minimo.dropna())
                _dm, _dn = difs.median(), evE.dif_minimo.median()
                caja(f"Frente a los medicamentos sin entradas en la misma ventana, el precio mediano {'cae' if _dm < 0 else 'sube'} "
                     f"{pct(abs(_dm), signo=False)} más (Wilcoxon p {'< 0,001' if w.pvalue < 0.001 else '= ' + fnum(w.pvalue, 3)}) y el mínimo {'cae' if _dn < 0 else 'sube'} "
                     f"{pct(abs(_dn), signo=False)} más (p {'< 0,001' if wm.pvalue < 0.001 else '= ' + fnum(wm.pvalue, 3)}). "
                     "El índice por producto casi no se mueve: los productos ya presentes no bajan su precio; el mercado se abarata "
                     "porque el entrante llega más barato.")

            medida_e = st.radio("Medida en la trayectoria", ["p_mediana", "p_min", "indice"], format_func=dp.MEDIDAS.get,
                                horizontal=True, key="med_e")
            trE = dp.trayectorias_entrada(precios, evE, k_e, medida_e)
            fig = go.Figure()
            for nm, d in trE.groupby("evento"):
                fig.add_trace(go.Scatter(x=d.mes_relativo, y=d.valor_rel, mode="lines", line=dict(color=SECONDARY_COLOR, width=0.8),
                                         opacity=0.25, showlegend=False, hovertemplate=f"{nm}<br>Mes %{{x}}: %{{y:.1f}}<extra></extra>"))
            if not trE.empty:
                med = trE.groupby("mes_relativo").valor_rel.median().reset_index()
                q1 = trE.groupby("mes_relativo").valor_rel.quantile(0.25).reset_index()
                q3 = trE.groupby("mes_relativo").valor_rel.quantile(0.75).reset_index()
                fig.add_trace(go.Scatter(x=q3.mes_relativo, y=q3.valor_rel, line=dict(width=0), showlegend=False, hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=q1.mes_relativo, y=q1.valor_rel, fill="tonexty", fillcolor="rgba(217,84,47,0.15)",
                                         line=dict(width=0), name="Rango intercuartílico", hoverinfo="skip"))
                fig.add_trace(go.Scatter(x=med.mes_relativo, y=med.valor_rel, mode="lines+markers", line=dict(color=SECONDARY_COLOR, width=3),
                                         name=f"Mediana ({trE.evento.nunique()} entradas)"))
            fig.add_vline(x=0, line_dash="dash", line_color="#52514e", annotation_text="Entrada")
            fig.add_hline(y=100, line_color="#c3c2b7")
            style_fig(fig, 430, f"{dp.MEDIDAS[medida_e]} alrededor de cada entrada", "Mes previo a la entrada = 100")
            fig.update_layout(hovermode="closest", xaxis_title="Meses respecto de la entrada")
            grafico(fig)

            st.markdown("#### Detalle por entrada")
            det = evE[["primer_mes", "molecula", "grupo_corto", "titular", "nombre", "clasificacion", "competidores_previos",
                       "emitido_a_mas_tardar", "rezago_meses_min", "var_mediana", "var_minimo", "var_indice", "ctrl_mediana",
                       "dif_mediana", "dif_minimo"]].sort_values("primer_mes")
            tabla(det, hide_index=True, column_config={
                "primer_mes": st.column_config.DateColumn("Entrada", format="MM/YYYY"), "molecula": "Molécula", "grupo_corto": "Grupo",
                "titular": "Titular entrante", "nombre": "Nombre comercial", "clasificacion": "Clasificación",
                "competidores_previos": "Competidores previos",
                "emitido_a_mas_tardar": st.column_config.DateColumn("Registro emitido a más tardar", format="MM/YYYY"),
                "rezago_meses_min": st.column_config.NumberColumn("Rezago registro→mercado (≥ meses)", format="%d"),
                "var_mediana": st.column_config.NumberColumn("Var. mediano", format="percent"),
                "var_minimo": st.column_config.NumberColumn("Var. mínimo", format="percent"),
                "var_indice": st.column_config.NumberColumn("Var. índice", format="percent"),
                "ctrl_mediana": st.column_config.NumberColumn("Control (mediano)", format="percent"),
                "dif_mediana": st.column_config.NumberColumn("Dif. mediano", format="percent"),
                "dif_minimo": st.column_config.NumberColumn("Dif. mínimo", format="percent")})
            st.download_button("⬇️ Descargar entradas (CSV)", det.to_csv(index=False).encode("utf-8-sig"), f"entradas_k{k_e}.csv", "text/csv")

            st.markdown("#### Por grupo de afinidad")
            pg = (evE.groupby("grupo_corto").agg(entradas=("n", "size"), medicamentos=("n", "nunique"),
                                                 var_mediana=("var_mediana", "median"), var_minimo=("var_minimo", "median"),
                                                 dif_mediana=("dif_mediana", "median")).reset_index().sort_values("var_mediana"))
            tabla(pg, hide_index=True, column_config={
                "grupo_corto": "Grupo", "entradas": "Entradas", "medicamentos": "Medicamentos",
                "var_mediana": st.column_config.NumberColumn("Mediana var. precio mediano", format="percent"),
                "var_minimo": st.column_config.NumberColumn("Mediana var. precio mínimo", format="percent"),
                "dif_mediana": st.column_config.NumberColumn("Mediana dif. vs control", format="percent")})

        st.markdown("#### Ritmo de entrada de nuevos competidores y normativa")
        st.caption("Así se evalúan las normas por el canal en que actuarían: la Ley 31738 y la Ley 32319 aceleran registros y la "
                   "Ley 32033 y el DS 020-2024-SA promueven genéricos. Una norma eficaz debería aumentar la tasa de entradas.")
        tasa = dp.tasa_entradas(ENT)
        fig = go.Figure(go.Bar(x=tasa.periodo, y=tasa.entradas, marker_color=[("#c3c2b7" if s else PRIMARY_COLOR) for s in tasa.sin_datos],
                               hovertemplate="%{x|%b-%Y}: %{y} entradas<extra></extra>", name="Entradas por trimestre"))
        add_breaks(fig)
        style_fig(fig, 340, None, "Nuevos competidores por trimestre")
        fig.update_layout(hovermode="closest", showlegend=False)
        grafico(fig)
        filas_t = []
        tq = tasa[~tasa.sin_datos]
        for kk, b in BREAKS.items():
            pre, post = tq[tq.periodo < b], tq[tq.periodo >= b]
            if len(pre) < 2 or len(post) < 2:
                continue
            npre, npost = int(pre.entradas.sum()), int(post.entradas.sum())
            r_pre, r_post = npre / len(pre), npost / len(post)
            # Comparación de tasas de Poisson condicionada al total (prueba binomial exacta)
            p_bin = _st.binomtest(npost, npre + npost, len(post) / (len(pre) + len(post))).pvalue if npre + npost > 0 else np.nan
            filas_t.append({"Corte": kk, "Norma(s)": LABELS[kk], "Trimestres pre/post": f"{len(pre)}/{len(post)}",
                            "Tasa pre (por trim.)": round(r_pre, 2), "Tasa post (por trim.)": round(r_post, 2),
                            "Razón de tasas": round(r_post / r_pre, 2) if r_pre > 0 else np.nan, "p (binomial exacta)": round(p_bin, 4)})
        if filas_t:
            tabla(pd.DataFrame(filas_t), hide_index=True)
        feb = ENT[(ENT.primer_mes == pd.Timestamp("2026-02-01")) & (ENT.tipo_entrada == "Nuevo competidor")]
        if not feb.empty:
            caja(f"En feb-2026 aparecen {len(feb)} nuevos competidores a la vez. La cobertura total del tablero de DIGEMID no aumentó ese "
                 "mes (alrededor de 10 600 registros reportan cada mes entre nov-2025 y ago-2026), de modo que no es un cambio de cobertura. "
                 f"Pero sus registros sanitarios se emitieron a más tardar entre {feb.emitido_a_mas_tardar.min():%m/%Y} y "
                 f"{feb.emitido_a_mas_tardar.max():%m/%Y} (rezago mínimo mediano de {int(feb.rezago_meses_min.median())} meses): son "
                 "lanzamientos o primeros reportes de productos ya registrados, no registros nuevos. Solo los emitidos desde "
                 "may-2025 podrían relacionarse con la Ley 32319. La fecha de emisión se infiere de la numeración correlativa de los "
                 "registros y debe confirmarse en la consulta de registro sanitario de DIGEMID.", "warn")

# ============================== TAB: Compra pública vs mercado ================
with tab_pub:
    st.subheader("Precio de compra del INEN frente al mercado privado")
    st.markdown("Compara el **precio operativo INEN** (Catálogo SISMED, ago-2026) con el precio privado del último mes disponible. "
                "Una razón mayor que 1 indica que el INEN paga más que el precio de venta en farmacias privadas.")
    ref_pub = st.radio("Referencia privada", ["Precio mínimo", "Precio mediano"], horizontal=True, key="ref_pub")
    ult = precios.sort_values("fecha").groupby("n").tail(1)
    cp = ult.merge(catalogo[["n"]], on="n")[["n", "medicamento", "molecula", "grupo_corto", "fuente", "precio_inen", "p_min", "p_mediana", "n_registros", "fecha"]]
    cp = cp.dropna(subset=["precio_inen"])
    cp = cp[cp.n.isin(precios.loc[precios[dim].isin(grupos_sel), "n"])]
    col_ref = "p_min" if ref_pub == "Precio mínimo" else "p_mediana"
    cp["razon"] = cp.precio_inen / cp[col_ref]
    cp["brecha_soles"] = cp.precio_inen - cp[col_ref]
    cp = cp.sort_values("razon")
    m1, m2, m3 = st.columns(3)
    m1.metric("Medicamentos comparables", len(cp))
    m2.metric("INEN paga más que el privado", int((cp.razon > 1).sum()))
    m3.metric("Razón mediana INEN / privado", fnum(cp.razon.median(), 2))
    fig = go.Figure(go.Bar(
        x=cp.razon, y=cp.medicamento.str.slice(0, 40), orientation="h",
        marker_color=[BAD_COLOR if r > 1.1 else (WARN_COLOR if r > 0.9 else OK_COLOR) for r in cp.razon],
        text=[fnum(r, 2) for r in cp.razon], textposition="outside",
        customdata=np.stack([cp.precio_inen, cp[col_ref], cp.fuente, cp.n_registros], axis=-1),
        hovertemplate="%{y}<br>INEN: S/ %{customdata[0]:,.2f}<br>Privado: S/ %{customdata[1]:,.2f}<br>%{customdata[2]} · "
                      "%{customdata[3]} registros<extra></extra>"))
    fig.add_vline(x=1, line_color="#52514e", line_dash="dash")
    fig.update_layout(height=max(420, 22 * len(cp)), template="opm", margin=dict(l=10, r=10, t=30, b=10),
                      xaxis_title=f"Precio operativo INEN / {ref_pub.lower()} privado", xaxis_type="log")
    grafico(fig)
    caja("Rojo: el INEN paga más de 10% por encima del privado; ámbar: dentro de ±10%; verde: paga menos. Escala logarítmica. "
         "Una razón alta en un medicamento multifuente es la señal más accionable: existe un competidor más barato en el mercado "
         "que podría usarse como referencia en la próxima compra. Verifique antes que la unidad del precio operativo coincida "
         "con la del precio privado (en alectinib la razón de 0,13 sugiere una diferencia de unidad).")
    tabla(cp[["medicamento", "grupo_corto", "fuente", "n_registros", "precio_inen", "p_min", "p_mediana", "razon", "brecha_soles"]],
          hide_index=True, column_config={
              "medicamento": "Medicamento", "grupo_corto": "Grupo", "fuente": "Mercado", "n_registros": "Registros que reportan",
              "precio_inen": st.column_config.NumberColumn("Precio INEN (S/)", format="%.2f"),
              "p_min": st.column_config.NumberColumn("Mínimo privado (S/)", format="%.2f"),
              "p_mediana": st.column_config.NumberColumn("Mediano privado (S/)", format="%.2f"),
              "razon": st.column_config.NumberColumn("Razón INEN/privado", format="%.2f"),
              "brecha_soles": st.column_config.NumberColumn("Brecha por unidad (S/)", format="%.2f")})
    st.download_button("⬇️ Descargar comparación (CSV)", cp.to_csv(index=False).encode("utf-8-sig"), "compra_publica_vs_mercado.csv", "text/csv")

# ============================== TAB 9: Normativa y metodología ===============
with tab_met:
    st.subheader("Cronología normativa")
    nt = norms_all.copy()
    fig = go.Figure()
    for eje, d in nt.groupby("eje"):
        fig.add_trace(go.Scatter(x=d.fecha_publicacion, y=[eje] * len(d), mode="markers+text", name=eje, text=d.id, textposition="top center",
                                 marker=dict(size=14, color=COL_EJE.get(eje, "#7D7D7D"), symbol="diamond"),
                                 customdata=np.stack([d.norma, d.relevancia, d.mecanismo], axis=-1),
                                 hovertemplate="%{customdata[0]}<br>Relevancia: %{customdata[1]}<br>%{customdata[2]}<extra></extra>"))
    fig.add_vrect(x0=MIN_M, x1=MAX_M, fillcolor="#2C5F8A", opacity=0.06, line_width=0, annotation_text="Serie de precios", annotation_position="top left")
    fig.update_layout(height=380, template="opm", showlegend=False, margin=dict(l=10, r=10, t=30, b=10))
    grafico(fig)
    tabla(nt[["id", "label", "norma", "fecha_publicacion", "fecha_efectiva", "eje", "mecanismo", "relevancia", "nota", "grupo_label"]],
          hide_index=True, column_config={"fecha_publicacion": st.column_config.DateColumn("Publicación", format="DD/MM/YYYY"),
                                          "fecha_efectiva": st.column_config.DateColumn("Corte", format="MM/YYYY")})
    st.caption("Editable sin tocar código en `data/norms.csv`. El corte se ubica en el mes de publicación (mes siguiente si se publicó "
               "después del día 20), con el mismo criterio del Observatorio del Financiamiento Oncológico.")

    st.subheader("Grupos de afinidad terapéutica")
    tabla(catalogo[["n", "medicamento", "molecula", "grupo", "mecanismo", "tipo_molecula", "via", "area", "fuente", "registros_total", "con_serie"]],
          hide_index=True, column_config={"con_serie": "Con serie de precios", "registros_total": "Registros sanitarios"})
    st.markdown("""
**Criterio de agrupación.** Mecanismo de acción y uso clínico (clasificación ATC L01, L02 y M05 adaptada), separando biológicos de
moléculas pequeñas y tumores sólidos de neoplasias hematológicas. Así cada grupo reúne productos con barreras de entrada y
competencia parecidas: los inhibidores de checkpoint y los anticuerpos son biológicos casi siempre monofuente; los TKI multiquinasa,
los inmunomoduladores y la quimioterapia ya tienen genéricos. La **estructura de mercado** (multifuente si hay más de un
registro sanitario de titulares distintos) se calcula de los propios datos.

**Medidas de precio.** El índice por producto es un índice encadenado de Jevons que compara cada producto consigo mismo. El precio
mediano y el mínimo de mercado son la mediana y el mínimo, entre registros sanitarios, del precio mensual reportado. Los grupos se
agregan con otro índice de Jevons encadenado sobre los medicamentos con dato en meses consecutivos.

**ITS de interrupciones múltiples** sobre el índice del grupo, con errores Newey-West (3 rezagos). **Quiebre automático** de dos
tramos con test de Chow, y **Chow en una fecha elegida** (norma o informe ETS). **DiD** de dos grupos sobre log(índice) con
chequeo de tendencias paralelas. **Mann-Kendall** (Hamed-Rao / Yue-Wang) con pendiente de Sen. Las funciones estadísticas son las
mismas del Observatorio del Financiamiento Oncológico (`stats_analysis.py`).

**Entrada de competidores.** Primer mes en que un registro sanitario nuevo reporta precio para un medicamento ya presente
(con 3 meses de margen desde el inicio de la serie). Es nuevo competidor si su titular no tenía registros previos en ese
medicamento. Se compara el promedio de k meses antes y después frente a los medicamentos sin entradas en ±k meses (Wilcoxon de
rangos con signo sobre la diferencia). La tasa trimestral de entradas antes y después de cada norma se compara con una prueba
binomial exacta (tasas de Poisson condicionadas al total). La fecha de emisión del registro se infiere de su numeración
correlativa: un registro se emitió a más tardar cuando aparece en el tablero cualquier registro de su serie con número mayor.

**Compra pública vs mercado.** Razón entre el precio operativo INEN (Catálogo SISMED ago-2026) y el precio mínimo o mediano
privado del último mes, por unidad.

**ETS.** Un informe se vincula con un medicamento cuando su título nombra la molécula. Se compara el promedio de los k meses previos
con el de los k posteriores al mes de publicación, frente a los medicamentos sin informe en ±k meses. La certeza se extrae del
dictamen y del texto de la conclusión.

**Limitaciones.** Precios nominales del canal privado minorista, con 1 a 3 establecimientos reportantes en muchos oncológicos. Sin
datos de ene a mar 2023 (interpolados). La normativa de financiamiento y las ETS actúan sobre la demanda pública, por lo que su efecto
esperado en el precio privado es indirecto. Normas próximas entre sí no se pueden separar con precisión. Los resultados son
asociaciones temporales, no efectos causales.
""")

# ---------------------------------------------------------------------------
# Pie de página
# ---------------------------------------------------------------------------
st.markdown(
    """<p class="source-note">Fuentes: DIGEMID/MINSA, tablero de indicador de precios de productos farmacéuticos (Observatorio Peruano de
Productos Farmacéuticos), extraído el 16/10/2026; INS, informes de Evaluación de Tecnología Sanitaria publicados en gob.pe; El Peruano
(normas legales). Precio operativo INEN: Catálogo SISMED ago-2026.</p>
<p class="source-note" style="text-align:center; margin-top:0.6rem;">© Luis A. Orrego Ferreyros, DDS, Econ., MCE, MMD, PhD(c), CQRM ·
Epidemiólogo y Economista de la Salud — INEN</p>""",
    unsafe_allow_html=True,
)
_cy = pd.Timestamp.today().year
_url = "https://[nombre-de-la-app].streamlit.app/"
st.markdown(
    f"""<div class="citation-box">
<h4>REFERENCIA WEB (formato AMA)</h4>
<p>Orrego-Ferreyros LA. Observatorio de Precios de Medicamentos Oncológicos de Alto Costo: Impacto Normativo y ETS. Instituto Nacional
de Enfermedades Neoplásicas. Published {_cy}. Accessed [Mes Día, Año]. {_url}</p>
<h4>WEB REFERENCE (AMA format)</h4>
<p>Orrego-Ferreyros LA. High-Cost Oncology Drug Price Observatory: Regulatory and HTA Impact. National Institute of Neoplastic
Diseases. Published {_cy}. Accessed [Month Day, Year]. {_url}</p>
</div>""",
    unsafe_allow_html=True,
)
_bib = (f"@misc{{orregoferreyros{_cy}opm,\n  author = {{Orrego-Ferreyros, Luis Alexander}},\n"
        "  title = {Observatorio de Precios de Medicamentos Oncol{\\'o}gicos de Alto Costo: Impacto Normativo y ETS},\n"
        f"  year = {{{_cy}}},\n  publisher = {{Instituto Nacional de Enfermedades Neopl{{\\'a}}sicas}},\n  url = {{{_url}}}\n}}\n")
_ris = ("TY  - ELEC\nAU  - Orrego-Ferreyros, Luis Alexander\nTI  - Observatorio de Precios de Medicamentos Oncológicos de Alto Costo: "
        f"Impacto Normativo y ETS\nPY  - {_cy}\nPB  - Instituto Nacional de Enfermedades Neoplásicas\nUR  - {_url}\nER  - \n")
d1, d2 = st.columns(2)
d1.download_button("⬇️ BibTeX (*.bib)", _bib, file_name="opm_inen.bib", mime="text/plain")
d2.download_button("⬇️ RIS (*.ris)", _ris, file_name="opm_inen.ris", mime="application/x-research-info-systems")
