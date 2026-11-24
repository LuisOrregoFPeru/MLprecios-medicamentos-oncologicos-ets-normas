"""Carga de datos y construcción de series de precios por medicamento y por grupo.

Las medidas de precio disponibles son tres:
  - "indice":    índice encadenado por producto (Jevons). Cambios de precio de cada producto
                 consigo mismo; no cambia cuando entra un competidor más barato.
  - "p_mediana": precio mediano de mercado entre registros sanitarios. Recoge la entrada
                 de genéricos y biosimilares.
  - "p_min":     precio mínimo de mercado.

Las series de grupo se agregan con un índice encadenado de Jevons: cada mes se toma la media
geométrica de las razones de precio (mes actual / mes previo) de los medicamentos del grupo con
dato en ambos meses, y se encadena (inicio del rango = 100). Así un medicamento que entra o sale
de la serie no produce saltos artificiales.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"

MEDIDAS = {
    "indice": "Índice por producto (sin efecto de entrada de competidores)",
    "p_mediana": "Precio mediano de mercado",
    "p_min": "Precio mínimo de mercado",
}
DIMENSIONES = {
    "grupo": "Afinidad terapéutica",
    "tipo_molecula": "Tipo de molécula",
    "via": "Vía de administración",
    "area": "Área clínica",
    "fuente": "Estructura de mercado (fuentes)",
}
DICTAMEN_ORDEN = ["A favor", "En contra", "Mixta", "Sin recomendación formal"]
GAP_MESES = [pd.Timestamp("2023-01-01"), pd.Timestamp("2023-02-01"), pd.Timestamp("2023-03-01")]


def load_catalogo() -> pd.DataFrame:
    return pd.read_csv(DATA / "medicamentos.csv")


def load_precios() -> pd.DataFrame:
    df = pd.read_csv(DATA / "precios_mensual.csv", parse_dates=["fecha"])
    return df.merge(load_catalogo(), on="n", how="left")


def load_registro() -> pd.DataFrame:
    df = pd.read_csv(DATA / "precios_registro.csv", parse_dates=["fecha"])
    df["exclusion"] = df["exclusion"].fillna("")
    return df


def load_norms() -> pd.DataFrame:
    return pd.read_csv(DATA / "norms.csv", parse_dates=["fecha_publicacion", "fecha_efectiva"])


def load_ets() -> tuple[pd.DataFrame, pd.DataFrame]:
    ets = pd.read_csv(DATA / "ets_informes.csv", parse_dates=["fecha_pub"])
    vin = pd.read_csv(DATA / "ets_vinculos.csv")
    vin = vin.merge(ets, on="id").merge(load_catalogo(), on="n", how="left")
    return ets, vin


# ---------------------------------------------------------------------------
# Normas
# ---------------------------------------------------------------------------

def breakpoint_view(norms: pd.DataFrame, grouped: bool) -> pd.DataFrame:
    """Una fila por norma, o una por grupo (fecha efectiva más tardía del grupo)."""
    cols = ["id", "label", "norma", "fecha_publicacion", "fecha_efectiva", "eje", "mecanismo", "relevancia", "nota"]
    if not grouped:
        return norms[cols].copy()
    rows = []
    for gid, g in norms.groupby("grupo_id", sort=False):
        g = g.sort_values("fecha_efectiva")
        rows.append({
            "id": gid, "label": g["grupo_label"].iloc[0], "norma": " + ".join(g["norma"]),
            "fecha_publicacion": g["fecha_publicacion"].max(), "fecha_efectiva": g["fecha_efectiva"].max(),
            "eje": " / ".join(dict.fromkeys(g["eje"])), "mecanismo": " ".join(g["mecanismo"]),
            "relevancia": g["relevancia"].iloc[g["relevancia"].map({"Alta": 0, "Media": 1, "Baja": 2, "Contexto": 3}).argmin()],
            "nota": " ".join(g["nota"].astype(str)),
        })
    return pd.DataFrame(rows)


def breaks_dict(view: pd.DataFrame) -> dict[str, pd.Timestamp]:
    return dict(zip(view["id"], view["fecha_efectiva"]))


# ---------------------------------------------------------------------------
# Series
# ---------------------------------------------------------------------------

def serie_medicamento(precios: pd.DataFrame, n: int, medida: str) -> pd.Series:
    s = precios[precios.n == n].set_index("fecha")[medida].sort_index()
    return s


def completar_mensual(s: pd.Series, interpolar: bool = True) -> pd.Series:
    """Reindexa a meses continuos. Los huecos (p. ej. ene-mar 2023) se interpolan en escala
    logarítmica para que el contador de tiempo del ITS no se desplace."""
    if s.empty:
        return s
    idx = pd.date_range(s.index.min(), s.index.max(), freq="MS")
    s = s.reindex(idx).astype(float)
    if interpolar:
        # Solo se rellenan huecos de hasta 3 meses (como ene-mar 2023); los más largos quedan vacíos
        # para no inventar una tendencia lineal donde el producto no reportó precio.
        falta = s.isna()
        run_id = (falta != falta.shift()).cumsum()
        largo = falta.groupby(run_id).transform("sum")
        rellenable = falta & (largo <= 3)
        interp = np.exp(np.log(s).interpolate(limit_area="inside"))
        s = s.where(~rellenable, interp)
    return s


def indice_grupo(precios: pd.DataFrame, ns: list[int], medida: str,
                 desde: pd.Timestamp | None = None, hasta: pd.Timestamp | None = None) -> pd.DataFrame:
    """Índice encadenado de Jevons del conjunto `ns` (inicio = 100), con el número de
    medicamentos que aporta cada mes."""
    d = precios[precios.n.isin(ns)][["n", "fecha", medida]].dropna()
    if desde is not None:
        d = d[d.fecha >= desde]
    if hasta is not None:
        d = d[d.fecha <= hasta]
    if d.empty:
        return pd.DataFrame(columns=["fecha", "indice", "n_med"])
    piv = d.pivot_table(index="fecha", columns="n", values=medida).sort_index()
    piv = piv.apply(lambda c: completar_mensual(c.dropna()) if c.notna().sum() > 1 else c)
    piv = piv.reindex(pd.date_range(piv.index.min(), piv.index.max(), freq="MS"))
    ratios = np.log(piv / piv.shift(1))
    paso = ratios.mean(axis=1, skipna=True).fillna(0.0)
    paso.iloc[0] = 0.0
    out = pd.DataFrame({"fecha": piv.index, "indice": 100 * np.exp(paso.cumsum()).values,
                        "n_med": piv.notna().sum(axis=1).values})
    out["interpolado"] = out.fecha.isin(GAP_MESES)
    return out


def series_por_dimension(precios: pd.DataFrame, dimension: str, valores: list[str], medida: str,
                         desde=None, hasta=None) -> dict[str, pd.DataFrame]:
    out = {}
    for v in valores:
        ns = precios.loc[precios[dimension] == v, "n"].unique().tolist()
        g = indice_grupo(precios, ns, medida, desde, hasta)
        if not g.empty:
            out[v] = g
    return out


def variacion_resumen(precios: pd.DataFrame, desde=None, hasta=None) -> pd.DataFrame:
    """Variación por medicamento entre el primer y el último mes del rango, en las 3 medidas."""
    d = precios.copy()
    if desde is not None:
        d = d[d.fecha >= desde]
    if hasta is not None:
        d = d[d.fecha <= hasta]
    filas = []
    for n, g in d.sort_values("fecha").groupby("n"):
        a, b = g.iloc[0], g.iloc[-1]
        filas.append({
            "n": n, "desde": a.fecha, "hasta": b.fecha, "meses": len(g),
            "var_indice": b.indice / a.indice - 1, "var_mediana": b.p_mediana / a.p_mediana - 1,
            "var_minimo": b.p_min / a.p_min - 1, "registros_ini": a.n_registros, "registros_fin": b.n_registros,
            "p_mediana_fin": b.p_mediana, "p_min_fin": b.p_min,
        })
    v = pd.DataFrame(filas)
    cat = load_catalogo()
    return v.merge(cat, on="n", how="left")


# ---------------------------------------------------------------------------
# ETS: estudio de eventos
# ---------------------------------------------------------------------------

def ventana(serie_n: pd.DataFrame, fecha: pd.Timestamp, k: int) -> dict:
    t = fecha.to_period("M").to_timestamp()
    pre = serie_n[(serie_n.fecha < t) & (serie_n.fecha >= t - pd.DateOffset(months=k))]
    post = serie_n[(serie_n.fecha > t) & (serie_n.fecha <= t + pd.DateOffset(months=k))]
    out = {"meses_pre": len(pre), "meses_post": len(post)}
    if len(pre) >= 2 and len(post) >= 2:
        for c, nombre in [("indice", "indice"), ("p_mediana", "mediana"), ("p_min", "minimo")]:
            out[f"{nombre}_pre"], out[f"{nombre}_post"] = pre[c].mean(), post[c].mean()
            out[f"var_{nombre}"] = out[f"{nombre}_post"] / out[f"{nombre}_pre"] - 1
        out["var_registros"] = post.n_registros.mean() - pre.n_registros.mean()
    return out


def estudio_eventos(precios: pd.DataFrame, vinc: pd.DataFrame, k: int = 6) -> pd.DataFrame:
    """Variación pre/post (k meses) de cada par informe–medicamento y la misma variación en los
    medicamentos sin informe ETS dentro de ±k meses (resto del listado), en la misma ventana."""
    filas = []
    fechas_ets = vinc.groupby("n")["fecha_pub"].apply(list).to_dict()
    grupos = {n: g for n, g in precios.groupby("n")}
    for _, r in vinc.iterrows():
        sn = grupos.get(r.n)
        if sn is None:
            continue
        w = ventana(sn, r.fecha_pub, k)
        if "var_indice" not in w:
            continue
        ctrl_all, ctrl_grp = [], []
        for n2, g2 in grupos.items():
            if n2 == r.n or any(abs((f - r.fecha_pub).days) <= k * 31 for f in fechas_ets.get(n2, [])):
                continue
            w2 = ventana(g2, r.fecha_pub, k)
            if "var_indice" in w2:
                ctrl_all.append(w2["var_indice"])
                if g2.grupo_id.iloc[0] == r.grupo_id:
                    ctrl_grp.append(w2["var_indice"])
        w["var_indice_resto"] = float(np.mean(ctrl_all)) if ctrl_all else np.nan
        w["var_indice_grupo"] = float(np.mean(ctrl_grp)) if ctrl_grp else np.nan
        w["n_control"] = len(ctrl_all)
        w["dif_vs_resto"] = w["var_indice"] - w["var_indice_resto"]
        filas.append({**r.to_dict(), **w})
    return pd.DataFrame(filas)


def trayectorias(precios: pd.DataFrame, eventos: pd.DataFrame, k: int = 6, medida: str = "indice") -> pd.DataFrame:
    filas = []
    for _, e in eventos.iterrows():
        sn = precios[precios.n == e.n].set_index("fecha")[medida]
        t = e.fecha_pub.to_period("M").to_timestamp()
        previos = sn[sn.index < t]
        if previos.empty:
            continue
        base = previos.iloc[-1]
        for rel in range(-k, k + 1):
            f = t + pd.DateOffset(months=rel)
            if f in sn.index:
                filas.append({"evento": f"{e.codigo} · {e.molecula}", "n": e.n, "grupo_id": e.grupo_id,
                              "dictamen_grupo": e.dictamen_grupo, "mes_relativo": rel, "valor_rel": 100 * sn[f] / base})
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# Entrada de competidores
# ---------------------------------------------------------------------------

def load_entradas() -> pd.DataFrame:
    p = DATA / "entradas.csv"
    if not p.exists():
        return pd.DataFrame()
    e = pd.read_csv(p, parse_dates=["primer_mes", "emitido_a_mas_tardar"])
    return e.merge(load_catalogo()[["n", "medicamento", "molecula", "grupo", "grupo_corto", "grupo_id", "fuente",
                                    "tipo_molecula", "via"]], on="n", how="left")


def estudio_entradas(precios: pd.DataFrame, entradas: pd.DataFrame, k: int = 6) -> pd.DataFrame:
    """Precio promedio de los k meses previos vs. los k posteriores al primer mes de cada entrada, y la misma
    variación en los medicamentos sin ninguna entrada en ±k meses (control contemporáneo)."""
    filas = []
    grupos = {n: g for n, g in precios.groupby("n")}
    fechas = entradas.groupby("n").primer_mes.apply(list).to_dict()
    for _, e in entradas.iterrows():
        sn = grupos.get(e.n)
        if sn is None:
            continue
        w = ventana(sn, e.primer_mes, k)
        if "var_indice" not in w:
            continue
        ctrl_med, ctrl_min = [], []
        for n2, g2 in grupos.items():
            if n2 == e.n or any(abs((f - e.primer_mes).days) <= k * 31 for f in fechas.get(n2, [])):
                continue
            w2 = ventana(g2, e.primer_mes, k)
            if "var_mediana" in w2:
                ctrl_med.append(w2["var_mediana"])
                ctrl_min.append(w2["var_minimo"])
        w["ctrl_mediana"] = float(np.mean(ctrl_med)) if ctrl_med else np.nan
        w["ctrl_minimo"] = float(np.mean(ctrl_min)) if ctrl_min else np.nan
        w["dif_mediana"] = w["var_mediana"] - w["ctrl_mediana"]
        w["dif_minimo"] = w["var_minimo"] - w["ctrl_minimo"]
        filas.append({**e.to_dict(), **w})
    return pd.DataFrame(filas)


def trayectorias_entrada(precios: pd.DataFrame, eventos: pd.DataFrame, k: int, medida: str) -> pd.DataFrame:
    filas = []
    for _, e in eventos.iterrows():
        sn = precios[precios.n == e.n].set_index("fecha")[medida]
        t = e.primer_mes
        previos = sn[sn.index < t]
        if previos.empty:
            continue
        base = previos.iloc[-1]
        for rel in range(-k, k + 1):
            f = t + pd.DateOffset(months=rel)
            if f in sn.index:
                filas.append({"evento": f"{e.molecula} · {e.titular} ({t:%m/%Y})", "tipo_entrada": e.tipo_entrada,
                              "mes_relativo": rel, "valor_rel": 100 * sn[f] / base})
    return pd.DataFrame(filas)


def tasa_entradas(entradas: pd.DataFrame, freq: str = "QS") -> pd.DataFrame:
    """Entradas de nuevos competidores por periodo (trimestre por defecto), con periodos vacíos en cero.
    El trimestre ene-mar 2023 no tiene datos de precios y queda marcado."""
    e = entradas[entradas.tipo_entrada == "Nuevo competidor"]
    s = e.set_index("primer_mes").resample(freq).size()
    idx = pd.date_range("2022-04-01", "2026-07-01", freq=freq)
    s = s.reindex(idx, fill_value=0)
    out = s.rename("entradas").reset_index().rename(columns={"index": "periodo"})
    out["sin_datos"] = out.periodo.eq(pd.Timestamp("2023-01-01"))
    return out
