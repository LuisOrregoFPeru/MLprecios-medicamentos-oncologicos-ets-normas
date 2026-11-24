"""Genera los CSV livianos de data/ a partir de los dos libros Excel de raw/.

    python prep.py

Entradas (raw/):
  - Precios_DIGEMID_alto_costo_2022_2026.xlsx  (hojas Resumen, Serie_mensual, Datos_registro)
  - ETS_INS_informes.xlsx                      (hoja Informes)
Salidas (data/):
  - medicamentos.csv     catálogo con grupos de afinidad terapéutica y atributos
  - precios_mensual.csv  serie mensual por medicamento
  - precios_registro.csv serie mensual por registro sanitario
  - ets_informes.csv     informes ETS del INS
  - ets_vinculos.csv     informe ETS x medicamento del listado
norms.csv se edita a mano (no lo genera este script).
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
RAW, OUT = BASE / "raw", BASE / "data"
PRECIOS = RAW / "Precios_DIGEMID_alto_costo_2022_2026.xlsx"
ETS = RAW / "ETS_INS_informes.xlsx"

# ---------------------------------------------------------------------------
# Catálogo: molécula, patrón para vincular ETS y grupo de afinidad terapéutica.
# Criterio de agrupación: mecanismo de acción y uso clínico (ATC L01/L02/M05),
# separando biológicos de moléculas pequeñas y tumores sólidos de hematología.
# ---------------------------------------------------------------------------
G = {
    "G1": "Inmunoterapia: inhibidores de checkpoint (anti-PD-1/PD-L1)",
    "G2": "Anticuerpos monoclonales y conjugados en tumores sólidos",
    "G3": "Anticuerpos e inmunoterapia en neoplasias hematológicas",
    "G4": "TKI orales dirigidos por biomarcador (ALK, EGFR, HER2, BRAF/MEK)",
    "G5": "TKI orales multiquinasa antiangiogénicos",
    "G6": "Terapia dirigida oral en neoplasias hematológicas (BTK, BCR-ABL, JAK, IMiD)",
    "G7": "Terapia endocrina y dirigida en mama y ovario (SERD, CDK4/6, PARP)",
    "G8": "Quimioterapia citotóxica y formulaciones especiales",
    "G9": "Terapia de soporte y otros",
}
G_CORTO = {"G1": "Checkpoint PD-1/PD-L1", "G2": "mAb tumores sólidos", "G3": "mAb hematología", "G4": "TKI por biomarcador",
           "G5": "TKI multiquinasa", "G6": "Dirigida oral hematología", "G7": "Mama/ovario endocrina-dirigida",
           "G8": "Quimioterapia citotóxica", "G9": "Soporte y otros"}

# n: (molécula, patrón ETS, grupo, blanco/mecanismo, tipo, vía, área)
CAT = {
    1: ("Alectinib", r"alectinib", "G4", "Inhibidor ALK", "Síntesis química", "Oral", "Tumores sólidos"),
    2: ("Amivantamab", r"amivantamab", "G2", "Biespecífico EGFR-MET", "Biológico", "Parenteral", "Tumores sólidos"),
    3: ("Asparaginasa", r"asparaginasa(?!.*pegilad)", "G8", "Enzima antileucémica", "Biológico", "Parenteral", "Hematología"),
    4: ("Atezolizumab", r"atezolizumab", "G1", "Anti-PD-L1", "Biológico", "Parenteral", "Tumores sólidos"),
    5: ("Avelumab", r"avelumab", "G1", "Anti-PD-L1", "Biológico", "Parenteral", "Tumores sólidos"),
    6: ("Bendamustina", r"bendamustina", "G8", "Alquilante", "Síntesis química", "Parenteral", "Hematología"),
    7: ("Bevacizumab", r"bevacizumab", "G2", "Anti-VEGF", "Biológico", "Parenteral", "Tumores sólidos"),
    8: ("Binimetinib", r"binimetinib", "G4", "Inhibidor MEK", "Síntesis química", "Oral", "Tumores sólidos"),
    9: ("Blinatumomab", r"blinatumomab", "G3", "Biespecífico CD19-CD3", "Biológico", "Parenteral", "Hematología"),
    10: ("Cabozantinib", r"cabozantinib", "G5", "Multiquinasa (MET, VEGFR, AXL)", "Síntesis química", "Oral", "Tumores sólidos"),
    11: ("Cetuximab", r"cetuximab", "G2", "Anti-EGFR", "Biológico", "Parenteral", "Tumores sólidos"),
    12: ("Clofarabina", r"clofarabina", "G8", "Antimetabolito (purina)", "Síntesis química", "Parenteral", "Hematología"),
    13: ("Doxorrubicina liposomal", r"doxorr?ubicina\s+liposomal", "G8", "Antraciclina liposomal", "Síntesis química", "Parenteral", "Tumores sólidos"),
    14: ("Daratumumab", r"daratumumab", "G3", "Anti-CD38", "Biológico", "Parenteral", "Hematología"),
    15: ("Daratumumab", r"daratumumab", "G3", "Anti-CD38", "Biológico", "Parenteral", "Hematología"),
    16: ("Denosumab", r"denosumab", "G9", "Anti-RANKL", "Biológico", "Parenteral", "Soporte"),
    17: ("Durvalumab", r"durvalumab", "G1", "Anti-PD-L1", "Biológico", "Parenteral", "Tumores sólidos"),
    18: ("Durvalumab", r"durvalumab", "G1", "Anti-PD-L1", "Biológico", "Parenteral", "Tumores sólidos"),
    19: ("Encorafenib", r"encorafenib", "G4", "Inhibidor BRAF", "Síntesis química", "Oral", "Tumores sólidos"),
    20: ("Foscarnet", r"foscarnet", "G9", "Antiviral", "Síntesis química", "Parenteral", "Soporte"),
    21: ("Fulvestrant", r"fulvestrant", "G7", "SERD", "Síntesis química", "Parenteral", "Tumores sólidos"),
    22: ("Ibrutinib", r"ibrutinib", "G6", "Inhibidor BTK", "Síntesis química", "Oral", "Hematología"),
    23: ("Idarrubicina", r"idarr?ubicina", "G8", "Antraciclina", "Síntesis química", "Parenteral", "Hematología"),
    24: ("Lapatinib", r"lapatinib", "G4", "Inhibidor HER2/EGFR", "Síntesis química", "Oral", "Tumores sólidos"),
    25: ("Lenalidomida", r"lenalidomida", "G6", "Inmunomodulador (IMiD)", "Síntesis química", "Oral", "Hematología"),
    26: ("Metoxaleno", r"metoxaleno", "G9", "Psoraleno (fotoféresis)", "Síntesis química", "Parenteral", "Soporte"),
    27: ("Nelarabina", r"nelarabina", "G8", "Antimetabolito (purina)", "Síntesis química", "Parenteral", "Hematología"),
    28: ("Nilotinib", r"nilotinib", "G6", "Inhibidor BCR-ABL", "Síntesis química", "Oral", "Hematología"),
    29: ("Nivolumab", r"nivolumab", "G1", "Anti-PD-1", "Biológico", "Parenteral", "Tumores sólidos"),
    30: ("Nivolumab", r"nivolumab", "G1", "Anti-PD-1", "Biológico", "Parenteral", "Tumores sólidos"),
    31: ("Olaparib", r"olaparib", "G7", "Inhibidor PARP", "Síntesis química", "Oral", "Tumores sólidos"),
    32: ("Osimertinib", r"osimertinib", "G4", "Inhibidor EGFR", "Síntesis química", "Oral", "Tumores sólidos"),
    33: ("Pazopanib", r"pazopanib", "G5", "Multiquinasa (VEGFR, PDGFR)", "Síntesis química", "Oral", "Tumores sólidos"),
    34: ("Pegaspargasa", r"pegaspargasa|asparaginasa.*pegilad", "G8", "Enzima antileucémica pegilada", "Biológico", "Parenteral", "Hematología"),
    35: ("Pembrolizumab", r"pembrolizumab", "G1", "Anti-PD-1", "Biológico", "Parenteral", "Tumores sólidos"),
    36: ("Pemetrexed", r"pemetrexed", "G8", "Antifolato", "Síntesis química", "Parenteral", "Tumores sólidos"),
    37: ("Pertuzumab", r"pertuzumab", "G2", "Anti-HER2", "Biológico", "Parenteral", "Tumores sólidos"),
    38: ("Plerixafor", r"plerixafor", "G9", "Antagonista CXCR4 (movilización)", "Síntesis química", "Parenteral", "Soporte"),
    39: ("Ribociclib", r"ribociclib", "G7", "Inhibidor CDK4/6", "Síntesis química", "Oral", "Tumores sólidos"),
    40: ("Rituximab SC", r"rituximab", "G3", "Anti-CD20", "Biológico", "Parenteral", "Hematología"),
    41: ("Ruxolitinib", r"ruxolitinib", "G6", "Inhibidor JAK1/2", "Síntesis química", "Oral", "Hematología"),
    42: ("Sorafenib", r"sorafenib", "G5", "Multiquinasa (VEGFR, RAF)", "Síntesis química", "Oral", "Tumores sólidos"),
    43: ("Trastuzumab deruxtecan", r"trastuzumab\s+deruxtecan", "G2", "Conjugado anti-HER2", "Biológico", "Parenteral", "Tumores sólidos"),
    44: ("Trióxido de arsénico", r"tri[oó]xido\s+de\s+ars[eé]nico", "G8", "Diferenciador (LPA)", "Síntesis química", "Parenteral", "Hematología"),
    45: ("Vemurafenib", r"vemurafenib", "G4", "Inhibidor BRAF", "Síntesis química", "Oral", "Tumores sólidos"),
}


def _norm(s) -> str:
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


def dictamen_grupo(d: str) -> str:
    d = str(d)
    if d.startswith("A favor"):
        return "A favor"
    if d.startswith("En contra") or d.startswith("No favorable"):
        return "En contra"
    if d.startswith("Mixta"):
        return "Mixta"
    return "Sin recomendación formal"


def certeza(dictamen: str, conclusion: str) -> str:
    t = _norm(f"{dictamen} {conclusion}")
    if "ausencia de evidencia" in t or "sin evidencia" in t:
        return "Ausencia de evidencia"
    if "muy baja" in t:
        return "Muy baja"
    if "certeza baja" in t or "certeza global de la evidencia baja" in t or "evidencia baja" in t:
        return "Baja"
    if "moderada" in t:
        return "Moderada"
    if "certeza alta" in t:
        return "Alta"
    return "No consignada"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    # ---------------- precios
    res = pd.read_excel(PRECIOS, "Resumen")
    res = res[pd.to_numeric(res["N°"], errors="coerce").notna()].copy()
    res["n"] = res["N°"].astype(int)
    serie = pd.read_excel(PRECIOS, "Serie_mensual").rename(columns={
        "N°": "n", "Medicamento": "medicamento", "Periodo": "fecha", "Registros que reportan": "n_registros",
        "Reportantes (suma)": "reportantes", "Precio mínimo (S/)": "p_min", "Precio mediano (S/)": "p_mediana",
        "Precio máximo (S/)": "p_max", "Índice encadenado (inicio=100)": "indice"})
    serie["fecha"] = pd.to_datetime(serie["fecha"]).dt.to_period("M").dt.to_timestamp()
    serie = serie[["n", "fecha", "n_registros", "reportantes", "p_min", "p_mediana", "p_max", "indice"]]
    serie.to_csv(OUT / "precios_mensual.csv", index=False)

    reg = pd.read_excel(PRECIOS, "Datos_registro").rename(columns={
        "N°": "n", "Periodo": "fecha", "Registro sanitario": "registro", "Unidad de análisis (RS | presentación | fracción)": "unidad",
        "Nombre comercial": "nombre", "Clasificación": "clasificacion", "Titular": "titular", "Fabricante": "fabricante",
        "País": "pais", "Reportantes": "reportantes", "Media (S/ unidad)": "media", "Mediana (S/ unidad)": "mediana",
        "Moda (S/ unidad)": "moda", "Exclusión": "exclusion"})
    reg["fecha"] = pd.to_datetime(reg["fecha"]).dt.to_period("M").dt.to_timestamp()
    reg["exclusion"] = reg["exclusion"].fillna("")
    reg.drop(columns=["Medicamento"]).to_csv(OUT / "precios_registro.csv", index=False)

    # ---------------- catálogo
    validos = reg[reg.exclusion == ""]
    n_reg = validos.groupby("n").registro.nunique()
    # Fuente: competidores reales = titulares distintos (dos registros del mismo titular no compiten).
    # Se normaliza el titular a su primera palabra significativa para no contar variantes de escritura
    # ("ROCHE FARMA (PERU) S.A." y "ROCHE FARMA PERU S.A.") como competidores distintos.
    def _tit_key(t):
        w = [x for x in re.sub(r"[^A-Z ]", " ", _norm(t).upper()).split() if x not in {"LABORATORIOS", "LABORATORIO", "DROGUERIA", "CORPORACION", "PHARMA", "S", "A", "SAC", "EIRL"}]
        return w[0] if w else ""
    alias = {"AGNOVARTIS": "NOVARTIS", "J": "JOHNSON", "JANSSEN": "JOHNSON"}
    validos = validos.assign(tit_key=validos.titular.map(lambda t: alias.get(_tit_key(t), _tit_key(t))))
    # Titular canónico por registro (el más frecuente) y competidores = titulares canónicos distintos.
    tit_reg = validos.groupby(["n", "registro"]).tit_key.agg(lambda s: s.value_counts().index[0]).reset_index()
    n_tit = tit_reg.groupby("n").tit_key.nunique()
    n_gen = validos[validos.clasificacion.str.upper() == "GENERICO"].groupby("n").registro.nunique()
    filas = []
    for _, r in res.iterrows():
        mol, pat, g, blanco, tipo, via, area = CAT[r.n]
        nr = int(n_reg.get(r.n, 0))
        filas.append({
            "n": r.n, "medicamento": str(r["Medicamento (listado INEN)"]).strip(), "molecula": mol, "patron_ets": pat,
            "grupo_id": g, "grupo": G[g], "grupo_corto": G_CORTO[g], "mecanismo": blanco, "tipo_molecula": tipo,
            "via": via, "area": area, "registros_total": nr,
            "titulares_total": int(n_tit.get(r.n, 0)),
            "fuente": "Sin datos" if nr == 0 else ("Multifuente" if (nr > 1 and n_tit.get(r.n, 0) > 1) else "Monofuente"),
            "registros_genericos": int(n_gen.get(r.n, 0)),
            "precio_inen": pd.to_numeric(r.get("Precio operativo INEN ago-2026 (S/)"), errors="coerce"),
            "con_serie": r.n in set(serie.n), "nota": r.get("Nota", ""),
        })
    cat = pd.DataFrame(filas)
    cat.to_csv(OUT / "medicamentos.csv", index=False)

    # ---------------- entrada de competidores
    # Una entrada es el primer mes en que un registro sanitario reporta precio para el medicamento, si ese
    # registro no estaba desde el inicio de la serie (se exigen 3 meses de margen). Es "nuevo competidor"
    # cuando su titular canónico no tenía ningún registro previo en ese medicamento; si no, es una
    # "extensión del mismo titular" (otra presentación o registro del mismo laboratorio).
    t0 = validos.fecha.min()
    prim = (validos.groupby(["n", "registro"])
            .agg(primer_mes=("fecha", "min"), titular=("titular", "first"), nombre=("nombre", "first"),
                 clasificacion=("clasificacion", "first"), pais=("pais", "first")).reset_index()
            .merge(tit_reg, on=["n", "registro"]))
    filas_e = []
    for _, e in prim[prim.primer_mes >= t0 + pd.DateOffset(months=3)].iterrows():
        previos = prim[(prim.n == e.n) & (prim.primer_mes < e.primer_mes)]
        if previos.empty:
            continue
        nuevo = e.tit_key not in set(previos.tit_key)
        filas_e.append({**e.to_dict(), "tipo_entrada": "Nuevo competidor" if nuevo else "Extensión del mismo titular",
                        "competidores_previos": previos.tit_key.nunique()})
    ent = pd.DataFrame(filas_e)

    # Fecha de emisión aproximada del registro: los números de registro se asignan en orden, así que un
    # registro fue emitido a más tardar cuando aparece en el tablero cualquier registro de su serie (EE, BE...)
    # con número igual o mayor. raw/registro_primer_mes_tablero.csv = primer mes de cada registro en TODO el
    # tablero DIGEMID (16 351 registros), extraído del mismo modelo Power BI.
    ref_path = RAW / "registro_primer_mes_tablero.csv"
    if ref_path.exists() and not ent.empty:
        ref = pd.read_csv(ref_path)
        ref["rs"] = ref.rs.astype(str).str.upper().str.strip()
        mm = ref.rs.str.extract(r"^([A-Z]+)(\d+)$")
        ref["pref"], ref["num"] = mm[0], pd.to_numeric(mm[1])
        ref["fmin"] = pd.to_datetime(ref.fmin, errors="coerce")
        ref = ref.dropna(subset=["num", "fmin"])

        def a_mas_tardar(rs):
            m2 = re.match(r"^([A-Z]+)(\d+)$", str(rs).upper())
            if not m2:
                return pd.NaT
            sub = ref[(ref.pref == m2.group(1)) & (ref.num >= int(m2.group(2)))]
            return sub.fmin.min() if not sub.empty else pd.NaT
        ent["emitido_a_mas_tardar"] = ent.registro.map(a_mas_tardar)
        ent["rezago_meses_min"] = ((ent.primer_mes.dt.year - ent.emitido_a_mas_tardar.dt.year) * 12
                                   + ent.primer_mes.dt.month - ent.emitido_a_mas_tardar.dt.month)
    ent.drop(columns=["tit_key"]).to_csv(OUT / "entradas.csv", index=False)

    # ---------------- ETS
    ets = pd.read_excel(ETS, "Informes").rename(columns={
        "N°": "id", "Código del informe": "codigo", "Tipo": "tipo", "Tecnología evaluada / población": "tecnologia",
        "Dictamen": "dictamen", "Conclusión / recomendación (texto del informe)": "conclusion", "Área terapéutica": "area",
        "Tipo de tecnología": "tipo_tec", "Solicitante (según listado)": "solicitante", "Fecha de publicación": "fecha_pub",
        "Año de publicación": "anio_pub", "Enlace al PDF": "pdf"})
    ets["fecha_pub"] = pd.to_datetime(ets["fecha_pub"])
    ets["dictamen_grupo"] = ets["dictamen"].map(dictamen_grupo)
    ets["certeza"] = [certeza(d, c) if g != "Sin recomendación formal" else "No aplica (síntesis de evidencia)"
                      for d, c, g in zip(ets.dictamen, ets.conclusion, ets.dictamen_grupo)]
    ets.to_csv(OUT / "ets_informes.csv", index=False)

    vin = []
    for _, e in ets.iterrows():
        t = _norm(e.tecnologia)
        hits = [n for n, v in CAT.items() if re.search(_norm(v[1]), t)]
        mols = {CAT[n][0] for n in hits}
        for n in hits:
            vin.append({"id": e.id, "n": n, "combinacion": len(mols) > 1})
    pd.DataFrame(vin).to_csv(OUT / "ets_vinculos.csv", index=False)
    print(f"{len(cat)} medicamentos, {serie.n.nunique()} con serie, {len(ets)} informes, {len(vin)} vínculos")


if __name__ == "__main__":
    main()
