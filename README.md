# 💊 Observatorio de Precios de Medicamentos Oncológicos de Alto Costo: Impacto Normativo y ETS
### High-Cost Oncology Drug Price Observatory: Regulatory and HTA Impact

Dashboard en **Streamlit** construido con la misma arquitectura y estilo que el
[Observatorio del Financiamiento Oncológico INEN](../inen-sisfissal-dashboard) (encabezado, panel de control,
cronología normativa editable en `data/norms.csv`, ITS multi-quiebre con Newey-West, Chow, DiD, Mann-Kendall).
Aquí la variable de resultado es el **precio de venta en farmacias privadas** (DIGEMID, Observatorio Peruano de
Productos Farmacéuticos) de 38 medicamentos de alto costo del listado INEN, mar-2022 a ago-2026.

> **Cita sugerida**
> Orrego-Ferreyros LA (2026). *Observatorio de Precios de Medicamentos Oncológicos de Alto Costo: Impacto Normativo y ETS*.
> Lima, Perú: Instituto Nacional de Enfermedades Neoplásicas.

## ✨ Qué analiza

- **Tres medidas de precio**: índice por producto (Jevons encadenado, sin efecto de entrada de competidores),
  precio mediano y precio mínimo de mercado (sí recogen la entrada de genéricos y biosimilares).
- **Agrupación por afinidad terapéutica** (9 grupos por mecanismo y uso clínico) o por tipo de molécula
  (biológico/síntesis), vía (oral/parenteral), área (tumores sólidos/hematología/soporte) y estructura de mercado
  (multifuente/monofuente, calculada de los registros sanitarios y titulares).
- **Impacto normativo**: ITS de interrupciones múltiples por grupo, comparación de cambios de nivel entre grupos
  (mapa de calor), quiebre automático, Chow en una norma o en un informe ETS, y DiD entre grupos
  (p. ej. multifuente vs monofuente para las normas de genéricos y de registro acelerado).
- **Entrada de competidores** (hallazgo central): estudio de eventos del precio mediano, mínimo e índice alrededor de cada
  entrada de un nuevo titular, contra medicamentos sin entradas; ritmo trimestral de entradas antes y después de cada norma;
  fecha de emisión aproximada de cada registro para distinguir lanzamientos de registros nuevos.
- **Compra pública vs mercado**: razón precio operativo INEN / precio privado mínimo o mediano por medicamento.
- **ETS del INS**: precio ±k meses alrededor de cada informe que evalúa un medicamento del listado, por dictamen
  y grupo, contrastado con el resto del listado y con el mismo grupo sin informe; cronología y texto de las
  conclusiones con su certeza de evidencia.
- **Variación y tendencia** por medicamento y grupo (Mann-Kendall Hamed-Rao y pendiente de Sen).

## 📁 Estructura

```
precios-normativa-dashboard/
├── app.py                 # Interfaz (9 pestañas)
├── data_processing.py     # Carga, índices de grupo, estudio de eventos ETS
├── stats_analysis.py      # Mismo módulo estadístico del Observatorio del Financiamiento Oncológico
├── prep.py                # Regenera data/*.csv desde raw/*.xlsx y define los grupos de afinidad
├── raw/                   # Libros fuente y primer mes de cada registro en todo el tablero DIGEMID
├── data/
│   ├── norms.csv          # Cronología normativa editable (puntos de corte)
│   ├── medicamentos.csv   # Catálogo con grupo de afinidad y atributos
│   ├── precios_mensual.csv, precios_registro.csv
│   ├── ets_informes.csv, ets_vinculos.csv
│   └── entradas.csv       # Entradas de competidores (generado por prep.py)
├── assets/opm_logo.svg
└── .streamlit/config.toml
```

## 🚀 Ejecutar y desplegar

```bash
pip install -r requirements.txt
streamlit run app.py
```
Para Streamlit Community Cloud: suba la carpeta a GitHub y elija `app.py` como archivo principal.

Para actualizar: reemplace los Excel de `raw/` y ejecute `python prep.py`. Las normas se editan en `data/norms.csv`
(columna `por_defecto` = 1 para incluirlas en el ITS al abrir la app; `grupo_id` agrupa normas cercanas). Los grupos
de afinidad se editan en el diccionario `CAT` de `prep.py`.

## 📐 Notas metodológicas

- Corte normativo en el mes de publicación (mes siguiente si se publicó después del día 20), igual que el observatorio hermano.
- El tablero de DIGEMID no tiene datos de ene a mar 2023: esos meses se interpolan en escala logarítmica; huecos de más de
  3 meses de un producto no se rellenan.
- Normas incluidas por defecto (relevancia alta y suficientes meses a ambos lados): RM 945/964-2022, Ley 31738, Ley 32033
  (con DU 005-2024 y RM 220-2024), DS 020-2024-SA (con DS 018-2024-SA, RM 820 y 885-2024) y Ley 32319.
- Las ETS y la normativa de financiamiento actúan sobre la demanda pública; el efecto sobre el precio privado es
  indirecto y los resultados son asociaciones temporales.
