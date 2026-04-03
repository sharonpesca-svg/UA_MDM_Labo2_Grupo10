# Requisitos: EDA + Dashboard Streamlit — PetFinder Adoption Prediction

> Revisá este documento antes de comenzar la construcción. Marcá secciones con ✅ / ❌ / 🔄 según aprobación.

---

## 1. Contexto del dataset

| Item | Detalle |
|---|---|
| Archivo fuente | `input/petfinder-adoption-prediction/train/train.csv` |
| Filas | 14.993 |
| Columnas | 24 |
| Target | `AdoptionSpeed` (ordinal 0–4) |
| Datos auxiliares | `breed_labels.csv`, `color_labels.csv`, `state_labels.csv` |

### Target — AdoptionSpeed
| Valor | Significado |
|---|---|
| 0 | Adoptado el mismo día (410 casos — clase muy minoritaria) |
| 1 | Adoptado en la primera semana (3.090) |
| 2 | Adoptado en el primer mes (4.037) |
| 3 | Adoptado en 2-3 meses (3.259) |
| 4 | Sin adopción después de 100 días (4.197 — clase mayoritaria) |

### Valores faltantes relevantes
- `Name`: 1.265 nulos (~8,4%) → indicador binario `tiene_nombre`
- `Description`: 13 nulos (~0,1%) → se excluyen del análisis de texto
- `PhotoAmt`: tipo float pero sin NaNs reales (puede redondearse a int)

### Observaciones clave del análisis inicial
- **84,5% de las mascotas son gratuitas** (Fee = 0); Fee llega hasta 3.000 MYR
- **Solo 574 perfiles tienen video** (3,8%), la mayoría tiene 0
- **Edad**: va de 0 a 255 meses — hay outliers evidentes (mascotas de 21 años)
- **Quantity**: hasta 20 mascotas en un solo perfil (3.428 perfiles son grupos)
- El dataset proviene de **15 estados de Malasia** (Selangor y KL concentran la mayoría)
- Las razas están codificadas con IDs — se decodificarán usando `breed_labels.csv`

---

## 2. Producto 1 — Notebook Jupyter de EDA

### Archivo de salida
`grupo10/EDA_Completo.ipynb` 

### Estructura de secciones

#### 0. Setup y carga de datos
- Importar librerías (pandas, numpy, matplotlib, seaborn, plotly, wordcloud)
- Cargar `train.csv` + labels auxiliares y hacer joins para decodificar IDs
- Mostrar shape, dtypes, y primeras filas

#### 1. Calidad de datos
- Mapa de nulos (heatmap con missingno o seaborn)
- Tipos de variable y cardinalidades
- Detección de outliers en Age, Fee, Quantity (boxplots)
- Distribución de `PhotoAmt` (float → ¿hay valores .5?)

#### 2. Variable target — AdoptionSpeed
- Distribución absoluta y relativa (barplot)
- Desequilibrio de clases: ratio clase mayoritaria vs minoritaria
- Agrupamiento binario sugerido: `rápido` (0–1) vs `lento` (2–4) — solo para exploración visual

#### 3. Tipo de animal — Perros vs Gatos
- Conteo y proporción de cada tipo
- AdoptionSpeed por tipo (barplot apilado / heatmap normalizado)
- ¿Qué tipo se adopta más rápido?

#### 4. Demografía — Edad, Género, Raza
- Distribución de Age por tipo (histograma + KDE)
- Age vs AdoptionSpeed (violin plot o boxplot por clase)
- Género: distribución general y vs AdoptionSpeed
- Top 15 razas primarias (Breed1) por frecuencia — separado por tipo
- Razas mixtas: proporción de perfiles con Breed2 > 0

#### 5. Características físicas
- MaturitySize y FurLength: distribución y relación con AdoptionSpeed
- Paleta de colores (Color1/2/3): ¿qué colores predominan? ¿influye en adopción?
- Combinaciones de color más frecuentes

#### 6. Estado de salud
- Distribución de Vaccinated, Dewormed, Sterilized, Health (countplot agrupado)
- Relación de cada variable de salud con AdoptionSpeed (heatmap de proporciones)
- ¿Las mascotas con las 3 vacunas/desparasitadas/esterilizadas se adoptan más rápido?
- Índice de salud compuesto (suma de estados positivos) vs AdoptionSpeed

#### 7. Media — Fotos y Videos
- Distribución de PhotoAmt (histograma, foco en 0–10)
- Impacto de tener fotos vs no tener (AdoptionSpeed por tramos: 0, 1-3, 4-7, 8+)
- ¿Tener video marca diferencia? (boxplot VideoAmt=0 vs >0)

#### 8. Fee — Costo de adopción
- Distribución del fee (log-scale por los outliers)
- Gratuito vs pago: AdoptionSpeed comparativo
- Fee por estado geográfico (¿dónde se cobra más?)

#### 9. Geografía — Estados de Malasia
- Distribución de perfiles por estado (barplot horizontal)
- AdoptionSpeed promedio por estado (heatmap o barplot)
- Cantidad de rescatistas únicos por estado

#### 10. Rescatistas (RescuerID)
- Distribución de perfiles por rescatista (¿hay rescatistas dominantes?)
- Top rescatistas por volumen y por tasa de adopción rápida

#### 11. Nombre de la mascota
- ¿Tiene nombre vs sin nombre? → impacto en AdoptionSpeed
- WordCloud de nombres — separado perros / gatos
- Nombres más frecuentes

#### 12. Descripción (texto libre)
- Longitud del texto (caracteres y palabras) vs AdoptionSpeed
- WordCloud del corpus completo y por clase de AdoptionSpeed
- ¿Los textos más largos se correlacionan con menor velocidad de adopción?

#### 13. Listings grupales (Quantity > 1)
- Proporción de listings individuales vs grupales
- AdoptionSpeed para grupos vs individuales
- Distribución de Quantity

#### 14. Análisis multivariado
- Matriz de correlación (variables numéricas)
- Pairplot de variables numéricas clave (Age, Fee, PhotoAmt, Quantity) coloreado por AdoptionSpeed
- Análisis de interacción: Type × Health × AdoptionSpeed

#### 15. Síntesis — Hallazgos principales
- Tabla resumen de los 5-8 factores con mayor impacto observado sobre AdoptionSpeed
- Conclusiones narrativas por sección

### Lineamientos técnicos del notebook
- Cada sección con celda markdown de título + interpretación posterior al gráfico
- Paleta de colores consistente (una por clase de AdoptionSpeed)
- Todos los gráficos con título, labels y fuente si aplica
- No usar datos de test en ninguna celda

---

## 3. Producto 2 — Dashboard Streamlit

### Archivo de salida
`grupo10/dashboard_petfinder.py` 

### Tecnologías
- `streamlit`, `plotly express`, `pandas`, `wordcloud` (render como imagen)

### Estructura de pantallas / secciones

#### Sidebar — Filtros globales
- Tipo de animal (Perro / Gato / Ambos)
- Estado geográfico (multiselect)
- Rango de edad (slider en meses)
- AdoptionSpeed (multiselect para filtrar clases)

Todos los gráficos del dashboard reaccionan a estos filtros.

#### Pestaña 1: Visión General
- KPIs en cards: total de mascotas, % adoptadas rápido (≤7 días), fee promedio, foto promedio
- Distribución del target (bar chart interactivo Plotly)
- Donut chart: Perros vs Gatos

#### Pestaña 2: Características de las Mascotas
- Histograma de Age con slider de bins (Plotly)
- Heatmap interactivo: variable categórica (selector) × AdoptionSpeed — proporciones normalizadas
  - Variables disponibles: Vaccinated, Dewormed, Sterilized, Gender, MaturitySize, FurLength, Health
- Barplot Top 10 razas por tipo con filtro por AdoptionSpeed

#### Pestaña 3: Geografía
- Barplot horizontal: distribución de mascotas por estado
- Tabla de métricas por estado: cantidad, fee promedio, AdoptionSpeed promedio, % adoptados en <7 días
- (Opcional si hay tiempo) Mapa choropleth de Malasia con `plotly` o `folium`

#### Pestaña 4: Media y Fee
- Scatter plot: PhotoAmt vs AdoptionSpeed (con jitter, coloreado por Type)
- Box plot: Fee por AdoptionSpeed (log scale toggle)
- Métrica: tasa de adopción rápida según tramo de fotos (0, 1-3, 4+)

#### Pestaña 5: Análisis de Texto
- WordCloud de Nombres (con toggle Perro/Gato) — renderizado como imagen en st.image
- Histograma de longitud de descripción (palabras) por AdoptionSpeed
- Top 20 palabras más frecuentes en descripciones (barplot, con stopwords en inglés/malayo)

### Lineamientos técnicos del dashboard
- El path al CSV debe ser configurable (constante al inicio del archivo o argumento)
- Caching con `@st.cache_data` en la función de carga y merge de datos
- Manejo de estados vacíos (si el filtro devuelve 0 filas, mostrar aviso)
- Instrucciones de ejecución en comentario al inicio: `streamlit run dashboard_petfinder.py`

---

## 4. Dependencias Python requeridas

```
pandas
numpy
matplotlib
seaborn
plotly
streamlit
wordcloud
```

*(missingno es opcional para el mapa de nulos)*

---

## 5. Preguntas abiertas para confirmar antes de construir

1. **Nombre de los archivos de salida** — ¿`04_EDA_Completo.ipynb` y `dashboard_petfinder.py` están bien, o preferís otros nombres? los nomres son: EDA_completo y dashboard_petfinder
2. **Sección de texto con NLP**: ¿querés incluir análisis de sentimiento sobre la descripción (requiere `textblob` o `transformers`)? Hay unos archivos json en la carpeta input/train_sentiment que entiendo que tienen ese analisis. se puede usar eso? 
3. **Mapa geográfico de Malasia**: ¿tenés acceso a un GeoJSON de los estados? Si no, lo reemplazamos por un barplot enriquecido. vamos con el barplot
4. **Idioma del notebook**: ¿celdas markdown en español o inglés? en español
5. **Integración con el notebook existente** (`03_Petfinder_EDA.ipynb`): ¿el nuevo EDA reemplaza ese notebook o lo complementa? este es un nuevo eda en la ruta grupo10/EDA_completo.ipynb
6. **Desagregación binaria del target**: ¿usamos `rápido` (0-1) vs `lento` (2-4) como vista simplificada en el dashboard, además de las 5 clases? no, dejemos las clases como estan

---

*Generado el 2026-04-01 — análisis basado en `train.csv` (14.993 filas, 24 columnas)*
