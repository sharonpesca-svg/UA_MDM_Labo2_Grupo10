

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
from io import BytesIO

warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────
# Configuración de página
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="PetFinder EDA Dashboard",
    page_icon="🐾",
    layout="wide",
)

BASE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'input', 'petfinder-adoption-prediction')

# Paleta de colores consistente para AdoptionSpeed
SPEED_COLORS = {
    0: '#2ecc71',
    1: '#27ae60',
    2: '#f39c12',
    3: '#e67e22',
    4: '#e74c3c',
}
SPEED_LABELS = {
    0: '0-Mismo día',
    1: '1-1ª semana',
    2: '2-1er mes',
    3: '3-2-3 meses',
    4: '4-Sin adopción',
}

# ──────────────────────────────────────────────
# Carga y preprocesamiento de datos (cacheado)
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    # Cargar train
    train = pd.read_csv(f'{BASE_PATH}/train/train.csv')

    # Cargar tablas de etiquetas
    breed_labels  = pd.read_csv(f'{BASE_PATH}/breed_labels.csv')
    color_labels  = pd.read_csv(f'{BASE_PATH}/color_labels.csv')
    state_labels  = pd.read_csv(f'{BASE_PATH}/state_labels.csv')

    # ── Decodificaciones ──────────────────────
    train['Type_name']  = train['Type'].map({1: 'Perro', 2: 'Gato'})

    breed_map = breed_labels.set_index('BreedID')['BreedName'].to_dict()
    train['Breed1_name'] = train['Breed1'].map(breed_map).fillna('Desconocida')

    color_map = color_labels.set_index('ColorID')['ColorName'].to_dict()
    train['Color1_name'] = train['Color1'].map(color_map).fillna('Desconocido')

    state_map = state_labels.set_index('StateID')['StateName'].to_dict()
    train['State_name'] = train['State'].map(state_map).fillna('Desconocido')

    train['Gender_name'] = train['Gender'].map({1: 'Macho', 2: 'Hembra', 3: 'Mixto'})
    train['Vaccinated_name']  = train['Vaccinated'].map({1: 'Sí', 2: 'No', 3: 'No sabe'})
    train['Dewormed_name']    = train['Dewormed'].map({1: 'Sí', 2: 'No', 3: 'No sabe'})
    train['Sterilized_name']  = train['Sterilized'].map({1: 'Sí', 2: 'No', 3: 'No sabe'})
    train['Health_name']      = train['Health'].map({1: 'Saludable', 2: 'Lesión menor', 3: 'Lesión grave'})
    train['MaturitySize_name']= train['MaturitySize'].map({1: 'Pequeño', 2: 'Mediano', 3: 'Grande', 4: 'Extra grande', 0: 'No aplica'})
    train['FurLength_name']   = train['FurLength'].map({1: 'Corto', 2: 'Mediano', 3: 'Largo', 0: 'No aplica'})

    # ── Features derivadas ────────────────────
    train['Tiene_nombre'] = train['Name'].apply(
        lambda x: False if (pd.isna(x) or str(x).strip().lower() in ['no name', 'no name yet']) else True
    )
    train['Desc_len_words'] = train['Description'].fillna('').apply(lambda x: len(str(x).split()))
    train['PhotoAmt'] = train['PhotoAmt'].astype(int)

    # Grupos de fotos
    def photo_group(n):
        if n == 0:
            return '0 fotos'
        elif n <= 3:
            return '1-3'
        elif n <= 7:
            return '4-7'
        else:
            return '8+'
    train['photo_group'] = train['PhotoAmt'].apply(photo_group)

    # Health score (suma de indicadores positivos)
    train['health_score'] = (
        (train['Vaccinated'] == 1).astype(int) +
        (train['Dewormed']   == 1).astype(int) +
        (train['Sterilized'] == 1).astype(int) +
        (train['Health']     == 1).astype(int)
    )

    # Etiqueta legible para AdoptionSpeed
    train['Speed_label'] = train['AdoptionSpeed'].map(SPEED_LABELS)

    return train


# ──────────────────────────────────────────────
# Carga inicial
# ──────────────────────────────────────────────
df_full = load_data()

# ──────────────────────────────────────────────
# SIDEBAR — Filtros globales
# ──────────────────────────────────────────────
st.sidebar.title("Filtros globales")

# Tipo de animal
tipo_sel = st.sidebar.radio("Tipo de animal", ["Todos", "Perro", "Gato"], index=0)

# Estado
all_states = sorted(df_full['State_name'].dropna().unique().tolist())
estados_sel = st.sidebar.multiselect("Estado", all_states, default=all_states)

# Rango de edad
age_min, age_max = st.sidebar.slider("Rango de edad (meses)", 0, 60, (0, 60))

# AdoptionSpeed
all_speeds = sorted(df_full['AdoptionSpeed'].unique().tolist())
speeds_sel = st.sidebar.multiselect(
    "AdoptionSpeed",
    options=all_speeds,
    default=all_speeds,
    format_func=lambda x: SPEED_LABELS[x]
)

# Aplicar filtros
df = df_full.copy()
if tipo_sel != "Todos":
    df = df[df['Type_name'] == tipo_sel]
if estados_sel:
    df = df[df['State_name'].isin(estados_sel)]
df = df[(df['Age'] >= age_min) & (df['Age'] <= age_max)]
if speeds_sel:
    df = df[df['AdoptionSpeed'].isin(speeds_sel)]

# Métrica de filas filtradas
st.sidebar.metric("Filas filtradas", f"{len(df):,}")

if len(df) == 0:
    st.warning("No hay datos con los filtros seleccionados.")
    st.stop()

# ──────────────────────────────────────────────
# Título principal
# ──────────────────────────────────────────────
st.title("PetFinder Adoption Prediction — Dashboard EDA")
st.caption("Dataset: PetFinder Malaysia | Grupo 10 — Laboratorio 2 MDM")

# ──────────────────────────────────────────────
# PESTAÑAS
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Visión General",
    "Características",
    "Geografía",
    "Media y Fee",
    "Texto",
])

# ══════════════════════════════════════════════
# PESTAÑA 1 — VISIÓN GENERAL
# ══════════════════════════════════════════════
with tab1:
    st.header("Visión General del Dataset")

    # KPI cards
    total = len(df)
    pct_rapido = (df['AdoptionSpeed'].isin([0, 1])).mean() * 100
    fee_mean = df.loc[df['Fee'] > 0, 'Fee'].mean() if (df['Fee'] > 0).any() else 0.0
    photo_mean = df['PhotoAmt'].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total mascotas", f"{total:,}")
    c2.metric("Adoptados en ≤7 días (Speed 0 ó 1)", f"{pct_rapido:.1f}%")
    c3.metric("Fee promedio (solo fee > 0)", f"RM {fee_mean:.1f}")
    c4.metric("Fotos promedio", f"{photo_mean:.2f}")

    st.divider()

    col_a, col_b = st.columns(2)

    # Bar chart de AdoptionSpeed
    with col_a:
        st.subheader("Distribución de AdoptionSpeed")
        speed_counts = df['AdoptionSpeed'].value_counts().sort_index().reset_index()
        speed_counts.columns = ['AdoptionSpeed', 'Count']
        speed_counts['Porcentaje'] = (speed_counts['Count'] / speed_counts['Count'].sum() * 100).round(1)
        speed_counts['Label']      = speed_counts['AdoptionSpeed'].map(SPEED_LABELS)
        speed_counts['Color']      = speed_counts['AdoptionSpeed'].map(SPEED_COLORS)

        fig_speed = px.bar(
            speed_counts,
            x='Label',
            y='Count',
            color='AdoptionSpeed',
            color_discrete_map=SPEED_COLORS,
            hover_data={'Count': True, 'Porcentaje': True, 'AdoptionSpeed': False, 'Label': False},
            labels={'Label': 'AdoptionSpeed', 'Count': 'Cantidad'},
            title='Distribución de AdoptionSpeed',
        )
        fig_speed.update_layout(showlegend=False)
        st.plotly_chart(fig_speed, use_container_width=True)

    # Pie chart Perros vs Gatos
    with col_b:
        st.subheader("Perros vs Gatos")
        type_counts = df['Type_name'].value_counts().reset_index()
        type_counts.columns = ['Tipo', 'Count']
        fig_pie = px.pie(
            type_counts,
            names='Tipo',
            values='Count',
            color='Tipo',
            color_discrete_map={'Perro': '#3498db', 'Gato': '#e74c3c'},
            title='Proporción Perros vs Gatos',
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# ══════════════════════════════════════════════
# PESTAÑA 2 — CARACTERÍSTICAS
# ══════════════════════════════════════════════
with tab2:
    st.header("Características de las Mascotas")

    # Histograma de Age
    st.subheader("Distribución de Edad")
    nbins_age = st.slider("Número de bins (Edad)", min_value=10, max_value=60, value=30, key='nbins_age')
    df_age = df[df['Age'] <= 60].copy()
    fig_age = px.histogram(
        df_age,
        x='Age',
        color='Type_name',
        nbins=nbins_age,
        barmode='overlay',
        opacity=0.7,
        color_discrete_map={'Perro': '#3498db', 'Gato': '#e74c3c'},
        labels={'Age': 'Edad (meses)', 'Type_name': 'Tipo'},
        title='Distribución de Edad por Tipo (Age <= 60 meses)',
    )
    st.plotly_chart(fig_age, use_container_width=True)

    st.divider()

    # Heatmap variable categórica × AdoptionSpeed
    st.subheader("Variable categórica vs AdoptionSpeed")
    cat_options = {
        'Vaccinated'   : 'Vaccinated_name',
        'Dewormed'     : 'Dewormed_name',
        'Sterilized'   : 'Sterilized_name',
        'Gender'       : 'Gender_name',
        'MaturitySize' : 'MaturitySize_name',
        'FurLength'    : 'FurLength_name',
        'Health'       : 'Health_name',
    }
    cat_sel = st.selectbox(
        "Seleccionar variable",
        options=list(cat_options.keys()),
        key='cat_sel'
    )
    cat_col = cat_options[cat_sel]

    pivot = (
        df.groupby([cat_col, 'AdoptionSpeed'])
          .size()
          .unstack(fill_value=0)
    )
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)
    pivot_norm.columns = [SPEED_LABELS[c] for c in pivot_norm.columns]

    fig_heat = px.imshow(
        pivot_norm,
        text_auto='.1%',
        color_continuous_scale='RdYlGn_r',
        aspect='auto',
        title=f'{cat_sel} × AdoptionSpeed (proporciones por fila)',
        labels={'x': 'AdoptionSpeed', 'y': cat_sel, 'color': 'Proporción'},
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    # Género: distribución y AdoptionSpeed
    st.subheader("Género: Distribución y AdoptionSpeed")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        gender_counts = df['Gender_name'].value_counts().reset_index()
        gender_counts.columns = ['Género', 'Count']
        fig_gender_pie = px.pie(
            gender_counts,
            names='Género',
            values='Count',
            color='Género',
            color_discrete_map={'Macho': '#3498db', 'Hembra': '#e74c3c', 'Mixto': '#f39c12'},
            title='Distribución por Género',
        )
        st.plotly_chart(fig_gender_pie, use_container_width=True)

    with col_g2:
        gender_speed = (
            df.groupby('Gender_name')['AdoptionSpeed']
              .mean()
              .reset_index()
              .rename(columns={'Gender_name': 'Género', 'AdoptionSpeed': 'AdoptionSpeed Promedio'})
        )
        fig_gender_speed = px.bar(
            gender_speed,
            x='Género',
            y='AdoptionSpeed Promedio',
            color='Género',
            color_discrete_map={'Macho': '#3498db', 'Hembra': '#e74c3c', 'Mixto': '#f39c12'},
            text='AdoptionSpeed Promedio',
            title='AdoptionSpeed promedio por Género (menor = más rápido)',
        )
        fig_gender_speed.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig_gender_speed.update_layout(showlegend=False, yaxis_range=[0, 4])
        st.plotly_chart(fig_gender_speed, use_container_width=True)

    # Heatmap género × AdoptionSpeed por tipo de animal
    gender_cross = pd.crosstab(
        df['Gender_name'], df['AdoptionSpeed'], normalize='index'
    ) * 100
    gender_cross.columns = [SPEED_LABELS[c] for c in gender_cross.columns]
    fig_gender_heat = px.imshow(
        gender_cross,
        text_auto='.1f',
        color_continuous_scale='RdYlGn_r',
        aspect='auto',
        title='Género × AdoptionSpeed (% por fila)',
        labels={'x': 'AdoptionSpeed', 'y': 'Género', 'color': '%'},
    )
    st.plotly_chart(fig_gender_heat, use_container_width=True)

    st.divider()

    # Top 10 razas por tipo
    st.subheader("Top 10 razas primarias")
    top_breeds = (
        df.groupby(['Breed1_name', 'Type_name'])
          .size()
          .reset_index(name='Count')
          .sort_values('Count', ascending=False)
          .head(20)
    )
    fig_breeds = px.bar(
        top_breeds,
        x='Count',
        y='Breed1_name',
        color='Type_name',
        orientation='h',
        color_discrete_map={'Perro': '#3498db', 'Gato': '#e74c3c'},
        labels={'Breed1_name': 'Raza', 'Count': 'Cantidad', 'Type_name': 'Tipo'},
        title='Top razas primarias por Tipo',
    )
    fig_breeds.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig_breeds, use_container_width=True)

# ══════════════════════════════════════════════
# PESTAÑA 3 — GEOGRAFÍA
# ══════════════════════════════════════════════
with tab3:
    st.header("Geografía — Estados de Malasia")

    # Barplot por estado
    state_counts = df['State_name'].value_counts().reset_index()
    state_counts.columns = ['Estado', 'Count']
    state_counts = state_counts.sort_values('Count', ascending=True)

    fig_states = px.bar(
        state_counts,
        x='Count',
        y='Estado',
        orientation='h',
        color='Count',
        color_continuous_scale='Blues',
        labels={'Count': 'Cantidad de mascotas', 'Estado': 'Estado'},
        title='Cantidad de mascotas por Estado (descendente)',
    )
    st.plotly_chart(fig_states, use_container_width=True)

    st.divider()

    # Tabla de métricas por estado
    st.subheader("Métricas por Estado")
    state_metrics = (
        df.groupby('State_name')
          .agg(
            N               = ('PetID', 'count'),
            Fee_promedio    = ('Fee', 'mean'),
            PhotoAmt_prom   = ('PhotoAmt', 'mean'),
            Speed_promedio  = ('AdoptionSpeed', 'mean'),
          )
          .reset_index()
    )
    state_metrics['Pct_rapido'] = (
        df[df['AdoptionSpeed'].isin([0, 1])]
          .groupby('State_name')['PetID']
          .count()
          .reindex(state_metrics['State_name'])
          .values / state_metrics['N'] * 100
    )
    state_metrics = state_metrics.sort_values('N', ascending=False)
    state_metrics.columns = ['Estado', 'N', 'Fee Promedio', 'PhotoAmt Promedio', 'AdoptionSpeed Promedio', '% Adopción Rápida (≤1)']
    for col in ['Fee Promedio', 'PhotoAmt Promedio', 'AdoptionSpeed Promedio', '% Adopción Rápida (≤1)']:
        state_metrics[col] = state_metrics[col].round(2)

    st.dataframe(state_metrics, use_container_width=True)

# ══════════════════════════════════════════════
# PESTAÑA 4 — MEDIA Y FEE
# ══════════════════════════════════════════════
with tab4:
    st.header("Media (Fotos/Videos) y Fee")

    col_left, col_right = st.columns(2)

    # Strip plot PhotoAmt vs AdoptionSpeed
    with col_left:
        st.subheader("Fotos por AdoptionSpeed")
        df_strip = df.copy()
        df_strip['Speed_label'] = df_strip['AdoptionSpeed'].map(SPEED_LABELS)
        fig_strip = px.strip(
            df_strip,
            x='Speed_label',
            y='PhotoAmt',
            color='Type_name',
            stripmode='overlay',
            color_discrete_map={'Perro': '#3498db', 'Gato': '#e74c3c'},
            labels={'Speed_label': 'AdoptionSpeed', 'PhotoAmt': 'Nº de fotos', 'Type_name': 'Tipo'},
            title='Distribución de fotos por AdoptionSpeed',
        )
        st.plotly_chart(fig_strip, use_container_width=True)

    # Boxplot Fee por AdoptionSpeed
    with col_right:
        st.subheader("Fee por AdoptionSpeed")
        log_scale = st.checkbox("Escala logarítmica en Fee", value=False, key='log_fee')
        df_fee = df[df['Fee'] > 0].copy()
        df_fee['Speed_label'] = df_fee['AdoptionSpeed'].map(SPEED_LABELS)
        fig_box_fee = px.box(
            df_fee,
            x='Speed_label',
            y='Fee',
            color='AdoptionSpeed',
            color_discrete_map=SPEED_COLORS,
            log_y=log_scale,
            labels={'Speed_label': 'AdoptionSpeed', 'Fee': 'Fee (RM)'},
            title='Distribución de Fee (solo fee > 0) por AdoptionSpeed',
        )
        fig_box_fee.update_layout(showlegend=False)
        st.plotly_chart(fig_box_fee, use_container_width=True)

    st.divider()

    # Barplot tasa de adopción rápida por photo_group
    st.subheader("Tasa de adopción rápida por grupo de fotos")
    photo_order = ['0 fotos', '1-3', '4-7', '8+']
    photo_rate = (
        df.groupby('photo_group')
          .apply(lambda g: (g['AdoptionSpeed'] <= 1).mean() * 100)
          .reindex(photo_order)
          .reset_index()
    )
    photo_rate.columns = ['photo_group', 'Tasa_rapida']
    fig_photo = px.bar(
        photo_rate,
        x='photo_group',
        y='Tasa_rapida',
        color='Tasa_rapida',
        color_continuous_scale='Greens',
        labels={'photo_group': 'Grupo de fotos', 'Tasa_rapida': '% Adopción rápida (Speed ≤ 1)'},
        title='Tasa de adopción rápida (Speed ≤ 1) por grupo de fotos',
        text='Tasa_rapida',
    )
    fig_photo.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    st.plotly_chart(fig_photo, use_container_width=True)

# ══════════════════════════════════════════════
# PESTAÑA 5 — TEXTO Y SENTIMIENTO
# ══════════════════════════════════════════════
with tab5:
    st.header("Análisis de Texto")

    # WordCloud
    st.subheader("Nube de palabras — Descripciones")
    wc_tipo = st.radio("Tipo para WordCloud", ["Todos", "Perro", "Gato"], horizontal=True, key='wc_tipo')

    try:
        from wordcloud import WordCloud, STOPWORDS

        if wc_tipo == "Todos":
            corpus = ' '.join(df['Description'].fillna('').tolist())
        elif wc_tipo == "Perro":
            corpus = ' '.join(df.loc[df['Type_name'] == 'Perro', 'Description'].fillna('').tolist())
        else:
            corpus = ' '.join(df.loc[df['Type_name'] == 'Gato', 'Description'].fillna('').tolist())

        stopwords_wc = set(STOPWORDS)
        wc = WordCloud(
            width=900,
            height=400,
            background_color='white',
            stopwords=stopwords_wc,
            max_words=150,
            colormap='viridis',
        ).generate(corpus if corpus.strip() else 'sin datos')

        buf = BytesIO()
        wc.to_image().save(buf, format='PNG')
        st.image(buf.getvalue(), use_column_width=True)
    except ImportError:
        st.info("Instalar la librería `wordcloud` para ver la nube de palabras: `pip install wordcloud`")

    st.divider()

    # Histograma Desc_len_words
    st.subheader("Longitud de descripción por AdoptionSpeed")
    df_desc = df.copy()
    df_desc['Speed_label'] = df_desc['AdoptionSpeed'].map(SPEED_LABELS)
    fig_desc = px.histogram(
        df_desc,
        x='Desc_len_words',
        color='AdoptionSpeed',
        color_discrete_map=SPEED_COLORS,
        nbins=40,
        barmode='overlay',
        opacity=0.7,
        labels={'Desc_len_words': 'Palabras en descripción', 'AdoptionSpeed': 'Speed'},
        title='Distribución de longitud de descripción por AdoptionSpeed',
    )
    st.plotly_chart(fig_desc, use_container_width=True)

