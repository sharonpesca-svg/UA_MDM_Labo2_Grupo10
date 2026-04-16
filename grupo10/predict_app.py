import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# ──────────────────────────────────────────────
# Configuración de página
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="PetFinder Prediction App",
    page_icon="🐾",
    layout="wide",
)

BASE_PATH = '../input/petfinder-adoption-prediction'

# Paleta de colores para AdoptionSpeed
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
# Cargar modelo y datos de referencia
# ──────────────────────────────────────────────
@st.cache_resource
def load_model_and_data():
    # Cargar modelo (LightGBM es mejor que CatBoost)
    model_path = 'model_test/lightgbm_petfinder.pkl'
    if not os.path.exists(model_path):
        st.error(f"Modelo LightGBM no encontrado. Ejecuta: python train_model.py --model lightgbm")
        st.stop()
    model = joblib.load(model_path)

    # Cargar tablas de etiquetas para decodificar
    breed_labels = pd.read_csv(os.path.join(BASE_PATH, 'breed_labels.csv'))
    color_labels = pd.read_csv(os.path.join(BASE_PATH, 'color_labels.csv'))
    state_labels = pd.read_csv(os.path.join(BASE_PATH, 'state_labels.csv'))

    return model, breed_labels, color_labels, state_labels

model, breed_labels, color_labels, state_labels = load_model_and_data()

# ──────────────────────────────────────────────
# Función de preprocesamiento (igual que en train_model.py)
# ──────────────────────────────────────────────
def preprocess_input(data):
    df = pd.DataFrame([data])

    # Decodificaciones (no necesarias para predicción, pero para consistencia)
    df['Type_name'] = df['Type'].map({1: 'Perro', 2: 'Gato'})
    df['Breed1_name'] = df['Breed1'].map(breed_labels.set_index('BreedID')['BreedName']).fillna('Desconocida')
    df['Color1_name'] = df['Color1'].map(color_labels.set_index('ColorID')['ColorName']).fillna('Desconocido')
    df['State_name'] = df['State'].map(state_labels.set_index('StateID')['StateName']).fillna('Desconocido')

    df['Gender_name'] = df['Gender'].map({1: 'Macho', 2: 'Hembra', 3: 'Mixto'})
    df['Vaccinated_name'] = df['Vaccinated'].map({1: 'Sí', 2: 'No', 3: 'No sabe'})
    df['Dewormed_name'] = df['Dewormed'].map({1: 'Sí', 2: 'No', 3: 'No sabe'})
    df['Sterilized_name'] = df['Sterilized'].map({1: 'Sí', 2: 'No', 3: 'No sabe'})
    df['Health_name'] = df['Health'].map({1: 'Saludable', 2: 'Lesión menor', 3: 'Lesión grave'})
    df['MaturitySize_name'] = df['MaturitySize'].map({1: 'Pequeño', 2: 'Mediano', 3: 'Grande', 4: 'Extra grande', 0: 'No aplica'})
    df['FurLength_name'] = df['FurLength'].map({1: 'Corto', 2: 'Mediano', 3: 'Largo', 0: 'No aplica'})

    # Features derivadas
    df['Tiene_nombre'] = df['Name'].apply(lambda x: False if (pd.isna(x) or str(x).strip().lower() in ['no name', 'no name yet']) else True)
    df['Desc_len_words'] = df['Description'].fillna('').apply(lambda x: len(str(x).split()))
    df['PhotoAmt'] = df['PhotoAmt'].astype(int)

    def photo_group(n):
        if n == 0:
            return 0  # Numérico, no string
        elif n <= 3:
            return 1
        elif n <= 7:
            return 2
        else:
            return 3
    df['photo_group'] = df['PhotoAmt'].apply(photo_group)

    df['health_score'] = (
        (df['Vaccinated'] == 1).astype(int) +
        (df['Dewormed'] == 1).astype(int) +
        (df['Sterilized'] == 1).astype(int) +
        (df['Health'] == 1).astype(int)
    )

    # Sentimiento (simulado con valores por defecto si no se proporciona)
    df['sent_score'] = df.get('sent_score', 0)
    df['sent_magnitude'] = df.get('sent_magnitude', 0)
    df['sent_num_sentences'] = df.get('sent_num_sentences', 0)
    df['sent_num_entities'] = df.get('sent_num_entities', 0)

    # Seleccionar features para modelo
    features = [
        'Type', 'Age', 'Breed1', 'Color1', 'State', 'Quantity', 'Fee', 'PhotoAmt', 'VideoAmt',
        'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health',
        'health_score', 'Desc_len_words', 'Tiene_nombre', 'sent_score', 'sent_magnitude',
        'sent_num_sentences', 'sent_num_entities', 'photo_group'
    ]

    df[features] = df[features].fillna(0)
    df_model = df[features]

    return df_model

# ──────────────────────────────────────────────
# Interfaz de usuario
# ──────────────────────────────────────────────
st.title("PetFinder Adoption Prediction — App de Predicción (LightGBM)")
st.caption("Modelo: LightGBM Classifier (Accuracy: 37%, F1: 0.334) | Predice AdoptionSpeed para nuevas mascotas")

st.header("Ingresa los datos de la mascota")

col1, col2, col3 = st.columns(3)

with col1:
    tipo = st.selectbox("Tipo", [1, 2], format_func=lambda x: "Perro" if x == 1 else "Gato")
    edad = st.slider("Edad (meses)", 0, 240, 12)
    breed1 = st.number_input("Breed1 (ID de raza)", 0, 307, 1)
    color1 = st.number_input("Color1 (ID de color)", 0, 7, 1)
    state = st.number_input("State (ID de estado)", 0, 14, 1)
    gender = st.selectbox("Gender", [1, 2, 3], format_func=lambda x: ["Macho", "Hembra", "Mixto"][x-1])

with col2:
    quantity = st.number_input("Quantity", 1, 20, 1)
    fee = st.slider("Fee (RM)", 0, 3000, 0)
    photo_amt = st.slider("Cantidad de fotos", 0, 30, 3)
    video_amt = st.slider("Cantidad de videos", 0, 10, 0)
    maturity_size = st.selectbox("MaturitySize", [0, 1, 2, 3, 4], format_func=lambda x: ["No aplica", "Pequeño", "Mediano", "Grande", "Extra grande"][x])

with col3:
    fur_length = st.selectbox("FurLength", [0, 1, 2, 3], format_func=lambda x: ["No aplica", "Corto", "Mediano", "Largo"][x])
    vaccinated = st.selectbox("Vaccinated", [(1, "Sí"), (2, "No"), (3, "No sabe")], format_func=lambda x: x[1], key="vacc") 
    vaccinated = vaccinated[0]
    dewormed = st.selectbox("Dewormed", [(1, "Sí"), (2, "No"), (3, "No sabe")], format_func=lambda x: x[1], key="dew")
    dewormed = dewormed[0]
    sterilized = st.selectbox("Sterilized", [(1, "Sí"), (2, "No"), (3, "No sabe")], format_func=lambda x: x[1], key="ster")
    sterilized = sterilized[0]
    health = st.selectbox("Health", [(1, "Saludable"), (2, "Lesión menor"), (3, "Lesión grave")], format_func=lambda x: x[1], key="health")
    health = health[0]

# Campos opcionales
st.subheader("Campos opcionales")
name = st.text_input("Nombre (opcional)", "")
description = st.text_area("Descripción (opcional)", "")
sent_score = st.slider("Sentimiento Score (opcional)", -1.0, 1.0, 0.0)
sent_magnitude = st.slider("Sentimiento Magnitud (opcional)", 0.0, 10.0, 0.0)

# Botón de predicción
if st.button("Predecir AdoptionSpeed"):
    # Preparar datos
    input_data = {
        'Type': tipo,
        'Name': name if name else None,
        'Age': edad,
        'Breed1': breed1,
        'Color1': color1,
        'State': state,
        'Gender': gender,
        'Quantity': quantity,
        'Fee': fee,
        'PhotoAmt': photo_amt,
        'VideoAmt': video_amt,
        'MaturitySize': maturity_size,
        'FurLength': fur_length,
        'Vaccinated': vaccinated,
        'Dewormed': dewormed,
        'Sterilized': sterilized,
        'Health': health,
        'Description': description,
        'sent_score': sent_score,
        'sent_magnitude': sent_magnitude,
        'sent_num_sentences': len(description.split()) if description else 0,
        'sent_num_entities': 0,  # Simplificado
    }

    # Preprocesar
    df_input = preprocess_input(input_data)

    # 🔍 DEBUG: Mostrar valores ingresados
    with st.expander("🔍 Debug - TODOS los valores enviados al modelo (23 features)"):
        st.write("DataFrame completo que se envía a LightGBM:")
        debug_full = df_input.copy()
        st.dataframe(debug_full, use_container_width=True)
        
        st.write("\nResumen de valores clave:")
        debug_df = pd.DataFrame({
            'Variable': [
                'Type', 'Age', 'PhotoAmt', 'Fee', 'Vaccinated', 
                'Dewormed', 'Sterilized', 'Health', 'health_score', 'photo_group'
            ],
            'Valor': [
                tipo, edad, photo_amt, fee, vaccinated,
                dewormed, sterilized, health, 
                df_input['health_score'].values[0],
                df_input['photo_group'].values[0]
            ]
        })
        st.dataframe(debug_df, use_container_width=True)

    # Predecir
    prediction = int(model.predict(df_input)[0])
    probabilities = model.predict_proba(df_input)[0]

    # Mostrar resultados
    st.success(f"**Predicción: AdoptionSpeed {prediction}** - {SPEED_LABELS[prediction]}")

    st.subheader("Probabilidades por clase")
    prob_df = pd.DataFrame({
        'AdoptionSpeed': list(SPEED_LABELS.keys()),
        'Etiqueta': list(SPEED_LABELS.values()),
        'Probabilidad': probabilities
    })
    prob_df['Probabilidad (%)'] = (prob_df['Probabilidad'] * 100).round(2)

    st.dataframe(prob_df[['AdoptionSpeed', 'Etiqueta', 'Probabilidad (%)']], use_container_width=True)

    # Gráfico de probabilidades
    import plotly.express as px
    fig = px.bar(
        prob_df,
        x='Etiqueta',
        y='Probabilidad',
        color='AdoptionSpeed',
        color_discrete_map=SPEED_COLORS,
        labels={'Etiqueta': 'AdoptionSpeed', 'Probabilidad': 'Probabilidad'},
        title='Probabilidades de AdoptionSpeed'
    )
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

st.info("Nota: Este modelo está entrenado con datos de PetFinder Malaysia. Las predicciones son estimaciones basadas en patrones históricos.")