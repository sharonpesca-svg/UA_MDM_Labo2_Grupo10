import pandas as pd
import joblib
import sys
import os

sys.path.append('.')

# Test case: valores MUY NEGATIVOS
test_data = {
    'Type': 1,
    'Age': 120,  # Muy viejo
    'Breed1': 150,
    'Color1': 5,
    'State': 41326,
    'Gender': 1,
    'Quantity': 5,
    'Fee': 1000,  # Muy caro
    'PhotoAmt': 0,  # SIN FOTOS
    'VideoAmt': 0,
    'MaturitySize': 4,
    'FurLength': 2,
    'Vaccinated': 2,  # NO vacunado
    'Dewormed': 2,  # NO desparasitado
    'Sterilized': 2,  # NO esterilizado
    'Health': 3,  # LESION GRAVE
    'Desc_len_words': 5,
    'Tiene_nombre': 0
}

# Cargar modelo e inputs
model = joblib.load('model_test/lightgbm_petfinder.pkl')
breed_labels = pd.read_csv('../input/petfinder-adoption-prediction/breed_labels.csv')
color_labels = pd.read_csv('../input/petfinder-adoption-prediction/color_labels.csv')
state_labels = pd.read_csv('../input/petfinder-adoption-prediction/state_labels.csv')

# Crear DataFrame con caso negativo
df = pd.DataFrame([test_data])

# Feature derivado: health_score
df['health_score'] = (
    (df['Vaccinated'] == 1).astype(int) +
    (df['Dewormed'] == 1).astype(int) +
    (df['Sterilized'] == 1).astype(int) +
    (df['Health'] == 1).astype(int)
)

# photo_group
def photo_group(n):
    if n == 0:
        return 0
    elif n <= 3:
        return 1
    elif n <= 7:
        return 2
    else:
        return 3

df['photo_group'] = df['PhotoAmt'].apply(photo_group)

# Decodificaciones para visualizar
df['Type_name'] = df['Type'].map({1: 'Perro', 2: 'Gato'})
df['Vaccinated_name'] = df['Vaccinated'].map({1: 'Sí', 2: 'No', 3: 'No sabe'})
df['Health_name'] = df['Health'].map({1: 'Saludable', 2: 'Lesión menor', 3: 'Lesión grave'})

# Sentiment por defecto
df['sent_score'] = 0
df['sent_magnitude'] = 0
df['sent_num_sentences'] = 0
df['sent_num_entities'] = 0

# Seleccionar features
features = [
    'Type', 'Age', 'Breed1', 'Color1', 'State', 'Quantity', 'Fee', 'PhotoAmt', 'VideoAmt',
    'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health',
    'health_score', 'Desc_len_words', 'Tiene_nombre', 'sent_score', 'sent_magnitude',
    'sent_num_sentences', 'sent_num_entities', 'photo_group'
]

df[features] = df[features].fillna(0)
df_model = df[features]

print("="*80)
print("TEST: CASO MUY NEGATIVO (debería dar Speed 3 o 4, NO 1)")
print("="*80)
print(f"\nValores ingresados:")
print(f"✗ Type: Perro")
print(f"✗ Age: 120 meses (MUY VIEJO)")
print(f"✗ Fee: $1000 (MUY CARO)")
print(f"✗ PhotoAmt: 0 (SIN FOTOS)")
print(f"✗ Vaccinated: 2 (NO)")
print(f"✗ Dewormed: 2 (NO)")
print(f"✗ Sterilized: 2 (NO)")
print(f"✗ Health: 3 (LESIÓN GRAVE)")
print(f"✗ health_score: {df['health_score'].values[0]} (debería ser 0)")

# Predicción
pred = model.predict(df_model)[0]
proba = model.predict_proba(df_model)[0]

print(f"\n" + "="*80)
print(f"PREDICCIÓN DEL MODELO:")
print(f"="*80)
print(f"AdoptionSpeed predicho: {pred}")

SPEED_LABELS = {
    0: "Mismo día",
    1: "1ª semana",
    2: "1er mes",
    3: "2-3 meses",
    4: "Más de 3 meses"
}

print(f"Interpretación: {SPEED_LABELS.get(pred, 'Unknown')}")
print(f"\nProbabilidades:")
for i in range(5):
    print(f"  Speed {i} ({SPEED_LABELS[i]}): {proba[i]*100:.2f}%")
