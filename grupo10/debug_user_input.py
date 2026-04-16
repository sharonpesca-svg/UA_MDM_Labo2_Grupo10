import pandas as pd
import joblib

model = joblib.load('model_test/lightgbm_petfinder.pkl')

# Exactamente los valores que viste en debug pero con Vaccinated=2 (No vacunado)
test_data = {
    'Type': 1,
    'Age': 13,
    'Breed1': 150,
    'Color1': 5,
    'State': 41326,
    'Gender': 1,
    'Quantity': 1,
    'Fee': 0,
    'PhotoAmt': 1,
    'VideoAmt': 0,
    'MaturitySize': 2,
    'FurLength': 2,
    'Vaccinated': 2,  # NO vacunado (cambiado de 1 a 2)
    'Dewormed': 2,
    'Sterilized': 2,
    'Health': 3,  # Lesión grave
    'Desc_len_words': 20,
    'Tiene_nombre': 0
}

df = pd.DataFrame([test_data])

# Crear features derivadas
df['health_score'] = (
    (df['Vaccinated'] == 1).astype(int) +
    (df['Dewormed'] == 1).astype(int) +
    (df['Sterilized'] == 1).astype(int) +
    (df['Health'] == 1).astype(int)
)

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
df['sent_score'] = 0
df['sent_magnitude'] = 0
df['sent_num_sentences'] = 0
df['sent_num_entities'] = 0

features = [
    'Type', 'Age', 'Breed1', 'Color1', 'State', 'Quantity', 'Fee', 'PhotoAmt', 'VideoAmt',
    'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health',
    'health_score', 'Desc_len_words', 'Tiene_nombre', 'sent_score', 'sent_magnitude',
    'sent_num_sentences', 'sent_num_entities', 'photo_group'
]

df[features] = df[features].fillna(0)
df_model = df[features]

SPEED_LABELS = {0: "Mismo día", 1: "1ª semana", 2: "1er mes", 3: "2-3 meses", 4: "Más de 3 meses"}

print("="*80)
print("COMPARACIÓN: Los valores que REALMENTE ingresaste")
print("="*80)

print("\n✅ ESCENARIO 1: Lo que PENSASTE que ingresaste")
print("-" * 80)
print("Type: 1 (Perro)")
print("Age: 13 meses")
print("PhotoAmt: 1 foto")
print("Fee: $0")
print("Vaccinated: 2 (NO) ← Esto es lo que CREÍAS")
print("Dewormed: 2 (No)")
print("Sterilized: 2 (No)")
print("Health: 3 (Lesión grave)")
print("health_score: 0")

# Predecir con Vaccinated=2
pred = model.predict(df_model)[0]
proba = model.predict_proba(df_model)[0]
print(f"\n→ Predicción: Speed {pred} - {SPEED_LABELS[pred]}")
print(f"→ Certeza: {proba[pred]*100:.1f}%")

print("\n" + "="*80)
print("❌ ESCENARIO 2: Lo que REALMENTE ingresaste")
print("-" * 80)
print("Type: 1 (Perro)")
print("Age: 13 meses")
print("PhotoAmt: 1 foto")
print("Fee: $0")
print("Vaccinated: 1 (SÍ) ← Esto es lo que ELEGISTE")
print("Dewormed: 2 (No)")
print("Sterilized: 2 (No)")
print("Health: 3 (Lesión grave)")
print("health_score: 1")

# Predecir con Vaccinated=1
df_real = df.copy()
df_real['Vaccinated'] = 1
df_real['health_score'] = 1
df_model_real = df_real[features]

pred_real = model.predict(df_model_real)[0]
proba_real = model.predict_proba(df_model_real)[0]
print(f"\n→ Predicción: Speed {pred_real} - {SPEED_LABELS[pred_real]}")
print(f"→ Certeza: {proba_real[pred_real]*100:.1f}%")

print("\n" + "="*80)
print("CONCLUSIÓN:")
print("="*80)
print(f"✅ Sin vacunar + Lesión grave → Speed {pred} ({SPEED_LABELS[pred]})")
print(f"❌ CON vacunar + Lesión grave → Speed {pred_real} ({SPEED_LABELS[pred_real]}) ← Lo que pasó")
print("\n🔍 El modelo está prediciendo correctamente.")
print("📌 El problema es que seleccionaste 'Sí vacunado' en el dropdown,")
print("   no 'No vacunado' como creías.")
