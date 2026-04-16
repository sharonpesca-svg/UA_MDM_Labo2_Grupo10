import pandas as pd
import joblib
import numpy as np

model = joblib.load('model_test/lightgbm_petfinder.pkl')

# Valores exactos que el usuario reportó
test_data = {
    'Type': 1,
    'Age': 66,  # VIEJO
    'Breed1': 150,
    'Color1': 5,
    'State': 41326,
    'Gender': 1,
    'Quantity': 1,
    'Fee': 0,
    'PhotoAmt': 2,  # Pocas fotos
    'VideoAmt': 0,
    'MaturitySize': 2,
    'FurLength': 2,
    'Vaccinated': 2,  # NO vacunado
    'Dewormed': 1,   # SÍ desparasitado
    'Sterilized': 2,  # NO esterilizado
    'Health': 3,   # LESION GRAVE
    'Desc_len_words': 20,
    'Tiene_nombre': 0
}

df = pd.DataFrame([test_data])

# Features derivadas
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

df_model = df[features].fillna(0)

print("="*80)
print("DEBUG: VALORES QUE REPORTASTE")
print("="*80)
print("\nValores procesados:")
print(f"Type: {df['Type'].values[0]}")
print(f"Age: {df['Age'].values[0]} meses")
print(f"PhotoAmt: {df['PhotoAmt'].values[0]}")
print(f"Fee: ${df['Fee'].values[0]}")
print(f"Vaccinated: {df['Vaccinated'].values[0]} (2=No)")
print(f"Dewormed: {df['Dewormed'].values[0]} (1=Sí)")
print(f"Sterilized: {df['Sterilized'].values[0]} (2=No)")
print(f"Health: {df['Health'].values[0]} (3=Lesión grave)")
print(f"health_score calculado: {df_model['health_score'].values[0]}")
print(f"photo_group calculado: {df_model['photo_group'].values[0]}")

# Mostrar el DataFrame que se envía al modelo
print("\n" + "="*80)
print("DataFrame enviado al modelo:")
print("="*80)
print(df_model)

# Predicción
pred = model.predict(df_model)[0]
proba = model.predict_proba(df_model)[0]

SPEED_LABELS = {0: "Mismo día", 1: "1ª semana", 2: "1er mes", 3: "2-3 meses", 4: "Más de 3 meses"}

print("\n" + "="*80)
print("PREDICCIÓN DEL MODELO:")
print("="*80)
print(f"AdoptionSpeed predicho: {pred}")
print(f"Interpretación: {SPEED_LABELS[pred]}")
print(f"\nProbabilidades:")
for i in range(5):
    print(f"  Speed {i} ({SPEED_LABELS[i]}): {proba[i]*100:.2f}%")

print("\n" + "="*80)
print("⚠️ ANÁLISIS:")
print("="*80)
if pred == 1:
    print("❌ EL MODELO ESTÁ PREDICIENDO SPEED=1 CON CARACTERÍSTICAS NEGATIVAS")
    print("   Esto sugiere un problema en:")
    print("   1. Los datos de entrenamiento")
    print("   2. La codificación de variables")
    print("   3. El modelo mismo está overfit o mal entrenado")
    
# Verificar feature importance (si es posible)
try:
    feature_importance = model.feature_importance()
    print("\n" + "="*80)
    print("IMPORTANCIA DE FEATURES (top 10):")
    print("="*80)
    importances = sorted(zip(features, feature_importance), key=lambda x: x[1], reverse=True)[:10]
    for feat, imp in importances:
        print(f"{feat:25} : {imp:8.2f}")
except Exception as e:
    print(f"\nNo se pudo obtener feature importance: {e}")
