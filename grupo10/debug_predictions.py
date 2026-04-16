import pandas as pd
import joblib
import os
from sklearn.preprocessing import LabelEncoder

BASE_PATH = '../input/petfinder-adoption-prediction'

# Cargar modelo
model_path = 'model_test/catboost_petfinder.pkl'
if not os.path.exists(model_path):
    print(f"Modelo no encontrado. Entrena primero con: python train_model.py")
    exit(1)

model = joblib.load(model_path)

SPEED_LABELS = {
    0: '0-Mismo día',
    1: '1-1ª semana',
    2: '2-1er mes',
    3: '3-2-3 meses',
    4: '4-Sin adopción',
}

# Casos de prueba (SIN photo_group, se genera automáticamente)
test_cases = [
    {
        'name': 'Gato sin fotos, sin salud',
        'Type': 2, 'Age': 100, 'Breed1': 264, 'Color1': 1, 'State': 1, 'Quantity': 1, 'Fee': 500, 'PhotoAmt': 0, 'VideoAmt': 0,
        'MaturitySize': 3, 'FurLength': 1, 'Vaccinated': 3, 'Dewormed': 3, 'Sterilized': 3, 'Health': 3,
        'health_score': 0, 'Desc_len_words': 0, 'Tiene_nombre': 0,
        'sent_score': 0, 'sent_magnitude': 0, 'sent_num_sentences': 0, 'sent_num_entities': 0, 'photo_group': 0
    },
    {
        'name': 'Perro Golden joven, fotos, sano',
        'Type': 1, 'Age': 6, 'Breed1': 115, 'Color1': 3, 'State': 1, 'Quantity': 1, 'Fee': 0, 'PhotoAmt': 30, 'VideoAmt': 0,
        'MaturitySize': 1, 'FurLength': 1, 'Vaccinated': 1, 'Dewormed': 1, 'Sterilized': 1, 'Health': 1,
        'health_score': 4, 'Desc_len_words': 50, 'Tiene_nombre': 1,
        'sent_score': 0.5, 'sent_magnitude': 3, 'sent_num_sentences': 5, 'sent_num_entities': 2, 'photo_group': 3
    },
    {
        'name': 'Perro promedio',
        'Type': 1, 'Age': 24, 'Breed1': 77, 'Color1': 2, 'State': 5, 'Quantity': 1, 'Fee': 100, 'PhotoAmt': 3, 'VideoAmt': 0,
        'MaturitySize': 2, 'FurLength': 2, 'Vaccinated': 1, 'Dewormed': 2, 'Sterilized': 1, 'Health': 1,
        'health_score': 2, 'Desc_len_words': 20, 'Tiene_nombre': 1,
        'sent_score': 0, 'sent_magnitude': 1, 'sent_num_sentences': 2, 'sent_num_entities': 0, 'photo_group': 1
    },
]

def photo_group(n):
    if n == 0:
        return '0 fotos'
    elif n <= 3:
        return '1-3'
    elif n <= 7:
        return '4-7'
    else:
        return '8+'

print("\n" + "="*80)
print("ANÁLISIS DE PREDICCIONES DEL MODELO")
print("="*80)

for test in test_cases:
    name = test.pop('name')
    
    # Asegurar orden exacto de features
    features_order = [
        'Type', 'Age', 'Breed1', 'Color1', 'State', 'Quantity', 'Fee', 'PhotoAmt', 'VideoAmt',
        'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health',
        'health_score', 'Desc_len_words', 'Tiene_nombre', 'sent_score', 'sent_magnitude',
        'sent_num_sentences', 'sent_num_entities', 'photo_group'
    ]
    
    test_ordered = {k: test[k] for k in features_order}
    df_test = pd.DataFrame([test_ordered])
    
    try:
        pred = int(model.predict(df_test)[0])
        proba = model.predict_proba(df_test)[0]
        
        print(f"\n📌 CASO: {name}")
        print(f"   PhotoAmt={test['PhotoAmt']}, Health={test['Health']}, Vaccinated={test['Vaccinated']}")
        print(f"   ✓ Predicción: {pred} - {SPEED_LABELS[pred]}")
        print(f"   Probabilidades:")
        for i, p in enumerate(proba):
            bar = "█" * int(p * 40)
            print(f"      {i} ({SPEED_LABELS[i]:20s}): {p*100:5.1f}% {bar}")
    except Exception as e:
        print(f"\n❌ Error en {name}: {str(e)}")

