import pandas as pd
import joblib
import sys
import os

sys.path.append('.')

model = joblib.load('model_test/lightgbm_petfinder.pkl')

# Casos de prueba variados
cases = [
    {
        'name': 'CASO 1: No vacunado, Lesión grave, sin fotos',
        'Type': 1, 'Age': 12, 'Breed1': 150, 'Color1': 5, 'State': 41326, 'Gender': 1,
        'Quantity': 1, 'Fee': 100, 'PhotoAmt': 0, 'VideoAmt': 0,
        'MaturitySize': 2, 'FurLength': 2, 'Vaccinated': 2, 'Dewormed': 2, 'Sterilized': 2, 'Health': 3,
        'Desc_len_words': 20, 'Tiene_nombre': 0
    },
    {
        'name': 'CASO 2: Vacunado, Saludable, 5 fotos',
        'Type': 1, 'Age': 12, 'Breed1': 150, 'Color1': 5, 'State': 41326, 'Gender': 1,
        'Quantity': 1, 'Fee': 100, 'PhotoAmt': 5, 'VideoAmt': 0,
        'MaturitySize': 2, 'FurLength': 2, 'Vaccinated': 1, 'Dewormed': 1, 'Sterilized': 1, 'Health': 1,
        'Desc_len_words': 50, 'Tiene_nombre': 1
    },
    {
        'name': 'CASO 3: Joven (1 mes), Saludable, 6 fotos',
        'Type': 2, 'Age': 1, 'Breed1': 265, 'Color1': 6, 'State': 41326, 'Gender': 2,
        'Quantity': 1, 'Fee': 0, 'PhotoAmt': 6, 'VideoAmt': 0,
        'MaturitySize': 1, 'FurLength': 2, 'Vaccinated': 2, 'Dewormed': 2, 'Sterilized': 2, 'Health': 1,
        'Desc_len_words': 50, 'Tiene_nombre': 1
    },
    {
        'name': 'CASO 4: Age 10, Sin vacunar, Lesion grave, 1 foto',
        'Type': 1, 'Age': 10, 'Breed1': 150, 'Color1': 5, 'State': 41326, 'Gender': 1,
        'Quantity': 1, 'Fee': 50, 'PhotoAmt': 1, 'VideoAmt': 0,
        'MaturitySize': 2, 'FurLength': 2, 'Vaccinated': 2, 'Dewormed': 2, 'Sterilized': 2, 'Health': 3,
        'Desc_len_words': 20, 'Tiene_nombre': 0
    }
]

SPEED_LABELS = {0: "Mismo día", 1: "1ª semana", 2: "1er mes", 3: "2-3 meses", 4: "Más de 3 meses"}

for case in cases:
    name = case.pop('name')
    
    # Crear DataFrame
    df = pd.DataFrame([case])
    
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
    
    pred = model.predict(df_model)[0]
    proba = model.predict_proba(df_model)[0]
    
    print("="*80)
    print(f"📦 {name}")
    print("="*80)
    print(f"Predicción: Speed {pred} - {SPEED_LABELS[pred]}")
    print(f"Certeza: {proba[pred]*100:.1f}%\n")
