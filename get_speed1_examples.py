import pandas as pd
import numpy as np

# Cargar datos de entrenamiento
train_df = pd.read_csv('../input/petfinder-adoption-prediction/train/train.csv')

print("="*80)
print("VARIABLES PARA PREDECIR ADOPTIONSTATUS = 1 (Adoptado en 1ª semana)")
print("="*80)

# Filtrar casos donde AdoptionSpeed == 1
speed_1 = train_df[train_df['AdoptionSpeed'] == 1]

print(f"\nTotal de casos con AdoptionSpeed=1: {len(speed_1)} ({len(speed_1)/len(train_df)*100:.1f}% del dataset)")
print("\n" + "="*80)
print("ESTADÍSTICAS DE FEATURES PARA SPEED=1:")
print("="*80)

# Mostrar algunos ejemplos
print("\n📊 EJEMPLOS REALES (primeros 5 casos con Speed=1):")
print("-" * 80)

ejemplo_features = ['Type', 'Age', 'Breed1', 'Color1', 'State', 'Gender', 'Quantity', 
                    'Fee', 'PhotoAmt', 'VideoAmt', 'MaturitySize', 'FurLength', 
                    'Vaccinated', 'Dewormed', 'Sterilized', 'Health']

for idx, (i, row) in enumerate(speed_1.head(5).iterrows()):
    print(f"\n✅ CASO #{idx+1}:")
    for feat in ejemplo_features:
        val = row[feat]
        print(f"  {feat:15} = {val}")

# Estadísticas agregadas
print("\n" + "="*80)
print("VALORES PROMEDIO/TÍPICOS PARA SPEED=1:")
print("="*80)

print(f"\n• Type (Tipo):")
print(f"  Perro: {(speed_1['Type']==1).sum()/len(speed_1)*100:.1f}%")
print(f"  Gato: {(speed_1['Type']==2).sum()/len(speed_1)*100:.1f}%")

print(f"\n• Age (Edad promedio): {speed_1['Age'].mean():.1f} meses (rango: {speed_1['Age'].min()}-{speed_1['Age'].max()})")

print(f"\n• PhotoAmt (Fotos promedio): {speed_1['PhotoAmt'].mean():.1f}")
print(f"• VideoAmt (Videos promedio): {speed_1['VideoAmt'].mean():.2f}")

print(f"\n• Fee (Tarifa promedio): ${speed_1['Fee'].mean():.0f}")

print(f"\n• Vaccinated: {(speed_1['Vaccinated']==1).sum()/len(speed_1)*100:.1f}% sí")
print(f"• Dewormed: {(speed_1['Dewormed']==1).sum()/len(speed_1)*100:.1f}% sí")
print(f"• Sterilized: {(speed_1['Sterilized']==1).sum()/len(speed_1)*100:.1f}% sí")

print(f"\n• Quantity (Cantidad promedio): {speed_1['Quantity'].mean():.2f}")
print(f"• MaturitySize (Más frecuente): {speed_1['MaturitySize'].mode()[0]}")
print(f"• FurLength (Más frecuente): {speed_1['FurLength'].mode()[0]}")

print("\n" + "="*80)
print("✅ VALORES RECOMENDADOS PARA OBTENER SPEED=1:")
print("="*80)

print("\nUsá estos valores en la app:")
print("""
Type: 1 (Perro)
Age: 12-24 meses
Breed1: 1-50 (razas comunes)
Color1: 1-10 (colores comunes)
State: varies
Gender: 1 (Macho)
Quantity: 1
Fee: 0-500
PhotoAmt: 4-7 fotos (importante!)
VideoAmt: 0-2
MaturitySize: 2 (Mediano)
FurLength: varies
Vaccinated: 1 (Sí)
Dewormed: 1 (Sí)
Sterilized: 0-1
Health: 1 (Saludable)
""")
