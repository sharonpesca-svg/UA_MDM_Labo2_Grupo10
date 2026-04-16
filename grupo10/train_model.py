import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, log_loss, classification_report
import joblib


BASE_PATH = '../input/petfinder-adoption-prediction'
SPEED_LABELS = {
    0: '0-Mismo día',
    1: '1-1ª semana',
    2: '2-1er mes',
    3: '3-2-3 meses',
    4: '4-Sin adopción',
}


def load_data(base_path=BASE_PATH):
    train = pd.read_csv(os.path.join(base_path, 'train', 'train.csv'))
    breed_labels = pd.read_csv(os.path.join(base_path, 'breed_labels.csv'))
    color_labels = pd.read_csv(os.path.join(base_path, 'color_labels.csv'))
    state_labels = pd.read_csv(os.path.join(base_path, 'state_labels.csv'))

    # Decodificaciones
    train['Type_name'] = train['Type'].map({1: 'Perro', 2: 'Gato'})
    train['Breed1_name'] = train['Breed1'].map(breed_labels.set_index('BreedID')['BreedName']).fillna('Desconocida')
    train['Color1_name'] = train['Color1'].map(color_labels.set_index('ColorID')['ColorName']).fillna('Desconocido')
    train['State_name'] = train['State'].map(state_labels.set_index('StateID')['StateName']).fillna('Desconocido')

    train['Gender_name'] = train['Gender'].map({1: 'Macho', 2: 'Hembra', 3: 'Mixto'})
    train['Vaccinated_name'] = train['Vaccinated'].map({1: 'Sí', 2: 'No', 3: 'No sabe'})
    train['Dewormed_name'] = train['Dewormed'].map({1: 'Sí', 2: 'No', 3: 'No sabe'})
    train['Sterilized_name'] = train['Sterilized'].map({1: 'Sí', 2: 'No', 3: 'No sabe'})
    train['Health_name'] = train['Health'].map({1: 'Saludable', 2: 'Lesión menor', 3: 'Lesión grave'})
    train['MaturitySize_name'] = train['MaturitySize'].map({1: 'Pequeño', 2: 'Mediano', 3: 'Grande', 4: 'Extra grande', 0: 'No aplica'})
    train['FurLength_name'] = train['FurLength'].map({1: 'Corto', 2: 'Mediano', 3: 'Largo', 0: 'No aplica'})

    train['Tiene_nombre'] = train['Name'].apply(lambda x: False if (pd.isna(x) or str(x).strip().lower() in ['no name', 'no name yet']) else True)
    train['Desc_len_words'] = train['Description'].fillna('').apply(lambda x: len(str(x).split()))
    train['PhotoAmt'] = train['PhotoAmt'].astype(int)

    def photo_group(n):
        if n == 0:
            return 0
        elif n <= 3:
            return 1
        elif n <= 7:
            return 2
        else:
            return 3

    train['photo_group'] = train['PhotoAmt'].apply(photo_group)

    train['health_score'] = (
        (train['Vaccinated'] == 1).astype(int) +
        (train['Dewormed'] == 1).astype(int) +
        (train['Sterilized'] == 1).astype(int) +
        (train['Health'] == 1).astype(int)
    )

    sentiment_path = os.path.join(base_path, 'train_sentiment')
    if os.path.isdir(sentiment_path):
        records = []
        for fname in os.listdir(sentiment_path):
            if fname.endswith('.json'):
                pet_id = fname.replace('.json', '')
                try:
                    with open(os.path.join(sentiment_path, fname), 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    ds = data.get('documentSentiment', {})
                    records.append({
                        'PetID': pet_id,
                        'sent_score': ds.get('score', np.nan),
                        'sent_magnitude': ds.get('magnitude', np.nan),
                        'sent_num_sentences': len(data.get('sentences', [])),
                        'sent_num_entities': len(data.get('entities', [])),
                    })
                except Exception:
                    continue

        if records:
            sent_df = pd.DataFrame(records)
            train = train.merge(sent_df, on='PetID', how='left')
        else:
            train['sent_score'] = np.nan
            train['sent_magnitude'] = np.nan
            train['sent_num_sentences'] = np.nan
            train['sent_num_entities'] = np.nan
    else:
        train['sent_score'] = np.nan
        train['sent_magnitude'] = np.nan
        train['sent_num_sentences'] = np.nan
        train['sent_num_entities'] = np.nan

    train['Speed_label'] = train['AdoptionSpeed'].map(SPEED_LABELS)
    return train


def build_features(df):
    df = df.copy()
    df = df.drop(columns=['Name', 'Description', 'RescuerID', 'PetID'], errors='ignore')

    df['sent_score'].fillna(0, inplace=True)
    df['sent_magnitude'].fillna(0, inplace=True)
    df['sent_num_sentences'].fillna(0, inplace=True)
    df['sent_num_entities'].fillna(0, inplace=True)

    # Selección de features
    features = [
        'Type', 'Age', 'Breed1', 'Color1', 'State', 'Quantity', 'Fee', 'PhotoAmt', 'VideoAmt',
        'MaturitySize', 'FurLength', 'Vaccinated', 'Dewormed', 'Sterilized', 'Health',
        'health_score', 'Desc_len_words', 'Tiene_nombre', 'sent_score', 'sent_magnitude',
        'sent_num_sentences', 'sent_num_entities', 'photo_group'
    ]

    # convertir boolean a int
    if 'Tiene_nombre' in df.columns:
        df['Tiene_nombre'] = df['Tiene_nombre'].astype(int)

    # Limpiar nulos en numéricos
    df[features] = df[features].fillna(0)

    # Label encode categóricas AQUÍ, antes de split
    categorical_cols = [
        'Type', 'Breed1', 'Color1', 'State', 'MaturitySize', 'FurLength',
        'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'photo_group'
    ]
    
    le_dict = {}
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le

    df_model = df[features]
    target = df['AdoptionSpeed']

    return df_model, target, le_dict


def train_and_evaluate(model_name='catboost', output_path='models'):
    df = load_data()
    X, y, le_dict = build_features(df)

    # Hold out split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # Categóricas para el modelo
    categorical_cols = [
        'Type', 'Breed1', 'Color1', 'State', 'MaturitySize', 'FurLength',
        'Vaccinated', 'Dewormed', 'Sterilized', 'Health', 'photo_group'
    ]
    
    # Para CatBoost, necesita índices de columnas categóricas
    cat_features_idx = [X.columns.get_loc(c) for c in categorical_cols if c in X.columns]

    os.makedirs(output_path, exist_ok=True)

    if model_name.lower() == 'catboost':
        try:
            from catboost import CatBoostClassifier
        except ImportError as e:
            raise ImportError("CatBoost no está instalado. Ejecuta: pip install catboost") from e
        model = CatBoostClassifier(
            iterations=500,  # Restaurado a valor original
            learning_rate=0.05,
            depth=6,
            loss_function='MultiClass',
            eval_metric='MultiClass',
            random_seed=42,
            verbose=50,
            class_weights={0: 5.0, 1: 5.0, 2: 1.0, 3: 2.0, 4: 0.5}  # Penalizar fuertemente clase 4
        )

        model.fit(X_train, y_train, cat_features=cat_features_idx, eval_set=(X_test, y_test), early_stopping_rounds=50)

    elif model_name.lower() == 'lightgbm':
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError("LightGBM no está instalado. Ejecuta: pip install lightgbm") from e

        # Ya están encoded desde build_features
        model = lgb.LGBMClassifier(
            objective='multiclass',
            num_class=5,
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=64,
            random_state=42,
            class_weight={0: 5.0, 1: 5.0, 2: 1.0, 3: 2.0, 4: 0.5}
        )

        # LightGBM no tiene early_stopping_rounds en scikit-learn wrapper, solo fit
        model.fit(X_train, y_train)

    else:
        raise ValueError('model_name debe ser catboost o lightgbm')

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    ll = log_loss(y_test, y_proba)
    report = classification_report(y_test, y_pred, digits=4)

    print(f"Modelo: {model_name}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 macro: {f1:.4f}")
    print(f"Log loss: {ll:.4f}")
    print("\nClassification report:\n", report)

    # Cross validação rápida
    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # cv_scores = []

    # for train_idx, valid_idx in skf.split(X, y):
    #     X_tr, X_val = X.iloc[train_idx], X.iloc[valid_idx]
    #     y_tr, y_val = y.iloc[train_idx], y.iloc[valid_idx]

    #     if model_name.lower() == 'catboost':
    #         model_cv = CatBoostClassifier(
    #             iterations=10,  # Reducido para CV rápida
    #             learning_rate=0.05,
    #             depth=6,
    #             loss_function='MultiClass',
    #             random_seed=42,
    #             verbose=0
    #         )
    #         model_cv.fit(X_tr, y_tr, cat_features=cat_features_idx, eval_set=(X_val, y_val), early_stopping_rounds=50)
    #     else:
    #         model_cv = lgb.LGBMClassifier(
    #             objective='multiclass',
    #             num_class=5,
    #             n_estimators=10,  # Reducido para CV rápida
    #             learning_rate=0.05,
    #             num_leaves=64,
    #             random_state=42
    #         )
    #         for col in categorical_cols:
    #             if col in X_tr.columns:
    #                 le = LabelEncoder()
    #                 X_tr[col] = le.fit_transform(X_tr[col].astype(str))
    #                 X_val[col] = le.transform(X_val[col].astype(str))
    #         model_cv.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], eval_metric='multi_logloss', early_stopping_rounds=50)

    #     cv_pred = model_cv.predict(X_val)
    #     cv_f1 = f1_score(y_val, cv_pred, average='macro')
    #     cv_scores.append(cv_f1)

    # print(f"CV F1 macro (5-fold): {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    print("CV omitido para guardar modelo rápido")

    model_file = os.path.join(output_path, f'{model_name}_petfinder.pkl')
    joblib.dump(model, model_file)
    print(f"Modelo guardado en: {model_file}")

    return model


if __name__ == '__main__':
    print("Iniciando entrenamiento...")
    parser = argparse.ArgumentParser(description='Entrena CatBoost/LightGBM para PetFinder AdoptionSpeed')
    parser.add_argument('--model', type=str, default='catboost', choices=['catboost', 'lightgbm'], help='Modelo a entrenar')
    parser.add_argument('--output', type=str, default='models', help='Carpeta para guardar el modelo')
    args = parser.parse_args()

    train_and_evaluate(model_name=args.model, output_path=args.output)
