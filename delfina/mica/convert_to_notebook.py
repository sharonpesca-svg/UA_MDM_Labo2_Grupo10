import json

cells = []

def code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source
    }

def md_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source
    }

cells.append(md_cell("# Modelado — TP Grupo 10\nLightGBM + Optuna (150 trials)"))

cells.append(code_cell(
    "import os\nimport numpy as np\nimport pandas as pd\nimport lightgbm as lgb\nimport optuna\nimport matplotlib.pyplot as plt\nimport seaborn as sns\nfrom sklearn.model_selection import StratifiedKFold, train_test_split\nfrom sklearn.metrics import cohen_kappa_score, accuracy_score, balanced_accuracy_score, confusion_matrix\nfrom sklearn.preprocessing import MinMaxScaler\nfrom joblib import dump, load"
))

cells.append(md_cell("## Rutas y parámetros"))

cells.append(code_cell(
    "BASE_DIR         = os.path.join(os.getcwd(), '..', '..')\n"
    "PATH_TRAIN_CLEAN = os.path.join(BASE_DIR, 'work/cleaned/train_clean.csv')\n"
    "PATH_TEST_CLEAN  = os.path.join(BASE_DIR, 'work/cleaned/test_clean.csv')\n"
    "PATH_TO_MODELS   = os.path.join(BASE_DIR, 'work/models')\n"
    "\n"
    "SEED    = 42\n"
    "N_FOLDS = 5\n"
    "\n"
    "FEATURES_SEL = [\n"
    "    'age_x_MaturitySize', 'Age', 'PhotoAmt', 'health_score',\n"
    "    'health_complete', 'rescuer_listings', 'Breed1', 'Sterilized',\n"
    "    'Breed2', 'photo_video_score'\n"
    "]"
))

cells.append(md_cell("## Carga de datos"))

cells.append(code_cell(
    "train = pd.read_csv(PATH_TRAIN_CLEAN)\n"
    "test  = pd.read_csv(PATH_TEST_CLEAN)\n"
    "print(f'Train: {train.shape} | Test: {test.shape}')"
))

cells.append(md_cell("## Feature Engineering"))

cells.append(code_cell(
    "def build_features(df, scaler=None, rescuer_counts=None, fit=True):\n"
    "    X = df.drop(columns=['AdoptionSpeed', 'PetID', 'RescuerID', 'Name', 'Description'])\n"
    "    X['health_score'] = (\n"
    "        (df['Vaccinated'] == 1).astype(int) +\n"
    "        (df['Dewormed']   == 1).astype(int) +\n"
    "        (df['Sterilized'] == 1).astype(int) +\n"
    "        (df['Health']     == 1).astype(int)\n"
    "    )\n"
    "    X['health_complete'] = (X['health_score'] == 4).astype(int)\n"
    "    X['age_x_MaturitySize'] = (X['Age'] / 12) * X['MaturitySize']\n"
    "    if fit:\n"
    "        scaler = MinMaxScaler()\n"
    "        X['age_x_MaturitySize'] = scaler.fit_transform(X[['age_x_MaturitySize']])\n"
    "    else:\n"
    "        X['age_x_MaturitySize'] = scaler.transform(X[['age_x_MaturitySize']])\n"
    "    X['photo_video_score'] = df['PhotoAmt'] + df['VideoAmt'] * 2\n"
    "    if fit:\n"
    "        rescuer_counts = df['RescuerID'].value_counts()\n"
    "        X['rescuer_listings'] = df['RescuerID'].map(rescuer_counts)\n"
    "    else:\n"
    "        X['rescuer_listings'] = df['RescuerID'].map(rescuer_counts).fillna(1)\n"
    "    return X[FEATURES_SEL], scaler, rescuer_counts\n"
    "\n"
    "X_train, scaler, rescuer_counts = build_features(train, fit=True)\n"
    "y_train = train['AdoptionSpeed']\n"
    "X_test, _, _ = build_features(test, scaler=scaler, rescuer_counts=rescuer_counts, fit=False)\n"
    "y_test = test['AdoptionSpeed']\n"
    "print(f'X_train: {X_train.shape} | X_test: {X_test.shape}')"
))

cells.append(md_cell("## Baseline"))

cells.append(code_cell(
    "skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)\n"
    "optuna.logging.set_verbosity(optuna.logging.WARNING)\n"
    "\n"
    "base_params = {\n"
    "    'objective': 'multiclass', 'num_class': 5, 'metric': 'multi_logloss',\n"
    "    'verbosity': -1, 'random_state': SEED, 'class_weight': 'balanced',\n"
    "}\n"
    "kappa_folds = []\n"
    "for fold, (idx_tr, idx_val) in enumerate(skf.split(X_train, y_train), 1):\n"
    "    X_tr, X_val = X_train.iloc[idx_tr], X_train.iloc[idx_val]\n"
    "    y_tr, y_val = y_train.iloc[idx_tr], y_train.iloc[idx_val]\n"
    "    m = lgb.LGBMClassifier(**base_params, n_estimators=500)\n"
    "    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],\n"
    "          callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])\n"
    "    k = cohen_kappa_score(y_val, m.predict(X_val), weights='quadratic')\n"
    "    kappa_folds.append(k)\n"
    "    print(f'  Fold {fold} — Kappa: {k:.4f}')\n"
    "print(f'\\nBaseline — Kappa medio: {np.mean(kappa_folds):.4f} ± {np.std(kappa_folds):.4f}')"
))

cells.append(md_cell("## Optuna — 150 trials"))

cells.append(code_cell(
    "def objective(trial):\n"
    "    params = {\n"
    "        'objective': 'multiclass', 'num_class': 5, 'metric': 'multi_logloss',\n"
    "        'verbosity': -1, 'random_state': SEED, 'class_weight': 'balanced',\n"
    "        'n_estimators': 500,\n"
    "        'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3, log=True),\n"
    "        'num_leaves':        trial.suggest_int('num_leaves', 20, 300),\n"
    "        'max_depth':         trial.suggest_int('max_depth', 3, 12),\n"
    "        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),\n"
    "        'subsample':         trial.suggest_float('subsample', 0.5, 1.0),\n"
    "        'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.5, 1.0),\n"
    "        'reg_alpha':         trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),\n"
    "        'reg_lambda':        trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),\n"
    "    }\n"
    "    kappas = []\n"
    "    for idx_tr, idx_val in skf.split(X_train, y_train):\n"
    "        X_tr, X_val = X_train.iloc[idx_tr], X_train.iloc[idx_val]\n"
    "        y_tr, y_val = y_train.iloc[idx_tr], y_train.iloc[idx_val]\n"
    "        m = lgb.LGBMClassifier(**params)\n"
    "        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],\n"
    "              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])\n"
    "        kappas.append(cohen_kappa_score(y_val, m.predict(X_val), weights='quadratic'))\n"
    "    return np.mean(kappas)\n"
    "\n"
    "db_path = os.path.abspath(os.path.join(BASE_DIR, 'work/db.sqlite3')).replace('\\\\', '/')\n"
    "storage_url = f'sqlite:///{db_path}'\n"
    "\n"
    "study = optuna.create_study(\n"
    "    direction='maximize',\n"
    "    sampler=optuna.samplers.TPESampler(seed=SEED),\n"
    "    storage=storage_url,\n"
    "    study_name='lgbm_kappa_tp10',\n"
    "    load_if_exists=True\n"
    ")\n"
    "study.optimize(objective, n_trials=150, show_progress_bar=True)\n"
    "\n"
    "print(f'\\nMejor Kappa: {study.best_value:.4f}')\n"
    "print(f'Mejores params: {study.best_params}')"
))

cells.append(md_cell("## Modelo final"))

cells.append(code_cell(
    "best_params = {\n"
    "    'objective': 'multiclass', 'num_class': 5, 'metric': 'multi_logloss',\n"
    "    'verbosity': -1, 'random_state': SEED, 'n_estimators': 1000,\n"
    "    'class_weight': 'balanced',\n"
    "    **study.best_params\n"
    "}\n"
    "X_tr_f, X_val_f, y_tr_f, y_val_f = train_test_split(\n"
    "    X_train, y_train, test_size=0.1, random_state=SEED, stratify=y_train\n"
    ")\n"
    "final_model = lgb.LGBMClassifier(**best_params)\n"
    "final_model.fit(X_tr_f, y_tr_f, eval_set=[(X_val_f, y_val_f)],\n"
    "                callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])\n"
    "os.makedirs(PATH_TO_MODELS, exist_ok=True)\n"
    "dump(final_model, os.path.join(PATH_TO_MODELS, 'lgbm_final_tp10.joblib'))\n"
    "print('Modelo guardado.')"
))

cells.append(md_cell("## Evaluación sobre test"))

cells.append(code_cell(
    "y_pred  = final_model.predict(X_test)\n"
    "kappa   = cohen_kappa_score(y_test, y_pred, weights='quadratic')\n"
    "acc     = accuracy_score(y_test, y_pred)\n"
    "bal_acc = balanced_accuracy_score(y_test, y_pred)\n"
    "\n"
    "print('=== Métricas sobre test ===')\n"
    "print(f'  Cohen Kappa (quadratic): {kappa:.4f}')\n"
    "print(f'  Accuracy:                {acc:.4f}')\n"
    "print(f'  Balanced Accuracy:       {bal_acc:.4f}')\n"
    "\n"
    "cm = confusion_matrix(y_test, y_pred)\n"
    "plt.figure(figsize=(7, 5))\n"
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',\n"
    "            xticklabels=[f'Pred {i}' for i in range(5)],\n"
    "            yticklabels=[f'Real {i}' for i in range(5)])\n"
    "plt.title('Matriz de confusión — LightGBM tuneado (150 trials)')\n"
    "plt.ylabel('Clase real')\n"
    "plt.xlabel('Clase predicha')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

cells.append(code_cell(
    "importance = pd.DataFrame({\n"
    "    'feature': FEATURES_SEL,\n"
    "    'importance': final_model.feature_importances_\n"
    "}).sort_values('importance', ascending=False)\n"
    "\n"
    "plt.figure(figsize=(8, 5))\n"
    "sns.barplot(data=importance, x='importance', y='feature', palette='viridis')\n"
    "plt.title('Importancia de features (gain)')\n"
    "plt.tight_layout()\n"
    "plt.show()"
))

cells.append(md_cell("## Predicciones sobre test"))

cells.append(code_cell(
    "resultado = test[['PetID']].copy()\n"
    "resultado['AdoptionSpeed_real']     = y_test.values\n"
    "resultado['AdoptionSpeed_predicho'] = y_pred\n"
    "resultado.to_csv(os.path.join(os.path.dirname(os.path.abspath('.')), 'delfina/mica/predicciones_test.csv'), index=False)\n"
    "print('Predicciones guardadas.')\n"
    "resultado.head(10)"
))

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "mi_entorno",
            "language": "python",
            "name": "mi_entorno"
        },
        "language_info": {"name": "python"}
    },
    "cells": cells
}

with open('4_modelado.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)

print("Notebook creado: 4_modelado.ipynb")
