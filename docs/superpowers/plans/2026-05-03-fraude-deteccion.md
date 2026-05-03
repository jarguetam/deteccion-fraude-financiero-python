# Detección de Fraude Financiero — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Construir un notebook Jupyter reproducible y documentado que entrene y evalúe un modelo ML de detección de fraude sobre `data/base.csv`, listo para entrega académica.

**Architecture:** Un único notebook con 6 módulos secuenciales — EDA → Preprocesamiento → Modelado → Evaluación → SHAP → Conclusiones. El split train/test se realiza primero; todas las transformaciones se ajustan solo sobre el train. Se comparan tres modelos (Regresión Logística baseline, Random Forest, XGBoost) y se selecciona el ganador aplicando el principio de parsimonia.

**Tech Stack:** Python 3.11+, pandas, numpy, scikit-learn, imbalanced-learn, xgboost, shap, matplotlib, seaborn, joblib

---

## Mapa de archivos

| Archivo | Acción | Responsabilidad |
|---|---|---|
| `requirements.txt` | Crear | Dependencias del proyecto |
| `notebooks/fraude_deteccion.ipynb` | Crear | Notebook principal con los 6 módulos |
| `models/` | Crear dir | Modelos serializados (.pkl) |

---

### Task 1: Entorno y dependencias

**Files:**
- Create: `requirements.txt`
- Create dir: `models/`

- [ ] **Step 1: Crear requirements.txt**

```
pandas==2.2.0
numpy==1.26.4
scikit-learn==1.4.0
imbalanced-learn==0.12.0
xgboost==2.0.3
shap==0.44.1
matplotlib==3.8.2
seaborn==0.13.2
joblib==1.3.2
jupyter==1.0.0
ipykernel==6.29.3
```

- [ ] **Step 2: Instalar dependencias**

```
pip install -r requirements.txt
```

Verificar que no hay errores. En Windows, si xgboost falla: `pip install xgboost --pre`.

- [ ] **Step 3: Crear directorio models/**

```
mkdir models
```

- [ ] **Step 4: Crear notebook vacío**

En VS Code: New File → `notebooks/fraude_deteccion.ipynb`. Seleccionar kernel Python 3.11+.

---

### Task 2: Módulo 0 — Configuración global

**Files:**
- Modify: `notebooks/fraude_deteccion.ipynb`

- [ ] **Step 1: Celda Markdown de portada**

```markdown
# Modelo de Detección de Fraude Financiero
## Caso Integrador Final — Programa de Especialización en Credit Scoring con Python

**Escenario:** Una institución financiera requiere fortalecer su sistema de gestión de riesgo
operacional mediante un modelo de ML capaz de identificar transacciones anómalas (fraude).

**Variable objetivo:** `fraud` — binaria (0 = legítima, 1 = fraude), tasa de incidencia ~2%.
```

- [ ] **Step 2: Celda de importaciones y configuración**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, roc_curve,
                              precision_recall_curve, average_precision_score,
                              f1_score, recall_score)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import shap
import joblib

RANDOM_STATE = 42
TEST_SIZE    = 0.20
np.random.seed(RANDOM_STATE)

sns.set_theme(style='whitegrid', palette='husl')
plt.rcParams['figure.figsize'] = (10, 5)

print("Librerías cargadas correctamente.")
```

Output esperado: `Librerías cargadas correctamente.`
Si hay `ImportError`, ejecutar `pip install <librería>` en terminal y reiniciar kernel.

- [ ] **Step 3: Celda de carga de datos**

```python
df = pd.read_csv('../data/base.csv')
print(f"Shape: {df.shape}")
print(f"\nColumnas: {df.columns.tolist()}")
df.head()
```

Output esperado: `Shape: (10000, 17)`

---

### Task 3: Módulo 1 — EDA (Distribución y estadísticos)

**Files:**
- Modify: `notebooks/fraude_deteccion.ipynb`

- [ ] **Step 1: Celda Markdown del módulo**

```markdown
## Módulo 1: Análisis Exploratorio de Datos (EDA)

> "Un buen modelo no nace en el algoritmo, nace en el EDA." — Instructor del programa.

El EDA es la base diagnóstica que justifica todas las decisiones técnicas posteriores.
Se analiza la calidad de los datos, el desbalance de clases, la distribución de variables
y el comportamiento diferenciado entre transacciones legítimas y fraudulentas.
```

- [ ] **Step 2: Estadísticos descriptivos y nulos**

```python
print("=== ESTADÍSTICOS DESCRIPTIVOS ===")
display(df.describe(percentiles=[.25, .50, .75, .90, .99]).round(2))

print("\n=== VALORES NULOS ===")
nulos = df.isnull().sum()
print(nulos[nulos > 0] if nulos.any() else "No se encontraron valores nulos.")
print(f"\nTipos de datos:\n{df.dtypes}")
```

- [ ] **Step 3: Distribución de la variable objetivo**

```python
conteo     = df['fraud'].value_counts()
pct_fraude = df['fraud'].mean() * 100

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(['Legítima (0)', 'Fraude (1)'], conteo.values,
            color=['steelblue', 'tomato'])
axes[0].set_title('Distribución de la variable objetivo')
axes[0].set_ylabel('Cantidad de transacciones')
for i, v in enumerate(conteo.values):
    axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')

axes[1].pie(conteo.values, labels=['Legítima', 'Fraude'],
            autopct='%1.2f%%', colors=['steelblue', 'tomato'], startangle=90)
axes[1].set_title('Proporción')

plt.suptitle(f'Desbalance de clases — Tasa de fraude: {pct_fraude:.2f}%',
             fontweight='bold')
plt.tight_layout()
plt.show()

print(f"\n→ {conteo[0]:,} legítimas vs {conteo[1]:,} fraudes ({pct_fraude:.2f}%)")
print("→ Estrategia requerida: SMOTE + métricas Recall/F1 (no Accuracy global)")
```

---

### Task 4: Módulo 1 — EDA (Outliers, correlaciones y comportamiento diferenciado)

**Files:**
- Modify: `notebooks/fraude_deteccion.ipynb`

- [ ] **Step 1: Distribuciones univariadas**

```python
num_cols = ['amount', 'client_credit_score', 'transaction_frequency',
            'customer_age', 'annual_income', 'account_balance',
            'num_previous_loans', 'customer_tenure', 'num_dependents',
            'education_level']

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    axes[i].hist(df[col], bins=30, color='steelblue', edgecolor='white', alpha=0.8)
    axes[i].set_title(col)
plt.suptitle('Distribuciones de variables numéricas', fontweight='bold')
plt.tight_layout()
plt.show()
```

- [ ] **Step 2: Outliers con boxplots**

```python
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    axes[i].boxplot(df[col], vert=True, patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.7))
    axes[i].set_title(col)
plt.suptitle('Detección de outliers (boxplots)', fontweight='bold')
plt.tight_layout()
plt.show()
print("→ No se eliminan registros. Outliers reales se conservan mediante Winsorización al p99.")
```

- [ ] **Step 3: Correlación de Spearman**

```python
corr = df[num_cols + ['fraud']].corr(method='spearman')

plt.figure(figsize=(12, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Correlación de Spearman\n'
          '(Se usa Spearman: más robusto ante outliers y relaciones no lineales que Pearson)',
          fontweight='bold')
plt.tight_layout()
plt.show()
```

- [ ] **Step 4: Comportamiento diferenciado fraud vs legítima**

```python
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    for label, color in [(0, 'steelblue'), (1, 'tomato')]:
        subset = df[df['fraud'] == label][col]
        axes[i].hist(subset, bins=25, alpha=0.6, density=True,
                     label=f'fraud={label}', color=color)
    axes[i].set_title(col)
    axes[i].legend(fontsize=7)
plt.suptitle('Distribución por clase: Legítima vs Fraude', fontweight='bold')
plt.tight_layout()
plt.show()
```

- [ ] **Step 5: Tasa de fraude por variable categórica**

```python
cat_cols = ['transaction_type', 'location', 'marital_status', 'housing_type']

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
for i, col in enumerate(cat_cols):
    tasa = df.groupby(col)['fraud'].mean().sort_values(ascending=False)
    tasa.plot(kind='bar', ax=axes[i], color='tomato', alpha=0.8, edgecolor='white')
    axes[i].set_title(f'Tasa de fraude por {col}')
    axes[i].set_ylabel('Proporción de fraude')
    axes[i].tick_params(axis='x', rotation=30)
    for p in axes[i].patches:
        axes[i].annotate(f'{p.get_height():.3f}',
                         (p.get_x() + p.get_width() / 2., p.get_height()),
                         ha='center', va='bottom', fontsize=9)
plt.suptitle('Tasa de fraude por variable categórica', fontweight='bold')
plt.tight_layout()
plt.show()
```

---

### Task 5: Módulo 2 — Preprocesamiento

**Files:**
- Modify: `notebooks/fraude_deteccion.ipynb`

- [ ] **Step 1: Celda Markdown del módulo**

```markdown
## Módulo 2: Preprocesamiento de Datos

**Regla de oro anti data leakage:** La división Train/Test se realiza PRIMERO.
Ninguna transformación (escalado, SMOTE, imputación, Winsorización) se aplica antes
de esta división. Los transformadores se ajustan exclusivamente sobre el conjunto de
entrenamiento (`.fit_transform()`) y luego se aplican al de prueba (`.transform()`).
```

- [ ] **Step 2: Feature engineering de timestamp**

```python
df['timestamp']  = pd.to_datetime(df['timestamp'])
df['hora']        = df['timestamp'].dt.hour
df['dia_semana']  = df['timestamp'].dt.dayofweek   # 0=lunes, 6=domingo
df['mes']         = df['timestamp'].dt.month

df = df.drop(columns=['transaction_id', 'timestamp'])

print("Feature engineering completado.")
print(f"Nuevas columnas: hora, dia_semana, mes")
print(f"Shape actualizado: {df.shape}")
```

- [ ] **Step 3: Split train/test estratificado**

```python
X = df.drop(columns=['fraud'])
y = df['fraud']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)

print(f"Train: {X_train.shape[0]:,} | Fraudes: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"Test:  {X_test.shape[0]:,}  | Fraudes: {y_test.sum()}  ({y_test.mean()*100:.2f}%)")
print("\n→ stratify=y garantiza la misma proporción de fraude en ambos conjuntos.")
```

- [ ] **Step 4: Definir columnas por tipo**

```python
num_features = ['amount', 'client_credit_score', 'transaction_frequency',
                'customer_age', 'annual_income', 'account_balance',
                'num_previous_loans', 'customer_tenure', 'num_dependents',
                'education_level', 'hora', 'dia_semana', 'mes']

cat_features = ['transaction_type', 'location', 'marital_status', 'housing_type']

print(f"Variables numéricas ({len(num_features)}): {num_features}")
print(f"Variables categóricas ({len(cat_features)}): {cat_features}")
print("\n→ education_level ya viene codificada ordinalmente (1-4), se trata como numérica.")
```

- [ ] **Step 5: Winsorización al p99 (train → aplicar a test)**

```python
winsor_limits = {}
for col in ['amount', 'annual_income', 'account_balance']:
    p99 = X_train[col].quantile(0.99)
    winsor_limits[col] = p99
    X_train[col] = X_train[col].clip(upper=p99)
    X_test[col]  = X_test[col].clip(upper=p99)

print("Winsorización al p99 aplicada:")
for col, lim in winsor_limits.items():
    print(f"  {col}: límite superior = {lim:,.2f}")
print("\n→ Se conservan todos los registros. Solo se acotan valores extremos.")
print("→ El límite se calcula sobre train y se aplica al test (anti data leakage).")
```

- [ ] **Step 6: Construir ColumnTransformer**

```python
preprocessor = ColumnTransformer(transformers=[
    ('num', Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler',  StandardScaler())
    ]), num_features),
    ('cat', Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot',  OneHotEncoder(drop='first', sparse_output=False,
                                  handle_unknown='ignore'))
    ]), cat_features)
])

print("ColumnTransformer definido.")
print("→ StandardScaler incluido para Regresión Logística.")
print("→ RF y XGBoost no lo requieren, pero no afecta su desempeño.")
```

- [ ] **Step 7: Aplicar transformador (fit sobre train, transform sobre test)**

```python
X_train_prep = preprocessor.fit_transform(X_train, y_train)
X_test_prep  = preprocessor.transform(X_test)

print(f"X_train_prep shape: {X_train_prep.shape}")
print(f"X_test_prep shape:  {X_test_prep.shape}")
print("\n→ .fit_transform() solo en train. .transform() en test.")
```

---

### Task 6: Módulo 3 — SMOTE y entrenamiento de modelos

**Files:**
- Modify: `notebooks/fraude_deteccion.ipynb`

- [ ] **Step 1: Celda Markdown del módulo**

```markdown
## Módulo 3: Estrategia ante Desbalance y Construcción del Modelo

**Estrategia de desbalance: SMOTE (Synthetic Minority Over-sampling Technique)**
SMOTE genera instancias sintéticas de la clase minoritaria (fraude) interpolando entre
muestras reales vecinas. Se aplica ÚNICAMENTE sobre el conjunto de entrenamiento
preprocesado. El test permanece con distribución real (~2% fraude).

Se entrenan tres modelos con StratifiedKFold(5) y RandomizedSearchCV:
- **Regresión Logística** (baseline) — modelo de referencia para parsimonia
- **Random Forest** — ensamble por Bagging
- **XGBoost** — ensamble por Boosting

**Principio de parsimonia:** Si RL obtiene métricas similares a RF/XGBoost (diferencia
F1 ≤ 0.03), se seleccionará RL por su mayor interpretabilidad ante reguladores.
```

- [ ] **Step 2: Aplicar SMOTE sobre train preprocesado**

```python
smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train_prep, y_train)

print(f"Antes de SMOTE  → {X_train_prep.shape[0]:,} muestras | fraudes: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print(f"Después de SMOTE → {X_train_res.shape[0]:,} muestras | fraudes: {y_train_res.sum()} ({y_train_res.mean()*100:.2f}%)")
print("\n→ SMOTE solo aplicado al train. El test mantiene distribución real.")
```

- [ ] **Step 3: Definir validación cruzada**

```python
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
print("StratifiedKFold(5) definido.")
print("→ Preserva la proporción de clases en cada fold durante el ajuste de hiperparámetros.")
```

- [ ] **Step 4: Regresión Logística baseline**

```python
param_grid_rl = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

search_rl = RandomizedSearchCV(
    LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    param_grid_rl, n_iter=6, cv=cv,
    scoring='f1', refit=True, random_state=RANDOM_STATE, n_jobs=-1
)
search_rl.fit(X_train_res, y_train_res)
best_rl = search_rl.best_estimator_

print(f"Mejor C para RL: {search_rl.best_params_['C']}")
print(f"F1 CV (train):   {search_rl.best_score_:.4f}")
```

- [ ] **Step 5: Random Forest**

```python
param_grid_rf = {
    'n_estimators':   [100, 200, 300],
    'max_depth':      [5, 10, 15, None],
    'min_samples_leaf': [1, 2, 5],
    'class_weight':   ['balanced']
}

search_rf = RandomizedSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    param_grid_rf, n_iter=20, cv=cv,
    scoring='f1', refit=True, random_state=RANDOM_STATE, n_jobs=-1
)
search_rf.fit(X_train_res, y_train_res)
best_rf = search_rf.best_estimator_

print(f"Mejores hiperparámetros RF: {search_rf.best_params_}")
print(f"F1 CV (train):              {search_rf.best_score_:.4f}")
```

- [ ] **Step 6: XGBoost**

```python
param_grid_xgb = {
    'n_estimators':    [100, 200, 300],
    'max_depth':       [3, 5, 7],
    'learning_rate':   [0.01, 0.05, 0.1, 0.2],
    'subsample':       [0.7, 0.8, 1.0],
    'colsample_bytree':[0.7, 0.8, 1.0]
}

search_xgb = RandomizedSearchCV(
    xgb.XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'),
    param_grid_xgb, n_iter=20, cv=cv,
    scoring='f1', refit=True, random_state=RANDOM_STATE, n_jobs=-1
)
search_xgb.fit(X_train_res, y_train_res)
best_xgb = search_xgb.best_estimator_

print(f"Mejores hiperparámetros XGB: {search_xgb.best_params_}")
print(f"F1 CV (train):               {search_xgb.best_score_:.4f}")
```

- [ ] **Step 7: Guardar modelos y preprocesador**

```python
joblib.dump(best_rl,        '../models/logistic_regression.pkl')
joblib.dump(best_rf,        '../models/random_forest.pkl')
joblib.dump(best_xgb,       '../models/xgboost.pkl')
joblib.dump(preprocessor,   '../models/preprocessor.pkl')

print("Modelos guardados en /models/")
```

---

### Task 7: Módulo 3 — Bono: Isolation Forest

**Files:**
- Modify: `notebooks/fraude_deteccion.ipynb`

- [ ] **Step 1: Celda Markdown**

```markdown
### Extensión Opcional: Isolation Forest (Modelo No Supervisado)
Se asume que la variable `fraud` no está disponible en el entrenamiento.
El Isolation Forest detecta anomalías basándose en cuán fácil es "aislar" un punto:
los fraudes son raros e inusuales, por lo que se aíslan con menos cortes aleatorios.
Esta extensión aplica al **bono de +5 puntos** en la dimensión de interpretabilidad.
```

- [ ] **Step 2: Entrenar y evaluar Isolation Forest**

```python
iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.02,       # proporción esperada de fraude
    random_state=RANDOM_STATE
)

# Solo se entrena con X_train_prep — sin usar y_train (no supervisado)
iso_forest.fit(X_train_prep)

# -1 = anomalía (fraude), 1 = normal → convertir a 0/1
iso_pred = np.where(iso_forest.predict(X_test_prep) == -1, 1, 0)

print("=== Isolation Forest (No Supervisado) ===")
print(classification_report(y_test, iso_pred, target_names=['Legítima', 'Fraude']))
print(f"Recall fraude: {recall_score(y_test, iso_pred):.4f}")
print(f"F1 fraude:     {f1_score(y_test, iso_pred):.4f}")
print("\n→ Sin acceso a la etiqueta 'fraud' durante el entrenamiento.")
```

---

### Task 8: Módulo 4 — Evaluación y selección de modelo

**Files:**
- Modify: `notebooks/fraude_deteccion.ipynb`

- [ ] **Step 1: Celda Markdown del módulo**

```markdown
## Módulo 4: Evaluación de Desempeño y Selección de Modelo

**Métricas priorizadas:** Recall y F1-Score sobre la clase positiva (fraude).
El Accuracy global no es informativo: un modelo que predice siempre "legítima"
obtendría ~98% de Accuracy sin detectar ningún fraude.

**Umbral:** Se analizará la curva Precision-Recall para identificar el umbral
óptimo. No se usa 0.5 por defecto.
```

- [ ] **Step 2: Función auxiliar de evaluación**

```python
def evaluar_modelo(nombre, modelo, X, y_true, umbral=0.5):
    y_proba = modelo.predict_proba(X)[:, 1]
    y_pred  = (y_proba >= umbral).astype(int)
    return {
        'Modelo':  nombre,
        'Recall':  recall_score(y_true, y_pred),
        'F1':      f1_score(y_true, y_pred),
        'AUC-ROC': roc_auc_score(y_true, y_proba),
        'AUC-PR':  average_precision_score(y_true, y_proba),
        'Umbral':  umbral
    }
```

- [ ] **Step 3: Matrices de confusión**

```python
modelos = [('Regresión Logística', best_rl),
           ('Random Forest',       best_rf),
           ('XGBoost',             best_xgb)]

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (nombre, modelo) in zip(axes, modelos):
    y_pred = modelo.predict(X_test_prep)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Legítima', 'Fraude'],
                yticklabels=['Legítima', 'Fraude'])
    ax.set_title(f'{nombre}\nFN={cm[1,0]} (fraudes no detectados)')
    ax.set_ylabel('Real'); ax.set_xlabel('Predicho')

plt.suptitle('Matrices de Confusión — Un Falso Negativo (fraude no detectado) tiene mayor costo',
             fontweight='bold')
plt.tight_layout()
plt.show()
```

- [ ] **Step 4: Curvas ROC y Precision-Recall**

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['blue', 'green', 'red']

for (nombre, modelo), color in zip(modelos, colors):
    y_proba = modelo.predict_proba(X_test_prep)[:, 1]

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[0].plot(fpr, tpr, color=color,
                 label=f'{nombre} (AUC={roc_auc_score(y_test, y_proba):.3f})')

    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    axes[1].plot(rec, prec, color=color,
                 label=f'{nombre} (AP={average_precision_score(y_test, y_proba):.3f})')

axes[0].plot([0,1],[0,1],'k--', alpha=0.3)
axes[0].set_title('Curva ROC'); axes[0].set_xlabel('FPR'); axes[0].set_ylabel('TPR')
axes[0].legend()

axes[1].axhline(y=0.02, color='black', linestyle='--', alpha=0.3, label='Baseline (~2%)')
axes[1].set_title('Curva Precision-Recall')
axes[1].set_xlabel('Recall'); axes[1].set_ylabel('Precision')
axes[1].legend()

plt.suptitle('Discriminación de modelos', fontweight='bold')
plt.tight_layout()
plt.show()
```

- [ ] **Step 5: Umbral óptimo (sobre el modelo con mejor AUC-PR)**

```python
# Calcular AUC-PR para cada modelo y elegir el mejor
auc_prs = {nombre: average_precision_score(y_test, modelo.predict_proba(X_test_prep)[:, 1])
           for nombre, modelo in modelos}
nombre_mejor_auc = max(auc_prs, key=auc_prs.get)
modelo_mejor_auc = dict(modelos)[nombre_mejor_auc]

y_proba_cand = modelo_mejor_auc.predict_proba(X_test_prep)[:, 1]
prec, rec, umbrales = precision_recall_curve(y_test, y_proba_cand)
f1_scores  = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-9)
idx_opt    = np.argmax(f1_scores)
umbral_opt = umbrales[idx_opt]

plt.figure(figsize=(10, 4))
plt.plot(umbrales, prec[:-1], label='Precision', color='blue')
plt.plot(umbrales, rec[:-1],  label='Recall',    color='green')
plt.plot(umbrales, f1_scores,  label='F1-Score',  color='red')
plt.axvline(x=umbral_opt, color='black', linestyle='--',
            label=f'Umbral óptimo = {umbral_opt:.3f}')
plt.title(f'Análisis de umbral — {nombre_mejor_auc}', fontweight='bold')
plt.xlabel('Umbral de decisión'); plt.legend()
plt.tight_layout()
plt.show()

print(f"Umbral óptimo: {umbral_opt:.4f}  |  Precision: {prec[idx_opt]:.4f}  |  Recall: {rec[idx_opt]:.4f}  |  F1: {f1_scores[idx_opt]:.4f}")
```

- [ ] **Step 6: Tabla comparativa y principio de parsimonia**

```python
resultados = [evaluar_modelo(n, m, X_test_prep, y_test, umbral=umbral_opt)
              for n, m in modelos]
df_res = pd.DataFrame(resultados).set_index('Modelo').round(4)
display(df_res.style
        .highlight_max(axis=0, color='lightgreen')
        .highlight_min(axis=0, color='#ffcccc'))

# Aplicar principio de parsimonia
f1_rl   = df_res.loc['Regresión Logística', 'F1']
mejor_ml = max(df_res.loc['Random Forest', 'F1'], df_res.loc['XGBoost', 'F1'])
diff    = mejor_ml - f1_rl

print("\n=== PRINCIPIO DE PARSIMONIA ===")
if diff <= 0.03:
    print(f"Diferencia F1 (RL vs mejor ML): {diff:.4f}")
    print("→ RL rinde de forma comparable. Se selecciona por parsimonia.")
    print("  Es más interpretable, auditable y preferida ante reguladores.")
    modelo_final  = best_rl
    nombre_final  = 'Regresión Logística'
else:
    print(f"Diferencia F1 (RL vs mejor ML): {diff:.4f}")
    print("→ Diferencia significativa. Se selecciona el modelo de mayor desempeño.")
    if df_res.loc['XGBoost', 'F1'] >= df_res.loc['Random Forest', 'F1']:
        modelo_final = best_xgb
        nombre_final = 'XGBoost'
    else:
        modelo_final = best_rf
        nombre_final = 'Random Forest'

print(f"\nMODELO FINAL SELECCIONADO: {nombre_final}")
```

- [ ] **Step 7: PSI (Population Stability Index)**

```python
def calcular_psi(score_train, score_test, bins=10):
    """PSI < 0.1: estable | 0.10-0.25: monitorear | > 0.25: inestable"""
    breaks   = np.unique(np.percentile(score_train, np.linspace(0, 100, bins + 1)))
    pct_tr   = np.histogram(score_train, bins=breaks)[0] / len(score_train) + 1e-6
    pct_te   = np.histogram(score_test,  bins=breaks)[0] / len(score_test)  + 1e-6
    return np.sum((pct_te - pct_tr) * np.log(pct_te / pct_tr))

psi_val = calcular_psi(
    modelo_final.predict_proba(X_train_res)[:, 1],
    modelo_final.predict_proba(X_test_prep)[:, 1]
)

estado = "ESTABLE" if psi_val < 0.1 else "MONITOREAR" if psi_val < 0.25 else "INESTABLE"
print(f"PSI del modelo final ({nombre_final}): {psi_val:.4f} → {estado}")
```

---

### Task 9: Módulo 5 — Explicabilidad SHAP

**Files:**
- Modify: `notebooks/fraude_deteccion.ipynb`

- [ ] **Step 1: Celda Markdown del módulo**

```markdown
## Módulo 5: Interpretabilidad con SHAP

SHAP (SHapley Additive exPlanations) está basado en la Teoría de Juegos.
Asigna a cada variable una contribución marginal exacta a la predicción.

- **Valor Base:** probabilidad promedio de fraude si no se supiera nada del cliente.
- **Valor SHAP:** cuánto suma o resta cada variable para llegar a la predicción final.

Se generan 3 visualizaciones:
1. **Feature Importance (barras):** importancia global
2. **Beeswarm plot:** magnitud y dirección (el "gráfico estrella")
3. **Waterfall plot:** explicación de un caso concreto de fraude
```

- [ ] **Step 2: Obtener nombres de features y calcular SHAP values**

```python
num_names  = num_features
cat_names  = list(preprocessor.named_transformers_['cat']
                  .named_steps['onehot'].get_feature_names_out(cat_features))
feat_names = num_names + cat_names

n_shap  = min(500, X_test_prep.shape[0])
X_shap  = X_test_prep[:n_shap]

# shap.Explainer auto-selecciona LinearExplainer para RL y TreeExplainer para RF/XGBoost
explainer   = shap.Explainer(modelo_final, X_train_res, feature_names=feat_names)
shap_values = explainer(X_shap)

print(f"SHAP calculado sobre {n_shap} muestras del test.")
print(f"Valor base (E[f(x)]): {shap_values.base_values.mean():.4f}")
```

- [ ] **Step 3: Feature Importance global (barras)**

```python
plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values, max_display=15, show=False)
plt.title('Feature Importance Global (SHAP)\n'
          'Variables que más influyen en la detección de fraude',
          fontweight='bold')
plt.tight_layout()
plt.show()
```

- [ ] **Step 4: Beeswarm plot global**

```python
plt.figure(figsize=(10, 8))
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.title('Beeswarm Plot — Magnitud y Dirección\n'
          'Rojo = valor alto de la variable | Azul = valor bajo',
          fontweight='bold')
plt.tight_layout()
plt.show()
print("→ Eje X: Impacto en predicción (>0 aumenta prob. fraude, <0 la reduce)")
print("→ Eje Y: Variables ordenadas por importancia global")
```

- [ ] **Step 5: Waterfall plot (un caso de fraude)**

```python
idx_fraudes = np.where(y_test.values[:n_shap] == 1)[0]
if len(idx_fraudes) > 0:
    idx  = idx_fraudes[0]
    prob = modelo_final.predict_proba(X_shap[idx:idx+1])[0, 1]
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[idx], max_display=12, show=False)
    plt.title(f'Waterfall Plot — Explicación Local\n'
              f'Caso de fraude detectado (prob={prob:.2%})',
              fontweight='bold')
    plt.tight_layout()
    plt.show()
    print(f"→ Probabilidad de fraude para este cliente: {prob:.2%}")
    print("→ Barras rojas: variables que aumentan el riesgo.")
    print("→ Barras azules: variables que reducen el riesgo.")
    print("→ Este gráfico puede mostrarse directamente al Comité de Riesgos.")
```

---

### Task 10: Módulo 6 — Conclusiones y entrega

**Files:**
- Modify: `notebooks/fraude_deteccion.ipynb`

- [ ] **Step 1: Celda Markdown del módulo**

```markdown
## Módulo 6: Conclusiones y Recomendaciones Operativas
```

- [ ] **Step 2: Resumen ejecutivo en código**

```python
y_pred_final = (modelo_final.predict_proba(X_test_prep)[:, 1] >= umbral_opt).astype(int)

print("=" * 65)
print("   RESUMEN EJECUTIVO — MODELO DE DETECCIÓN DE FRAUDE")
print("=" * 65)
print(f"\n DATASET: 10,000 transacciones | {y.sum()} fraudes ({y.mean()*100:.2f}%)")
print(f"  Desbalance tratado con SMOTE sobre el conjunto de entrenamiento.\n")
print(f" MODELOS EVALUADOS: Regresión Logística, Random Forest, XGBoost")
print(f"  Optimizados con RandomizedSearchCV + StratifiedKFold(5).\n")
print(f" MODELO SELECCIONADO: {nombre_final}")
print(f"  Umbral operativo: {umbral_opt:.4f} (optimizado sobre curva Precision-Recall)\n")
print(classification_report(y_test, y_pred_final, target_names=['Legítima', 'Fraude']))
estado_psi = "Estable" if psi_val < 0.1 else "Monitorear" if psi_val < 0.25 else "Revisar"
print(f" ESTABILIDAD (PSI): {psi_val:.4f} — {estado_psi}")
print(f"\n RECOMENDACIÓN OPERATIVA:")
print(f"  - Implementar umbral {umbral_opt:.4f} en producción.")
print(f"  - Monitorear PSI mensualmente. Si PSI > 0.25, reentrenar.")
print(f"  - Revisar falsos negativos trimestralmente.")
print("=" * 65)
```

- [ ] **Step 3: Ejecutar notebook completo y verificar**

```
Kernel → Restart & Run All
```

Verificar:
- Todas las celdas corren sin errores de arriba a abajo
- `Shape: (10000, 17)` en la carga de datos
- El resumen ejecutivo final se imprime sin excepciones
- El directorio `models/` contiene: `logistic_regression.pkl`, `random_forest.pkl`, `xgboost.pkl`, `preprocessor.pkl`

---

## Verificación final (checklist de rúbrica)

- [ ] `random_state=42` presente en: split, SMOTE, RL, RF, XGBoost, StratifiedKFold
- [ ] SMOTE aplicado únicamente después del split, sobre `X_train_prep`
- [ ] `.fit_transform()` solo en train; `.transform()` en test
- [ ] Winsorización calculada sobre train, aplicada a test
- [ ] Métricas reportadas: Recall y F1-Score (no solo Accuracy)
- [ ] Umbral óptimo calculado sobre curva PR (no 0.5)
- [ ] Principio de parsimonia documentado con justificación en celda Markdown
- [ ] Tres gráficos SHAP: barras, beeswarm, waterfall
- [ ] Cada módulo tiene celda Markdown con el "porqué" de las decisiones
- [ ] Notebook ejecuta de principio a fin sin errores (`Restart & Run All`)
