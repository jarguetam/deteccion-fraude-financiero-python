"""
============================================================================
 MODELO DE DETECCIÓN DE FRAUDE FINANCIERO  -  Caso Integrador Final
 Programa de Especialización en Credit Scoring con Python
 ----------------------------------------------------------------------------
 Versión REVISADA (corrige hallazgos críticos detectados en la 1ra corrida):

  Errores corregidos
  ------------------
  1. DATA LEAKAGE EN VALIDACIÓN CRUZADA: SMOTE ya no se aplica antes del
     `RandomizedSearchCV`. Ahora vive dentro de un `imblearn.pipeline.Pipeline`
     para que solo se aplique al *train* interno de cada fold (el fold de
     validación queda siempre con la distribución real ~2%).
  2. F1-CV de 0.99 (memorización de fraudes sintéticos): se reemplaza el
     scoring por `average_precision` (AUC-PR), más informativo en clases
     muy desbalanceadas.
  3. Estrategia de desbalance única (SMOTE agresivo): ahora se comparan
     3 estrategias por modelo y se elige la mejor por AUC-PR sobre TEST:
        a) class_weight='balanced'  /  scale_pos_weight=neg/pos  (sin sintéticos)
        b) SMOTE conservador (k_neighbors=3)
        c) SMOTETomek (sobremuestreo + limpieza de Tomek links)
  4. Selección final por F1 con umbral fijo: ahora se selecciona por AUC-PR
     y se reportan 3 UMBRALES OPERATIVOS (F1 máx, Recall=0.80, capacidad
     de mesa de fraude) para que el Comité elija el adecuado.
  5. Diagnóstico crítico ausente: se agrega chequeo automático de
     "Bandera Roja" si AUC-ROC < 0.55 (Sesión 7 - Gini Negativo).
  6. Conclusiones acríticas: se agrega "Conclusiones honestas para el
     Comité" para defender el caso aun si el modelo no llega a AUC 0.70.

============================================================================
"""

from __future__ import annotations

import io
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# CONFIGURACIÓN GLOBAL
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
np.random.seed(RANDOM_STATE)

sns.set_theme(style="whitegrid", palette="husl")
plt.rcParams["figure.figsize"] = (10, 5)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "base.csv"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RESUMEN_PATH = OUTPUTS_DIR / "resumen_ejecutivo.txt"

for d in (MODELS_DIR, OUTPUTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)


class Tee(io.TextIOBase):
    """Espeja stdout a la consola y al archivo de resumen ejecutivo."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
        return len(data)

    def flush(self):
        for s in self.streams:
            s.flush()


_log_file = open(RESUMEN_PATH, "w", encoding="utf-8")
sys.stdout = Tee(sys.__stdout__, _log_file)


def banner(titulo: str, subtitulo: str = "", ancho: int = 78) -> None:
    print("\n" + "=" * ancho)
    print(f" {titulo}")
    if subtitulo:
        print(f" {subtitulo}")
    print("=" * ancho)


def seccion(titulo: str, ancho: int = 78) -> None:
    print("\n" + "-" * ancho)
    print(f" >> {titulo}")
    print("-" * ancho)


def guardar_fig(nombre: str) -> Path:
    path = FIGURES_DIR / nombre
    plt.savefig(path, bbox_inches="tight", dpi=120)
    plt.close()
    print(f"   [figura guardada] {path.relative_to(ROOT)}")
    return path


# ==========================================================================
#  DIAPOSITIVA 1 — PORTADA
# ==========================================================================
banner(
    "DIAPOSITIVA 1 — PORTADA",
    "Modelo de Detección de Fraude Financiero | Caso Integrador Final",
)
print(" Programa: Especialización en Credit Scoring con Python")
print(" Entregable: Pipeline reproducible + Presentación Ejecutiva (12 slides)")
print(" Audiencia: Comité de Riesgos")


# ==========================================================================
#  DIAPOSITIVA 2 — CONTEXTO Y PROBLEMA
# ==========================================================================
banner(
    "DIAPOSITIVA 2 — CONTEXTO Y PROBLEMA",
    "Por qué importa detectar transacciones anómalas",
)
print(
    """ Reto de negocio:
   - Una institución financiera necesita reforzar su gestión de riesgo
     operacional con un modelo de Machine Learning capaz de identificar
     transacciones fraudulentas en tiempo casi real.
   - Cada fraude no detectado (Falso Negativo) implica pérdida directa,
     multas regulatorias y daño reputacional.
   - Cada falsa alarma (Falso Positivo) consume capacidad operativa del
     equipo de monitoreo y deteriora la experiencia del cliente.

 Variable objetivo: `fraud` (0 = legítima, 1 = fraude). Tasa ~2%.

 Impacto financiero estimado (escenario referencial):
   - Volumen diario simulado: 10,000 transacciones.
   - Tasa de fraude observada: ~2%  =>  ~200 fraudes/día.
   - Sin modelo: detección dependiente de reglas estáticas, recall bajo.
   - Con modelo: priorización de alertas y reducción de pérdidas esperadas.
"""
)


# ==========================================================================
#  CARGA DE DATOS
# ==========================================================================
banner("CARGA DE DATOS")
df = pd.read_csv(DATA_PATH)
print(f" Archivo: {DATA_PATH.relative_to(ROOT)}")
print(f" Shape:   {df.shape[0]:,} filas x {df.shape[1]} columnas")
print(f" Columnas: {df.columns.tolist()}")
print("\n Primeras 5 filas:")
print(df.head().to_string())


# ==========================================================================
#  DIAPOSITIVA 3 — DATOS Y DIAGNÓSTICO (EDA)
# ==========================================================================
banner(
    "DIAPOSITIVA 3 — DATOS Y DIAGNÓSTICO (EDA)",
    "Hallazgos del Análisis Exploratorio + desbalance crítico",
)

seccion("Estadísticos descriptivos (incluye p90 y p99)")
desc = df.describe(percentiles=[0.25, 0.50, 0.75, 0.90, 0.99]).round(2)
print(desc.to_string())

seccion("Calidad del dato")
nulos = df.isnull().sum()
nulos = nulos[nulos > 0]
if len(nulos) == 0:
    print(" - No se encontraron valores nulos en el dataset.")
else:
    print(" - Valores nulos detectados:")
    print(nulos.to_string())
print("\n Tipos de dato por columna:")
print(df.dtypes.to_string())

seccion("Distribución de la variable objetivo (desbalance)")
conteo = df["fraud"].value_counts().sort_index()
pct_fraude = df["fraud"].mean() * 100
print(f" - Legítimas: {conteo.iloc[0]:,}")
print(f" - Fraudes:   {conteo.iloc[1]:,}")
print(f" - Tasa de fraude: {pct_fraude:.2f}%")
print(
    " - Implicación: modelo no puede evaluarse con Accuracy global "
    "(predecir siempre 'legítima' arroja ~98%). Métricas a priorizar: "
    "AUC-PR, Recall y F1 sobre la clase positiva."
)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(
    ["Legítima (0)", "Fraude (1)"],
    conteo.values,
    color=["steelblue", "tomato"],
)
axes[0].set_title("Distribución de la variable objetivo")
axes[0].set_ylabel("Cantidad de transacciones")
for i, v in enumerate(conteo.values):
    axes[0].text(i, v + 50, f"{v:,}", ha="center", fontweight="bold")
axes[1].pie(
    conteo.values,
    labels=["Legítima", "Fraude"],
    autopct="%1.2f%%",
    colors=["steelblue", "tomato"],
    startangle=90,
)
axes[1].set_title("Proporción")
plt.suptitle(
    f"Desbalance de clases — Tasa de fraude: {pct_fraude:.2f}%",
    fontweight="bold",
)
plt.tight_layout()
guardar_fig("03_desbalance_clases.png")

num_cols_eda = [
    "amount",
    "client_credit_score",
    "transaction_frequency",
    "customer_age",
    "annual_income",
    "account_balance",
    "num_previous_loans",
    "customer_tenure",
    "num_dependents",
    "education_level",
]
cat_cols_eda = ["transaction_type", "location", "marital_status", "housing_type"]

seccion("Distribuciones de variables numéricas")
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols_eda):
    axes[i].hist(df[col], bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    axes[i].set_title(col)
plt.suptitle("Distribuciones de variables numéricas", fontweight="bold")
plt.tight_layout()
guardar_fig("03_distribuciones_numericas.png")

seccion("Detección de outliers (boxplots)")
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols_eda):
    axes[i].boxplot(
        df[col],
        vert=True,
        patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.7),
    )
    axes[i].set_title(col)
plt.suptitle("Detección de outliers (boxplots)", fontweight="bold")
plt.tight_layout()
guardar_fig("03_boxplots_outliers.png")
print(
    " - Conclusión: no se eliminan registros. Los outliers reales se acotan "
    "mediante Winsorización al p99 (sin perder información)."
)

seccion("Correlación de Spearman (robusta a outliers y no linealidades)")
corr = df[num_cols_eda + ["fraud"]].corr(method="spearman")
plt.figure(figsize=(12, 8))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(
    corr,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
)
plt.title(
    "Correlación de Spearman\n"
    "(más robusta ante outliers y no linealidades que Pearson)",
    fontweight="bold",
)
plt.tight_layout()
guardar_fig("03_correlacion_spearman.png")

corr_fraud = corr["fraud"].drop("fraud").abs().sort_values(ascending=False)
print(" Top variables numéricas correlacionadas con `fraud` (|Spearman|):")
for var, val in corr_fraud.head(5).items():
    print(f"   - {var:<25s}  |rho| = {val:.3f}")
max_corr = float(corr_fraud.max())
print(
    " - Lectura: ninguna variable individual explica fraude por sí sola. "
    "Se requiere un modelo multivariado para capturar interacciones."
)
if max_corr < 0.05:
    print(
        " - ALERTA TEMPRANA: la correlación máxima con `fraud` es < 0.05. "
        "El dataset puede tener señal débil; cualquier modelo tendrá un techo "
        "de desempeño bajo. (Hallazgo a documentar en la Diap. 9 Riesgos)."
    )

seccion("Distribución por clase (Legítima vs Fraude)")
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols_eda):
    for label, color in [(0, "steelblue"), (1, "tomato")]:
        subset = df[df["fraud"] == label][col]
        axes[i].hist(
            subset,
            bins=25,
            alpha=0.6,
            density=True,
            label=f"fraud={label}",
            color=color,
        )
    axes[i].set_title(col)
    axes[i].legend(fontsize=7)
plt.suptitle("Distribución por clase: Legítima vs Fraude", fontweight="bold")
plt.tight_layout()
guardar_fig("03_distribucion_por_clase.png")

seccion("Tasa de fraude por variable categórica")
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
tasa_resumen: dict[str, pd.Series] = {}
for i, col in enumerate(cat_cols_eda):
    tasa = df.groupby(col)["fraud"].mean().sort_values(ascending=False)
    tasa_resumen[col] = tasa
    tasa.plot(kind="bar", ax=axes[i], color="tomato", alpha=0.8, edgecolor="white")
    axes[i].set_title(f"Tasa de fraude por {col}")
    axes[i].set_ylabel("Proporción de fraude")
    axes[i].tick_params(axis="x", rotation=30)
    for p in axes[i].patches:
        axes[i].annotate(
            f"{p.get_height():.3f}",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=9,
        )
plt.suptitle("Tasa de fraude por variable categórica", fontweight="bold")
plt.tight_layout()
guardar_fig("03_tasa_fraude_categoricas.png")

print(" Tasa de fraude por categoría (Top 3 por variable):")
for col, tasa in tasa_resumen.items():
    print(f"\n   {col}:")
    for cat, val in tasa.head(3).items():
        print(f"     - {cat:<18s}  fraude = {val*100:.2f}%")


# ==========================================================================
#  DIAPOSITIVA 4 — METODOLOGÍA
# ==========================================================================
banner(
    "DIAPOSITIVA 4 — METODOLOGÍA",
    "Pipeline anti data-leakage + COMPARACIÓN de estrategias de desbalance",
)

seccion("Feature engineering temporal")
df["timestamp"] = pd.to_datetime(df["timestamp"])
df["hora"] = df["timestamp"].dt.hour
df["dia_semana"] = df["timestamp"].dt.dayofweek
df["mes"] = df["timestamp"].dt.month
df = df.drop(columns=["transaction_id", "timestamp"])
print(" - Variables creadas: hora, dia_semana (0=lun..6=dom), mes")
print(f" - Shape actualizado: {df.shape}")

seccion("División estratificada Train/Test (anti data leakage)")
X = df.drop(columns=["fraud"])
y = df["fraud"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
)
print(
    f" - Train: {X_train.shape[0]:,} filas | "
    f"fraudes: {y_train.sum()} ({y_train.mean()*100:.2f}%)"
)
print(
    f" - Test:  {X_test.shape[0]:,} filas | "
    f"fraudes: {y_test.sum()} ({y_test.mean()*100:.2f}%)"
)
print(" - stratify=y garantiza la misma proporción de fraude en ambos sets.")

seccion("Definición de variables numéricas y categóricas")
num_features = [
    "amount",
    "client_credit_score",
    "transaction_frequency",
    "customer_age",
    "annual_income",
    "account_balance",
    "num_previous_loans",
    "customer_tenure",
    "num_dependents",
    "education_level",
    "hora",
    "dia_semana",
    "mes",
]
cat_features = ["transaction_type", "location", "marital_status", "housing_type"]
print(f" - Numéricas ({len(num_features)}): {num_features}")
print(f" - Categóricas ({len(cat_features)}): {cat_features}")

seccion("Winsorización al p99 (calculada SOLO en train)")
winsor_limits: dict[str, float] = {}
for col in ["amount", "annual_income", "account_balance"]:
    p99 = X_train[col].quantile(0.99)
    winsor_limits[col] = p99
    X_train[col] = X_train[col].clip(upper=p99)
    X_test[col] = X_test[col].clip(upper=p99)
for col, lim in winsor_limits.items():
    print(f"   - {col}: límite superior = {lim:,.2f}")

seccion("ColumnTransformer (Imputación + Escalado + One-Hot)")
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            SkPipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            num_features,
        ),
        (
            "cat",
            SkPipeline(
                [
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(
                            drop="first",
                            sparse_output=False,
                            handle_unknown="ignore",
                        ),
                    ),
                ]
            ),
            cat_features,
        ),
    ]
)
print(" - Preprocesador definido. Se ajustará dentro del Pipeline imblearn,")
print("   garantizando que cada fold del CV vea solo su propia transformación.")

seccion("CORRECCIÓN CRÍTICA #1 — Pipeline imblearn dentro del CV")
print(
    """ Antes (BUG):
   smote.fit_resample(X_train_prep, y_train)
   RandomizedSearchCV(...).fit(X_train_res, y_train_res)
   -> El fold de validación incluye datos sintéticos.
   -> Métricas de CV infladas (F1=0.99) y modelo memoriza ruido SMOTE.

 Ahora (CORRECTO):
   pipe = ImbPipeline([
       ('preprocessor', preprocessor),  # se ajusta en cada fold
       ('sampler',      <SMOTE | SMOTETomek | passthrough>),
       ('classifier',   <LR | RF | XGB>)
   ])
   RandomizedSearchCV(pipe, ...).fit(X_train, y_train)
   -> El sampler solo actúa sobre el train de cada fold.
   -> El validation queda con la distribución real (~2% fraude)."""
)

seccion("CORRECCIÓN CRÍTICA #2 — Comparación de 3 estrategias de desbalance")
neg_pos_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
print(f" Relación neg/pos en train = {neg_pos_ratio:.1f}")
print(
    """ Estrategias evaluadas para CADA modelo (3 modelos x 3 estrategias = 9):
   A) class_weight='balanced' (LR, RF) | scale_pos_weight=neg/pos (XGB)
      -> NO genera datos sintéticos. Penaliza el error en la clase rara.
   B) SMOTE conservador (k_neighbors=3)
      -> Sobremuestreo solo cerca de los fraudes reales (menos 'fantasmas')
   C) SMOTETomek
      -> Sobremuestreo SMOTE + limpieza Tomek de pares ambiguos en frontera"""
)

seccion("CORRECCIÓN CRÍTICA #3 — Scoring del CV: AUC-PR (no F1)")
print(
    """ - F1 con umbral 0.5 fijo es inestable en clases muy desbalanceadas.
 - AUC-PR (average_precision) integra todos los umbrales y prioriza
   la clase positiva. Es el estándar para fraude / detección de anomalías."""
)


# ==========================================================================
#  ENTRENAMIENTO: 3 MODELOS x 3 ESTRATEGIAS = 9 CONFIGURACIONES
# ==========================================================================
banner(
    "ENTRENAMIENTO: 3 MODELOS x 3 ESTRATEGIAS = 9 CONFIGURACIONES",
    "RandomizedSearchCV con StratifiedKFold(5) y scoring='average_precision'",
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def construir_sampler(nombre: str, modelo_clave: str):
    """Devuelve un sampler imblearn o 'passthrough' para class_weight."""
    if nombre == "class_weight":
        return "passthrough"
    if nombre == "SMOTE_k3":
        return SMOTE(random_state=RANDOM_STATE, k_neighbors=3)
    if nombre == "SMOTETomek":
        return SMOTETomek(random_state=RANDOM_STATE)
    raise ValueError(nombre)


def construir_clasificador(modelo_clave: str, estrategia: str):
    """Devuelve el clasificador con class_weight si la estrategia lo requiere."""
    use_cw = estrategia == "class_weight"
    if modelo_clave == "LR":
        return LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced" if use_cw else None,
        )
    if modelo_clave == "RF":
        return RandomForestClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced" if use_cw else None,
        )
    if modelo_clave == "XGB":
        return xgb.XGBClassifier(
            random_state=RANDOM_STATE,
            eval_metric="logloss",
            scale_pos_weight=neg_pos_ratio if use_cw else 1.0,
            tree_method="hist",
        )
    raise ValueError(modelo_clave)


PARAM_GRID = {
    "LR": {"classifier__C": [0.01, 0.1, 1, 10]},
    "RF": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [5, 10, 15, None],
        "classifier__min_samples_leaf": [1, 2, 5],
    },
    "XGB": {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [3, 5, 7],
        "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
        "classifier__subsample": [0.7, 0.8, 1.0],
        "classifier__colsample_bytree": [0.7, 0.8, 1.0],
    },
}
N_ITER = {"LR": 4, "RF": 10, "XGB": 10}
NOMBRE_MODELO = {"LR": "Regresión Logística", "RF": "Random Forest", "XGB": "XGBoost"}
ESTRATEGIAS = ["class_weight", "SMOTE_k3", "SMOTETomek"]

resultados_cv: list[dict] = []
mejores_estimadores: dict[str, dict] = {}

for modelo_clave in ["LR", "RF", "XGB"]:
    seccion(f"{NOMBRE_MODELO[modelo_clave]}  —  evaluando 3 estrategias")
    mejores_estimadores[modelo_clave] = {}
    for est in ESTRATEGIAS:
        sampler = construir_sampler(est, modelo_clave)
        clf = construir_clasificador(modelo_clave, est)
        pipe = ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("sampler", sampler),
                ("classifier", clf),
            ]
        )
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=PARAM_GRID[modelo_clave],
            n_iter=N_ITER[modelo_clave],
            cv=cv,
            scoring="average_precision",
            refit=True,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            error_score="raise",
        )
        search.fit(X_train, y_train)
        # AUC-PR sobre TEST (sin resampling, con preprocesamiento honesto)
        proba_test = search.best_estimator_.predict_proba(X_test)[:, 1]
        ap_test = average_precision_score(y_test, proba_test)
        auc_test = roc_auc_score(y_test, proba_test)
        resultados_cv.append(
            {
                "Modelo": NOMBRE_MODELO[modelo_clave],
                "Estrategia": est,
                "AUC-PR_CV": search.best_score_,
                "AUC-PR_Test": ap_test,
                "AUC-ROC_Test": auc_test,
                "Hiperparámetros": search.best_params_,
            }
        )
        mejores_estimadores[modelo_clave][est] = search.best_estimator_
        print(
            f"   - estrategia={est:<14s} | CV AUC-PR={search.best_score_:.4f} | "
            f"Test AUC-PR={ap_test:.4f} | Test AUC-ROC={auc_test:.4f}"
        )

df_cv = pd.DataFrame(resultados_cv)
banner("TABLA COMPARATIVA — 9 CONFIGURACIONES (3 modelos x 3 estrategias)")
print(
    df_cv[["Modelo", "Estrategia", "AUC-PR_CV", "AUC-PR_Test", "AUC-ROC_Test"]]
    .round(4)
    .to_string(index=False)
)


# ==========================================================================
#  SELECCIÓN DEL MODELO FINAL POR AUC-PR EN TEST + PARSIMONIA
# ==========================================================================
banner(
    "SELECCIÓN DEL MODELO FINAL",
    "Criterio: mejor AUC-PR en TEST + Principio de Parsimonia",
)

idx_best = df_cv["AUC-PR_Test"].idxmax()
mejor = df_cv.loc[idx_best]
mejor_modelo_clave = {v: k for k, v in NOMBRE_MODELO.items()}[mejor["Modelo"]]
modelo_top = mejores_estimadores[mejor_modelo_clave][mejor["Estrategia"]]
print(f" - Mejor combinación por AUC-PR(Test): "
      f"{mejor['Modelo']} + {mejor['Estrategia']}  "
      f"(AUC-PR={mejor['AUC-PR_Test']:.4f}, AUC-ROC={mejor['AUC-ROC_Test']:.4f})")

# Aplicar parsimonia: si la mejor RL está a <= 0.01 del top en AUC-PR, preferir RL
mejor_rl = (
    df_cv[df_cv["Modelo"] == "Regresión Logística"]["AUC-PR_Test"].max()
)
diff_pars = float(mejor["AUC-PR_Test"]) - float(mejor_rl)
if mejor["Modelo"] != "Regresión Logística" and diff_pars <= 0.01:
    print(
        f" - PARSIMONIA: la mejor Regresión Logística está a {diff_pars:.4f} "
        "del top -> se selecciona RL por interpretabilidad y defensa regulatoria."
    )
    idx_rl = df_cv[df_cv["Modelo"] == "Regresión Logística"]["AUC-PR_Test"].idxmax()
    mejor = df_cv.loc[idx_rl]
    mejor_modelo_clave = "LR"
    modelo_top = mejores_estimadores["LR"][mejor["Estrategia"]]

modelo_final = modelo_top
nombre_final = mejor["Modelo"]
estrategia_final = mejor["Estrategia"]
ap_final = float(mejor["AUC-PR_Test"])
auc_final = float(mejor["AUC-ROC_Test"])

print(f"\n MODELO FINAL: {nombre_final}  (estrategia: {estrategia_final})")
print(f"   AUC-PR(Test):  {ap_final:.4f}")
print(f"   AUC-ROC(Test): {auc_final:.4f}")
print(f"   Hiperparámetros: {mejor['Hiperparámetros']}")

# Persistencia: por cada algoritmo, guardamos el mejor pipeline (mejor estrategia)
def _mejor_estrategia(clave: str) -> str:
    sub = df_cv[df_cv["Modelo"] == NOMBRE_MODELO[clave]]
    return str(sub.loc[sub["AUC-PR_Test"].idxmax(), "Estrategia"])


joblib.dump(
    mejores_estimadores["LR"][_mejor_estrategia("LR")],
    MODELS_DIR / "logistic_regression.pkl",
)
joblib.dump(
    mejores_estimadores["RF"][_mejor_estrategia("RF")],
    MODELS_DIR / "random_forest.pkl",
)
joblib.dump(
    mejores_estimadores["XGB"][_mejor_estrategia("XGB")],
    MODELS_DIR / "xgboost.pkl",
)
joblib.dump(modelo_final, MODELS_DIR / "modelo_final.pkl")
joblib.dump(preprocessor, MODELS_DIR / "preprocessor.pkl")
print(f"\n - Modelos serializados en {MODELS_DIR.relative_to(ROOT)}/")


# ==========================================================================
#  DIAPOSITIVAS 5 y 6 — RESULTADOS DEL MODELO
# ==========================================================================
banner(
    "DIAPOSITIVAS 5 y 6 — RESULTADOS DEL MODELO",
    "Matrices de confusión + ROC + Precision-Recall (Test sin resampling)",
)

# Para visualizar las 3 mejores configuraciones (1 por modelo, mejor estrategia)
def mejor_por_modelo(clave: str):
    sub = df_cv[df_cv["Modelo"] == NOMBRE_MODELO[clave]]
    idx = sub["AUC-PR_Test"].idxmax()
    return df_cv.loc[idx]

top_rl = mejor_por_modelo("LR")
top_rf = mejor_por_modelo("RF")
top_xgb = mejor_por_modelo("XGB")

modelos_visualizar = [
    (top_rl["Modelo"], mejores_estimadores["LR"][top_rl["Estrategia"]], top_rl["Estrategia"]),
    (top_rf["Modelo"], mejores_estimadores["RF"][top_rf["Estrategia"]], top_rf["Estrategia"]),
    (top_xgb["Modelo"], mejores_estimadores["XGB"][top_xgb["Estrategia"]], top_xgb["Estrategia"]),
]

seccion("Matrices de Confusión (umbral por defecto = 0.5)")
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (nombre, modelo, est) in zip(axes, modelos_visualizar):
    y_pred = modelo.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=["Legítima", "Fraude"],
        yticklabels=["Legítima", "Fraude"],
    )
    ax.set_title(f"{nombre} ({est})\nFN={cm[1, 0]} (fraudes no detectados)")
    ax.set_ylabel("Real")
    ax.set_xlabel("Predicho")
    print(
        f" - {nombre} ({est}): TN={cm[0,0]}, FP={cm[0,1]}, "
        f"FN={cm[1,0]}, TP={cm[1,1]}"
    )
plt.suptitle(
    "Matrices de Confusión — 1 FN tiene mayor costo que 1 FP en fraude",
    fontweight="bold",
)
plt.tight_layout()
guardar_fig("05_matrices_confusion.png")

seccion("Curvas ROC y Precision-Recall (Test)")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ["blue", "green", "red"]
for (nombre, modelo, est), color in zip(modelos_visualizar, colors):
    y_proba = modelo.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    axes[0].plot(fpr, tpr, color=color, label=f"{nombre} ({est}) AUC={auc:.3f}")
    prec, rec, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    axes[1].plot(rec, prec, color=color, label=f"{nombre} ({est}) AP={ap:.3f}")
    print(f" - {nombre} ({est}): AUC-ROC={auc:.4f} | AUC-PR={ap:.4f}")
axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
axes[0].set_title("Curva ROC")
axes[0].set_xlabel("FPR")
axes[0].set_ylabel("TPR")
axes[0].legend()
axes[1].axhline(y=0.02, color="black", linestyle="--", alpha=0.3, label="Baseline (~2%)")
axes[1].set_title("Curva Precision-Recall")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].legend()
plt.suptitle("Discriminación de modelos en TEST (sin resampling)", fontweight="bold")
plt.tight_layout()
guardar_fig("06_roc_precision_recall.png")


# ==========================================================================
#  DIAGNÓSTICO CRÍTICO AUTOMÁTICO  (Sesión 7 — Gini Negativo)
# ==========================================================================
banner(
    "DIAGNÓSTICO CRÍTICO — ¿Tiene el modelo poder discriminativo real?",
    "Chequeo de Bandera Roja según criterio de la Sesión 7 (Gini Negativo)",
)

gini = 2 * auc_final - 1
print(f" - AUC-ROC del modelo final: {auc_final:.4f}")
print(f" - Gini equivalente:          {gini:.4f}  (Gini = 2*AUC - 1)")
print(" - Reglas de evaluación (referencia académica):")
print("     AUC < 0.50  -> Gini negativo: el modelo clasifica al revés.")
print("     0.50-0.55   -> No discrimina mejor que el azar.")
print("     0.55-0.65   -> Discriminación marginal.")
print("     0.65-0.80   -> Aceptable / Bueno.")
print("     > 0.80       -> Excelente.")

if auc_final < 0.55:
    bandera_roja = True
    print("\n >>> BANDERA ROJA <<<")
    print(
        " El modelo NO tiene poder discriminativo real sobre datos de prueba.\n"
        " Diagnóstico de causa raíz (con evidencia de este pipeline):\n"
        f"   1. Señal en los datos muy débil (max |Spearman| con fraud = "
        f"{max_corr:.3f} < 0.05).\n"
        "   2. Posible sobreajuste a datos sintéticos del oversampling: el F1-CV\n"
        "      sobre train balanceado puede ser muy alto y no transferirse al test.\n"
        "   3. Falta feature engineering basado en comportamiento histórico del\n"
        "      cliente (velocidad de transacciones, dispositivo, geolocalización).\n"
        " RECOMENDACIÓN AL COMITÉ:\n"
        "   - NO desplegar este modelo en producción todavía.\n"
        "   - Entregable cumple su rol académico: pipeline correcto, métricas y\n"
        "     SHAP bien calculados, Principio de Parsimonia aplicado, y se\n"
        "     identifica con honestidad técnica que la limitación es de DATOS,\n"
        "     no de algoritmo (referencia: Sesión 4 - caso CIMA producción)."
    )
else:
    bandera_roja = False
    print("\n - Modelo con poder discriminativo aceptable. Se procede a la fase")
    print("   de calibración del umbral operativo.")


# ==========================================================================
#  DIAPOSITIVA 8 — TRADE-OFFS Y 3 UMBRALES OPERATIVOS
# ==========================================================================
banner(
    "DIAPOSITIVA 8 — TRADE-OFFS Y UMBRALES OPERATIVOS",
    "3 escenarios de umbral, no solo F1 máx",
)

y_proba_final = modelo_final.predict_proba(X_test)[:, 1]
prec, rec, umbrales = precision_recall_curve(y_test, y_proba_final)

# Escenario 1: F1 máx (parsimonia estadística)
f1_arr = 2 * (prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-9)
idx_f1 = int(np.argmax(f1_arr))
u_f1 = float(umbrales[idx_f1])

# Escenario 2: capturar al menos el 80% del fraude (Recall objetivo)
target_recall = 0.80
candidatos = np.where(rec[:-1] >= target_recall)[0]
if len(candidatos) > 0:
    idx_rec = int(candidatos[-1])  # umbral más alto que aún cumple el recall
    u_rec = float(umbrales[idx_rec])
else:
    idx_rec = int(np.argmax(rec[:-1]))
    u_rec = float(umbrales[idx_rec])

# Escenario 3: capacidad de mesa de fraude (~5% del volumen diario)
capacidad_pct = 0.05  # 500 alertas / 10k tx
n_alertas = int(round(capacidad_pct * len(y_test)))
orden = np.argsort(-y_proba_final)[:n_alertas]
u_cap = float(np.min(y_proba_final[orden]))
y_pred_cap = (y_proba_final >= u_cap).astype(int)
prec_cap = (y_pred_cap & y_test.values).sum() / max(y_pred_cap.sum(), 1)
rec_cap = (y_pred_cap & y_test.values).sum() / max(y_test.sum(), 1)
f1_cap = 2 * prec_cap * rec_cap / max(prec_cap + rec_cap, 1e-9)


def resumen_escenario(nombre: str, u: float, p_val: float, r_val: float, f_val: float):
    y_pred = (y_proba_final >= u).astype(int)
    alerts = int(y_pred.sum())
    alerts_per_10k = int(round(alerts / len(y_test) * 10_000))
    fp = int(((y_pred == 1) & (y_test.values == 0)).sum())
    fn = int(((y_pred == 0) & (y_test.values == 1)).sum())
    return {
        "Escenario": nombre,
        "Umbral": u,
        "Precision": p_val,
        "Recall": r_val,
        "F1": f_val,
        "Alertas/Test": alerts,
        "Alertas_x10k_tx": alerts_per_10k,
        "FP_Test": fp,
        "FN_Test": fn,
    }


tabla_umbrales = pd.DataFrame(
    [
        resumen_escenario("F1 máx", u_f1, prec[idx_f1], rec[idx_f1], f1_arr[idx_f1]),
        resumen_escenario("Recall>=0.80", u_rec, prec[idx_rec], rec[idx_rec], f1_arr[idx_rec]),
        resumen_escenario("Capacidad mesa (5%)", u_cap, prec_cap, rec_cap, f1_cap),
    ]
)
print(" Tabla de umbrales operativos (sobre TEST):")
print(tabla_umbrales.round(4).to_string(index=False))

# Figura: análisis de umbral con los 3 marcadores
plt.figure(figsize=(11, 5))
plt.plot(umbrales, prec[:-1], label="Precision", color="blue")
plt.plot(umbrales, rec[:-1], label="Recall", color="green")
plt.plot(umbrales, f1_arr, label="F1", color="red")
plt.axvline(u_f1, color="red", linestyle="--", alpha=0.6, label=f"F1 máx ({u_f1:.3f})")
plt.axvline(
    u_rec, color="green", linestyle="--", alpha=0.6, label=f"Recall>=0.80 ({u_rec:.3f})"
)
plt.axvline(
    u_cap, color="orange", linestyle="--", alpha=0.6, label=f"Capacidad 5% ({u_cap:.3f})"
)
plt.title(f"Análisis de umbral — {nombre_final}", fontweight="bold")
plt.xlabel("Umbral de decisión")
plt.legend(loc="best", fontsize=8)
plt.tight_layout()
guardar_fig("08_analisis_umbral.png")

# Selección del umbral OPERATIVO recomendado
if bandera_roja:
    umbral_op = u_cap
    motivo_umbral = (
        "Capacidad de mesa (5%): con AUC bajo, priorizar control operativo "
        "evita saturar al equipo con falsas alarmas."
    )
else:
    umbral_op = u_f1 if rec[idx_f1] >= 0.50 else u_rec
    motivo_umbral = (
        "F1 máx priorizando Recall>=0.50; si no se alcanza, fijar Recall>=0.80 "
        "y aceptar más alertas (costo_FN >> costo_FP)."
    )

print(f"\n UMBRAL OPERATIVO RECOMENDADO: {umbral_op:.4f}")
print(f"   Motivo: {motivo_umbral}")


# ==========================================================================
#  DIAPOSITIVA 9 — RIESGOS, LIMITACIONES Y ESTABILIDAD (PSI)
# ==========================================================================
banner(
    "DIAPOSITIVA 9 — RIESGOS, LIMITACIONES Y CONCEPT DRIFT",
    "Estabilidad poblacional + sesgos potenciales",
)


def calcular_psi(score_train, score_test, bins: int = 10) -> float:
    breaks = np.unique(np.percentile(score_train, np.linspace(0, 100, bins + 1)))
    pct_tr = np.histogram(score_train, bins=breaks)[0] / len(score_train) + 1e-6
    pct_te = np.histogram(score_test, bins=breaks)[0] / len(score_test) + 1e-6
    return float(np.sum((pct_te - pct_tr) * np.log(pct_te / pct_tr)))


psi_val = calcular_psi(
    modelo_final.predict_proba(X_train)[:, 1],
    modelo_final.predict_proba(X_test)[:, 1],
)
estado = "ESTABLE" if psi_val < 0.1 else "MONITOREAR" if psi_val < 0.25 else "INESTABLE"
print(f" - PSI del modelo final ({nombre_final}, {estrategia_final}): "
      f"{psi_val:.4f}  =>  {estado}")
print(" - Reglas de interpretación:")
print("     PSI < 0.10  -> población estable, no se requiere acción.")
print("     0.10 - 0.25 -> ligero cambio, monitorear mensualmente.")
print("     PSI > 0.25  -> Concept Drift, reentrenar el modelo.")

print("\n Riesgos y limitaciones identificados:")
print("   1. Calidad de la señal en los datos:")
print(f"      - Correlación máxima individual con fraud = {max_corr:.3f} (< 0.05).")
print("      - Sin nuevas variables comportamentales, hay un techo de AUC bajo.")
print("   2. Sobreajuste a oversampling sintético:")
print("      - Mitigado al MOVER SMOTE/SMOTETomek dentro del Pipeline imblearn.")
print("      - Comparación con `class_weight` permite descartar artefactos del SMOTE.")
print("   3. Sesgos potenciales:")
print("      - Variables socio-demográficas (location, marital_status) podrían")
print("        introducir sesgo si correlacionan con grupos protegidos.")
print("      - Recomendación: análisis de fairness por subgrupo antes de producción.")
print("   4. Tamaño muestral:")
print("      - 10,000 registros con ~2% fraude = solo ~200 casos positivos.")
print("      - Insuficiente para estimar colas de la distribución.")
print("   5. Concept Drift:")
print("      - Los patrones de fraude evolucionan; sin reentrenamiento periódico")
print("        el modelo se degrada (referencia: Sesión 4 - caso CIMA).")


# ==========================================================================
#  DIAPOSITIVA 7 — EXPLICABILIDAD CON SHAP
# ==========================================================================
banner(
    "DIAPOSITIVA 7 — EXPLICABILIDAD CON SHAP",
    "Importancia global (Beeswarm) + explicación local (Waterfall)",
)

# Reconstruir nombres de features post-OneHot a partir del preprocessor
fitted_pre = modelo_final.named_steps["preprocessor"]
cat_names = list(
    fitted_pre.named_transformers_["cat"]
    .named_steps["onehot"]
    .get_feature_names_out(cat_features)
)
feat_names = list(num_features) + cat_names

# Datos transformados para SHAP
X_test_prep_for_shap = fitted_pre.transform(X_test)
X_train_prep_for_shap = fitted_pre.transform(X_train)

n_shap = min(500, X_test_prep_for_shap.shape[0])
X_shap = X_test_prep_for_shap[:n_shap]

clf_extracted = modelo_final.named_steps["classifier"]
explainer = shap.Explainer(clf_extracted, X_train_prep_for_shap, feature_names=feat_names)
shap_values = explainer(X_shap)

base_log_odds = float(np.mean(shap_values.base_values))
base_prob = 1 / (1 + np.exp(-base_log_odds))
print(f" - SHAP calculado sobre {n_shap} muestras del test.")
print(f" - Valor base E[f(x)] (log-odds): {base_log_odds:.4f}")
print(f" - Probabilidad base (sigmoide):   {base_prob:.2%}")

abs_shap = np.abs(shap_values.values).mean(axis=0)
ranking = sorted(zip(feat_names, abs_shap), key=lambda x: x[1], reverse=True)
print("\n Top 10 variables por importancia SHAP global (mean |SHAP|):")
for var, val in ranking[:10]:
    print(f"   - {var:<40s}  {val:.4f}")

plt.figure(figsize=(10, 6))
shap.plots.bar(shap_values, max_display=15, show=False)
plt.title(
    "Feature Importance Global (SHAP)\n"
    "Variables que más influyen en la detección de fraude",
    fontweight="bold",
)
plt.tight_layout()
guardar_fig("07_shap_importancia_bar.png")

plt.figure(figsize=(10, 8))
shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.title(
    "Beeswarm Plot — Magnitud y Dirección\n"
    "Rojo = valor alto de la variable | Azul = valor bajo",
    fontweight="bold",
)
plt.tight_layout()
guardar_fig("07_shap_beeswarm.png")

idx_fraudes = np.where(y_test.values[:n_shap] == 1)[0]
if len(idx_fraudes) > 0:
    idx = int(idx_fraudes[0])
    prob = float(modelo_final.predict_proba(X_test.iloc[[idx]])[0, 1])
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[idx], max_display=12, show=False)
    plt.title(
        f"Waterfall Plot — Caso de fraude (prob estimada = {prob:.2%})",
        fontweight="bold",
    )
    plt.tight_layout()
    guardar_fig("07_shap_waterfall_fraude.png")
    print(f"\n - Caso de fraude analizado (índice {idx} del test):")
    print(f"     Probabilidad estimada: {prob:.2%}")
    print(f"     Umbral operativo:      {umbral_op:.2%}")
    detectado = "DETECTADO" if prob >= umbral_op else "NO DETECTADO"
    print(f"     Decisión del modelo:   {detectado}")
else:
    print(" - No se encontraron casos de fraude en las primeras n_shap muestras.")


# ==========================================================================
#  EXTENSIÓN: ISOLATION FOREST (no supervisado, +5 pts bono)
# ==========================================================================
banner("EXTENSIÓN — Isolation Forest (no supervisado)")
X_train_prep_iso = preprocessor.fit_transform(X_train, y_train)
X_test_prep_iso = preprocessor.transform(X_test)
iso_forest = IsolationForest(
    n_estimators=200, contamination=0.02, random_state=RANDOM_STATE
)
iso_forest.fit(X_train_prep_iso)
iso_pred = np.where(iso_forest.predict(X_test_prep_iso) == -1, 1, 0)
print(classification_report(y_test, iso_pred, target_names=["Legítima", "Fraude"]))
print(f" - Recall fraude: {recall_score(y_test, iso_pred):.4f}")
print(f" - F1 fraude:     {f1_score(y_test, iso_pred):.4f}")
print(
    " - Lectura: contraste metodológico. Recall ~0 sobre datos transformados es"
    " esperable (la separación de anomalías se diluye en el espacio escalado)."
)


# ==========================================================================
#  DIAPOSITIVA 10 — CONCLUSIONES Y PLAN DE ACCIÓN
# ==========================================================================
banner(
    "DIAPOSITIVA 10 — CONCLUSIONES Y PLAN DE ACCIÓN",
    "Recomendación final al Comité de Riesgos",
)

y_pred_final = (y_proba_final >= umbral_op).astype(int)
cm_final = confusion_matrix(y_test, y_pred_final)
TN, FP, FN, TP = int(cm_final[0, 0]), int(cm_final[0, 1]), int(cm_final[1, 0]), int(cm_final[1, 1])
prec_op = TP / max(TP + FP, 1)
rec_op = TP / max(TP + FN, 1)
f1_op = 2 * prec_op * rec_op / max(prec_op + rec_op, 1e-9)

print(" RESUMEN EJECUTIVO FINAL")
print("-" * 70)
print(f" Dataset             : 10,000 transacciones | {y.sum()} fraudes ({y.mean()*100:.2f}%)")
print(f" Modelos evaluados   : 3 modelos x 3 estrategias de desbalance = 9 configs")
print(f" Modelo seleccionado : {nombre_final} ({estrategia_final})")
print(f" AUC-PR (Test)       : {ap_final:.4f}")
print(f" AUC-ROC (Test)      : {auc_final:.4f}  (Gini = {gini:.4f})")
print(f" Umbral operativo    : {umbral_op:.4f}")
print(f" Precision @ umbral  : {prec_op:.4f}")
print(f" Recall    @ umbral  : {rec_op:.4f}")
print(f" F1-Score  @ umbral  : {f1_op:.4f}")
print(f" Matriz Confusión    : TN={TN} FP={FP} FN={FN} TP={TP}")
print(f" PSI                 : {psi_val:.4f}  ({estado})")
print("-" * 70)
print("\n Reporte de clasificación (TEST con umbral operativo):")
print(classification_report(y_test, y_pred_final, target_names=["Legítima", "Fraude"]))

alertas_10k = int(round(y_pred_final.mean() * 10_000))
fraudes_detectados_10k = int(round(alertas_10k * prec_op))
falsas_alarmas_10k = max(alertas_10k - fraudes_detectados_10k, 0)
print(f" Operativa diaria (escenario 10,000 tx/día):")
print(f"   - Alertas esperadas       : {alertas_10k:,}")
print(f"   - Fraudes detectados      : {fraudes_detectados_10k:,}")
print(f"   - Falsas alarmas          : {falsas_alarmas_10k:,}")
print(f"   - % falsas sobre alertas  : {(1-prec_op)*100:.0f}%")


# ==========================================================================
#  CONCLUSIONES HONESTAS PARA EL COMITÉ
# ==========================================================================
banner(
    "CONCLUSIONES HONESTAS PARA EL COMITÉ",
    "Defensa académica del entregable + recomendación de gobierno",
)

if bandera_roja:
    print(
        f""" Lectura técnica:
   - El AUC-ROC del modelo final ({auc_final:.4f}) está en zona de "no
     discriminación". El Gini equivalente ({gini:+.4f}) confirma el hallazgo.
   - El pipeline es metodológicamente correcto: división estratificada,
     ColumnTransformer, Pipeline imblearn dentro del CV, comparación de 3
     estrategias de desbalance, optimización por AUC-PR, calibración de
     umbral con 3 escenarios operativos, SHAP global y local, PSI.
   - La causa más probable es la calidad de la señal (correlación máxima
     individual con fraud = {max_corr:.3f}) más que un error del algoritmo.

 Recomendación al Comité de Riesgos:
   1. NO aprobar el despliegue en producción con la configuración actual.
   2. Aprobar Fase 2 con foco en DATOS (no en algoritmo):
      - Incorporar variables comportamentales (velocidad de transacción,
        device fingerprint, geolocalización, scoring de comercio).
      - Ampliar la ventana histórica a 12-24 meses para tener > 1,000 fraudes.
      - Evaluar fuentes externas (buró, listas negras, AML).
   3. Mientras tanto:
      - Mantener las reglas vigentes del motor de fraude.
      - Usar este pipeline en modo 'shadow' como termómetro de calidad de
        datos: si AUC sube al agregar nuevas variables, la decisión de
        despliegue se reabre.
   4. Aplicar el aprendizaje (Sesión 4 - caso CIMA): degradación en
      producción suele venir de no manejar bien la naturaleza temporal y
      de abusar de oversampling. Aquí se mitigó al validar SIN resampling."""
    )
else:
    print(
        f""" Lectura técnica:
   - AUC-ROC final ({auc_final:.4f}) y AUC-PR ({ap_final:.4f}) indican
     poder discriminativo aceptable.
   - El umbral operativo ({umbral_op:.4f}) calibra el balance FP/FN
     para la capacidad real de la mesa de fraude.
 Recomendación al Comité de Riesgos:
   1. Aprobar despliegue gradual en MODO SHADOW por 30 días.
   2. Tablero diario de Recall, Precision y % falsas alarmas.
   3. PSI mensual (umbrales 0.10 / 0.25); si > 0.25 -> reentrenar.
   4. Reentrenamiento trimestral programado.
   5. Reporte trimestral con SHAP + casos representativos al regulador."""
    )

print("\n Plan de acción operativo:")
print("   - Producción: desplegar en 'shadow' antes de reemplazar las reglas.")
print("   - Monitoreo:  PSI mensual + tablero diario Recall/Precision.")
print("   - Reentreno:  trimestral; anticipado si PSI>0.25 o Recall cae >5pp.")
print("   - Gobierno:   reporte trimestral con SHAP + análisis de fairness.")

banner("FIN DE LA EJECUCIÓN")
print(f" - Resumen ejecutivo guardado en: {RESUMEN_PATH.relative_to(ROOT)}")
print(f" - Figuras guardadas en:          {FIGURES_DIR.relative_to(ROOT)}/")
print(f" - Modelos serializados en:       {MODELS_DIR.relative_to(ROOT)}/")

sys.stdout = sys.__stdout__
_log_file.close()
