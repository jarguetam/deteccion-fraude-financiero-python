# Spec: Modelo de Detección de Fraude Financiero

**Fecha:** 2026-05-03
**Proyecto:** Caso Integrador Final — Programa de Especialización en Credit Scoring con Python
**Entregable principal:** `notebooks/fraude_deteccion.ipynb`
**Dataset:** `data/base.csv` (10,000 registros, 17 variables, ~2% fraude)

---

## Contexto

Una institución financiera requiere un sistema de ML para identificar transacciones anómalas. El modelo debe ser presentado ante un Comité de Riesgos. La rúbrica evalúa notebook técnico (30%) y presentación ejecutiva (70%), pero el notebook es el primer entregable.

La variable objetivo `fraud` es binaria (0 = legítima, 1 = fraude) con tasa de incidencia ~2% — desbalance crítico que debe ser abordado explícitamente.

---

## Decisiones de diseño

### Estructura
Un único notebook (`notebooks/fraude_deteccion.ipynb`) con 6 módulos secuenciales. Cada módulo inicia con una celda Markdown que documenta el razonamiento técnico de las decisiones (vale 10% de la nota).

### Semilla aleatoria
`random_state = 42` fijo en todos los procesos estocásticos (split, SMOTE, modelos, CV).

### Anti data leakage
El split train/test se realiza **antes** de cualquier transformación. Los transformadores (imputer, scaler, Winsorización) se ajustan con `.fit_transform()` sobre train y se aplican con `.transform()` sobre test.

### Principio de parsimonia
Se incluye Regresión Logística como baseline. Si sus métricas son similares a RF o XGBoost, se selecciona RL como modelo final por su mayor interpretabilidad ante reguladores.

---

## Módulos del notebook

### Módulo 0 — Configuración
- Importación de librerías
- Constante `RANDOM_STATE = 42`
- Carga de `data/base.csv`

### Módulo 1 — EDA

| Análisis | Técnica |
|---|---|
| Estadísticos descriptivos | `describe()`, percentiles |
| Distribución variable objetivo | `value_counts()`, gráfico de barras |
| Análisis univariado | Histogramas + boxplots |
| Outliers | IQR + boxplots |
| Correlaciones | Spearman (no Pearson — más robusto ante outliers y no linealidades) |
| Comportamiento diferenciado | Distribuciones separadas `fraud=0` vs `fraud=1` |

**Nota:** Los modelos de árbol (RF, XGBoost) son robustos ante outliers; la RL no. Esto informará la decisión de Winsorización.

### Módulo 2 — Preprocesamiento

Orden estricto:
1. Feature engineering de `timestamp` → `hora`, `dia_semana`, `mes`
2. **División train/test** (`stratify=y`, 80/20, `random_state=42`)
3. Verificación y manejo de nulos con `SimpleImputer` (mediana numérica, moda categórica)
4. Winsorización al p99 sobre train, mismo límite aplicado a test
5. Encoding:
   - One-Hot (`drop='first'`): `transaction_type`, `location`, `marital_status`, `housing_type`
   - Ordinal existente: `education_level` (ya codificada 1-4, se usa tal cual)
6. `StandardScaler` solo para la pipeline de Regresión Logística; RF y XGBoost no lo requieren

### Módulo 3 — Modelado

**Estrategia de desbalance:** SMOTE aplicado únicamente sobre `X_train` / `y_train` después del split.

```python
X_train_res, y_train_res = SMOTE(random_state=42).fit_resample(X_train, y_train)
```

**Modelos entrenados:**

| Modelo | Escalado | Hiperparámetros clave |
|---|---|---|
| Regresión Logística (baseline) | StandardScaler | `C`, `max_iter` |
| Random Forest | No | `n_estimators`, `max_depth`, `min_samples_leaf` |
| XGBoost | No | `n_estimators`, `max_depth`, `learning_rate`, `scale_pos_weight` |

**Validación:** `RandomizedSearchCV` con `StratifiedKFold(n_splits=5)`, optimizando por F1 sobre clase positiva.

**Bono (+5 pts):** Isolation Forest entrenado sin `fraud` — comparar anomalías predichas vs valores reales.

### Módulo 4 — Evaluación y selección

**Métricas reportadas:**
- Recall (clase 1) — prioridad: detectar el máximo de fraudes
- F1-Score (clase 1)
- AUC-ROC
- AUC-PR (más informativa que ROC en clases desbalanceadas)
- PSI (Population Stability Index: Train vs Test)

**Visualizaciones:**
- Matriz de confusión para cada modelo
- Curvas ROC y Precision-Recall superpuestas
- Umbral óptimo identificado sobre curva Precision-Recall (no 0.5)

**Tabla comparativa** de los tres modelos. Aplicación del **principio de parsimonia**: si RL obtiene métricas similares, se selecciona como modelo final con justificación documentada.

### Módulo 5 — Explicabilidad (SHAP)

Aplicado sobre el modelo ganador:

| Gráfico | Alcance | Propósito |
|---|---|---|
| Feature Importance (barras) | Global | Variables más influyentes |
| Beeswarm plot | Global | Magnitud y dirección de cada variable |
| Waterfall plot | Local (1 caso fraude) | Explicación individual para el Comité |

### Módulo 6 — Conclusiones

- Resumen de hallazgos EDA
- Modelo seleccionado con justificación
- Umbral operativo recomendado e impacto en negocio
- Alerta de concept drift: usar PSI en producción para detectar degradación del modelo

---

## Estructura de archivos

```
deteccion-fraude-financiero-python/
├── data/
│   └── base.csv
├── notebooks/
│   └── fraude_deteccion.ipynb
├── models/
│   └── (modelos serializados con joblib)
└── docs/
    └── superpowers/specs/
        └── 2026-05-03-fraude-deteccion-design.md
```

---

## Criterios de calidad (rúbrica)

| Criterio | Acción en el notebook |
|---|---|
| Sin data leakage | Split primero, transformar después |
| Reproducibilidad | `random_state=42` en todo |
| Documentación | Celdas Markdown con el "porqué" de cada decisión |
| Métricas correctas | Recall y F1 sobre clase positiva, no Accuracy |
| Umbral óptimo | Analizar curva PR, no usar 0.5 |
| Parsimonia | Comparar RL vs RF vs XGBoost, elegir el más simple competitivo |
| SHAP | Beeswarm global + Waterfall local |
| Bono | Isolation Forest (+5 pts en interpretabilidad) |
