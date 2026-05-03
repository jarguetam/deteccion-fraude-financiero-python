# Detección de Fraude Financiero con Machine Learning

Proyecto insignia de portafolio orientado a **riesgo financiero**: construye, evalúa y explica un pipeline de detección de fraude con enfoque en métricas de negocio, control de data leakage y decisión operativa por umbrales.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Boosting-green)
![SHAP](https://img.shields.io/badge/Explainability-SHAP-purple)
![Status](https://img.shields.io/badge/Status-Academic%20Showcase-success)

## Resumen Ejecutivo

- **Problema**: detectar transacciones fraudulentas en un dataset altamente desbalanceado (~2% fraude).
- **Enfoque**: evaluación rigurosa con `Pipeline` de `imblearn`, validación cruzada estratificada y comparación de estrategias de desbalance.
- **Valor de negocio**: prioriza decisiones para Comité de Riesgos (trade-off entre falsos negativos y falsas alarmas).
- **Resultado clave**: el pipeline quedó metodológicamente robusto y explicable, pero la señal del dataset es débil para despliegue productivo inmediato.

## Stack Tecnológico

- Python 3.13
- pandas, numpy
- scikit-learn
- imbalanced-learn
- xgboost
- shap
- matplotlib, seaborn
- jupyter notebook

## Estructura del Proyecto

```text
data/                  Dataset base
models/                Modelos serializados (.pkl)
notebooks/             Notebook principal del caso
outputs/
  figures/             Visualizaciones para la presentación ejecutiva
  resumen_ejecutivo.txt Salida consolidada en consola
scritps/
  modelo_fraude.py     Script ejecutable end-to-end
requirements.txt
README.md
```

## Metodología (versión corregida)

El proyecto incluye una segunda iteración de mejora metodológica para eliminar sesgos de evaluación:

1. **Split estratificado train/test** antes de transformaciones.
2. **Preprocesamiento** con `ColumnTransformer` (imputación + escalado + one-hot).
3. **Control de leakage**: `SMOTE` y `SMOTETomek` se aplican **dentro** de `ImbPipeline` durante CV.
4. **Comparativa 3x3**:
   - Modelos: Regresión Logística, Random Forest, XGBoost.
   - Estrategias de desbalance: `class_weight`, `SMOTE_k3`, `SMOTETomek`.
5. **Selección por AUC-PR en test** + criterio de parsimonia.
6. **Umbral operativo** con 3 escenarios:
   - F1 máximo
   - Recall objetivo (>= 0.80)
   - Capacidad operativa (5% de alertas)
7. **Explicabilidad y estabilidad**:
   - SHAP global (bar + beeswarm) y local (waterfall).
   - PSI para monitoreo de drift.

## Resultados Relevantes

Resultados del último run validado:

- Mejor combinación por AUC-PR en test: **Random Forest + class_weight**.
- Aplicando parsimonia: **Regresión Logística + SMOTETomek** como modelo final defendible.
- Métricas del modelo final:
  - AUC-PR (test): `0.0242`
  - AUC-ROC (test): `0.4487`
  - Gini: `-0.1026`
  - PSI: `0.0133` (estable)

### Lectura técnica honesta

- El pipeline es correcto y auditable.
- El bajo AUC indica **poca señal predictiva en los datos actuales**, no solo problema de algoritmo.
- Recomendación: usar en modo shadow y abrir fase de mejora de datos antes de producción.

## Visuales Generadas

El proyecto genera automáticamente figuras listas para presentación:

- `outputs/figures/03_desbalance_clases.png`
- `outputs/figures/05_matrices_confusion.png`
- `outputs/figures/06_roc_precision_recall.png`
- `outputs/figures/07_shap_beeswarm.png`
- `outputs/figures/07_shap_waterfall_fraude.png`
- `outputs/figures/08_analisis_umbral.png`

Y un consolidado en texto:

- `outputs/resumen_ejecutivo.txt`

## Ejecución

### 1) Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2) Ejecutar pipeline completo

```bash
python scritps/modelo_fraude.py
```

### 3) Ejecutar notebook

Abrir y correr:

```text
notebooks/fraude_deteccion.ipynb
```

## Qué hace este proyecto especial

- Traduce ML técnico a lenguaje de negocio para Comité de Riesgos.
- No maquilla métricas: muestra trade-offs y limitaciones reales.
- Incluye explicabilidad (SHAP) y monitoreo (PSI) desde el diseño.
- Documenta hallazgos críticos (AUC/Gini) con criterio académico y de producción.

## Roadmap (Fase 2)

Para llevarlo a nivel productivo:

- Ingeniería de variables comportamentales (velocidad transaccional, dispositivo, geo).
- Mayor ventana histórica y más eventos positivos.
- Validación temporal estricta (time split / backtesting).
- Calibración de probabilidad y función de costo explícita (FN >> FP).
- Monitoreo continuo con alertas de drift y fairness por subgrupos.

## Autor

**Josué Argueta**  
Proyecto integrador en Credit Scoring y Riesgo con Python.

