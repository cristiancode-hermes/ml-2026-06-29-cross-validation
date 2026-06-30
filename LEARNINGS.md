# Learnings — 2026-06-29

## Qué se hizo
Proyecto de Validación Cruzada y Métricas Avanzadas usando un dataset simulado de Telecom Customer Churn (4,000 clientes, ~20% churn). Se cubrieron 9 pasos desde el análisis de desbalanceo hasta GridSearchCV con validación cruzada. Se compararon 4 modelos (Logistic Regression, KNN, Decision Tree, Random Forest) y se optimizaron hiperparámetros con 5-fold Stratified CV.

## Conceptos nuevos

- [x] K-Fold Cross Validation: dividir en K partes, entrenar K veces, promediar resultados
- [x] Stratified K-Fold: mantener proporción de clases en cada fold (estándar para clasificación)
- [x] Diferencia entre K-Fold y Stratified K-Fold
- [x] cross_val_score: evaluar modelos con CV en una línea
- [x] Accuracy vs Precision vs Recall vs F1-Score vs ROC-AUC
- [x] Matriz de Confusión: VP, FP, FN, VN
- [x] Curva ROC y Área Bajo la Curva (AUC)
- [x] Curva Precision-Recall (mejor que ROC para datos desbalanceados)
- [x] Learning Curves: diagnosticar sesgo vs varianza
- [x] GridSearchCV con validación cruzada
- [ ] *Pendiente: Pipelines (paquete completo)*
- [ ] *Pendiente: Clustering (K-Means)*

## Dificultades encontradas

1. **Sintaxis de f-strings con comillas anidadas**: En Python 3.13, `f"{'text':>8}"` funciona bien, pero hay que tener cuidado con el anidamiento de `'` dentro de `"` y viceversa.
2. **Destructuring for loop con enumerate + split**: El patrón `for i, (train_idx, val_idx) in enumerate(kf.split(...))` requiere paréntesis exactos — un paréntesis extra causa SyntaxError.
3. **Sobreajuste de learning curves**: Los scores de entrenamiento son mucho más altos que los de validación, indicando que Random Forest con parámetros default sobreajusta en este dataset.
4. **GridSearch lento con 27 combinaciones**: 27 × 5 folds = 135 entrenamientos. Con n_jobs=-1 se acelera pero igual toma tiempo.

## Notas para el próximo proyecto

- El próximo skill es **Pipelines**: crear un proyecto que encadene preprocessing + modelo en un solo Pipeline de scikit-learn
- Asegurarse de cubrir ColumnTransformer para mezclar datos numéricos y categóricos
- Incluir validación cruzada DENTRO del Pipeline (no hacer preprocessing fuera)
- Dataset ideal: uno con features mixtos (numéricos + categóricos), tal vez el dataset de Titanic o uno de viviendas
