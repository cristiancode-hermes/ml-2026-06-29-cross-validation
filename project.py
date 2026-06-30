#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Validación Cruzada y Métricas Avanzadas                                   ║
║  Nivel 3 — ML Clásico con scikit-learn                                     ║
║  Dataset: Telecom Customer Churn (simulado realista)                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

CONTENIDO DEL PROYECTO:
  1. Generar datos realistas de churn telefónico (20% churn)
  2. Análisis exploratorio + desbalanceo de clases
  3. Train/Test simple vs K-Fold Cross Validation
  4. Stratified K-Fold para datos desbalanceados
  5. Múltiples métricas: Accuracy, Precision, Recall, F1, ROC-AUC
  6. Curvas de aprendizaje (learning curves)
  7. Curva ROC y Precision-Recall
  8. Comparación de modelos con CV
  9. Grid Search con validación cruzada
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # 🔥 Modo sin pantalla (para servidor)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ─── Configuración ──────────────────────────────────────────────────────────
# Usamos semilla fija para que los resultados sean REPRODUCIBLES
np.random.seed(42)
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
DOCS_DIR = BASE_DIR / 'docs'
DATA_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

# Colores profesionales para gráficos
COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2E', '#3A7CA5']
sns.set_style('whitegrid')
plt.rcParams.update({'figure.max_open_warning': 0, 'font.size': 11})

print("=" * 70)
print("  📊  VALIDACIÓN CRUZADA Y MÉTRICAS AVANZADAS")
print("  🏢  Dataset: Telecom Customer Churn (simulado)")
print("=" * 70)

# ============================================================================
#  PASO 1: GENERAR DATOS REALISTAS DE CHURN TELECOM
# ============================================================================
#  💡 Creamos datos sintéticos pero con patrones realistas:
#     - Contratos mes a mes → más churn (la gente se cansa y se va)
#     - Fibra óptica → más churn (competencia ofrece mejores precios)
#     - Pago con cheque electrónico → más churn (clientes menos estables)
#     - Más tenure (antigüedad) → menos churn (fidelización)
#     - Soporte técnico → menos churn (clientes satisfechos)
# ============================================================================

print("\n📦 PASO 1: Generando datos de churn telecom...")

N_CLIENTES = 4000  # 4000 clientes simulados

def generar_churn_realista():
    """Genera datos de churn con patrones realistas del mundo real."""
    
    # --- Variables categóricas base ---
    contract_types = np.random.choice(
        ['Month-to-month', 'One year', 'Two year'], 
        size=N_CLIENTES, 
        p=[0.55, 0.25, 0.20]  # 💡 55% están en contrato mensual (alta rotación)
    )
    
    internet_services = np.random.choice(
        ['Fiber optic', 'DSL', 'No'], 
        size=N_CLIENTES, 
        p=[0.44, 0.34, 0.22]  # 💡 44% fibra, 34% DSL, 22% sin internet
    )
    
    payment_methods = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
        size=N_CLIENTES,
        p=[0.34, 0.22, 0.24, 0.20]
    )
    
    # --- Variables binarias ---
    senior = np.random.choice([0, 1], size=N_CLIENTES, p=[0.84, 0.16])
    partner = np.random.choice([0, 1], size=N_CLIENTES, p=[0.48, 0.52])
    dependents = np.random.choice([0, 1], size=N_CLIENTES, p=[0.70, 0.30])
    paperless = np.random.choice([0, 1], size=N_CLIENTES, p=[0.40, 0.60])
    
    # --- Variables numéricas con correlaciones realistas ---
    # Tenure: clientes con más antigüedad tienen más servicios y pagan más
    tenure = np.random.exponential(scale=24, size=N_CLIENTES).astype(int)
    tenure = np.clip(tenure, 1, 72)  # entre 1 y 72 meses
    
    # Monthly charges: base $20 + extras según servicios
    monthly_charges = 20 + np.random.normal(30, 15, N_CLIENTES)
    monthly_charges = np.clip(monthly_charges, 18, 120).round(2)
    
    # Total charges = monthly * tenure con algo de ruido
    total_charges = (monthly_charges * tenure * 
                     np.random.uniform(0.9, 1.1, N_CLIENTES)).round(2)
    
    # --- Servicios adicionales (dependen de internet_service) ---
    online_security = np.full(N_CLIENTES, 'No internet')
    online_backup = np.full(N_CLIENTES, 'No internet')
    device_protection = np.full(N_CLIENTES, 'No internet')
    tech_support = np.full(N_CLIENTES, 'No internet')
    streaming_tv = np.full(N_CLIENTES, 'No internet')
    streaming_movies = np.full(N_CLIENTES, 'No internet')
    
    for i in range(N_CLIENTES):
        if internet_services[i] != 'No':
            online_security[i] = np.random.choice(['Yes', 'No'], p=[0.55, 0.45])
            online_backup[i] = np.random.choice(['Yes', 'No'], p=[0.60, 0.40])
            device_protection[i] = np.random.choice(['Yes', 'No'], p=[0.55, 0.45])
            tech_support[i] = np.random.choice(['Yes', 'No'], p=[0.50, 0.50])
            streaming_tv[i] = np.random.choice(['Yes', 'No'], p=[0.65, 0.35])
            streaming_movies[i] = np.random.choice(['Yes', 'No'], p=[0.65, 0.35])
    
    # --- Calcular probabilidad de churn (basada en patrones REALES) ---
    # Cada característica suma o resta a la probabilidad base
    churn_prob = np.full(N_CLIENTES, 0.15)  # probabilidad base 15%
    
    for i in range(N_CLIENTES):
        p = 0.15  # probabilidad base
        
        # 💡 Contratos mes a mes = 3x más probabilidad de irse
        if contract_types[i] == 'Month-to-month':
            p += 0.20
        elif contract_types[i] == 'One year':
            p += 0.05
        else:
            p -= 0.05  # contrato largo = cliente fiel
        
        # 💡 Fibra óptica: más competencia, más churn
        if internet_services[i] == 'Fiber optic':
            p += 0.10
        elif internet_services[i] == 'No':
            p -= 0.03
        
        # 💡 Cheque electrónico: clientes menos estables financieramente
        if payment_methods[i] == 'Electronic check':
            p += 0.08
        elif payment_methods[i] in ['Bank transfer', 'Credit card']:
            p -= 0.03
        
        # 💡 Antigüedad: a más tenure, menos churn
        p -= 0.002 * tenure[i]  # -0.2% por mes de antigüedad
        
        # 💡 Servicios que retienen clientes
        if online_security[i] == 'Yes':
            p -= 0.04
        if tech_support[i] == 'Yes':
            p -= 0.06
        if partner[i] == 1:
            p -= 0.03
        if dependents[i] == 1:
            p -= 0.04
        
        # 💡 Senior citizens tienen menos movilidad
        if senior[i] == 1:
            p -= 0.05
        
        # 💡 Paperless billing = más churn (menos contacto personal)
        if paperless[i] == 1:
            p += 0.03
        
        # 💡 Monthly charges altos → más churn (gastan más, se quejan más)
        if monthly_charges[i] > 80:
            p += 0.06
        elif monthly_charges[i] < 35:
            p -= 0.02
        
        churn_prob[i] = np.clip(p, 0.01, 0.95)
    
    # Asignar Churn según la probabilidad
    churn = (np.random.random(N_CLIENTES) < churn_prob).astype(int)
    
    # --- Armar DataFrame ---
    df = pd.DataFrame({
        'customer_id': [f'CUST-{i+1:05d}' for i in range(N_CLIENTES)],
        'gender': np.random.choice(['Male', 'Female'], N_CLIENTES),
        'senior_citizen': senior,
        'partner': partner,
        'dependents': dependents,
        'tenure': tenure,
        'contract_type': contract_types,
        'payment_method': payment_methods,
        'paperless_billing': paperless,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'internet_service': internet_services,
        'online_security': online_security,
        'online_backup': online_backup,
        'device_protection': device_protection,
        'tech_support': tech_support,
        'streaming_tv': streaming_tv,
        'streaming_movies': streaming_movies,
        'churn': churn
    })
    
    return df

# 🔥 Generar el dataset
df = generar_churn_realista()

# Guardar a CSV
csv_path = DATA_DIR / 'telecom_churn.csv'
df.to_csv(csv_path, index=False)
print(f"  ✅ Datos generados: {df.shape[0]} clientes, {df.shape[1]} columnas")
print(f"  📁 Guardado en: {csv_path}")

# ─── Mostrar primeras filas ──────────────────────────────────────────────
print("\n📋 Primeros 5 clientes:")
print(df.head().to_string())
print(f"\n📊 Dimensiones: {df.shape}")

# ============================================================================
#  PASO 2: ANÁLISIS DE DESBALANCEO DE CLASES
# ============================================================================
#  💡 En telecom, normalmente ~20-27% de clientes se van (churn).
#     Esto crea un DESBALANCEO: la clase minoritaria es la que nos interesa.
#     Si un modelo dice "nadie se va", acierta el 80%... ¡pero no sirve!
#     Por eso necesitamos métricas más sofisticadas.
# ============================================================================

print("\n" + "=" * 70)
print("  📊 PASO 2: ANÁLISIS DE DESBALANCEO DE CLASES")
print("=" * 70)

# Contar clientes que se fueron vs se quedaron
churn_counts = df['churn'].value_counts()
churn_pcts = df['churn'].value_counts(normalize=True) * 100

print(f"\n  📊 Distribución de Churn:")
print(f"     • No se fueron (0): {churn_counts[0]:>5} clientes ({churn_pcts[0]:.1f}%)")
print(f"     • Se fueron  (1): {churn_counts[1]:>5} clientes ({churn_pcts[1]:.1f}%)")

# 💡 Esto es clave: ¿cuánto desbalanceo tenemos?
# ratio de desbalanceo = clase_mayoritaria / clase_minoritaria
imbalance_ratio = churn_counts[0] / churn_counts[1]
print(f"\n  ⚠️  Ratio de desbalanceo: {imbalance_ratio:.1f}:1")
print(f"     (por cada cliente que se fue, {imbalance_ratio:.0f} se quedaron)")

if imbalance_ratio > 3:
    print("     🔴 Desbalanceo SIGNIFICATIVO — necesitamos métricas especiales")
elif imbalance_ratio > 1.5:
    print("     🟡 Desbalanceo MODERADO — ojo con accuracy")
else:
    print("     🟢 Desbalanceo leve")

# ─── Visualización del desbalanceo ──────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico de barras
colors_bar = ['#2E86AB', '#C73E1D']
bars = axes[0].bar(['No Churn (0)', 'Churn (1)'], churn_counts.values, 
                    color=colors_bar, edgecolor='white', linewidth=2)
axes[0].bar_label(bars, labels=[f'{v}\n({p:.1f}%)' for v, p in 
                  zip(churn_counts.values, churn_pcts.values)], fontsize=12, fontweight='bold')
axes[0].set_title('Distribución de Churn', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Cantidad de clientes')
axes[0].set_ylim(0, churn_counts.max() * 1.25)

# Gráfico de torta
wedges, texts, autotexts = axes[1].pie(
    churn_counts.values, labels=['No Churn', 'Churn'], 
    autopct='%1.1f%%', colors=colors_bar, startangle=90,
    explode=(0.02, 0.08), shadow=False,
    textprops={'fontsize': 13, 'fontweight': 'bold'}
)
autotexts[1].set_color('white')
axes[1].set_title('Proporción de clases', fontsize=14, fontweight='bold')

plt.suptitle('🎯 Paso 2: ¿Están balanceadas las clases?', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(DOCS_DIR / '01_desbalanceo_clases.png', dpi=150, bbox_inches='tight')
plt.close()
print("  📈 Gráfico guardado: docs/01_desbalanceo_clases.png")

# ============================================================================
#  PASO 3: TRAIN/TEST SIMPLE vs VALIDACIÓN CRUZADA
# ============================================================================
#  💡 Train/Test Split: partimos los datos una sola vez (80% train, 20% test).
#     Problema: la partición puede tener "suerte" o "mala suerte".
#     
#  💡 Validación Cruzada (K-Fold): partimos los datos en K partes (folds).
#     Entrenamos K veces, cada vez con K-1 folds y validamos con el fold restante.
#     La métrica final es el PROMEDIO de las K iteraciones.
#     Esto da una estimación más ROBUSTA del rendimiento real.
# ============================================================================

print("\n" + "=" * 70)
print("  🔬 PASO 3: TRAIN/TEST SIMPLE vs K-FOLD CROSS VALIDATION")
print("=" * 70)

from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, learning_curve, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)

# ─── Preparar los datos para ML ──────────────────────────────────────────
# 💡 Convertimos variables categóricas a números (Label Encoding)
#    scikit-learn solo entiende números, no texto como "Month-to-month"
print("\n🔧 Preparando datos para ML...")

df_ml = df.copy()

# Columnas categóricas a codificar
cat_cols = ['gender', 'contract_type', 'payment_method', 'internet_service',
            'online_security', 'online_backup', 'device_protection', 
            'tech_support', 'streaming_tv', 'streaming_movies']

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col].astype(str))
    label_encoders[col] = le
    print(f"  🔤 {col}: {len(le.classes_)} categorías → {list(le.classes_)}")

# Seleccionar features (X) y target (y)
feature_cols = ['gender', 'senior_citizen', 'partner', 'dependents', 'tenure',
                'contract_type', 'payment_method', 'paperless_billing',
                'monthly_charges', 'total_charges', 'internet_service',
                'online_security', 'online_backup', 'device_protection',
                'tech_support', 'streaming_tv', 'streaming_movies']

X = df_ml[feature_cols].values
y = df_ml['churn'].values

print(f"\n  ✅ Features (X): {X.shape} — {len(feature_cols)} variables")
print(f"  ✅ Target  (y): {y.shape} — Churn (0/1)")
print(f"  📋 Features: {', '.join(feature_cols)}")

# ─── Train/Test Split Simple ─────────────────────────────────────────────
# 💡 Un solo split: rápido, pero puede ser engañoso si tenemos mala suerte
print("\n" + "-" * 60)
print("  🎲 MÉTODO 1: Train/Test Split SIMPLE (una sola partición)")
print("-" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y  # 💡 stratify mantiene proporción
)

# Estandarizar (importante para modelos basados en distancias)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar Random Forest
rf_simple = RandomForestClassifier(n_estimators=100, random_state=42)
rf_simple.fit(X_train_scaled, y_train)
y_pred = rf_simple.predict(X_test_scaled)
y_proba = rf_simple.predict_proba(X_test_scaled)[:, 1]

# Métricas del split simple
acc_simple = accuracy_score(y_test, y_pred)
prec_simple = precision_score(y_test, y_pred)
rec_simple = recall_score(y_test, y_pred)
f1_simple = f1_score(y_test, y_pred)
auc_simple = roc_auc_score(y_test, y_proba)

print(f"\n  📊 Resultados (una sola partición):")
print(f"     Accuracy:  {acc_simple:.4f}  ({acc_simple*100:.1f}%)")
print(f"     Precision: {prec_simple:.4f}  ({prec_simple*100:.1f}%)")
print(f"     Recall:    {rec_simple:.4f}  ({rec_simple*100:.1f}%)")
print(f"     F1-Score:  {f1_simple:.4f}  ({f1_simple*100:.1f}%)")
print(f"     ROC-AUC:   {auc_simple:.4f}  ({auc_simple*100:.1f}%)")

# ─── K-Fold Cross Validation ────────────────────────────────────────────
# 💡 K-Fold: dividimos en K=5 partes. Entrenamos 5 veces.
#    Cada vez, 4 partes para entrenar, 1 para validar.
#    Al final promediamos las 5 puntuaciones → más confiable.
print("\n" + "-" * 60)
print("  🔄 MÉTODO 2: K-Fold Cross Validation (K=5)")
print("-" * 60)

# Creamos el objeto KFold con 5 splits
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# cross_val_score hace TODO automáticamente:
#   1. Divide en 5 folds
#   2. Entrena el modelo 5 veces
#   3. Calcula la métrica en cada fold
#   4. Devuelve un array con 5 resultados
scores_acc = cross_val_score(rf_simple, X_train, y_train, cv=kf, scoring='accuracy')
scores_f1 = cross_val_score(rf_simple, X_train, y_train, cv=kf, scoring='f1')
scores_auc = cross_val_score(rf_simple, X_train, y_train, cv=kf, scoring='roc_auc')

print(f"\n  📊 Resultados K-Fold (5 folds) — Accuracy:")
print(f"     Fold 1: {scores_acc[0]:.4f}")
print(f"     Fold 2: {scores_acc[1]:.4f}")
print(f"     Fold 3: {scores_acc[2]:.4f}")
print(f"     Fold 4: {scores_acc[3]:.4f}")
print(f"     Fold 5: {scores_acc[4]:.4f}")
print(f"     ─────────────────────────────")
print(f"     Media:   {scores_acc.mean():.4f}  ± {scores_acc.std():.4f}")

print(f"\n  📊 Resultados K-Fold — F1-Score:")
print(f"     Fold 1: {scores_f1[0]:.4f}")
print(f"     Fold 2: {scores_f1[1]:.4f}")
print(f"     Fold 3: {scores_f1[2]:.4f}")
print(f"     Fold 4: {scores_f1[3]:.4f}")
print(f"     Fold 5: {scores_f1[4]:.4f}")
print(f"     ─────────────────────────────")
print(f"     Media:   {scores_f1.mean():.4f}  ± {scores_f1.std():.4f}")

print(f"\n  📊 Resultados K-Fold — ROC-AUC:")
print(f"     Fold 1: {scores_auc[0]:.4f}")
print(f"     Fold 2: {scores_auc[1]:.4f}")
print(f"     Fold 3: {scores_auc[2]:.4f}")
print(f"     Fold 4: {scores_auc[3]:.4f}")
print(f"     Fold 5: {scores_auc[4]:.4f}")
print(f"     ─────────────────────────────")
print(f"     Media:   {scores_auc.mean():.4f}  ± {scores_auc.std():.4f}")

# ─── Comparación visual: Split Simple vs K-Fold ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gráfico 1: Comparación de métricas
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
simple_vals = [acc_simple, prec_simple, rec_simple, f1_simple, auc_simple]
kfold_vals = [scores_acc.mean(), None, None, scores_f1.mean(), scores_auc.mean()]

x_pos = np.arange(len(metrics_names))
width = 0.35

bars1 = axes[0].bar(x_pos - width/2, simple_vals, width, label='Train/Test Simple', 
                     color='#2E86AB', edgecolor='white', linewidth=1.5)
bars2 = axes[0].bar(x_pos + width/2, 
                     [v if v is not None else 0 for v in kfold_vals], 
                     width, label='K-Fold CV (media ± std)', 
                     color='#A23B72', edgecolor='white', linewidth=1.5)

# Añadir valores en las barras
for bar, val in zip(bars1, simple_vals):
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, [v for v in kfold_vals if v is not None]):
    if val is not None:
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

axes[0].set_ylim(0, 1.15)
axes[0].set_ylabel('Puntuación')
axes[0].set_title('Comparación: Split Simple vs K-Fold CV', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(axis='y', alpha=0.3)

# Gráfico 2: Distribución de scores K-Fold (mostrar variabilidad)
fold_data = {
    'Accuracy': scores_acc,
    'F1-Score': scores_f1,
    'ROC-AUC': scores_auc
}
x_pos2 = np.arange(len(fold_data))
colors_fold = ['#2E86AB', '#A23B72', '#F18F01']

for i, (name, scores) in enumerate(fold_data.items()):
    # Puntos individuales
    jitter = np.random.normal(0, 0.04, len(scores))
    axes[1].scatter([i]*len(scores), scores, s=80, c=colors_fold[i], 
                     edgecolors='white', linewidth=1.5, zorder=5)
    # Media
    axes[1].scatter(i, scores.mean(), s=200, marker='D', c=colors_fold[i], 
                     edgecolors='black', linewidth=2, zorder=6)
    # Línea de media
    axes[1].hlines(scores.mean(), i-0.3, i+0.3, colors='black', linestyles='--', 
                    linewidth=1.5, alpha=0.7)
    # Anotación con media ± std
    axes[1].text(i, scores.mean() - 0.12, f'{scores.mean():.3f}±{scores.std():.3f}',
                  ha='center', fontsize=9, fontweight='bold', 
                  bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.8))

axes[1].set_xticks(x_pos2)
axes[1].set_xticklabels(list(fold_data.keys()))
axes[1].set_ylabel('Puntuación en cada fold')
axes[1].set_title('Variabilidad entre Folds (K=5)', fontsize=13, fontweight='bold')
axes[1].set_ylim(0.4, 1.05)
axes[1].grid(axis='y', alpha=0.3)

plt.suptitle('🔬 Paso 3: Train/Test Simple vs K-Fold Cross Validation', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(DOCS_DIR / '02_simple_vs_kfold.png', dpi=150, bbox_inches='tight')
plt.close()
print("  📈 Gráfico guardado: docs/02_simple_vs_kfold.png")

# ============================================================================
#  PASO 4: STRATIFIED K-FOLD — CUANDO LAS CLASES ESTÁN DESBALANCEADAS
# ============================================================================
#  💡 K-Fold normal puede fallar si una clase es minoritaria:
#     algún fold podría tener CASI PURA la clase mayoritaria.
#     Stratified K-Fold mantiene la misma proporción de clases en CADA fold.
#     ¡Es el estándar para clasificación con desbalanceo!
# ============================================================================

print("\n" + "=" * 70)
print("  🎯 PASO 4: K-FOLD NORMAL vs STRATIFIED K-FOLD")
print("=" * 70)

# Comparar la proporción de churn en cada fold
kf_normal = KFold(n_splits=5, shuffle=True, random_state=42)
kf_strat = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n📊 Proporción de Churn en cada fold:")
print(f"{'Fold':>8} {'K-Fold Normal':>16} {'Stratified':>16} {'Diferencia':>16}")
print("-" * 60)

proporciones_normal = []
proporciones_strat = []

for i, (train_idx, val_idx) in enumerate(kf_normal.split(X_train, y_train)):
    y_val_normal = y_train[val_idx]
    pct_normal = y_val_normal.mean() * 100
    proporciones_normal.append(pct_normal)
    
    (train_idx2, val_idx2) = list(kf_strat.split(X_train, y_train))[i]
    y_val_strat = y_train[val_idx2]
    pct_strat = y_val_strat.mean() * 100
    proporciones_strat.append(pct_strat)
    
    print(f"  Fold {i+1}: {pct_normal:>10.2f}% {pct_strat:>13.2f}% {pct_strat - pct_normal:>12.2f}%")

print("-" * 60)
pct_total = y_train.mean() * 100
print(f"  {'Dataset completo':>8}: {pct_total:>10.2f}% {'—':>16}")

# Visualización de la comparación
fig, ax = plt.subplots(figsize=(10, 6))

x_folds = np.arange(1, 6)
ax.plot(x_folds, proporciones_normal, 'o-', color='#C73E1D', linewidth=2.5, 
        markersize=10, label='K-Fold Normal', markerfacecolor='white', markeredgewidth=2.5)
ax.plot(x_folds, proporciones_strat, 's-', color='#2E86AB', linewidth=2.5, 
        markersize=10, label='Stratified K-Fold', markerfacecolor='white', markeredgewidth=2.5)
ax.axhline(y=pct_total, color='green', linestyle='--', linewidth=1.5, 
           alpha=0.7, label=f'Proporción real ({pct_total:.1f}%)')

# Rellenar área entre normal y real
ax.fill_between(x_folds, proporciones_normal, pct_total, alpha=0.15, color='red', 
                 label='Desviación (Normal)')
ax.fill_between(x_folds, proporciones_strat, pct_total, alpha=0.15, color='blue',
                 label='Desviación (Stratified)')

ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('% de Churn en el fold', fontsize=12)
ax.set_title('🎯 Stratified K-Fold mantiene la proporción de clases', fontsize=14, fontweight='bold')
ax.set_xticks(x_folds)
ax.legend(fontsize=10, loc='best')
ax.grid(alpha=0.3)

# Anotación explicativa
ax.text(0.5, 0.01, 
        'Stratified asegura que CADA fold tenga ~ mismo % de Churn → métricas más confiables',
        transform=ax.transAxes, ha='center', fontsize=11, fontstyle='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig(DOCS_DIR / '03_stratified_kfold.png', dpi=150, bbox_inches='tight')
plt.close()
print("  📈 Gráfico guardado: docs/03_stratified_kfold.png")

# ─── Ahora con Stratified K-Fold en serio ────────────────────────────────
print("\n🔄 Usando Stratified K-Fold (el estándar correcto)...")

strat_scores_acc = cross_val_score(rf_simple, X_train, y_train, 
                                    cv=kf_strat, scoring='accuracy')
strat_scores_f1 = cross_val_score(rf_simple, X_train, y_train, 
                                   cv=kf_strat, scoring='f1')
strat_scores_auc = cross_val_score(rf_simple, X_train, y_train, 
                                    cv=kf_strat, scoring='roc_auc')

print(f"\n  📊 Stratified K-Fold Results:")
print(f"     Accuracy: {strat_scores_acc.mean():.4f} ± {strat_scores_acc.std():.4f}")
print(f"     F1:       {strat_scores_f1.mean():.4f} ± {strat_scores_f1.std():.4f}")
print(f"     ROC-AUC:  {strat_scores_auc.mean():.4f} ± {strat_scores_auc.std():.4f}")

# ============================================================================
#  PASO 5: MATRIZ DE CONFUSIÓN Y MÉTRICAS AVANZADAS
# ============================================================================
#  💡 La matriz de confusión nos dice DÓNDE se equivoca el modelo:
#     - Verdaderos Positivos (VP): churns que PREDIJIMOS BIEN
#     - Falsos Positivos (FP): clientes que dijimos que se iban pero NO
#     - Falsos Negativos (FN): churns que NO DETECTAMOS (¡los más peligrosos!)
#     - Verdaderos Negativos (VN): no-churns que PREDIJIMOS BIEN
# ============================================================================

print("\n" + "=" * 70)
print("  📋 PASO 5: MATRIZ DE CONFUSIÓN Y MÉTRICAS AVANZADAS")
print("=" * 70)

# Entrenar con el split estratificado para el análisis detallado
# 💡 Usamos los datos de entrenamiento con validación cruzada para elegir
#    el modelo, luego evaluamos en test para tener una estimación imparcial
rf_final = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
rf_final.fit(X_train_scaled, y_train)

y_pred_final = rf_final.predict(X_test_scaled)
y_proba_final = rf_final.predict_proba(X_test_scaled)[:, 1]

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred_final)
tn, fp, fn, tp = cm.ravel()

print(f"\n  📊 Matriz de Confusión (en test set):")
print(f"  ┌────────────────────────────────────────────────────────┐")
print(f"  │                      │  PREDICCIÓN    │  PREDICCIÓN   │")
print(f"  │                      │  No Churn (0)  │  Churn (1)    │")
print(f"  ├──────────────────────┼────────────────┼───────────────┤")
print(f"  │ REAL: No Churn (0)   │  {tn:>6} (VN)     │  {fp:>6} (FP)    │")
print(f"  │ REAL: Churn (1)      │  {fn:>6} (FN)     │  {tp:>6} (VP)    │")
print(f"  └──────────────────────┴────────────────┴───────────────┘")

# ---- Explicación de cada métrica ----
# Accuracy: (VP + VN) / Total — ¿qué % acertamos en total?
acc_final = accuracy_score(y_test, y_pred_final)
total = tn + fp + fn + tp
print(f"\n  📐 FÓRMULAS DE MÉTRICAS:")

print(f"\n  ✅ Accuracy  = (VP + VN) / Total")
print(f"               = ({tp} + {tn}) / ({total})")
print(f"               = {acc_final:.4f} = {acc_final*100:.1f}%")
print(f"     ❌ PROBLEMA: con clases desbalanceadas, accuracy miente.")
print(f"     Si el modelo predice 'nadie se va', accuracy = {(tn / total)*100:.1f}%")
print(f"     ¡Pero no detectaría ningún churn!")

# Precision: VP / (VP + FP) — de los que dije que se iban, ¿cuántos se fueron?
prec_final = precision_score(y_test, y_pred_final)
print(f"\n  🎯 Precision = VP / (VP + FP)")
print(f"               = {tp} / ({tp} + {fp})")
print(f"               = {prec_final:.4f} = {prec_final*100:.1f}%")
print(f"     💡 Cuando PREDIGO que un cliente se va, ¿qué tan seguro estoy?")
print(f"     Útil cuando: los FP son caros (ej: enviar ofertas a quien no necesita)")

# Recall: VP / (VP + FN) — de los que se fueron, ¿cuántos detecté?
rec_final = recall_score(y_test, y_pred_final)
print(f"\n  🎯 Recall    = VP / (VP + FN)")
print(f"               = {tp} / ({tp} + {fn})")
print(f"               = {rec_final:.4f} = {rec_final*100:.1f}%")
print(f"     💡 De TODOS los clientes que se fueron, ¿cuántos atrapé?")
print(f"     Útil cuando: los FN son MUY caros (ej: detectar fraudes)")

# F1-Score: media armónica de Precision y Recall
f1_final = f1_score(y_test, y_pred_final)
print(f"\n  🎯 F1-Score  = 2 * (Precision * Recall) / (Precision + Recall)")
print(f"               = 2 * ({prec_final:.3f} * {rec_final:.3f}) / ({prec_final:.3f} + {rec_final:.3f})")
print(f"               = {f1_final:.4f} = {f1_final*100:.1f}%")
print(f"     💡 Balance entre Precision y Recall. La métrica ESTÁNDAR")
print(f"        para clasificación con desbalanceo.")

# ROC-AUC: qué tan bien separa las clases el modelo
auc_final = roc_auc_score(y_test, y_proba_final)
print(f"\n  🎯 ROC-AUC   = Área bajo la curva ROC")
print(f"               = {auc_final:.4f} = {auc_final*100:.1f}%")
print(f"     💡 1.0 = perfecto, 0.5 = aleatorio, 0.0 = todo mal")
print(f"     Mide la capacidad del modelo de SEPARAR clases.")

# ─── Matriz de confusión visual ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
            xticklabels=['No Churn (0)', 'Churn (1)'],
            yticklabels=['No Churn (0)', 'Churn (1)'],
            annot_kws={'fontsize': 18, 'fontweight': 'bold'},
            linewidths=2, linecolor='white', ax=ax)
ax.set_xlabel('PREDICCIÓN', fontsize=13, fontweight='bold')
ax.set_ylabel('REALIDAD', fontsize=13, fontweight='bold')
ax.set_title('📋 Matriz de Confusión — Churn Telecom', fontsize=15, fontweight='bold')

# Añadir anotaciones contextuales
ax.text(0.5, -0.15, 
        'VN = No-churn correctos | FP = Falsas alarmas | FN = ¡Churns no detectados! | VP = Churns detectados',
        transform=ax.transAxes, ha='center', fontsize=10, fontstyle='italic',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig(DOCS_DIR / '04_matriz_confusion.png', dpi=150, bbox_inches='tight')
plt.close()
print("  📈 Gráfico guardado: docs/04_matriz_confusion.png")

# ─── Reporte de clasificación completo ───────────────────────────────────
print("\n\n📋 Reporte de Clasificación Detallado:")
print("=" * 50)
print(classification_report(y_test, y_pred_final, 
                            target_names=['No Churn', 'Churn']))

# ============================================================================
#  PASO 6: CURVA ROC Y PRECISION-RECALL
# ============================================================================
#  💡 Curva ROC: muestra la relación entre TPR (Recall) y FPR (1-Especificidad)
#     al variar el umbral de decisión. El área bajo la curva (AUC) es la métrica.
#     
#  💡 Curva Precision-Recall: más útil cuando hay DESBALANCEO.
#     Muestra el trade-off entre Precision y Recall al variar el umbral.
#     Un modelo bueno tiene curva cerca de la esquina superior derecha.
# ============================================================================

print("\n" + "=" * 70)
print("  📈 PASO 6: CURVA ROC Y CURVA PRECISION-RECALL")
print("=" * 70)

# --- Curva ROC ---
fpr, tpr, thresholds_roc = roc_curve(y_test, y_proba_final)

# --- Curva Precision-Recall ---
precision_pts, recall_pts, thresholds_pr = precision_recall_curve(y_test, y_proba_final)

# Calcular línea base (modelo aleatorio)
base_recall = np.linspace(0, 1, 100)
base_precision = np.full(100, y_test.mean())  # proporción de churn en test

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico 1: Curva ROC
axes[0].plot(fpr, tpr, color='#2E86AB', linewidth=3, label=f'Random Forest (AUC = {auc_final:.3f})')
axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1.5, alpha=0.5, label='Modelo aleatorio (AUC = 0.5)')
axes[0].fill_between(fpr, tpr, alpha=0.15, color='#2E86AB')

# Punto del clasificador default (umbral = 0.5)
idx_default = np.argmin(np.abs(thresholds_roc - 0.5))
axes[0].scatter(fpr[idx_default], tpr[idx_default], s=200, c='#C73E1D', 
                 edgecolors='white', linewidth=2, zorder=5,
                 label=f'Umbral=0.5 (TPR={tpr[idx_default]:.3f}, FPR={fpr[idx_default]:.3f})')

axes[0].set_xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
axes[0].set_ylabel('Tasa de Verdaderos Positivos (TPR = Recall)', fontsize=12)
axes[0].set_title('📈 Curva ROC', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=9, loc='lower right')
axes[0].grid(alpha=0.3)
axes[0].set_xlim(-0.02, 1.02)
axes[0].set_ylim(-0.02, 1.02)

# Gráfico 2: Curva Precision-Recall
axes[1].plot(recall_pts, precision_pts, color='#A23B72', linewidth=3, 
             label=f'Random Forest (F1={f1_final:.3f})')
axes[1].axhline(y=y_test.mean(), color='gray', linestyle='--', linewidth=1.5, alpha=0.7,
                 label=f'Línea base (siempre predecir churn = {y_test.mean():.3f})')
axes[1].fill_between(recall_pts, precision_pts, alpha=0.15, color='#A23B72')

# Punto default
axes[1].scatter(rec_final, prec_final, s=200, c='#F18F01', 
                 edgecolors='white', linewidth=2, zorder=5,
                 label=f'Umbral=0.5 (Recall={rec_final:.3f}, Precision={prec_final:.3f})')

axes[1].set_xlabel('Recall', fontsize=12)
axes[1].set_ylabel('Precision', fontsize=12)
axes[1].set_title('📈 Curva Precision-Recall', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=9, loc='lower right')
axes[1].grid(alpha=0.3)
axes[1].set_xlim(-0.02, 1.02)
axes[1].set_ylim(-0.02, 1.02)

plt.suptitle('📈 Paso 6: Evaluación Visual del Modelo', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(DOCS_DIR / '05_roc_pr_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  📈 Gráfico guardado: docs/05_roc_pr_curves.png")

# ============================================================================
#  PASO 7: CURVAS DE APRENDIZAJE (LEARNING CURVES)
# ============================================================================
#  💡 Learning curves: muestran cómo mejora el modelo al añadir más datos.
#     - Train score alto + Test score bajo = SOBREAJUSTE (mucha varianza)
#     - Ambos scores bajos = SUBAJUSTE (mucho sesgo)
#     - Ambos scores altos y cerca = ajuste ideal
#     Si las curvas NO se juntan al final, añadir más datos AYUDARÍA.
# ============================================================================

print("\n" + "=" * 70)
print("  📈 PASO 7: CURVAS DE APRENDIZAJE (LEARNING CURVES)")
print("=" * 70)

# Calcular learning curves
train_sizes, train_scores, test_scores = learning_curve(
    rf_final, X_train, y_train, cv=kf_strat, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),  # 10 puntos: 10% a 100% de datos
    scoring='f1', random_state=42
)

# Calcular medias y desviaciones
train_mean = train_scores.mean(axis=1)  # media del F1 en train para cada tamaño
train_std = train_scores.std(axis=1)     # desviación estándar
test_mean = test_scores.mean(axis=1)
test_std = test_scores.std(axis=1)

# Mostrar en consola
print(f"\n  📊 Learning Curves (F1-Score por tamaño de training set):")
print(f"  {'% datos':>8} {'Train F1':>10} {'Test F1':>10} {'Diferencia':>12}")
print("-" * 45)
for i, ts in enumerate(train_sizes):
    diff = train_mean[i] - test_mean[i]
    print(f"  {ts/len(X_train)*100:>6.0f}%:  {train_mean[i]:.4f}      {test_mean[i]:.4f}     {diff:+.4f}")

# Último punto (con todos los datos)
final_diff = train_mean[-1] - test_mean[-1]
print("-" * 45)
print(f"  💡 Con {train_sizes[-1]} muestras de entrenamiento:")
print(f"     F1 en Train: {train_mean[-1]:.4f}")
print(f"     F1 en Test:  {test_mean[-1]:.4f}")
print(f"     Diferencia:  {final_diff:+.4f}")

if final_diff > 0.15:
    print(f"  ⚠️  SOBREAJUSTE detectado (diferencia > 0.15). ¿Más datos o regularizar?")
elif final_diff > 0.05:
    print(f"  🟡 SOBREAJUSTE LEVE. Podría mejorar con más datos.")
else:
    print(f"  🟢 Ajuste razonable. El modelo generaliza bien.")

# ─── Visualizar Learning Curves ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

# Train scores
ax.plot(train_sizes, train_mean, 'o-', color='#2E86AB', linewidth=2.5, 
        markersize=8, label='Train Score (F1)', markerfacecolor='white', markeredgewidth=2)
ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                alpha=0.15, color='#2E86AB')

# Test scores
ax.plot(train_sizes, test_mean, 's-', color='#C73E1D', linewidth=2.5, 
        markersize=8, label='Validation Score (F1)', markerfacecolor='white', markeredgewidth=2)
ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, 
                alpha=0.15, color='#C73E1D')

ax.set_xlabel('Muestras de entrenamiento', fontsize=12)
ax.set_ylabel('F1-Score', fontsize=12)
ax.set_title('📈 Curvas de Aprendizaje — ¿Ayudaría más datos?', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(alpha=0.3)
ax.set_ylim(0.4, 1.02)

# Línea de referencia en F1=1.0
ax.axhline(y=1.0, color='green', linestyle=':', alpha=0.3)

# Anotaciones explicativas
ax.annotate('Sobreajuste si\ntrain >> val', xy=(train_sizes[-1], train_mean[-1]),
            xytext=(train_sizes[-1]*0.6, train_mean[-1]+0.08),
            fontsize=10, fontweight='bold', color='#2E86AB',
            arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=2))

ax.annotate('Generalización\nen test', xy=(train_sizes[-1], test_mean[-1]),
            xytext=(train_sizes[-1]*0.6, test_mean[-1]-0.08),
            fontsize=10, fontweight='bold', color='#C73E1D',
            arrowprops=dict(arrowstyle='->', color='#C73E1D', lw=2))

plt.tight_layout()
plt.savefig(DOCS_DIR / '06_learning_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  📈 Gráfico guardado: docs/06_learning_curves.png")

# ============================================================================
#  PASO 8: COMPARAR MÚLTIPLES MODELOS CON VALIDACIÓN CRUZADA
# ============================================================================
#  💡 Comparamos 4 modelos con la MISMA validación cruzada:
#     - Regresión Logística (rápida, interpretable)
#     - KNN (vecinos más cercanos)
#     - Árbol de Decisión (interpretable)
#     - Random Forest (potente, ensemble)
#     El ganador es el que tiene mejor F1 promedio con menor desviación.
# ============================================================================

print("\n" + "=" * 70)
print("  ⚔️  PASO 8: COMPARACIÓN DE MODELOS CON VALIDACIÓN CRUZADA")
print("=" * 70)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# Definir modelos
modelos = {
    'Regresión Logística': LogisticRegression(random_state=42, max_iter=1000),
    'KNN (k=7)': KNeighborsClassifier(n_neighbors=7),
    'Árbol Decisión (d=6)': DecisionTreeClassifier(max_depth=6, random_state=42),
    'Random Forest (200)': RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42)
}

resultados = []

print(f"\n  {'Modelo':<25} {'Accuracy':>10} {'F1-Score':>10} {'ROC-AUC':>10} {'Estabilidad':>15}")
print("-" * 70)

for nombre, modelo in modelos.items():
    # 💡 cross_val_score con Stratified K-Fold para TODOS los modelos
    scores_acc = cross_val_score(modelo, X_train, y_train, cv=kf_strat, scoring='accuracy')
    scores_f1 = cross_val_score(modelo, X_train, y_train, cv=kf_strat, scoring='f1')
    scores_auc = cross_val_score(modelo, X_train, y_train, cv=kf_strat, scoring='roc_auc')
    
    # La estabilidad es 1 - (std / mean) — qué tan consistentes son los scores
    estabilidad = 1 - (scores_f1.std() / max(scores_f1.mean(), 0.001))
    
    resultados.append({
        'Modelo': nombre,
        'Accuracy': scores_acc.mean(),
        'F1-Score': scores_f1.mean(),
        'ROC-AUC': scores_auc.mean(),
        'Estabilidad': estabilidad,
        'F1_std': scores_f1.std()
    })
    
    print(f"  {nombre:<25} {scores_acc.mean():>8.4f}  {scores_f1.mean():>8.4f}  {scores_auc.mean():>8.4f}  {estabilidad:>10.2f}")

# Convertir a DataFrame y mostrar
df_resultados = pd.DataFrame(resultados)
df_resultados = df_resultados.sort_values('F1-Score', ascending=False)

print(f"\n  🏆 Ranking por F1-Score:")
for i, (_, row) in enumerate(df_resultados.iterrows(), 1):
    medal = {1: '🥇', 2: '🥈', 3: '🥉'}.get(i, f'  {i}.')
    print(f"     {medal} {row['Modelo']}: F1 = {row['F1-Score']:.4f}, "
          f"ROC-AUC = {row['ROC-AUC']:.4f}, Estabilidad = {row['Estabilidad']:.2f}")

# ─── Visualización de comparación de modelos ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Gráfico 1: Barras agrupadas por métrica
x = np.arange(len(df_resultados))
width = 0.25

metrica_colors = {'Accuracy': '#2E86AB', 'F1-Score': '#A23B72', 'ROC-AUC': '#F18F01'}
for i, metrica in enumerate(['Accuracy', 'F1-Score', 'ROC-AUC']):
    axes[0].bar(x + (i-1)*width, df_resultados[metrica].values, width,
                label=metrica, color=metrica_colors[metrica], edgecolor='white', linewidth=1.5)

axes[0].set_xticks(x)
axes[0].set_xticklabels(df_resultados['Modelo'], fontsize=10, rotation=15)
axes[0].set_ylabel('Puntuación', fontsize=12)
axes[0].set_title('⚔️ Comparación de Modelos (CV)', fontsize=14, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].set_ylim(0.4, 1.05)
axes[0].grid(axis='y', alpha=0.3)

# Gráfico 2: F1-Score con barras de error (desviación entre folds)
mejor_modelo = df_resultados.iloc[0]['Modelo']
colors_bar = ['#2E86AB' if m != mejor_modelo else '#C73E1D' for m in df_resultados['Modelo']]

bars = axes[1].barh(range(len(df_resultados)), df_resultados['F1-Score'].values, 
                     xerr=df_resultados['F1_std'].values, 
                     color=colors_bar, edgecolor='white', linewidth=1.5,
                     height=0.6, capsize=5)

axes[1].set_yticks(range(len(df_resultados)))
axes[1].set_yticklabels(df_resultados['Modelo'], fontsize=10)
axes[1].set_xlabel('F1-Score (con desviación entre folds)', fontsize=12)
axes[1].set_title('🏆 Ranking: F1-Score con Variabilidad', fontsize=14, fontweight='bold')
axes[1].axvline(x=df_resultados.iloc[0]['F1-Score'], color='#C73E1D', 
                 linestyle='--', alpha=0.4, linewidth=1)
axes[1].grid(axis='x', alpha=0.3)
axes[1].set_xlim(0.4, 1.05)

# Anotar ganador
axes[1].text(df_resultados.iloc[0]['F1-Score'] + 0.02, 0,
             f'🥇 {mejor_modelo}', fontsize=12, fontweight='bold', color='#C73E1D')

plt.suptitle('⚔️ Paso 8: Competencia de Modelos — ¿Quién gana?', fontsize=15, y=1.02)
plt.tight_layout()
plt.savefig(DOCS_DIR / '07_comparacion_modelos.png', dpi=150, bbox_inches='tight')
plt.close()
print("  📈 Gráfico guardado: docs/07_comparacion_modelos.png")

# ============================================================================
#  PASO 9: GRID SEARCH CON VALIDACIÓN CRUZADA
# ============================================================================
#  💡 GridSearchCV: busca la MEJOR combinación de hiperparámetros
#     usando validación cruzada para evaluar CADA combinación.
#     Así no hacemos trampa (no usamos test set para elegir parámetros).
# ============================================================================

print("\n" + "=" * 70)
print("  🔍 PASO 9: GRID SEARCH CON VALIDACIÓN CRUZADA")
print("=" * 70)

# Definir el grid de hiperparámetros a probar
# 💡 Cada combinación se evalúa con 5-fold Stratified CV
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [6, 10, 15],
    'min_samples_split': [2, 5, 10]
}

print(f"\n  🔧 Grid de hiperparámetros a explorar:")
for param, values in param_grid.items():
    print(f"     {param}: {values}")
print(f"  Total combinaciones: {3 * 3 * 3} = 27")
print(f"  Total entrenamientos: 27 × 5 folds = 135 🤯")

# 💡 GridSearchCV con 5-fold Stratified
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=5,  # 5-fold CV
    scoring='f1',  # optimizar F1 (no accuracy)
    n_jobs=-1,  # usar todos los CPUs
    verbose=1,
    return_train_score=True
)

print(f"\n  🔍 Buscando mejores hiperparámetros (esto toma unos segundos)...")
grid_search.fit(X_train, y_train)

# Mostrar resultados
print(f"\n  🏆 Mejores hiperparámetros encontrados:")
for param, value in grid_search.best_params_.items():
    print(f"     {param}: {value}")
print(f"  Mejor F1 (CV): {grid_search.best_score_:.4f}")

# Tabla de resultados
df_grid = pd.DataFrame(grid_search.cv_results_)
print(f"\n  📊 Top 5 combinaciones:")
top5 = df_grid.sort_values('rank_test_score').head(5)
for i, (_, row) in enumerate(top5.iterrows(), 1):
    params = ', '.join([f'{k}={v}' for k, v in row['params'].items()])
    print(f"     {i}. [{params}] → F1 = {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")

# Evaluar el mejor modelo en TEST (por FIN)
best_rf = grid_search.best_estimator_
best_rf.fit(X_train_scaled, y_train)
y_pred_best = best_rf.predict(X_test_scaled)
y_proba_best = best_rf.predict_proba(X_test_scaled)[:, 1]

acc_best = accuracy_score(y_test, y_pred_best)
f1_best = f1_score(y_test, y_pred_best)
auc_best = roc_auc_score(y_test, y_proba_best)

print(f"\n  🏆 Modelo final (GridSearchCV + CV + Test):")
print(f"     Accuracy: {acc_best:.4f}  (antes: {acc_final:.4f})")
print(f"     F1:       {f1_best:.4f}  (antes: {f1_final:.4f})")
print(f"     ROC-AUC:  {auc_best:.4f}  (antes: {auc_final:.4f})")

if f1_best > f1_final:
    print(f"  ✅ ¡Mejora de {((f1_best - f1_final) / f1_final * 100):.1f}% en F1 gracias a GridSearchCV!")

# ─── Comparación final ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))

final_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
final_before = [acc_final, prec_final, rec_final, f1_final, auc_final]
final_after = [acc_best, 
               precision_score(y_test, y_pred_best),
               recall_score(y_test, y_pred_best),
               f1_best, auc_best]

x_final = np.arange(len(final_metrics))
width = 0.35

bars_before = ax.bar(x_final - width/2, final_before, width, 
                      label='Random Forest (default)', color='#2E86AB', 
                      edgecolor='white', linewidth=1.5)
bars_after = ax.bar(x_final + width/2, final_after, width,
                     label=f'RF con GridSearchCV (best: {grid_search.best_params_})', 
                     color='#C73E1D', edgecolor='white', linewidth=1.5)

# Etiquetas de mejora
for i, (before, after) in enumerate(zip(final_before, final_after)):
    mejora = ((after - before) / max(before, 0.001)) * 100
    color = 'green' if mejora > 0 else 'red'
    ax.text(x_final[i], max(before, after) + 0.02, 
            f'{mejora:+.1f}%', ha='center', fontsize=10, fontweight='bold', color=color)

ax.set_xticks(x_final)
ax.set_xticklabels(final_metrics, fontsize=11)
ax.set_ylabel('Puntuación', fontsize=12)
ax.set_title('🏆 Antes vs Después: GridSearchCV con Validación Cruzada', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.set_ylim(0.3, 1.15)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(DOCS_DIR / '08_grid_search_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  📈 Gráfico guardado: docs/08_grid_search_comparison.png")

# ============================================================================
#  🧠 RESUMEN FINAL — QUÉ APRENDISTE HOY
# ============================================================================

print("\n" + "=" * 70)
print("  🧠 RESUMEN: LO QUE APRENDISTE HOY")
print("=" * 70)

print(r"""
  📊 VALIDACIÓN CRUZADA (Cross Validation):
     ══════════════════════════════════════
     • Train/Test Split: una sola partición → rápido pero variable
     • K-Fold CV: K particiones, K entrenamientos → más robusto
     • Stratified K-Fold: mantiene proporción de clases → ¡el estándar!

  📐 MÉTRICAS AVANZADAS:
     ══════════════════════
     • Accuracy: fácil de engañar con datos desbalanceados
     • Precision: de los que predije, ¿cuántos acerté?
     • Recall: de los que realmente son, ¿cuántos detecté?
     • F1-Score: balance entre Precision y Recall
     • ROC-AUC: capacidad de separar clases

  📈 CURVAS DE APRENDIZAJE:
     ════════════════════════
     • Train score >> Test score → SOBREAJUSTE (mucha varianza)
     • Ambos scores bajos → SUBAJUSTE (mucho sesgo)
     • Curvas se acercan al final → más datos ayudarían

  🔍 GRID SEARCH CV:
     ═══════════════════
     • Búsqueda exhaustiva de hiperparámetros
     • Cada combinación evaluada con CV
     • Nunca usamos test para elegir parámetros
""")

# Mostrar resultados finales
print(f"\n  📊 RESULTADOS FINALES DEL PROYECTO:")
print(f"  ┌────────────────────────────────────────────────────────────────┐")
print(f"  │ Dataset: Telecom Churn — {N_CLIENTES} clientes, {len(feature_cols)} features        │")
print(f"  │ Desbalanceo: ~{churn_pcts[1]:.1f}% churn ({imbalance_ratio:.1f}:1)                      │")
print(f"  │ Modelo ganador: {df_resultados.iloc[0]['Modelo']:<30s}│")
print(f"  │ Mejor F1 (CV): {grid_search.best_score_:.4f}                                       │")
print(f"  │ F1 en TEST:    {f1_best:.4f}                                       │")
print(f"  │ ROC-AUC TEST:  {auc_best:.4f}                                       │")
print(f"  └────────────────────────────────────────────────────────────────┘")

print(f"\n  📁 Proyecto guardado en: {BASE_DIR}")
print(f"  📊 {len(list(DOCS_DIR.glob('*.png')))} visualizaciones generadas en docs/")
print(f"  📊 Tabla resumen guardada en: docs/comparativa_metricas.csv")

# Guardar tabla resumen
df_resultados.to_csv(DOCS_DIR / 'comparativa_metricas.csv', index=False)
print("\n" + "=" * 70)
print("  ✅ ¡PROYECTO COMPLETADO!")
print("  🚀 Siguiente: Pipelines en scikit-learn (Nivel 3)")
print("=" * 70)
