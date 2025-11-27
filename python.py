import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split 

# 1. Cargar el archivo application_.parquet
file_path = 'application_.parquet'  # ¡CORREGIDO!
df = pd.read_parquet(file_path)

print("--- Primeras Filas ---")
print(df.head())

print("\n--- Información General ---")
df.info()


# Variables que usaremos para el clustering (Matriz X)
features = [
    'AMT_INCOME_TOTAL', 
    'AMT_CREDIT', 
    'DAYS_BIRTH', 
    'NAME_INCOME_TYPE', 
    'NAME_EDUCATION_TYPE'
]

# Seleccionar solo las características y la variable TARGET para el análisis posterior
df_cluster = df[features + ['TARGET']].copy()

# --- 1. Tratamiento de Nulos (Imputación Simple) ---
# Para simplificar y avanzar, rellenaremos los nulos de las columnas numéricas con la mediana 
# y los nulos de las categóricas con el valor 'Missing'.

for col in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'DAYS_BIRTH']:
    median_val = df_cluster[col].median()
    df_cluster[col].fillna(median_val, inplace=True)

for col in ['NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE']:
    df_cluster[col].fillna('Missing', inplace=True)

# --- 2. Ingeniería de Características (Edad) ---
# Convertir DAYS_BIRTH a AGE en años (dividiendo por -365 para obtener un valor positivo)
df_cluster['AGE'] = df_cluster['DAYS_BIRTH'] / -365
df_cluster.drop('DAYS_BIRTH', axis=1, inplace=True)

# --- 3. Codificación de Variables Categóricas (One-Hot Encoding) ---
# K-Means necesita números. Convertimos las categóricas a variables dummy.
df_processed = pd.get_dummies(df_cluster.drop('TARGET', axis=1), drop_first=True)

# --- 4. Escalado de Datos ---
# Estandarizar las variables para que todas tengan media 0 y desviación estándar 1
# Esto es CRUCIAL para K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_processed)

print("\n--- Vista Previa de la Matriz Escalada (X_scaled) ---")
print("Forma del dataset preprocesado:", X_scaled.shape)
print("Primeras filas de la matriz escalada:")
print(X_scaled[:5])


## --- 4.1. Método del Codo para Encontrar K Óptimo ---

# Rangos de K a probar (ejemplo: de 1 a 10 clusters)
k_range = range(1, 11)
inertia = [] # Lista para almacenar la inercia para cada k

print("\n--- Ejecutando Método del Codo (Calculando Inercia)... ---")

for k in k_range:
    # Inicializar y ajustar el modelo K-Means
    # n_init='auto' asegura que el algoritmo elija el mejor centroide inicial
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    print(f"Calculado k={k}...")

# Graficar los resultados
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia, marker='o')
plt.title('Método del Codo para K-Means')
plt.xlabel('Número de Clusters (k)')
plt.ylabel('Inercia')
plt.xticks(k_range)
plt.grid(True)

# Guardar el gráfico para el análisis posterior
elbow_plot_filename = 'elbow_method_plot.png'
plt.savefig(elbow_plot_filename)
print(f"\nGráfico del Método del Codo guardado como: {elbow_plot_filename}")
plt.close() # Cierra la figura para no mostrarla en la consola

## --- 4.2. Selección y Entrenamiento del Modelo Final ---

# Basado en la práctica común y la expectativa de un dataset grande, 
# asumiremos un valor inicial de K=4. Luego lo ajustaremos visualmente.
K_FINAL = 4 

print(f"\n--- Entrenando Modelo K-Means Final con K = {K_FINAL} ---")

kmeans_final = KMeans(n_clusters=K_FINAL, random_state=42, n_init='auto')
# Entrenar el modelo y obtener las etiquetas de cluster para cada cliente
cluster_labels = kmeans_final.fit_predict(X_scaled)

# Asignar las etiquetas de cluster al DataFrame original para el análisis
df_cluster['CLUSTER_LABEL'] = cluster_labels

print(f"Etiquetas de cluster generadas. Se han creado {K_FINAL} clusters.")
print("Primeras 5 etiquetas:", df_cluster['CLUSTER_LABEL'].head())

## --- 5.1. Análisis del Riesgo (TARGET) por Cluster ---

# Calcular la tasa de morosidad (mean del TARGET) para cada cluster
cluster_risk = df_cluster.groupby('CLUSTER_LABEL')['TARGET'].agg(['count', 'mean']).reset_index()
cluster_risk.rename(columns={'count': 'Num_Clientes', 'mean': 'Tasa_Morosidad'}, inplace=True)

# La tasa de morosidad generalmente es baja (e.g., 5-10% en el dataset completo)
# Multiplicar por 100 y redondear para mejor visualización
cluster_risk['Tasa_Morosidad_pct'] = (cluster_risk['Tasa_Morosidad'] * 100).round(2)

print("\n--- Tasa de Morosidad por Cluster ---")
print(cluster_risk)

# Crear un gráfico de barras para visualizar la diferencia en la tasa de morosidad
plt.figure(figsize=(8, 5))
sns.barplot(x='CLUSTER_LABEL', y='Tasa_Morosidad_pct', data=cluster_risk)
plt.title('Tasa de Morosidad (%) por Cluster de K-Means')
plt.xlabel('Etiqueta de Cluster')
plt.ylabel('Tasa de Morosidad (%)')
# Guardar el gráfico
risk_plot_filename = 'cluster_risk_plot.png'
plt.savefig(risk_plot_filename)
print(f"\nGráfico de Riesgo por Cluster guardado como: {risk_plot_filename}")
plt.close()


## --- 5.2. Caracterización del Perfil de Cada Cluster ---

# Variables a analizar (las 5 originales más la edad calculada)
analysis_features = [
    'AMT_INCOME_TOTAL', 
    'AMT_CREDIT', 
    'AGE', 
    'NAME_INCOME_TYPE', 
    'NAME_EDUCATION_TYPE',
    'CLUSTER_LABEL'
]

# Recargar el DataFrame con las transformaciones de ingeniería (AGE)
df_profile = df_cluster[analysis_features].copy() 

print("\n--- Resumen de Perfiles por Cluster (Variables Numéricas) ---")
# Comparar la media de las variables numéricas entre los clusters
profile_numeric = df_profile.groupby('CLUSTER_LABEL')[['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AGE']].mean().round(0)
print(profile_numeric)

print("\n--- Resumen de Perfiles por Cluster (Variables Categóricas - Moda) ---")
# Identificar la Moda (valor más frecuente) para las variables categóricas
def get_mode(series):
    return series.mode()[0]

profile_categorical = df_profile.groupby('CLUSTER_LABEL').agg({
    'NAME_INCOME_TYPE': get_mode,
    'NAME_EDUCATION_TYPE': get_mode
})
print(profile_categorical)




# 2. Asumir que ya tienes una columna TARGET (0: No Mora, 1: Mora)
# y que tienes una forma de identificar el conjunto de ENTRENAMIENTO.
# **IMPORTANTE:** Debes usar solo el conjunto de entrenamiento.
# Si tu dataset ya está solo en entrenamiento, puedes ignorar el filtro.
# Si no, necesitas el filtro. Por ahora, asumiremos que el df es solo el conjunto de entrenamiento.