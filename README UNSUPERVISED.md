# Análisis de Clientes y Riesgo con Aprendizaje No Supervisado (K-Means)

## [cite_start]1. Descripción de la Técnica Utilizada y Justificación de la Elección [cite: 37]

Se implementó el **Análisis de Clusters** utilizando el algoritmo **K-Means** sobre el conjunto de entrenamiento del dataset de scoring crediticio, utilizando solo el archivo `application_.parquet`.

* **Técnica:** K-Means es un algoritmo de clustering que agrupa observaciones en 'K' grupos, minimizando la inercia (la suma de las distancias cuadradas dentro del cluster).
* **Objetivo:** El propósito es identificar **segmentos de clientes homogéneos** que exhiban un **riesgo crediticio diferenciado** (tasas de morosidad distintas). Esto complementa el modelo supervisado al revelar la estructura subyacente de la población de solicitantes.
* **Variables Utilizadas:** Se seleccionaron cinco característicass clave que definen el perfil demográfico y financiero: `AMT_INCOME_TOTAL`, `AMT_CREDIT`, `DAYS_BIRTH` (transformada a AGE), `NAME_INCOME_TYPE`, y `NAME_EDUCATION_TYPE`. Estas variables fueron preprocesadas (imputación de nulos, One-Hot Encoding y estandarización) para cumplir con los requisitos del algoritmo K-Means.

---

## [cite_start]2. Instrucciones de Ejecución Claras del Código [cite: 38]

El código para la implementación se encuentra en el archivo `python.py`.

1.  **Dependencias:** Asegúrese de tener instaladas las siguientes librerías de Python: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `pyarrow`.
2.  **Preparación de Datos:** El archivo de datos **`application_.parquet`** debe estar ubicado en el mismo directorio que el script `python.py`.
3.  **Ejecución:** Ejecute el script directamente desde la terminal en el directorio de trabajo:
    ```bash
    python python.py
    ```
4.  **Resultados:** El script imprime el análisis de perfiles y de riesgo en la consola, y genera los siguientes archivos gráficos: `elbow_method_plot.png` y `cluster_risk_plot.png`.

---

## [cite_start]3. Análisis e Interpretación de Resultados [cite: 39]

El modelo K-Means, ajustado con $K=4$ (basado en el método del Codo), identificó segmentos que tienen un riesgo de morosidad significativamente diferente, lo que valida la utilidad del agrupamiento.

### A. Tasa de Morosidad por Cluster (Vinculación con Modelo Supervisado)

| Cluster | Num. Clientes | Tasa de Morosidad (%) | Categoría de Riesgo |
| :---: | :---: | :---: | :--- |
| **0** | 175,056 | **9.85%** | Base/Riesgo Alto (Working Class) |
| **1** | 55,108 | 5.39% | Bajo Riesgo (Pensionados) |
| **2** | 22 | **36.36%** | **Anomalía / Riesgo Extremo** |
| **3** | 77,325 | 5.95% | Bajo Riesgo (Ingreso Alto/Educación Superior) |



### B. Perfil de los Clusters

| Cluster | Ingreso Promedio (USD) | Edad Promedio | Tipo de Ingreso (Moda) | Educación (Moda) |
| :---: | :---: | :---: | :--- | :--- |
| **0** | 160,513 | 41 años | **Working** | Secondary / secondary special |
| **1** | 135,615 | **60 años** | **Pensioner** | Secondary / secondary special |
| **2** | 110,536 | 44 años | **Unemployed** | Secondary / secondary special |
| **3** | **211,219** | 39 años | **Working** | **Higher education** |

### Interpretación de los Hallazgos
1.  **Segmentación por Riesgo:** El *clustering* separó claramente grupos de bajo riesgo (Cluster 1 y 3, ambos con morosidad $\sim 5-6\%$) de grupos de alto riesgo (Cluster 0, $\sim 9.85\%$).
2.  **Identificación de Patrones Ocultos:**
    * **Bajo Riesgo:** La baja morosidad se asocia a la estabilidad de la pensión (Cluster 1: edad alta y `Pensioner`) o a una alta capacidad económica (Cluster 3: ingreso superior y `Higher education`).
    * **Anomalía de Riesgo:** El **Cluster 2** representa una **anomalía**. Aunque es diminuto (22 clientes), tiene una tasa de morosidad del $36.36\%$, estando compuesto mayoritariamente por clientes **desempleados**. Esto sugiere que el estado laboral es un factor de riesgo extremo, aislado por K-Means.

---

## [cite_start]4. Discusión sobre si el Método Aplicado Podría o No Incorporarse al Proyecto Final [cite: 40]

**El análisis de K-Means y sus resultados deben incorporarse al proyecto final.**

[cite_start]La segmentación no supervisada mejora la solidez del modelo supervisado y la toma de decisiones, cumpliendo con la capacidad de vincular los hallazgos con el modelo de *scoring* supervisado[cite: 44].

### [cite_start]Razones Técnicas para la Incorporación: [cite: 45]

* **Ingeniería de Características:** La **etiqueta de *cluster*** (`CLUSTER_LABEL`) puede ser añadida como una **variable categórica adicional** (*feature*) al *dataset* de entrenamiento del modelo supervisado. Dado que esta variable resume la información de perfil y está fuertemente correlacionada con el riesgo (tasas de morosidad dispares), mejorará el poder predictivo del modelo final.
* **Tratamiento de Anomalías y Sesgos:** El **Cluster 2 (Desempleados)** es una señal de alerta clara.
    * Estos 22 casos (o cualquier otro cliente clasificado en este *cluster* de alto riesgo) pueden ser **excluidos del entrenamiento** para evitar que sesguen la optimización del modelo base.
    * Alternativamente, cualquier solicitud clasificada en el Cluster 2 podría ser enviada directamente a **rechazo automático o revisión manual**, independientemente de lo que prediga el modelo de *scoring* supervisado, debido a su riesgo intrínseco.