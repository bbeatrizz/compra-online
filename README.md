# 🛒 Online Shoppers Purchasing Intention – Machine Learning Classification Project

Este proyecto consiste en el desarrollo y comparación de modelos de aprendizaje automático supervisado para predecir si un usuario que visita una tienda online tiene intención de realizar una compra. 
Se ha utilizado el dataset **Online Shoppers Purchasing Intention**, que contiene variables de comportamiento, navegación y sesión de usuarios en un sitio web de comercio electrónico.

## 📌 Objetivo

Clasificar a los usuarios en función de la probabilidad de generar ingresos, con el fin de aplicar estrategias específicas de marketing y optimizar la conversión.

## 🧠 Técnicas utilizadas

- Regresión Logística
- Redes Neuronales Artificiales (MLPClassifier)
- Bosques Aleatorios (Random Forest)
- Validación cruzada (Cross-validation)
- Ajuste de hiperparámetros con GridSearchCV
- Métricas de evaluación: ROC AUC, accuracy, precisión, recall, matriz de confusión

## 🛠️ Herramientas y librerías

- Python
- pandas, numpy
- scikit-learn
- seaborn, matplotlib
- Jupyter Notebook

## 📊 Proceso

1. **Carga y exploración de datos**
2. **Tratamiento de variables categóricas y binarias**
3. **Análisis visual y correlacional**
4. **Estandarización y división en train/test**
5. **Entrenamiento de modelos y ajuste de hiperparámetros**
6. **Evaluación comparativa**
7. **Validación cruzada para ver estabilidad de los modelos**

## 📈 Resultados

El modelo que ofreció el mejor rendimiento general fue **Random Forest**, con una **precisión de 0.89** y una **exactitud (accuracy) de 0.74**. Se analizó su rendimiento con matriz de confusión y curvas ROC, confirmando que era el modelo más robusto frente al conjunto de test.

## 🗂️ Dataset

**Fuente**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)  
**Nombre**: `online_shoppers_intention.csv`

El dataset incluye 18 variables relacionadas con el comportamiento de navegación, duración de la sesión, tasa de rebote, visitas anteriores, tipo de visitante, día de la semana, entre otras.

## 🧾 Conclusiones

- La conversión en ventas puede modelarse eficazmente usando técnicas de clasificación.
- Es clave preprocesar bien los datos (normalización, codificación, selección de features).
- Random Forest mostró una mejor capacidad predictiva con un balance adecuado entre precisión y recall.
