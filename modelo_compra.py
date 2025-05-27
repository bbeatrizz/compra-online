#!/usr/bin/env python
# coding: utf-8

# # Caso practico. Aprendizaje automático

# ## Online Shoppers Purchasing Intention

# Trabajamos como científicos de datos para una empresa de retail que, debido al cambio en los hábitos de consumo de los clientes, está potenciando ampliamente el servicio de venta online. La empresa quiere realizar un modelo de aprendizaje automático para clasificar a los clientes en función de la probabilidad de generar ingresos al comprar en la web. El objetivo es realizar una serie de acciones específicas para los clientes que es más probable que hagan compras en la web.

# ## Carga de librerías:

# In[57]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from IPython.core.display import display, HTML
import warnings
warnings.filterwarnings('ignore')


# ## Carga de datos:

# In[2]:


XY = pd.read_csv('online_shoppers_intention.csv', sep=',', index_col=False)
XY.head()


# In[3]:


print('El número de filas en el dataset es: {}'.format(XY.shape[0]))
print('El número de columnas en el dataset es: {}'.format(XY.shape[1]))
print('Nombre de las variables: {}'.format(list(XY.columns)))


# ## Procesamiento de datos:

# ## Missings: 

# In[4]:


XY.isnull().sum()


# Comprobamos que no hay ningún valor faltante por lo que continuamos con el procesamiento de datos.

# ## De categóricas a numéricas:

# Para poder introducir las variables al modelo neceistamos convertirlas de categóricas a numéricas, para ello disponemos de diferentes modelos:

# In[5]:


# Sacamos la lista de variables categóricas
XY.select_dtypes(exclude=['number']).columns


# ### month

# In[6]:


XY['Month'].value_counts()


# La variable meses no la cambiamos a numérica, sin embargo cambiaremos el mes de junio que esta mal abreviado y ajustaremos el orden de los meses

# In[7]:


XY.loc[XY['Month'] == 'June', 'Month'] = 'Jun'


# In[8]:


nuevo_orden = ['Feb', 'Mar', 'May', 'Jun','Jul', 'Aug', 'Sep',
               'Oct', 'Nov', 'Dec']

#la convertimos en categórica

XY['Month'] = pd.Categorical(XY['Month'], categories = nuevo_orden, ordered = True)


# In[9]:


XY.groupby('Month').agg('count')


# ### Visitor Type

# In[10]:


XY['VisitorType'].value_counts()


# In[11]:


# En este caso creamos dummies para la variable categórica
dummy_df = pd.get_dummies(XY['VisitorType'], prefix='VisitorType')
XY = pd.concat([XY,dummy_df], axis=1)
XY = XY.drop('VisitorType', axis=1)
XY


# ### Weekend

# Las variables booleanas las cambiamos con astype

# In[12]:


XY['Weekend'].value_counts()


# In[13]:


XY['Weekend'] = XY['Weekend'].astype(int)


# In[14]:


XY['Weekend'].value_counts()


# ### Revenue

# Codificamos la variable target que en nuestro caso se llama Revenue

# In[15]:


XY['Revenue'].value_counts()


# In[16]:


#Lo transdormamos a un número entero para modificarlo directamente en 0 y 1.
XY['Revenue'] = XY['Revenue'].astype(int)


# In[17]:


XY['Revenue'].value_counts()


# In[18]:


#Confirmamos que no hay más tipos númericos
XY.select_dtypes(exclude=['number']).columns


# ## Visualizaciones y correlaciones

# In[19]:


XY.describe()


# ### Division del dataset en features y target

# In[20]:


# Eliminamos también la variable Month para poder tener solo las variables numéricas
X = XY.drop(columns = ['Revenue','Month'], axis=1)
Y = XY['Revenue']


#  ### Boxplots:

# In[21]:


# Para poder tener una buena visualización del boxplot normalizamos
X_normalizado = (X-X.mean())/X.std()
X_normalizado.head()


# In[22]:


plt.figure(figsize=(15,7))
ax= sns.boxplot(data= X_normalizado)
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.title('Gráfico cajas y bigotes de las variables independientes X')
plt.ylabel('Variable normalizada')
_ = plt.xlabel('Nombre de las variables')


# Como observamos todas las variables tienen valores atípicos o outliers, lo que nos puede afectar si el modelo elegido es sensible a dichos valores.
# 

# ### Histogramas: 

# In[23]:


plt.figure(figsize=(18,20))
n = 0
for i, column in enumerate(X.columns):
    n+=1
    plt.subplot(6, 3, n)
    sns.distplot(X[column], bins=30)
    plt.title('Distribución var {}'.format(column))
plt.show()


# Las variables no parecen seguir una distribución normal o Gaussianas.

# ### Matriz de correlaciones:

# In[24]:


matriz_correlaciones = XY.corr(method = 'pearson')
n_ticks = len(XY.columns)
plt.figure( figsize=(9, 9) )
plt.xticks(range(n_ticks), XY.columns, rotation='vertical')
plt.yticks(range(n_ticks), XY.columns)
plt.colorbar(plt.imshow(matriz_correlaciones, interpolation='nearest', 
                            vmin=-1., vmax=1., 
                            cmap=plt.get_cmap('Reds')))
_ = plt.title('Matriz de correlaciones de Pearson')


# In[25]:


correlaciones_Revenue = matriz_correlaciones.values[ -1, : -1]
indices_inversos =  abs(correlaciones_Revenue[ : ]).argsort()[ : : -1]

diccionario = {}

for nombre, correlacion in zip( X.columns[indices_inversos], list(correlaciones_Revenue[indices_inversos] ) ):
    diccionario[nombre] = correlacion
pd.DataFrame.from_dict(diccionario, orient='index', columns=['Correlación con Revenue'])


# ## Estandarización de los datos: 

# In[26]:


# Es un proceso necesario para algunos modelos
obj_escalar = StandardScaler()
X_estandarizado = obj_escalar.fit_transform(X)


# ## División en train y test

# Dividimos en dos conjuntos, uno para poder entrenar los modelos(train) y otro (test) para poder validar al final y poder obtener una métrica que podamos usar para ver cual es el mejor modelo.

# In[27]:


X_train, X_test, Y_train, Y_test = train_test_split(X_estandarizado, Y, test_size=0.2, random_state=0)


# # Modelo Lineal - Regresión Logística

# In[30]:


# Ajustamos al modelo con los parametros por defecto
log_regression = LogisticRegression()
log_regression.fit(X_train, Y_train)


# ### Optimizando C y class_weight con GridSearch

# Este método se utiliza para optimizar los parámetros aplicando todas sus combinaciones posibles. Hay diferentes parámetros que se pueden aplicar, en este caso usaremos C y Class_weight para controlar el desequilibrio de clases.

# In[35]:


parametros = {"C": [0., 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008,0.009], 
              "class_weight":['balanced', None]}
y_test_pred_prob = modelo_gs.predict_proba(X_test) = GridSearchCV(log_regression, param_grid = parametros,
                        cv= 5, scoring='roc_auc')
log_regression_gs.fit(X_train, Y_train)


# In[36]:


# Mostramos los mejores parámetros:
print(log_regression_gs.best_params_, "\nROC AUC: {}".format(round(log_regression_gs.best_score_,2)))


# In[39]:


# Calculamos las probabilidades en el conjunto de Test y representamos la 
# curva ROC de dichas predicciones

y_test_pred_prob = log_regression_gs.predict_proba(X_test)
preds = y_test_pred_prob[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

plt.figure(figsize=(10,7))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[72]:


y_umbralizadas = 1*(y_test_pred_prob[:, 1] > 0.5)

print(u"Matriz de confusión\n", metrics.confusion_matrix(Y_test, y_umbralizadas))
print("\nAccuracy\t{}".format(round(metrics.accuracy_score(Y_test, y_umbralizadas),2)))  
print("Sensitividad\t{}".format(round(metrics.recall_score(Y_test, y_umbralizadas),2)))
print(u"Precisión\t{}".format(round(metrics.precision_score(Y_test, y_umbralizadas),2)))


# # Modelo de Redes Neuronales - MLPClassifier

# Se usa la librería sklearn en la cual encontramos un clasificador llamado MLPClassifier que es lo que se denomina Multi-layer Perceptron Classifier. 

# In[41]:


MLPC_modelo = MLPClassifier(random_state=1, max_iter=100)


# In[42]:


# Ajustamos el modelo
MLPC_modelo.fit(X_train, Y_train)


# ### Mejores parámetros: 

# In[43]:


# Buscamos los mejores parámetros

parametros = {'solver': ['lbfgs'], 
              'max_iter': [100,200,300,500], 
              'alpha': 10.0 ** -np.arange(1, 3), #
              'hidden_layer_sizes':np.arange(30, 35),
              'random_state':[0]}
MLPC_modelo_gs = GridSearchCV(MLPC_modelo, param_grid = parametros, cv =3,
                             scoring='roc_auc', n_jobs=-1, verbose=10)
MLPC_modelo_gs.fit(X_train, Y_train)


# In[44]:


# Mostramos los mejores parámetros:
print(MLPC_modelo_gs.best_params_, "\nROC AUC: {}".format(round(MLPC_modelo_gs.best_score_,2)))


# In[45]:


MLPC_modelo_mejor = MLPClassifier(**MLPC_modelo_gs.best_params_, verbose=10)


# In[46]:


MLPC_modelo_mejor.fit(X_train, Y_train)


# In[48]:


y_test_pred_prob_2 = MLPC_modelo_mejor.predict_proba(X_test) 

preds = y_test_pred_prob_2[:,1]
fpr, tpr, threshold = metrics.roc_curve(Y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

plt.figure(figsize=(10,7))
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# ### Umbralizar las predicciones: 

# In[70]:


umbral = 0.5
y_umbralizadas = 1*(y_test_pred_prob_2[:, 1] > umbral)


# In[71]:


print(u"Matriz de confusión\n", metrics.confusion_matrix(Y_test, y_umbralizadas))
print("\nAccuracy\t{}".format(round(metrics.accuracy_score(Y_test, y_umbralizadas),2)))  
print("Sensitividad\t{}".format(round(metrics.recall_score(Y_test, y_umbralizadas),2)))
print(u"Precisión\t{}".format(round(metrics.precision_score(Y_test, y_umbralizadas),2)))  


# # Modelo Clasificación - Random forest

# In[50]:


RForest_model = RandomForestClassifier()
RForest_model.fit(X_train, Y_train)


# ### Mejores parámetros: 

# In[53]:


parametros = {'n_estimators': [10, 50, 100, 200, 300],
             'max_depth': [5, 7, 9, 11],
             'max_features': [10, 'sqrt']}
RForest_model_gs = GridSearchCV(RForest_model, parametros, cv = 3, n_jobs=-1)
RForest_model_gs.fit(X_train, Y_train)


# In[54]:


# Mostramos los mejores parámetros:
print(RForest_model_gs.best_params_, "\nROC AUC: {}".format(round(RForest_model_gs.best_score_,2)))


# In[73]:


y_test_pred_prob_3 = RForest_model_gs.predict_proba(X_test)
y_umbralizadas = 1*(y_test_pred_prob_3[:, 1] > 0.5)

print(u"Matriz de confusión\n", metrics.confusion_matrix(Y_test, y_umbralizadas))
print("\nAccuracy\t{}".format(round(metrics.accuracy_score(Y_test, y_umbralizadas),2)))  
print("Sensitividad\t{}".format(round(metrics.recall_score(Y_test, y_umbralizadas),2)))
print(u"Precisión\t{}".format(round(metrics.precision_score(Y_test, y_umbralizadas),2)))


# ## Accuracy de los modelos

# In[55]:


train_scores = {
    'Regresión Logística': log_regression_gs.score(X_train, Y_train),
    'MLPClassifier': MLPC_modelo_mejor.score(X_train, Y_train),
    'Random Forest': RForest_model_gs.score(X_train, Y_train),
}

test_scores = {
    'Regresión Logística': log_regression_gs.score(X_test, Y_test),
    'MLPClassifier': MLPC_modelo_mejor.score(X_test, Y_test),
    'Random Forest': RForest_model_gs.score(X_test, Y_test),
}


# In[56]:


train_df = pd.DataFrame(train_scores.items(), columns=['Algoritmo', 'Train Score'])
test_df = pd.DataFrame(test_scores.items(), columns=['Algoritmo', 'Test Score'])

score_df = pd.merge(train_df, test_df, on='Algoritmo')
display(score_df)


# ## Validación cruzada para cada modelo

# In[67]:


RLogistica_sores = cross_val_score(log_regression_gs, X_test, Y_test, cv=10)
MLP_scores = cross_val_score(MLPC_modelo_mejor,X_test, Y_test, cv=10)
RForest_scores = cross_val_score(RForest_model_gs, X_test, Y_test, cv=10)


# In[66]:


print('- El valor de Validación Cruzada del modelo de Regresión logistica es: {}'
      .format(RLogistica_sores.mean()),
     '\n- El valor de Validación Cruzada del modelo de MLPClassifier es: {}'
      .format(MLP_scores.mean()),
      '\n- El valor de Validación Cruzada del modelo de Random Forest es: {}'
      .format(RForest_scores.mean())
     )


# Observando los valores tras analizar los modelos observamos que el algoritmo de Random Forest es el que mejores datos nos da, con una precisión del 0,74 y una exactitud del 0,89.

# In[ ]:




