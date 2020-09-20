#!/usr/bin/env python
# coding: utf-8

# **Attraverso la libreria Scikit-learn si costruisce un modello di apprendimento automatico utilizzando l'algoritmo k-Nearest Neighbors per prevedere se i pazienti esaminati hanno il diabete o meno. **

# In[1]:


#importazione librerie 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')


# In[2]:


#caricamento del dataset
df = pd.read_csv('diabetes.csv')

#prime 5 righe del dataset
df.head()


# In[3]:


#struttura
df.shape


# Si hanno 768 righe e 9 colonne. Le prime 8 colonne rappresentano le feature , l'ultima è l'etichettatura. L'etichetta 1 
# indica che la persona in questione ha il diabete 0 altrimenti.

# In[4]:


#creazione della matrice di feature e target
X = df.drop('Outcome',axis=1).values
y = df['Outcome'].values


# Divisione dei dati in training set e test set. 
# Il classificatore viene addestrato sul training set e le previsioni sul test set. Poi vengono confrontste le previsioni con le etichette note.
# 

# In[5]:


#importing train_test_split
from sklearn.model_selection import train_test_split


# La dimesione del test set è pari al 40% del dataset totale.

# In[6]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42, stratify=y)


# creazione di un classificatore mediante k-Nearest Neighbors.
# 
# Osservazione dell'accuratezza per differnti valori di k.

# In[7]:


#import KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier

#memorizzazioen del training e test accuracies
neighbors = np.arange(1,9)
train_accuracy =np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

for i,k in enumerate(neighbors):
   
    knn = KNeighborsClassifier(n_neighbors=k)
    
    
    knn.fit(X_train, y_train)
    
    #accuratezza del training set
    train_accuracy[i] = knn.score(X_train, y_train)
    
    #accuratezza del test set
    test_accuracy[i] = knn.score(X_test, y_test) 


# In[8]:


#Generate plot
plt.title('k-NN variazione del numero di vicini')
plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
plt.plot(neighbors, train_accuracy, label='Training accuracy')
plt.legend()
plt.xlabel('numero di vicini')
plt.ylabel('accuratezza')
plt.show()


# La massima accuratezza di ha per k=7

# In[9]:



knn = KNeighborsClassifier(n_neighbors=7)


# In[10]:



knn.fit(X_train,y_train)


# In[11]:


#accuratezza
knn.score(X_test,y_test)


# **Matrice di confusione**
# 

# In[12]:


#import confusion_matrix
from sklearn.metrics import confusion_matrix


# In[13]:


#predizioni ottenute con il classificatore di cui sopra
y_pred = knn.predict(X_test)


# In[14]:


confusion_matrix(y_test,y_pred)


# Dalla matrice di confusione di ottengono i seguenti dati:
# 
# True negative = 165 ->pazienti sani , classificati sani
# 
# False positive = 36 ->pazienti sani classificati malati
# 
# True postive = 60 -> pazienti malati classificati malati
# 
# Fasle negative = 47 ->pazienti malati classificati sani

# matrice di confusione ottenuta con pandas

# In[15]:


pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True)


# **Classification Report di Scikit-learn**
# 
# 

# In[16]:


#import classification_report
from sklearn.metrics import classification_report


# In[17]:


print(classification_report(y_test,y_pred))


# **ROC (Reciever Operating Charecteristic) curve**
# 
# Da una curva ROC è possibile notare:
# 
# 1) tradeoff trasensibilità e specificità, un incremento della sensibilità è accompagnato da un decremento della specificità.
# 
# 2) Più la curva segue il bordo superiore dello spazio ROC, più accurato è il test
# 
# 3)Più la curva si avvicina alla diagonale di 45 gradi dello spazio ROC, meno preciso è il test.
# 
# 4) L'area sotto la curva è un misura dell'accuratezza.

# In[18]:


y_pred_proba = knn.predict_proba(X_test)[:,1]


# In[19]:


from sklearn.metrics import roc_curve


# In[20]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)


# In[21]:


plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label='Knn')
plt.xlabel('fpr')
plt.ylabel('tpr')
plt.title('Knn(n_neighbors=7) ROC curve')
plt.show()


# In[22]:


#area sotto la curca ROC
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test,y_pred_proba)


# **Cross Validation**
# Le prestazioni del modello dipendono dal modo in cui i dati vengono suddivisi. Non sappiamo nulla della capacità di generalizzare del modello.
# La cross validation viene utilizzata per ottenera una misura più accurata dell'efficacia del modello.
# In particolare utiliizzando k fold cross validation il campione originale viene suddiviso in modo casuale in k sottocampioni di uguale dimensione. 
# Di questi, un singolo sottocampione viene conservato come dati di convalida per testare il modello e i restanti k-1 sottocampioni vengono utilizzati come dati di addestramento. 
# Il processo di cross validation viene quindi ripetuto k volte , con ciascuno dei k sottocampioni utilizzato esattamente una volta come dati di convalida. I k risultati delle pieghe possono quindi essere mediati (o altrimenti combinati) per produrre una singola stima.
# Tutte le osservazioni vengono utilizzate sia per l'addestramento che per la convalida e ogni osservazione viene utilizzata per la convalida esattamente una volta.

# **Scelta dell'iperparametro k**
# 
# Ci sono metodi più sofisticati per la scelta di k come iperparametro, rispetto a quello sopra utilizzato. 
# 
# Il modo migliore, consiste nel testare una serie di valori di k e scegliere quello che offre le 
# prestazioni migliori selezionandoli attraverso la cross validation.
# 
# Questo può essere agevolmete fatto utilizzando Scikit-learn e quindi Grid Search cross-validation.
# 
# 

# In[23]:


#import GridSearchCV
from sklearn.model_selection import GridSearchCV


# In[24]:


#nel caso di knn il parametro da regolare è n_neighbors
param_grid = {'n_neighbors':np.arange(1,50)}


# In[25]:


knn = KNeighborsClassifier()
knn_cv= GridSearchCV(knn,param_grid,cv=5)
knn_cv.fit(X,y)


# In[26]:


knn_cv.best_score_


# In[27]:


knn_cv.best_params_


# Il miglior parametro è k=14 con una precisione del 76%

# --end--

# In[ ]:





# In[ ]:




