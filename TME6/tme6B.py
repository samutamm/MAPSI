# -*- coding: utf-8 -*-

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

#data = pkl.load(file("TME6_lettres.pkl","rb"))
with open('TME6_lettres.pkl', 'rb') as f:
    data = pkl.load(f, encoding='latin1') 
X = np.array(data.get('letters')) # récupération des données sur les lettres
Y = np.array(data.get('labels')) # récupération des étiquettes associées

# affichage d'une lettre
def tracerLettre(let):
    a = -let*np.pi/180; # conversion en rad
    coord = np.array([[0, 0]]); # point initial
    for i in range(len(a)):
        x = np.array([[1, 0]]);
        rot = np.array([[np.cos(a[i]), -np.sin(a[i])],[ np.sin(a[i]),np.cos(a[i])]])
        xr = x.dot(rot) # application de la rotation
        coord = np.vstack((coord,xr+coord[-1,:]))
    plt.figure()
    plt.plot(coord[:,0],coord[:,1])
    plt.savefig("exlettre.png")
    return

def discretise(Xcont, d):
    intervalle = 360 / d;
    return np.array([np.floor(x/intervalle) for x in Xcont]);

def groupByLabel( y):
    index = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        index.append(ind)
    return index;

def learnMarkovModel(Xc, d):
    #A = np.zeros((d,d))
    #Pi = np.zeros(d)
    A = np.ones((d,d))
    Pi = np.ones(d)
    for element in Xc:
        last = np.int(element[0]);
        Pi[last] += 1;
        for i in range(1, element.size):
            current = np.int(element[i]);
            A[last][current] += 1.0;
            last = current;
    
    A_norm = A/np.maximum(A.sum(1).reshape(d,1),1) # normalisation
    Pi = Pi/Pi.sum()
    return(Pi, A_norm)
    
def probaSequence(s,Pi,A):
    last = np.int(s[0]);
    proba = np.log(Pi[last]);
    for i in range(1,len(s)):
        current = np.int(s[i]);
        proba += np.log(A[last][current]);
        last = current;
    return proba;

def calculProba(d, Xlocal1, Ylocal1, Xlocal2):   
    Xd1 = discretise(Xlocal1,d)  # application de la discrétisation
    Xd2 = discretise(Xlocal2,d)
    index = groupByLabel(Ylocal1)  # groupement des signaux par classe
    models = []
    for cl in range(len(np.unique(Ylocal1))): # parcours de toutes les classes et optimisation des modèles
        models.append(learnMarkovModel(Xd1[index[cl]], d))
    
    proba = np.array([
        [probaSequence(Xd2[i], models[cl][0], models[cl][1]) for i in range(len(Xlocal2))] 
        for cl in range(len(np.unique(Ylocal1)))
    ])
    return proba

### Evaluation des performances ###

# Char à numero
proba = calculProba(20, X, Y, X)
Ynum = np.zeros(Y.shape);
for num,char in enumerate(np.unique(Y)):
    Ynum[Y==char] = num;
    
pred = proba.argmax(0);
print("teste avec le méme donnée")
print(np.where(pred != Ynum, 0.,1.).mean())

### Evalution juste ###

# separation app/test, pc=ratio de points en apprentissage
def separeTrainTest(y, pc):
    indTrain = []
    indTest = []
    for i in np.unique(y): # pour toutes les classes
        ind, = np.where(y==i)
        n = len(ind)
        indTrain.append(ind[np.random.permutation(n)][:int(np.floor(pc*n))])
        indTest.append(np.setdiff1d(ind, indTrain[-1]))
    return indTrain, indTest
# exemple d'utilisation
itrain,itest = separeTrainTest(Y,0.8)

def iTrain():
    ia = []
    for i in itrain:
        ia += i.tolist()    
    it = []
    for i in itest:
        it += i.tolist()
    return (ia, it)
    
ia,it=iTrain()

# Char à numero
Ynum = np.zeros(Y[it].shape);
for num,char in enumerate(np.unique(Y)):
    Ynum[Y[it]==char] = num;
    
# change dimension et regarde l'effet sur les resultats
predictions = [] 
for d in range(10,90,10):
    Xd_test = discretise(X[it], d)
    
    proba = calculProba(d, X[ia], Y[ia], X[it])
    
    pred = proba.argmax(0);
    print("teste avec le separation à training et test, d =",d)
    print(np.where(pred != Ynum, 0.,1.).mean())
    predictions.append(pred);

def visualise_conf(verite, prediction, Ylocal):
    conf = np.zeros((26,26))
    for cl in range (len(Ynum)):
        v = int(verite[cl])
        p = prediction[cl]    
        conf[v][p] += 1
        
    plt.figure()
    plt.imshow(conf, interpolation='nearest')
    plt.colorbar()
    plt.xticks(np.arange(26),np.unique(Ylocal))
    plt.yticks(np.arange(26),np.unique(Ylocal))
    plt.xlabel(u'Vérité terrain')
    plt.ylabel(u'Prédiction')
    plt.savefig("mat_conf_lettres.png")

def prochaineEtat(dist):
    r = np.random.uniform()
    cs = dist.cumsum()
    i = 0
    while(cs[i] < r):
        i += 1
    return i

def generate(pi, model, N):
    liste = []
    current = prochaineEtat(pi)
    for i in range(N):
        liste.append(current)
        dist = model[current]
        current = prochaineEtat(dist)
    return liste

#Evaluation qualitative
pred = predictions[2] # d = 20
visualise_conf(Ynum, pred, Y);

#Modèle génératif
newa = generate(models[0][0],models[0][1], 25) # generation d'une séquence d'états
intervalle = 360./d # pour passer des états => valeur d'angles
newa_continu = np.array([i*intervalle for i in newa]) # conv int => double
tracerLettre(newa_continu)
