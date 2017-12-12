#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle as pkl
#from sklearn import hmm # import obligatoire
from markov_tools import *
import matplotlib.pyplot as plt

with open('genome_genes.pkl', 'rb') as f:
        data = pkl.load(f, encoding='latin1')
#data = pkl.load(file("genome_genes.pkl","rb"))

Xgenes  = data.get("genes") #Les genes, une array de arrays
Genome = data.get("genome") #le premier million de bp de Coli
Annotation = data.get("annotation") ##l'annotation sur le genome
##0 = non codant, 1 = gene sur le brin positif

### Quelques constantes
DNA = ["A", "C", "G", "T"]
stop_codons = ["TAA", "TAG", "TGA"]

#Ex2 Sous question 1
#l'esperance de la loi géométrique est 1 / p donc pour p = a :
a = 1.0 / 200
#Sous question 2
#Pour b, on  prend la longueur moyenne d'un gène :
longueur=0
for gene in Xgenes:
    longueur += len(gene)
longueur = longueur / len(Xgenes)
b = 3.0 / longueur #idem a avec taille 3 genes :

#Ex2 sous question 3, distribution des 4 lettres sur le Genome
def distriInter(Genome):
    dist = np.zeros((1,4))
    for lettre in Genome:
        dist[0][lettre] += 1
    return dist / len(Genome)

Binter = distriInter(Genome)
print(Binter)

#Ex2 sous question 4, distribution des codons
#note : 1 codon = 3 genes successifs. Donc 3 valeurs et 4 lettres
def distriCodons(Xgenes, n):
    dist = np.zeros((3,n))
    for gene in Xgenes:
        for i in range(3, len(gene)-3):#on ignore le 1er et dernier codon
            index = i % 3 #
            dist[index][gene[i]] += 1
    return np.array([line/np.sum(line) for line in dist])

Bgene = distriCodons(Xgenes, 4)
print(Bgene)

#Ex2 sous question 5, donnée par l'enoncée
Pi_m1 = np.array([1, 0, 0, 0])
A_m1 =  np.array([[1-a, a  , 0, 0],
                  [0  , 0  , 1, 0],
                  [0  , 0  , 0, 1],
                  [b  , 1-b, 0, 0]])

B_m1 = np.vstack((Binter, Bgene))
print("B_m1")
print(B_m1)
pred, vsbce = viterbi(Genome,Pi_m1,A_m1,B_m1)
#vsbce contient la log vsbce
#pred contient la sequence des etats predits (valeurs entieres entre 0 et 3)

#on peut regarder la proportion de positions bien predites
#en passant les etats codant a 1
sp = pred
sp[np.where(sp>=1)] = 1
percpred1 = float(np.sum(sp == Annotation) )/ len(Annotation)

print('percpred1 = ' + repr(percpred1))

positions = pred[range(6000)]
verite = Annotation[range(6000)]
plt.plot(list(range(6000)),  positions, "r.", label='prediction')
plt.plot(list(range(6000)),  verite, "b.", label='annotation')
plt.legend()
plt.show()

#Ex3 sous question 1
#modèle avec codon start et codon stop, ecrire la matrice

#creer une ligne de taille taille remplie de 0 sauf à index avec valeur val1
def cligne(index, val1, taille):
    A = [0 for i in range(taille)]
    A[index] = val1
    return A

#creer une ligne de taille taille remplie de 0 
#avec ind1 avec valeur val1 et ind2 avec valeur val2
def cligne2(ind1, val1, ind2, val2, taille):
    A = [0 for i in range(taille)]
    A[ind1] = val1
    A[ind2] = val2
    return A

t = 12
Pi_m2 = np.array(cligne(0,1,t))
#A_m2 = np.array([cligne2(0, 1-a,1,a, t),cligne(2,1,t),cligne(3,1,t),cligne(4,1,t),
#                 cligne(5,1,t),cligne(6,1,t),cligne2(4,1-b,7,b,t),cligne2(8,0.5,9,0.5,t),
#                 cligne2(10,0.5,11,0.5,t), cligne(11,1,t),cligne(0,1,t),cligne(0,1,t) 
#])
A_m2 = np.array([[1-a,a,0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0,1-b,0, 0, b, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 0, 0, 0,0.5,0.5,0,0],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,0.5,0.5],
                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

#print('Pi_m2 : ', Pi_m2)
#print('A_m2 : ', A_m2)

#Codon start : ATG, GTG, TTG
Bstart = np.array([[0.83, 0, 0.14, 0.03],
                   [0, 0, 0, 1],
                   [0, 0, 1, 0]])

#Codon stop : TAA, TAG, TGA
Bstop = np.array([[0, 0, 0, 1], #T 1er
                  [1, 0, 0, 0], #A 2e
                  [0, 0, 1, 0], #G 
                  [0, 0, 1, 0], #G
                  [1, 0, 0, 0]])#A

B_m2 = np.vstack((Binter, Bstart, Bgene, Bstop))
#print(B_m2)
pred2, vsbce2 = viterbi(Genome,Pi_m2,A_m2,B_m2)

sp2 = pred2
sp2[np.where(sp2>=1)] = 1
percpred2 = float(np.sum(sp2 == Annotation) )/ len(Annotation)
print('percpred2 =', percpred2)
#Amelioration peu marquante, comment expliquez-vous cela?

#Ex3 sous question 4
#Graphe
positions2 = pred2[range(6000)]
plt.plot(list(range(6000)),  positions, "r.", label='prediction 1')
plt.plot(list(range(6000)),  positions2, "g.", label='prediction 2')
plt.plot(list(range(6000)),  verite, "b.", label='annotation')
plt.legend()
plt.show()

#Ex4 (Optionnel)
#Necessite la bibliotèque qui ne marche pas...

#model3 = hmm.MultinomialHMM(n_s_m2, n_iter = 100, thresh = 1e-6,   #les paramètres de la  
#                            params = "es")    #l'estimation est faite pour les emission "e" et le start "s"
#model3._set_transmat(A_m2)
#model3.fit(G)
#
#model3.transmat_  # afficher la matrice estimée