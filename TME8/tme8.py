#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle as pkl
#from sklearn import hmm # import obligatoire

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
b = 1.0 / longueur #idem a :
"""
#enoncee
n_states_m1 = 4
# syntaxe objet python: créer un objet HMM
model1 = hmm.MultinomialHMM(n_components = n_states_m1)

Pi_m1 = np.array([1, 0, 0, 0]) ##on commence dans l'intergenique
A_m1 = np.array([[1-a, a  , 0, 0], 
                 [0  , 0  , 1, 0],
                 [0  , 0  , 0, 1],
                 [b  , 1-b, 0, 0 ]])

# paramétrage de l'objet 
model1._set_startprob(Pi_m1)
model1._set_transmat(A_m1)
# [cf question d'après pour la détermination]
#model1._set_emissionprob(B_m1)
"""
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
def distriCodons(Xgenes):
    dist = np.zeros((3,4))
    for gene in Xgenes:
        for i in range(3, len(gene)-3):#on ignore le 1er et dernier codon
            index = i % 3 #
            dist[index][gene[i]] += 1
    return np.array([line/np.sum(line) for line in dist])

Bgene = distriCodons(Xgenes)
print(Bgene)

#Ex2 sous question 5, donnée par l'enoncée
"""B_m1 = np.vstack((Binter, Bgene))

model1._set_emissionprob(B_m1)

vsbce, pred = model1.decode(Genome) 
#vsbce contient la log vsbce
#pred contient la sequence des etats predits (valeurs entieres entre 0 et 3)

#on peut regarder la proportion de positions bien predites
#en passant les etats codant a 1
sp = pred
sp[np.where(sp>=1)] = 1
percpred1 = float(np.sum(sp == Annotation) )/ len(Annotation)

print(percpred1)
#Out[10]:  0.636212

#Ex2 sous question 6, afficher graphique la classification de la CMC avec 
# l'annotation pour les 6000 premières positions du génome

"""


#Ex3 sous question 1
#modèle avec codon start et codon stop

#creer une ligne de taille taille remplie de 0 sauf à index avec valeur 1
def creerligne(index, taille):
    A = [0 for i in range(taille)]
    A[index] = 1
    return A

taille2 = 12
Pi_m2 = np.array(creerligne(0,taille2))
print('Pi_m2 : ', Pi_m2)


