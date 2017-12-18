#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""

TME 10

"""

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import pdb # debugging

def tirages(N, a, b, sig):
    X = np.random.rand(N)
    e = sig * np.random.randn(N)
    return X, a*X + b + e

#yquad=ax^2+bx+c+ϵ, ϵ∼N(0,σ)
def tiragesQuadratique(N, a, b, c, sig):
    X = np.random.rand(N)
    e = sig * np.random.randn(N)
    return X, a*X**2 + b*X + c + e

a = 6.
b = -1.
N = 100
sig = .4
X, Y = tirages(N, a, b, sig)

#Estimation de parametres probabilistes
#Lineaire : ax+b
def estimationParamProba(X, Y, sig):
    cov = np.cov([X,Y])#/sig**2#comme ratio sig n'annule apres...
    a_proba = cov[0,1]/cov[0,0]
    b_proba = Y.mean() - a_proba * X.mean()
    return a_proba, b_proba

a_proba, b_proba = estimationParamProba(X, Y, sig)
print("paramètres probabilistes:", a_proba, b_proba)

t=np.array([0,1])
plt.figure()
plt.scatter(X,Y)
plt.plot(t, a_proba*t + b_proba, 'r')
plt.show()

#Estimation au sens des moindres carres
Xmc = np.hstack((X.reshape(N,1), np.ones((N,1))))

def estimationMoindresCarres(X,Y):
    return np.linalg.solve(X.T.dot(X), X.T.dot(Y))

wstar = estimationMoindresCarres(Xmc, Y)
print("paramètres moindres carres:", wstar[0], wstar[1])

def testSansBiais(N,a,b):
    x,y=tirages(N, a, b, 0.)
    a_p, b_p = estimationParamProba(x, y, 0.)
    t=np.array([0,1])
    plt.figure()
    plt.scatter(x,y)
    plt.plot(t, a_p*t + b_p, 'r')
    #Les donnees ne se dispersent pas

#testSansBiais(N,a,b)

#---------------------------------------------#
#    Optimisation par descente de gradient    #
#---------------------------------------------#   

def coutFonction(X, Y, w):
    e = np.dot(X,w) - Y
    return np.dot(e.T, e)

def optimDescGradient(X, Y,w, eps=5e-3, N=30):
    h = 0.001
    #eps = 5e-3
    nIterations = N
    #w = np.zeros(X.shape[1]) # init à 0
    allw = [w]
    for i in range(nIterations):
        #dérivation
        derives = []
        for i in range(0, X.shape[1]):
            w2 = w.copy()
            w2[i] += h
            v = (coutFonction(X,Y,w2) - coutFonction(X,Y,w)) / h
            derives.append(v)
        derives = np.array(derives)
        w = w - eps * derives
        allw.append(w)
        #print(w)
    
    allw = np.array(allw)
    return allw

print("desc de gradient")
print(optimDescGradient(Xmc, Y, np.zeros(Xmc.shape[1]),5e-3)[-1])
print(optimDescGradient(Xmc, Y, np.zeros(Xmc.shape[1]), 8e-3)[-1]) #eps = 8e-3 souvent trop grand
print(optimDescGradient(Xmc, Y, np.zeros(Xmc.shape[1]),3e-3,)[-1])
print(optimDescGradient(Xmc, Y, np.ones(Xmc.shape[1]), 5e-3, 100)[-1])

def gradient_visualisation(X, Y, wstar, allw):
    # tracer de l'espace des couts
    ngrid = 20
    w1range = np.linspace(-0.5, 8, ngrid)
    w2range = np.linspace(-1.5, 1.5, ngrid)
    w1,w2 = np.meshgrid(w1range,w2range)
    
    cost = np.array([[np.log(((X.dot(np.array([w1i,w2j]))-Y)**2).sum()) for w1i in w1range] for w2j in w2range])
    
    plt.figure()
    plt.contour(w1, w2, cost)
    plt.scatter(wstar[0], wstar[1],c='r')
    plt.plot(allw[:,0],allw[:,1],'b+-' ,lw=2 )
    
allw = optimDescGradient(Xmc, Y, np.zeros(Xmc.shape[1]), 2e-3, 50)
gradient_visualisation(Xmc, Y, wstar, allw)

#----------------------------------------------------#
#    Extension non-linéaire (solution analytique)    #
#----------------------------------------------------#   

#yquad=ax2+bx+c+ϵ, ϵ∼N(0,σ)
c = 1
Xq, Yq = tiragesQuadratique(N, a, b, c, sig)

#Estimation au sens des moindres carres
Xqmc = np.hstack(((Xq**2).reshape(N,1), Xq.reshape(N,1), np.ones((N,1))))
wstar2 = estimationMoindresCarres(Xqmc, Yq)

print(wstar2)
x=np.linspace(0,1,N)
y_pred = wstar2[0]*(x**2) + (wstar2[1]*x) + wstar2[2]
plt.figure()
plt.scatter(Xq,Yq)
plt.plot(x, y_pred, 'r')
plt.show()

# R2 ne serve pas le modele non-linéaire 
#def R2(Y, Yp):
#    Ym = Y.mean()
#    ve = ((Y - Yp) ** 2).sum()
#    vr = ((Y - Ym) ** 2).sum()
#    return (ve / vr)

def reconstruction_error(Y, Yp):
    return np.sqrt(sum((Yq-Yp)**2))

print("error de reconstruction:")
print(reconstruction_error(Yq, y_pred))

#----------------------------------------------------#
#                   Données réelles                  #
#----------------------------------------------------#

red_wine = np.loadtxt("winequality/winequality-red.csv", delimiter=';', skiprows=1)
N,d = data.shape # extraction des dimensions
pcTrain  = 0.7 # 70% des données en apprentissage
allindex = np.random.permutation(N)
indTrain = allindex[:int(pcTrain*N)]
indTest = allindex[int(pcTrain*N):]
X = data[indTrain,:-1] # pas la dernière colonne (= note à prédire)
Y = data[indTrain,-1]  # dernière colonne (= note à prédire)
# Echantillon de test (pour la validation des résultats)
XT = data[indTest,:-1] # pas la dernière colonne (= note à prédire)
YT = data[indTest,-1]  # dernière colonne (= note à prédire)


