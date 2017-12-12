#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

TME 9

"""

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

def tirage(m):
    return [np.random.uniform(-1,1) * m,np.random.uniform(-1,1) * m]

def monteCarlo(N):
    tirages = np.array([tirage(1) for i in range(N)])    
    X = tirages[:,0]
    Y = tirages[:,1]    
    pi = 4 * (np.where(np.sqrt(X**2 + Y**2) <= 1)[0].size) / N
    return (pi,X,Y)

#plt.figure()

# trace le carré
plt.plot([-1, -1, 1, 1], [-1, 1, 1, -1], '-')

# trace le cercle
x = np.linspace(-1, 1, 100)
y = np.sqrt(1- x*x)
plt.plot(x, y, 'b')
plt.plot(x, -y, 'b')

# estimation par Monte Carlo
pi, x, y = monteCarlo(int(1e4))

# trace les points dans le cercle et hors du cercle
#dist = x*x + y*y 
#plt.plot(x[dist <=1], y[dist <=1], "go")
#plt.plot(x[dist>1], y[dist>1], "ro")
#plt.show()

# si vos fichiers sont dans un repertoire "ressources"
with open("./countWar.pkl", 'rb') as f:
    (count, mu, A) = pkl.load(f, encoding='latin1')
    
with open("./fichierHash.pkl", 'rb') as f:
    chars2index = pkl.load(f, encoding='latin1')

with open("./secret.txt", 'r') as f:
    secret = f.read()[0:-1] # -1 pour supprimer le saut de ligne
 
def swapF(d1):
    d2 = {}
    for cle in d1.keys():
        d2[d1[cle]] = cle
    return d2
     
tau = {'a' : 'b', 'b' : 'c', 'c' : 'a', 'd' : 'd' }
print(swapF(tau))

def decrypt(mess, d1):
    ret_str = ""
    for c in mess:
        nouvel_char = d1[c];
        ret_str = ret_str + nouvel_char
    return ret_str
 
#le prochaine ne fonctionne pas :D
#chars2index = dict(zip(np.array(list(count.keys())), np.arange(len(count.keys()))))

def  logLikelihood(mess, mu, A, chars2index):
    proba = 0
    precedant = chars2index[mess[0]]
    proba = proba + np.log(mu[precedant])
    for i in range(1, len(mess)):
        new = chars2index[mess[i]]
        proba = proba + np.log(A[precedant, new])
        precedant = new
    return proba
    
print("abcd = ", logLikelihood( "abcd", mu, A, chars2index ))
print("dcba = ", logLikelihood( "dcba", mu, A, chars2index ))

def MetropolisHastings(mess, mu, A, tau, N, chars2index):
    return "TODO"