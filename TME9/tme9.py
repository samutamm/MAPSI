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

with open("./secret2.txt", 'r') as f:
    secret2 = f.read()[0:-1] # -1 pour supprimer le saut de ligne
 
def swapF(d1):
    c1 = np.random.choice(list(d1.keys()))
    c2 = np.random.choice(list(d1.keys()))
    d2 = dict(d1)
    d2[c1] = d1[c2]
    d2[c2] = d1[c1]
    return d2
     
tau = {'a' : 'b', 'b' : 'c', 'c' : 'a', 'd' : 'd' }
print(swapF(tau))

def decrypt(mess, dictionnaire):
    return ''.join([dictionnaire[c] for c in mess])
 
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
    plus_vs = logLikelihood(mess, mu, A, chars2index);
    message_decode = mess;
    for i in range(0,N):
        tau2 = swapF(tau);
        decode = decrypt(mess, tau2);
        vs = logLikelihood(decode, mu, A, chars2index);
        V = min(0, vs - plus_vs)
        U = np.log(np.random.uniform())
        if U < V:
            print("likelihood: %, V: %",vs,V)
            tau = tau2
            plus_vs = vs;
            message_decode = decode;
    return message_decode, plus_vs

def identityTau (count):
    tau = {}
    for k in list(count.keys ()):
        tau[k] = k
    return tau

def creeTau(count):
    freqKeys = np.array(list(count.keys()))
    freqVal  = np.array(list(count.values()))
    # indice des caracteres: +freq => - freq dans la references
    rankFreq = (-freqVal).argsort()
    
    # analyse mess. secret: indice les + freq => - freq
    cles = np.array(list(set(secret2))) # tous les caracteres de secret2
    rankSecret = np.argsort(-np.array([secret2.count(c) for c in cles]))
    # ATTENTION: 37 cles dans secret, 77 en général... On ne code que les caractères les plus frequents de mu, tant pis pour les autres
    # alignement des + freq dans mu VS + freq dans secret
    tau_init = dict([(cles[rankSecret[i]], freqKeys[rankFreq[i]]) for i in range(len(rankSecret))])
    return tau_init

print(MetropolisHastings( secret2, mu, A, creeTau(count), 50000, chars2index))