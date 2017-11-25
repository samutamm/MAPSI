#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
# truc pour un affichage plus convivial des matrices numpy
np.set_printoptions(precision=2, linewidth=320)
plt.close('all')

#data = pkl.load(file("ressources/lettres.pkl","rb"))
with open('lettres.pkl', 'rb') as f:
    data = pkl.load(f, encoding='latin1') 
X = np.array(data.get('letters'))
Y = np.array(data.get('labels'))

nCl = 26

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

def initGD(X,N):
    return np.array([np.floor(np.linspace(0,N-.00000001,len(x))) for x in X]);

def learnMarkovModel(Xc, d):
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

def learnHMM(allx, allq, N, K, initTo0=False):
    if initTo0:
        A = np.zeros((N,N))
        B = np.zeros((N,K))
        Pi = np.zeros(N)
    else:
        eps = 1e-8
        A = np.ones((N,N))*eps
        B = np.ones((N,K))*eps
        Pi = np.ones(N)*eps
    
    for i in range(allx.size):
        observations = allx[i];
        hidden_states = allq[i];
        last_state = np.int(hidden_states[0]);
        Pi[last_state] += 1;
        B[0][np.int(observations[0])] += 1;
        for j in range(1, observations.size):
            #PARTIE A
            current_state = np.int(hidden_states[j]);
            A[last_state][current_state] += 1.0;
            last_state = current_state;
            #PARTIE B
            observation = np.int(observations[j]);
            B[current_state][observation] += 1.0;
            
    A = A/np.maximum(A.sum(1).reshape(N,1),1) # normalisation
    B = B/np.maximum(B.sum(1).reshape(N,1),1) # normalisation
    Pi = Pi/Pi.sum()
    return (Pi, A,B);

K = 10 # discrétisation (=10 observations possibles)
N = 5  # 5 états possibles (de 0 à 4 en python) 
# Xd = angles observés discrétisés    
Xd = discretise(X,K)
GD = initGD(Xd,N);
Pi, A, B = learnHMM(Xd[Y=='a'],GD[Y=='a'],N,K, True)

def viterbi(x,Pi,A,B):
    N,T = B.shape;
    sigma = np.zeros((N,x.size));
    phi = np.zeros((N,x.size));
    #Initialisation
    for i in range(N):
        obs = np.int(x[0])
        sigma[i][0] = np.log(Pi[i]) + np.log(B[i][obs]);# if (Pi[i] == 0 or B[i][obs] == 0) else np.log(Pi[i]) + np.log(B[i][obs]);
        phi[i][0] = -1;
    #Recursion
    for t in range(1,x.size):
        for i in range(N):
            obs = np.int(x[t]);
            s_t_last = sigma[:, t-1];
            a_i = np.log(A[:, i]);
            sums = np.array([sigma[prev_i][t-1] + np.log(A[prev_i][i]) for prev_i in range(N)])
            sigma[i][t] = sums.max() + np.log(B[i][obs]);
            phi[i][t] = sums.argmax()
    p_est = sigma[:,-1].max()
    s_est = [phi[:,j].argmax() for j in range(phi.shape[1])]
    return (s_est, p_est, sigma, phi)
            
s_est, p_est, s, p = viterbi(Xd[0], Pi, A, B)
print("p_est: ", p_est)