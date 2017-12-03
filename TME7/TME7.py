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

def learnHMM(allx, alls, N, K, initTo0=False):
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
        states = alls[i];
        last_state = np.int(states[0]);
        Pi[last_state] += 1;
        B[0][np.int(observations[0])] += 1;
        for j in range(1, observations.size):
            #PARTIE A
            current_state = np.int(states[j]);
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
#Xd = discretise(X,K)
#GD = initGD(Xd,N);
#Pi, A, B = learnHMM(Xd[Y=='a'],GD[Y=='a'],N,K, True)


def viterbi(x,Pi,A,B):
    N,T = B.shape;
    delta = np.zeros((N,x.size));
    phi = np.zeros((N,x.size));
    #Initialisation
    for i in range(N):
        obs = np.int(x[0])
        delta[i][0] = np.log(Pi[i]) + np.log(B[i][obs]);
        phi[i][0] = -1;
    #Recursion
    for t in range(1,x.size):
        observation = np.int(x[t]);
        s_t_last = delta[:, t-1];
        for i in range(N):
            a_i = np.log(A[:, i])
            sums = np.array([delta[prev_i][t-1] + np.log(A[prev_i][i]) for prev_i in range(N)])
            delta[i][t] = sums.max() + np.log(B[i][observation]);
            phi[i][t] = sums.argmax()
    p_est = delta[:,-1].max()
    s_est = np.zeros(phi.shape[1])
    s_est[(phi.shape[1]-1)] = phi[:,-1].max()
    for i in range(phi.shape[1]-2, 0, -1):
        last = np.int(s_est[i+1])
        s_est[i] = phi[last, np.int(i+1)]
    return (s_est, p_est)
            
s_est, p_est = viterbi(Xd[0], Pi, A, B)

def log_matrice(V):
    return np.array([0 if v == 0 else np.log(v) for v in V])        
        
def baumwelch(X, Y, N, K):
    Xd = discretise(X,K)
    GD = initGD(X,N);
    index = groupByLabel(Y)
    limite=0.0001
    convergence = False
    signaux = GD
    classes = np.unique(Y)
    L=[1]
    probas_estime={}
    while not (convergence):
        modeles = []
        probas_estime={}
        #probas_estime = np.zeros(nClasse)
        cpt=0
        for cl in classes:
            probas_estime[cl] = []
            M = learnHMM(Xd[Y==cl], signaux[index[cpt]], N, K, True)
            modeles.append(M)
            proba_temp = []
            classe_resultat = []
            for x in Xd[Y == cl]:
                viterbi_res = viterbi(x, M[0], M[1], M[2])
                proba_temp.append(viterbi_res[1])
                classe_resultat.append(viterbi_res[0])
                
            signaux[index[cpt]] = classe_resultat
            probas_estime[cl] = proba_temp
            cpt +=1
        log_lk = 0
        for c in classes:
            for i in range(len(X[Y == c])):
                log_lk += probas_estime[c][i]
        
        print("maximum de vraisemblance: ",L[-1])
        convergence = ((L[-1] - log_lk) / L[-1] < limite)
        L.append(log_lk)
    return (modeles, L)



#########################
## Partie evaluation ####
#########################

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

# x part of X, M modeles from BaumWelch
#def predic(x, M):
#    maxi = -1* float('inf')
#    index = None
#    for cl in range(len(M)):
#        m = M[cl]
#        proba, s = viterbi(x, m[0], m[1], m[2])
#        if proba > maxi:
#            maxi = proba
#            index = cl
#    print(index)
#    #print(argmax([viterbi(x, M[0], M[1], M[2]])[1] for c in range(...)))#OU placer [c]?
#    return maxi, index

def iTrain():
    itrain,itest = separeTrainTest(Y,0.8)
    ia = []
    for i in itrain:
        ia += i.tolist()    
    it = []
    for i in itest:
        it += i.tolist()
    return (ia, it)

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
    
ia,it=iTrain()

# aprentissage avec les donnée training
modeles, L = baumwelch(X[ia], Y[ia], N, K)

# Char à numero
Ynum = np.zeros(Y[it].shape);
for num,char in enumerate(np.unique(Y)):
    Ynum[Y[it]==char] = num;

#predictions avec les donnée test
Xd = discretise(X[it], K)
proba = np.array([
    [viterbi(x, modeles[cl][0], modeles[cl][1], modeles[cl][2])[1] for x in Xd] 
    for cl in range(len(np.unique(Y)))
])

pred = proba.argmax(0);
print("teste avec le separation à training et test")
print(np.where(pred != Ynum, 0.,1.).mean())

# visualisation
visualise_conf(Ynum, pred, Y)