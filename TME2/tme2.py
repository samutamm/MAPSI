#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:55:10 2017

"""
import numpy as np
import math as m
import matplotlib.pyplot as plt

def bernoulli(p):
    return np.random.rand() <= p;

def binomiale(n, p):
    return sum([bernoulli(p) for i in range(n)]);

# 3. Histogramme de la loi binomiale
n = 20
tableau_1000_cases = np.array([binomiale(n, 0.5) for i in range(1000)]);
plt.hist(tableau_1000_cases, n)

# Visualisation d'indépendances
# 1. Loi normale centrée

def densite_normale(x, sigma):
    return (1/(m.sqrt(2*m.pi)*sigma))*m.e**(-0.5 * (x/sigma)**2);

def normale ( k, sigma ):
    if k % 2 == 0:
        raise ValueError ( 'le nombre k doit etre impair' )
    x = np.linspace(-2*sigma, 2*sigma, k)
    return np.array([densite_normale(item, sigma) for item in x]);

plt.plot(range(41), normale(41, 5));

#
# 2. Distribution de probabilité affine

def proba_affine ( k, slope ):
    if k % 2 == 0:
        raise ValueError ( 'le nombre k doit etre impair' )
    if abs ( slope  ) > 2. / ( k * k ):
        raise ValueError ( 'la pente est trop raide : pente max = ' + 
        str ( 2. / ( k * k ) ) )
    return np.array([1/k + (i - (k-1)/2) * slope for i in range(k)]);

plt.plot(proba_affine(11, 0.000001), range(11));

#
# 3. Distribution jointe

def Pxy(arr1, arr2):
    return np.array([arr2 * i1 for i1 in arr1]);

PA = np.array ( [0.2, 0.7, 0.1] )
PB = np.array ( [0.4, 0.4, 0.2] )
print(Pxy(PA, PB));

# 4. Affichage de la distribution jointe

from mpl_toolkits.mplot3d import Axes3D

def dessine ( P_jointe ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = np.linspace ( -3, 3, P_jointe.shape[0] )
    y = np.linspace ( -3, 3, P_jointe.shape[1] )
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, P_jointe, rstride=1, cstride=1 )
    ax.set_xlabel('A')
    ax.set_ylabel('B')
    ax.set_zlabel('P(A) * P(B)')
    plt.show ()

dessine(Pxy(PA, PB));

#binomial, normale


# affine, normale
n = 101
test_affine = proba_affine(n, 0.0001);
test_normale = normale(n, 3);
dessine(Pxy(test_affine,test_normale));
# inverse
dessine(Pxy(test_normale, test_affine));

# Indépendances conditionnelles
# creation de P(X,Y,Z,T)
P_XYZT = np.array([[[[ 0.0192,  0.1728],
                     [ 0.0384,  0.0096]],

                    [[ 0.0768,  0.0512],
                     [ 0.016 ,  0.016 ]]],

                   [[[ 0.0144,  0.1296],
                     [ 0.0288,  0.0072]],

                    [[ 0.2016,  0.1344],
                     [ 0.042 ,  0.042 ]]]])

P_YZ = np.zeros((2,2));
for x in range(0, 2):
    for y in range(0,2):
        for z in range(0,2):
            for t in range(0,2):
                P_YZ[y][z] += P_XYZT[x][y][z][t]
print(P_YZ);
print("-------------------")
P_XTcondYZ=np.zeros((2,2,2,2));
for x in range(0, 2):
    for y in range(0,2):
        for z in range(0,2):
            for t in range(0,2):
                P_XTcondYZ[x][y][z][t] = (P_XYZT[x][y][z][t])/P_YZ[y][z]
print(P_XTcondYZ);
print("-------------------")

P_XcondYZ = np.zeros((2,2,2));
for x in range(0, 2):
    for y in range(0,2):
        for z in range(0,2):
            for t in range(0,2):
                P_XcondYZ[x][y][z] += P_XTcondYZ[x][y][z][t]
print(P_XcondYZ);
print("-------------------")

P_TcondYZ = np.zeros((2,2,2));
for x in range(0, 2):
    for y in range(0,2):
        for z in range(0,2):
            for t in range(0,2):
                P_TcondYZ[t][y][z] += P_XTcondYZ[x][y][z][t]
print(P_TcondYZ);
print("-------------------")

# P(X,T|Y,Z)=P(X|Y,Z)×P(T|Y,Z) VRAIE
for x in range(0, 2):
    for y in range(0,2):
        for z in range(0,2):
            for t in range(0,2):
                left = P_XTcondYZ[x][y][z][t]
                right = P_XcondYZ[x][y][z] * P_TcondYZ[t][y][z]
                if abs(left - right) < 0.0001:
                    print(True)
                else:
                    print(False)

P_XYZ= np.zeros((2,2,2));
for x in range(0, 2):
    for y in range(0,2):
        for z in range(0,2):
            for t in range(0,2):
                P_XYZ [x][y][z] += P_XYZT[x][y][z][t]

P_X= np.zeros(2);
P_YZ= np.zeros((2,2));
for x in range(0, 2):
    for y in range(0,2):
        for z in range(0,2):
            P_X[x] += P_XYZ[x][y][z]
            P_YZ[y][z] += P_XYZ[x][y][z]
print(P_X)
print(P_YZ)

# P(X,Y,Z)=P(X)×P(Y,Z) PAS VRAIE
for x in range(0, 2):
    for y in range(0,2):
        for z in range(0,2):
            left = P_XYZ[x][y][z]
            right = P_X[x] * P_YZ[y][z]
            if abs(left - right) < 0.0001:
                print(True)
            else:
                print(False)


# Indépendances conditionnelles et consommation mémoire
# 1. Package de manipulation de probabilités
                
import pyAgrum as gum
import pyAgrum.lib.ipython as gnb
#mport pyAgrum.lib.notebook as gnb

def read_file ( filename ):
    """
    Renvoie les variables aléatoires et la probabilité contenues dans le
    fichier dont le nom est passé en argument.
    """
    Pjointe = gum.Potential ()
    variables = []

    fic = open ( filename, 'r' )
    # on rajoute les variables dans le potentiel
    nb_vars = int ( fic.readline () )
    for i in range ( nb_vars ):
        name, domsize = fic.readline ().split ()
        variable = gum.LabelizedVariable(name,name,int (domsize))
        variables.append ( variable )
        Pjointe.add(variable)

    # on rajoute les valeurs de proba dans le potentiel
    cpt = []
    for line in fic:
        cpt.append ( float(line) )
    Pjointe.fillWith(np.array ( cpt ) )

    fic.close ()
    return np.array ( variables ), Pjointe

variables, Pjointe = read_file('asia.txt');

#3. Test d'indépendance conditionnelle

def remove_var(var_list, to_remove):
    try:    
        var_list.remove(to_remove.name());
    except ValueError:
        print('Trying to remove '+ to_remove.name() + ' from list ' + str(var_list));
        raise ValueError('A very specific bad thing happened')

def conditional_indep(potential, varX, varY, ensZ, epsilon):
    sum_out = list(map(lambda x: x.name(), potential.variablesSequence()));
    remove_var(sum_out, varX);
    remove_var(sum_out, varY);
    for varZ in ensZ:
        remove_var(sum_out, varZ);
    P_XYZ = potential.margSumOut(sum_out);
    #P_XYZ = potential.margSumIn([x.name() for x in [varX,varY] + ensZ])
    P_XZ = P_XYZ.margSumOut([varY.name()]);
    P_YZ = P_XYZ.margSumOut([varX.name()]);
    #P_Z = P_XZ.margSumOut([varX.name()]);
    P_Z = P_YZ.margSumOut([varY.name()]);
    P_XcondZ = P_XZ / P_Z;
    P_YcondZ = P_YZ / P_Z;
    Q_XYcondZ = P_XYZ - (P_XcondZ * P_YcondZ);
    #return Q_XYcondZ.abs().max();
    return Q_XYcondZ.abs().max() < epsilon;
    
    
# testing
visit_to_Asia = variables[0];
tuberculosis = variables[1];
smoking = variables[2];
lung_cancer = variables[3];
tuberculosis_or_lung_cancer = variables[4];
bronchitis = variables[5];
positive_Xray = variables[6];
dyspnoea = variables[7];

# visiblement il fonctionné pas trés bien ca, parce que je trouve pas aucun combination¨¨
# qui soit independant sachant qc autre.
conditional_indep(Pjointe, 
                  visit_to_Asia, 
                  tuberculosis, 
                  [tuberculosis_or_lung_cancer], 0.1)

# 4. Compactage de probabilités conditionnelles

def compact_conditional_proba(potential, varXin, epsilon):
    K = potential.variablesSequence();
    for varXij in potential.variablesSequence():
        K_without_Xij = list(filter(lambda item: item.name() != varXij.name(), K))
        if varXin.name() == varXij.name():
            continue
        if varXij.name() not in list(map(lambda x: x.name(), K_without_Xij)):
            continue
        if conditional_indep(potential, varXin, varXij, K_without_Xij, epsilon):
            K = K_without_Xij;
    left = gum.Potential();
    right = gum.Potential();
    left.add(varXin);
    for variable in K:
        if not left.contains(variable): left.add(variable);
        right.add(variable);
    return left / right # P(Xin|K)

X_in = positive_Xray;
proba = compact_conditional_proba(Pjointe, X_in, 0.5);
proba = proba.putFirst(X_in.name())
gnb.showPotential ( proba )
# visualisation ne donne pas truc comprehensible :( 
# bug soit dans conditional_indep, soit dans compact_conditional_proba

# 5. Création d'un réseau bayésien

#bayesien network

bn=gum.BayesNet('WaterSprinkler')
print(bn)
c=bn.add(gum.LabelizedVariable('c','cloudy ?',2))
print(c)
s, r, w = [ bn.add(name, 2) for name in "srw" ] #bn.add(name, 2) === bn.add(gum.LabelizedVariable(name, name, 2))
print (s,r,w)
print (bn)
for link in [(c,r),(s,w),(r,w)]:
    bn.addArc(*link)
print(bn)
bn.cpt(c).fillWith([0.5,0.5])
bn.cpt(s)[:]=[ [0.5,0.5],[0.9,0.1]]
bn.cpt(w)[0,0,:] = [1, 0] # r=0,s=0
bn.cpt(w)[0,1,:] = [0.1, 0.9] # r=0,s=1
bn.cpt(w)[1,0,:] = [0.1, 0.9] # r=1,s=0
bn.cpt(w)[1,1,:] = [0.01, 0.99] # r=1,s=1

bn.cpt(r)[{'c':0}]=[0.8,0.2]
bn.cpt(r)[{'c':1}]=[0.2,0.8]

ie=gum.LazyPropagation(bn)
ie.makeInference()
print(ie.posterior(w))

#evidence
ie.setEvidence({'s':0, 'c': 0})
ie.makeInference()
print(ie.posterior(w))
ie.setEvidence({'s': [0.5, 1], 'c': [1, 0]})
ie.makeInference()
print(ie.posterior(w))

gnb.showInference(bn,evs={})