#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 10:52:56 2017

"""

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

fname = "dataVelib2.pkl"
f= open(fname, "rb")
data = pkl.load(f)
f.close()


def is_valable_station(station):
    number = station['number'] // 1000
    return 1 <= number and number <= 20 and station['status'] == 'OPEN'

def parse_arrondissement(address):
    return address.split(" ")[-2]

parse_arr = np.vectorize(parse_arrondissement)

stations = np.array([s for s in data if is_valable_station(s)])

altitude =  np.array([s['alt'] for s in stations])
addresses =  np.array([s['address'] for s in stations])
bike_stands =  np.array([s['bike_stands'] for s in stations])
available_bike_stands =  np.array([s['available_bike_stands'] for s in stations])
available_bikes =  np.array([s['available_bikes'] for s in stations])

matrix = np.column_stack((parse_arr(addresses), altitude, bike_stands, available_bike_stands))

# proba P(Ar)
arrondissements = matrix[:, 0]
valeurs = set(arrondissements)
proba_ar = np.array(
        [np.where(arrondissements == valeur)[0].size / arrondissements.size 
        for valeur in valeurs])
print("Arrondissements: ")
print(valeurs)
print("P(Ar) = ")
print(proba_ar)

# proba P(Al)
nIntervalles = 30
effectif_al_total = plt.hist(altitude, nIntervalles)
proba_al_intevalles = effectif_al_total[1]
proba_al = np.array([effectif / effectif_al_total[0].sum() for effectif in effectif_al_total[0]])
print("Intervalles d'altitude: ")
print(proba_al_intevalles)
print("P(Al) = ")
print(proba_al)

# station pleine
# proba P(Sp | Al)
station_full = available_bike_stands == 0
altitude_de_sp = altitude[station_full]
effectif_sp_al = np.histogram(altitude_de_sp, bins=proba_al_intevalles)
# P(Sp | Al) = P(Sp and Al) / P(Al)
proba_sp_sachant_al = effectif_sp_al[0] / effectif_al_total[0]
print("P(Sp | Al) = ")
print(proba_sp_sachant_al)

# au moins 2 velos disponible
# sachant altitude
# P(Vd | Al)
two_or_more_bikes = available_bikes >= 2
effectif_vd_al = np.histogram(altitude[two_or_more_bikes], bins=proba_al_intevalles)
proba_vd_sachant_al = effectif_vd_al[0] / effectif_al_total[0]
print("P(Vd | Al) = ")
print(proba_vd_sachant_al)

# au moins 2 velos disponible
# sachant arrondissement
# P(Vd | Ar)
proba_vd_sachant_ar = np.array(
        [np.where(arrondissements[two_or_more_bikes] == valeur)[0].size /
         np.where(arrondissements == valeur)[0].size 
        for valeur in valeurs])
print("P(Vd | Ar) = ")
print(proba_vd_sachant_ar)

### ---------------------- ###

# Tracer un histogramme

alt = effectif_al_total[1]
intervalle = effectif_al_total[1][1]-effectif_al_total[1][0]
pAlt = effectif_al_total[0]/effectif_al_total[0].sum()
#pAlt /= intervalle % Sum ne va pas étre 1, si on divide par pAlt
plt.bar((alt[1:]+alt[:-1])/2,pAlt, alt[1]-alt[0])
plt.show()


#E[P[Vd|Al]]
print("E[P[Vd|Al]] = ")
E = proba_vd_sachant_al.sum() / proba_vd_sachant_al.size
print(str(E))

### ---------------------- ###

# Tracer la population des stations
#
coordonnees = np.array([(s['position']['lng'], s['position']['lat']) for s in stations])
x1 = coordonnees[:,0] # recuperation des coordonnées 
x2 = coordonnees[:,1]
# définition de tous les styles (pour distinguer les arrondissements)
style = [(s,c) for s in "o^+*" for c in "byrmck" ] 

# tracé de la figure
plt.figure()
arr_list = list(set(arrondissements));
for i in range(0,21):
    ind, = np.where(arrondissements == arr_list[i])
    # scatter c'est plus joli pour ce type d'affichage
    plt.scatter(x1[ind],x2[ind],marker=style[i][0],c=style[i][1],linewidths=0)

plt.axis('equal') # astuce pour que les axes aient les mêmes espacements
plt.legend(range(1,21), fontsize=10)
plt.savefig("carteArrondissements.pdf")
plt.show();

### ---------------------- ###
# Tracer disponibilité

full_stations = np.where(available_bike_stands == 0)[0]
empty_stations = np.where(available_bikes < 2)[0]
not_empty_nor_full_stations = np.array([i for i in np.arange(0, stations.size)
                                        if i not in np.concatenate((full_stations, empty_stations))])

plt.scatter(x1[full_stations],x2[full_stations],marker=style[0][0],c='r',linewidths=0)
plt.scatter(x1[empty_stations],x2[empty_stations],marker=style[1][0],c='y',linewidths=0)
plt.scatter(x1[not_empty_nor_full_stations],x2[not_empty_nor_full_stations],marker=style[2][0],c='g',linewidths=0); 
plt.show()

# Moyenne, Médiane..

average_altitude = altitude.mean()
less_than_average = np.where(altitude < average_altitude)[0];
plt.scatter(x1[less_than_average],x2[less_than_average],marker=style[1][0],c='m',linewidths=0)

mediane_altitude = np.sort(altitude)[altitude.size // 2]
more_than_mediane = np.where(altitude > mediane_altitude)[0];
plt.scatter(x1[more_than_mediane],x2[more_than_mediane],marker=style[1][0],c='k',linewidths=0)

# Tests de corrélation
# Corr(bike available, altitude)
bike_available = np.where(np.logical_or(available_bike_stands == 0, available_bikes == 0),0,1)
bike_available_corr_altitude = np.corrcoef(altitude, bike_available)
print("Corr(bike available, altitude)")
print(bike_available_corr_altitude)

arrondissement_to_int = np.vectorize(lambda x: int(x))
bike_available_corr_arrondissement = np.corrcoef(arrondissement_to_int(arrondissements), bike_available)
print("Corr(bike available, arrodissement)")
print(bike_available_corr_arrondissement)

print("Quel facteur est le plus lié au fait qu'(au moins) un vélo soit disponible dans une station?")
print("Altitude est plus lié au fait qu'au moins un vélo soit disponible dans une station.")
