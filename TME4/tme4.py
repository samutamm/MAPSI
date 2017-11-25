
# -*- coding: utf-8 -*-

import numpy as np
from math import *
from pylab import *
import matplotlib.pyplot as plt

def read_file ( filename ):
    """
    Lit le fichier contenant les données du geyser Old Faithful
    """
    # lecture de l'en-tête
    infile = open ( filename, "r" )
    for ligne in infile:
        if ligne.find ( "eruptions waiting" ) != -1:
            break

    # ici, on a la liste des temps d'éruption et des délais d'irruptions
    data = []
    for ligne in infile:
        nb_ligne, eruption, waiting = [ float (x) for x in ligne.split () ]
        data.append ( eruption )
        data.append ( waiting )
    infile.close ()

    # transformation de la liste en tableau 2D
    data = np.asarray ( data )
    data.shape = ( int ( data.size / 2 ), 2 )

    return data

data = read_file ( "2015_tme4_faithful.txt" )

def partie_densite(var, mu, sigma):
    return ((var-mu) / sigma)**2;

def normale_bidim (x, z, params):
    a=1.0/(2*math.pi*params[2]*params[3]*math.sqrt(1-params[4]**2))
    b=-0.5/(1-params[4]**2)
        
    c = -2 * params[4] * (((x - params[0])*(z - params[1])) / (params[2]*params[3] ) )
    x_den = partie_densite(x, params[0], params[2])
    z_den = partie_densite(z, params[1], params[3])
        
    return a * np.exp(b * (x_den + c + z_den))
    

def dessine_1_normale ( params ):
    # récupération des paramètres
    mu_x, mu_z, sigma_x, sigma_z, rho = params

    # on détermine les coordonnées des coins de la figure
    x_min = mu_x - 2 * sigma_x
    x_max = mu_x + 2 * sigma_x
    z_min = mu_z - 2 * sigma_z
    z_max = mu_z + 2 * sigma_z

    # création de la grille
    x = np.linspace ( x_min, x_max, 100 )
    z = np.linspace ( z_min, z_max, 100 )
    X, Z = np.meshgrid(x, z)

    # calcul des normales
    norm = X.copy ()
    for i in range ( x.shape[0] ):
        for j in range ( z.shape[0] ):
            norm[i,j] = normale_bidim ( x[i], z[j], params )

    # affichage
    fig = plt.figure ()
    plt.contour ( X, Z, norm, cmap=cm.autumn )
    plt.show ()

def dessine_normales ( data, params, weights, bounds, ax ):
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]

    x_min = bounds[0]
    x_max = bounds[1]
    z_min = bounds[2]
    z_max = bounds[3]

    nb_x = nb_z = 100
    x = np.linspace ( x_min, x_max, nb_x )
    z = np.linspace ( z_min, z_max, nb_z )
    X, Z = np.meshgrid(x, z)

    norm0 = np.zeros ( (nb_x,nb_z) )
    for j in range ( nb_z ):
        for i in range ( nb_x ):
            norm0[j,i] = normale_bidim ( x[i], z[j], params[0] )
    norm1 = np.zeros ( (nb_x,nb_z) )
    for j in range ( nb_z ):
        for i in range ( nb_x ):
             norm1[j,i] = normale_bidim ( x[i], z[j], params[1] )

    ax.contour ( X, Z, norm0, cmap=cm.winter, alpha = 0.5 )
    ax.contour ( X, Z, norm1, cmap=cm.autumn, alpha = 0.5 )
    for point in data:
        ax.plot ( point[0], point[1], 'k+' )


def find_bounds ( data, params ):
    mu_x0, mu_z0, sigma_x0, sigma_z0, rho0 = params[0]
    mu_x1, mu_z1, sigma_x1, sigma_z1, rho1 = params[1]

    x_min = min ( mu_x0 - 2 * sigma_x0, mu_x1 - 2 * sigma_x1, data[:,0].min() )
    x_max = max ( mu_x0 + 2 * sigma_x0, mu_x1 + 2 * sigma_x1, data[:,0].max() )
    z_min = min ( mu_z0 - 2 * sigma_z0, mu_z1 - 2 * sigma_z1, data[:,1].min() )
    z_max = max ( mu_z0 + 2 * sigma_z0, mu_z1 + 2 * sigma_z1, data[:,1].max() )

    return ( x_min, x_max, z_min, z_max )


mean1 = data[:,0].mean ()
mean2 = data[:,1].mean ()
std1  = data[:,0].std ()
std2  = data[:,1].std ()

#params = np.array ( [(mean1 - 0.2, mean2 - 1, std1, std2, 0),
#                     (mean1 + 0.2, mean2 + 1, std1, std2, 0)] )
#weights = np.array ( [0.4, 0.6] )
#bounds = find_bounds ( data, params )
#
#fig = plt.figure ()
#ax = fig.add_subplot(111)
#dessine_normales ( data, params, weights, bounds, ax )
#plt.show ()

def Q_i (data, current_params, current_weights ):
    alpha0 = np.array([normale_bidim (point[0], point[1], current_params[0]) for point in data])*current_weights[0]
    alpha1 = np.array([normale_bidim (point[0], point[1], current_params[1]) for point in data])*current_weights[1]
    #print(alpha1)
    q0=alpha0/(alpha0+alpha1)
    q1=alpha1/(alpha0+alpha1)f
    return np.array([[q0[i], q1[i]] for i in range(q0.size)]) 

current_params = np.array([[ 3.2194684, 67.83748075, 1.16527301, 13.9245876,  0.9070348 ],
                           [ 3.75499261, 73.9440348, 1.04650191, 12.48307362, 0.88083712]])
current_weights = np.array ( [ 0.49896815, 0.50103185] )    
    
T = Q_i ( data, current_params, current_weights )

def M_step ( data, Q, current_params, current_weights ):
    Q0=Q[:, 0]
    Q1=Q[:, 1]
    sumQ0=Q0.sum() 
    sumQ1=Q1.sum()
    
    pi0=sumQ0/(sumQ0+sumQ1)
    pi1=sumQ1/(sumQ0+sumQ1)
    
    xi=data[:,0]
    zi=data[:,1]
    
    mux0=(Q0*xi).sum()/sumQ0
    mux1=(Q1*xi).sum()/sumQ1
    
    muz0=(Q0*zi).sum()/sumQ0
    muz1=(Q1*zi).sum()/sumQ1
    
    sigmax0=math.sqrt((Q0*(xi-mux0)**2).sum()/sumQ0)
    sigmax1=math.sqrt((Q1*(xi-mux1)**2).sum()/sumQ1)
    
    sigmaz0=math.sqrt((Q0*(zi-muz0)**2).sum()/sumQ0)
    sigmaz1=math.sqrt((Q1*(zi-muz1)**2).sum()/sumQ1)
    
    p0=(( Q0*((xi-mux0)*(zi-muz0)/(sigmax0*sigmaz0))).sum())/sumQ0
    p1=(( Q1*((xi-mux1)*(zi-muz1)/(sigmax1*sigmaz1))).sum())/sumQ1
    
    return (np.array([
                [mux0, muz0, sigmax0, sigmaz0, p0],[mux1, muz1, sigmax1, sigmaz1, p1]        
            ]),
            np.array([pi0, pi1]))
    
# calcul des bornes pour contenir toutes les lois normales calculées
def find_video_bounds ( data, res_EM ):
    bounds = np.asarray ( find_bounds ( data, res_EM[0][0] ) )
    for param in res_EM:
        new_bound = find_bounds ( data, param[0] )
        for i in [0,2]:
            bounds[i] = min ( bounds[i], new_bound[i] )
        for i in [1,3]:
            bounds[i] = max ( bounds[i], new_bound[i] )
    return bounds


import matplotlib.animation as animation

# la fonction appelée à chaque pas de temps pour créer l'animation
def animate ( i ):
    ax.cla ()
    dessine_normales (data, res_EM[i][0], res_EM[i][1], bounds, ax)
    ax.text(5, 40, 'step = ' + str ( i ))
    print("step animate = %d" % ( i ))


    
#current_params = array([(2.51460515, 60.12832316, 0.90428702, 11.66108819, 0.86533355),
#                        (4.2893485,  79.76680985, 0.52047055,  7.04450242, 0.58358284)])
#current_weights = array([ 0.45165145,  0.54834855])
#Q = Q_i ( data, current_params, current_weights )
#print(M_step ( data, Q, current_params, current_weights ))

mean1 = data[:,0].mean ()
mean2 = data[:,1].mean ()
std1  = data[:,0].std ()
std2  = data[:,1].std ()
params = np.array ( [(mean1 - 0.2, mean2 - 1, std1, std2, 0),
                     (mean1 + 0.2, mean2 + 1, std1, std2, 0)] )
weights = np.array ( [ 0.5, 0.5 ] )

res_EM = []

for i in range(20):    
    Q = Q_i ( data, params, weights );
    M = M_step ( data, Q, params, weights );
    params = M[0];
    weights = M[1];
    res_EM.append((params, weights));


bounds = find_video_bounds ( data, res_EM )
# création de l'animation : tout d'abord on crée la figure qui sera animée
fig = plt.figure ()
ax = fig.gca (xlim=(bounds[0], bounds[1]), ylim=(bounds[2], bounds[3]))
# exécution de l'animation
anim = animation.FuncAnimation(fig, animate, 
                               frames = len ( res_EM ), interval=1000 )
plt.show ()
Fwriter = animation.FFMpegWriter(bitrate=4000)
anim.save('old_faithful.avi', Fwriter)

