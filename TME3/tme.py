
import numpy as np
import matplotlib.pyplot as plt
import math

def read_file ( filename ):
    # 
    infile = open ( filename, "r" )    
    nb_classes, nb_features = [ int( x ) for x in infile.readline().split() ]

    # 
    #
    data = np.empty ( 10, dtype=object )  
    filler = np.frompyfunc(lambda x: list(), 1, 1)
    filler( data, data )

    #
    for ligne in infile:
        champs = ligne.split ()
        if len ( champs ) == nb_features + 1:
            classe = int ( champs.pop ( 0 ) )
            data[classe].append ( list ( map ( lambda x: float(x), champs ) ) )
    infile.close ()

    # 
    output  = np.empty ( 10, dtype=object )
    filler2 = np.frompyfunc(lambda x: np.asarray (x), 1, 1)
    filler2 ( data, output )

    return output

def display_image ( X ):
    #
    if X.size != 256:
        raise ValueError ( "kqsdqsd" )

    
    Y = X / X.max ()
    img = np.zeros ( ( Y.size, 3 ) )
    for i in range ( 3 ):
        img[:,i] = X

    img.shape = (16,16,3)


    plt.imshow( img )
    plt.show ()
    
data = read_file ("usps_train.txt")
test_data = read_file ("test_data.txt")


def learnML_class_parameters(table):
    return (np.mean(table, axis=0),np.var(table, axis=0))

def learnML_all_parameters(data):
    return np.array([learnML_class_parameters(classe) for classe in data])

print(learnML_class_parameters(data[0]))

print(learnML_all_parameters(data))

def log_likelihood(image,table):
    a = 0    
    for i in range(256):
        if table[1][i] != 0:
            a += -0.5*math.log(2*math.pi*table[1][i])-0.5*(image[i]-table[0][i])**2 / table[1][i]
    return a
    
    
parameters = learnML_all_parameters ( data )
print(log_likelihood ( test_data[2][3], parameters[1] ))

def log_likelihoods(image,tables):
    return np.array([ log_likelihood ( image, tables[i] ) for i in range (10) ])

print(log_likelihoods ( test_data[1][5], parameters ))

def classify_image(image, tables):
    return log_likelihoods(image, tables).argmax()

def classify_one_class(test_data, class_params, class_index):
    results = np.array([classify_image(image, parameters) for image in test_data[class_index]]);
    classification_results = np.array([(results == classification_res).sum() 
                                        for classification_res in range(10)])
    return classification_results / results.size;

def classify_all_images(images, classes):
    return np.array([classify_one_class(images, classes, i) for i in range(10)]);

T = classify_all_images(test_data, parameters);

from mpl_toolkits.mplot3d import Axes3D

def dessine ( classified_matrix ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = y = np.linspace ( 0, 9, 10 )
    X, Y = np.meshgrid(x, y)
    ax.plot_surface(X, Y, classified_matrix, rstride = 1, cstride=1 )

dessine(T)