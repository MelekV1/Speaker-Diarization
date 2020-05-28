import time
import seaborn
import numpy, scipy, matplotlib.pyplot as plt
import librosa,librosa.display
import numpy as np
from math import *


def importer(fichier):
    return np.loadtxt(fichier)

def distance(v1, v2):
    return np.linalg.norm(v1-v2)

def kmeans(k, epsilon=0.000001):
    registre_moyenne= []
    dataset = importer('dataset.txt')
    nbinstances, taillefeatures = dataset.shape
    moyennes = dataset[np.random.randint(0, nbinstances - 1, size=k)]
    registre_moyenne.append(moyennes)
    prevmoyenes = np.zeros(moyennes.shape)
    cluster_vers =[0]*nbinstances
    convergance = distance(moyennes, prevmoyenes)
    iteration = 0
    while convergance > epsilon:
        iteration += 1
        convergance = distance(moyennes, prevmoyenes)
        prevmoyenes = moyennes
        for index_instance, instance in enumerate( dataset ):
            dist_vec = np.zeros( ( k , 1 ) )
            for index_prototype, prototype in enumerate(moyennes):
                dist_vec[index_prototype] = distance(prototype,instance)
            cluster_vers[index_instance] = np.argmin(dist_vec)

        tmp = np.zeros((k, taillefeatures))
        for index in range(len(moyennes)):
            instances_close = [i for i in range(len(cluster_vers)) if cluster_vers[i] == index]
            m = np.mean(dataset[instances_close], axis=0)
            tmp[index, :] = m
        moyennes = tmp
        registre_moyenne.append(tmp)
    return moyennes, registre_moyenne, cluster_vers

def sections(seq , fe, pas=0.5 ):
    longeur_pas=int(pas*fe);
    end=int(longeur_signal - longeur_signal%longeur_pas);
    parole_ou_non=[0]*int(longeur_signal/longeur_pas) ;

    for isection in range ( int(longeur_signal/longeur_pas) ):
        decision=estimate(seq[isection*longeur_pas :longeur_pas*isection + longeur_pas ]);
        parole_ou_non[isection]=decision;

    if longeur_signal%pas!=0:
        parole_ou_non.append(parole_ou_non[end-1])
    return parole_ou_non;

def estimate(morceau):
    Gain_seuil=0.05;
    long_morceau=len(morceau);
    gain_moyenne=sum([abs(i) for i in morceau] ) / long_morceau;
    if gain_moyenne>Gain_seuil:
        return 1
    else:
        return 0;

def reconstituer(avant,reference,pas=0.5):
    apres=[0]*len(avant);
    longeur_pas=int(pas*fe);

    for ires in range(len(reference)):
        if reference[ires]==1:
            apres[ires*longeur_pas:(ires+1)*longeur_pas]=avant[ires*longeur_pas:(ires+1)*longeur_pas];
    return apres;

def step_to_clusters(sequence,silent_speech,fe,pas=0.5):
    intervals=dict();intervalsData=dict();
    counter=0;longeur_pas=int(pas*fe);
    for iinterval in range (len(silent_speech) ):
        if silent_speech[iinterval] !=0 :
            intervals[counter]=( longeur_pas*iinterval,(iinterval+1)*longeur_pas)
            intervalsData[counter]=sequence[iinterval*longeur_pas:(iinterval+1)*longeur_pas]
            counter+=1
    return intervals,intervalsData

def MFCC_Emphasis(valuable_data):
    emphasised=list()
    dataset  = open("dataset.txt", "w")
    for i in valuable_data:
        mfcc=librosa.feature.mfcc(y=np.array(valuable_data[i] ) , sr=fe)
        to_treat=np.array(mfcc)[0]
        to_treat=[valuable_data[i][0]/sqrt(sum( [j*j for j in to_treat ] ) ) ,valuable_data[i][-1]/sqrt(sum( [j*j for j in to_treat] ) )]
        databrute=str(to_treat[0])+" "+str(to_treat[1])+"\n"
        dataset.write(databrute)
        emphasised.append( to_treat )
    return

def visualiser(dataset, registre, appartenance):
    locuteur = ['m', 'c']
    fig, ax = plt.subplots()
    for index in range(dataset.shape[0]):
        plus_proche = [i for i in range(len(appartenance)) if appartenance[i] == index]
        for instance_index in plus_proche:
            ax.plot(dataset[instance_index][0], dataset[instance_index][1], (locuteur[index] + 'o'))
    points = []
    for index, moyenne in enumerate(registre):
        for inner, item in enumerate(moyenne):
            if index == 0:
                points.append(ax.plot(item[0], item[1], 'ko')[0])
            else:
                points[inner].set_data(item[0], item[1])
                print("MoyenneCluster {} {}".format(index, item))

                plt.pause(0.8)
                pass
            pass
        pass
    return

def cluster():
    dataset = importer('dataset.txt')
    moyenne, registre, appartient = kmeans(2)
    visualiser(dataset, registre, appartient)
def print_speech_info(suite_bin,intervals):
    print("-----------------------------Module de segmentation------------------------------")
    print("Le nombre de segments dans ce fichiers audio..........:",len(parole_ou_non))
    print("Le nombre de segments contenant de la parole..........:",len(list(filter(lambda x: x != 0 , parole_ou_non))))
    print("----------------------------------------------------------------------------------")
    print("---------------------------Module d'extraction de ''features''--------------------")
    print("les intervalles parole en phase Extraction MFCC ..:",len(intervals)," intervalles")
    print("segment   |intervalle de donne dans l'audio")
    for interv in intervals:
        print(interv,"   | [ ",intervals[interv][0],",",intervals[interv][1],"]")
    print("---------------------------Donnees de parole enregistré --------------------------")

    print("----------------------------------------------------------------------------------")
    print("--------------------------------Module de 'Clustering'----------------------------")
    cluster()
    print("----------------------------------------------------------------------------------")
    return
def Affiche_segment_MFCCs(valuable_data):
    for i in valuable_data:
        mfccs=librosa.feature.mfcc(y=np.array(valuable_data[i] ) , sr=fe)
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mfccs, x_axis='time')
        plt.title(( "segment"+str(i) ))
        plt.colorbar()
        plt.tight_layout()

seq, fe = librosa.load('Shetty_and_Kobe.wav');
longeur_signal=len(seq)
seq_list=[piece for piece in seq]
parole_ou_non=sections(seq,fe);
intervals,valuable_data=step_to_clusters(seq_list,parole_ou_non,fe,pas=0.5)
MFCC_Emphasis(valuable_data)
print_speech_info(parole_ou_non,intervals)

"""
speech=numpy.array(reconstituer(seq,parole_ou_non))
plt.figure()
plt.title(' Parole+Silence')
librosa.display.waveplot(seq, fe)
plt.figure()
plt.title(' Parole +silence eliminé' )
librosa.display.waveplot(speech,fe)
plt.show()
"""
plt.show()
