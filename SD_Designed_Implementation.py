import time
import seaborn
import numpy, scipy, matplotlib.pyplot as plt
import librosa,librosa.display
import numpy as np


seq, fe = librosa.load('Shetty_and_Kobe.wav');
longeur_signal=len(seq)
seq_list=[piece for piece in seq]

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
    gain_moyenne=sum(abs(morceau)) / long_morceau;
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
def print_speech_info(suite_bin):
    print("-----------------------------Module de segmentation------------------------------")
    print("Le nombre de segments dans ce fichiers audio..........:",len(parole_ou_non))
    print("Le nombre de segments contenant de la parole..........:",len(list(filter(lambda x: x != 0 , parole_ou_non))))
    print("----------------------------------------------------------------------------------")
    print("---------------------------Module d'extraction de ''features''--------------------")
    print("----------------------------------------------------------------------------------")
    print("--------------------------------Module de 'Clustering'----------------------------")
    print("----------------------------------------------------------------------------------")
def step_to_clusters(sequence,silent_speech,fe,pas=0.5):
    intervals=dict();intervalsData=dict();
    counter=0;longeur_pas=int(pas*fe);
    for iinterval in range (len(silent_speech) ):
        if iinterval !=0 :
            intervals[counter]=( pas*iinterval,(pas+1)*iinterval )
            intervalsData[counter]=sequence[iinterval*longeur_pas:(iinterval+1)*longeur_pas]
            counter+=1
    return intervals,intervalsData

parole_ou_non=sections(seq,fe);


"""
speech=numpy.array(reconstituer(seq,parole_ou_non))
plt.figure()
plt.title(' Parole+Silence')
librosa.display.waveplot(seq, fe)

plt.figure()
plt.title(' Parole +silence eliminé' )
librosa.display.waveplot(speech,fe)


"""
intervals,valuable_data=step_to_clusters(seq_list,parole_ou_non,fe,pas=0.5)
for i in valuable_data:
    mfcc=librosa.feature.mfcc(y=np.array(valuable_data[i] ) , sr=fe)
    print(mfcc.shape)
    print(np.array(mfcc))
    #print([float("{:.2f}".format(x)) for x in np.array(mfcc)])
"""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time')
    plt.title("Module d'extraction MFCCs")
    plt.colorbar()
    plt.tight_layout()
    plt.show()
"""

print_speech_info(parole_ou_non)
