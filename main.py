from Bio import SeqIO
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from Levenshtein import distance
from sklearn.neighbors import KNeighborsClassifier
from calcolatePerformance import calcolateModel
from sequenceAlignment import sequenceAlignment
import numpy as np

# Definiamo qualche categoria di proteine sulle quali lavorare
"""
Definiamo qualche categoria di proteine sulle quali lavorare:
1. Enzimi
2. Recettori
3. Immunoglobuline
4. Fattori di Trascrizione
"""
sequences = []
classes = ["Enzyme", "Receptor", "TF", "IG"]
# Enzimi
fasta_file = "C:/Users/39334/Desktop/Poli/BioQuants/ProjectBioQuants/EnzymeHuman.fasta"
for record in SeqIO.parse(fasta_file, "fasta"):
    sequences.append({"ID": record.id, "Sequence": str(record.seq), "Function": "Enzyme"})

# Recettori
fasta_file = "C:/Users/39334/Desktop/Poli/BioQuants/ProjectBioQuants/ReceptorHuman.fasta"
for record in SeqIO.parse(fasta_file, "fasta"):
    sequences.append({"ID": record.id, "Sequence": str(record.seq), "Function": "Receptor"})

# Immonuglobuline IG
fasta_file = "C:/Users/39334/Desktop/Poli/BioQuants/ProjectBioQuants/IGHuman.fasta"
for record in SeqIO.parse(fasta_file, "fasta"):
    sequences.append({"ID": record.id, "Sequence": str(record.seq), "Function": "IG"})

# Fattori di Trascrizione TF
fasta_file = "C:/Users/39334/Desktop/Poli/BioQuants/ProjectBioQuants/IGHuman.fasta"
for record in SeqIO.parse(fasta_file, "fasta"):
    sequences.append({"ID": record.id, "Sequence": str(record.seq), "Function": "TF"})

# Creazione DataFrame
df = pd.DataFrame(sequences)

# PreProcessing

# Creo un dataset ridotto perchè ho troppi dati
sequences_reduced, _, y_reduced, _ = train_test_split(df['Sequence'], df['Function'], test_size=0.99)
dfReducedString = []
for i in range(len(sequences_reduced)):
    dfReducedString.append({"Sequence": sequences_reduced.iat[i], "Function": y_reduced.iat[i]})

df = pd.DataFrame(dfReducedString)

leSequence = LabelEncoder()
df['SequenceTransform'] = leSequence.fit_transform(df['Sequence'])


###### Random Forest #############################
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['SequenceTransform'], df['Function'], test_size=0.3)
# per funzionare gli alberi hanno bisogno di array non series
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()
X_train = X_train.reshape(-1, 1)
X_test = X_test.reshape(-1, 1)
# Fit a model to the training data
model = RandomForestClassifier()
model.fit(X_train, y_train)
# Make predictions on the testing data
y_pred = model.predict(X_test)
staticisRFC = calcolateModel(y_test, y_pred, classes)

############################# KNN ############################################
# idea: proteine con sequenze simili hanno funzioni simili -> usiamo distanza di Levenshtein!
X_train, X_test, y_train, y_test = train_test_split(df['Sequence'], df['Function'], test_size=0.3)
# per funzionare gli alberi hanno bisogno di array non series
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()
X_train = X_train.reshape(-1, 1)
# y_train = y_train.reshape(-1,1)
X_test = X_test.reshape(-1, 1)
distanceTrainTrain = np.zeros((X_train.size, X_train.size))
for i in range(X_train.size):
    for j in range(i, X_train.size):
        xi = X_train[i]
        xj = X_train[j]
        xi = xi[0]
        xj = xj[0]
        distanceTrainTrain[i, j] = distance(xi, xj)
        distanceTrainTrain[j, i] = distanceTrainTrain[i, j]

distanceTestTrain = np.zeros((X_test.size, X_train.size))
for i in range(X_test.size):
    for j in range(X_train.size):
        xi = X_test[i]
        xj = X_train[j]
        xi = xi[0]
        xj = xj[0]
        distanceTestTrain[i, j] = distance(xi, xj)

neigh = KNeighborsClassifier(n_neighbors=3, metric="precomputed")
neigh.fit(distanceTrainTrain, y_train)
y_predKNN = neigh.predict(distanceTestTrain)

staticisKNN = calcolateModel(y_test, y_predKNN, classes)

################################# Proviamo KNN con Sequence Aligment

X_train, X_test, y_train, y_test = train_test_split(df['Sequence'], df['Function'], test_size=0.3)
# per funzionare gli alberi hanno bisogno di array non series
X_train = X_train.to_numpy()
y_train = y_train.to_numpy()
X_test = X_test.to_numpy()
y_test = y_test.to_numpy()
X_train = X_train.reshape(-1, 1)
# y_train = y_train.reshape(-1,1)
X_test = X_test.reshape(-1, 1)
distanceTrainTrain = np.zeros((X_train.size, X_train.size))
for i in range(X_train.size):
    for j in range(i, X_train.size):
        xi = X_train[i]
        xj = X_train[j]
        xi = xi[0]
        xj = xj[0]
        distanceTrainTrain[i, j] = sequenceAlignment(xi, xj)
        distanceTrainTrain[j, i] = distanceTrainTrain[i, j]

distanceTestTrain = np.zeros((X_test.size, X_train.size))
for i in range(X_test.size):
    for j in range(X_train.size):
        xi = X_test[i]
        xj = X_train[j]
        xi = xi[0]
        xj = xj[0]
        distanceTestTrain[i, j] = sequenceAlignment(xi, xj)

neigh = KNeighborsClassifier(n_neighbors=3, metric="precomputed")
neigh.fit(distanceTrainTrain, y_train)
y_predKNN = neigh.predict(distanceTestTrain)

staticisKNN = calcolateModel(y_test, y_predKNN, classes)
