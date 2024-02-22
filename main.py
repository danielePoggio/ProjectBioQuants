from Bio import SeqIO
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from methods import knn
from methods import randomForest

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
sequences_reduced, _, y_reduced, _ = train_test_split(df['Sequence'], df['Function'], test_size=0.9)
dfReducedString = []
for i in range(len(sequences_reduced)):
    dfReducedString.append({"Sequence": sequences_reduced.iat[i], "Function": y_reduced.iat[i]})

df = pd.DataFrame(dfReducedString)

leSequence = LabelEncoder()
df['SequenceTransform'] = leSequence.fit_transform(df['Sequence'])

# Random Forest #
statRF = randomForest(dataframe=df, classes=classes, flagplt=True)
print(statRF)

# KNN #
# idea: proteine con sequenze simili hanno funzioni simili -> usiamo distanza di Levenshtein!
# valutiamo il numero migliore di vicini da utilizzare
nTest = 7
accuracyVec = np.zeros((nTest, 1))

for i in range(nTest):
    k = i + 3
    accuracy, other = knn(df, k, classes, False)
    accuracyVec[i] = accuracy

testSample = np.linspace(3, nTest+2, nTest)
fig, ax = plt.subplots()  # Create a figure containing a single axes.
ax.plot(testSample, accuracyVec)  # Plot some data on the axes.
plt.show()

k = 8

accuracy, classes_stat = knn(df, k, classes, True)
print(classes_stat)

# Proviamo KNN con Sequence Aligment #


