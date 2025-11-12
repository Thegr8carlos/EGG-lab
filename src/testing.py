import matplotlib.pyplot as plt


import numpy as np

# get the data and labels

data = np.load("Aux/nieto_inner_speech/sub-01/ses-01/eeg/sub-01_ses-01_task-innerspeech_eeg.npy")
print(data.shape)
labels = np.load("Aux/nieto_inner_speech/sub-01/ses-01/eeg/Labels/sub-01_ses-01_task-innerspeech_eeg.npy")
print(labels.shape)
print(labels)

flatten = labels.flatten()
plt.plot(flatten)
plt.title("Labels")
plt.savefig("labels_plot.png")
print("Plot guardado como labels_plot.png")