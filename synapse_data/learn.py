import pickle,statistics
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import pandas as pd

from sklearn.model_selection import train_test_split
import numpy


with open('labels.pkl', 'rb') as f:
    labels = pickle.load(f)

with open('features.pkl', 'rb') as f:
    features = pickle.load(f)

new_label = []
#Simplify labels
for element in labels:
    res = element.values.tolist()[1:]
    the_sum = sum(res)/len(res)
    label = 0
    if the_sum > 2.5:
        label = 1
    if the_sum <= 2.5:
        label = 0
    new_label.append(label)

#Simplify Features
i = 0
to_remove = []
new_features = []
for element in features:
    i += 1
    if element.empty:
        to_remove.append(i)
        new_features.append(1000)
        continue
    try:
        res = pd.Series(element.GSR).values.tolist()
        # average_GSR = statistics.stdev(res)
        average_GSR = statistics.mean(res)
        res = pd.Series(element.heart_rate).values.tolist()
        # average_hr = statistics.stdev(res)
        average_hr = statistics.mean(res)
        res = pd.Series(element.motion).values.tolist()
        # average_motion = statistics.stdev(res)
        average_motion = statistics.mean(res)
        new_features.append([average_GSR,average_motion,average_hr])
    except:
        to_remove.append(i)
        new_features.append(1000)
        continue
for element in to_remove:
    new_label[element] = 1000

#Combine Feature and Labels
# i = 0
# to_remove = []
# new_label = []
# new_features = []
# for element in features:
#     if element.empty:
#         i += 1
#         continue
#     for element2 in element.values.tolist():
#         new_features.append([element2[0],element2[1],element2[2]])
#         res = labels[i].values.tolist()[1:]
#         the_sum = sum(res)
#         label = 0
#         if the_sum > 30:
#             label = 1
#         if the_sum <= 30:
#             label = 0
#         new_label.append(label)
#         # new_label.append(labels[i][4])
#     i += 1
# for element in to_remove:
#     new_label[element] = 1000

# new_label = list(filter((1000).__ne__, new_label))
# new_features = list(filter((1000).__ne__, new_features))
# print(new_features)
print(len(new_features))
print(len(new_label))

#PANAS_5 worked best. 
new_labels = []
# for element in labels:
#     new_labels.append(element[4])
labels = new_label
features = new_features
feature_train = features[:int((len(features)+1)*0.7)]
feature_test = features[int((len(features)+1)*0.3):]
labels_train = labels[:int((len(labels)+1)*0.7)]
labels_test = labels[int((len(labels)+1)*0.3):]
# print(np.array(feature_test))
# print(np.array(labels_test))
# exit()
norm_abalone_model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation='relu'),
  tf.keras.layers.Dense(10, activation='relu'),
#   tf.keras.layers.Dense(10, activation='relu'),
  tf.keras.layers.Dense(10, activation='sigmoid'),
  tf.keras.layers.Dense(1, activation = 'softmax')])


# input_data = Input(shape=(None,), ragged=True)
# tf.keras.layers.Dense(32),
norm_abalone_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])
norm_abalone_model.fit(np.array(feature_train), np.array(labels_train),epochs = 100)

test_loss, test_acc = norm_abalone_model.evaluate(np.array(feature_test), np.array(labels_test), verbose=2)
print('\nTest accuracy:', test_acc)

'''
All results: Tensor --> each PANAS Value: Max accuracy was 56%
Means of each feature --> binary label: 52.5%
Modes of each feature --> binary label: 52.5%
Population and sample Stdev of each feature --> binary label: 52.5%
Adding label to every feature --> accuracy = 42.25%
Messing around with dense layers = no significant difference
Timeseries to use one feature to predict next: Loss = 0.6992
See visualizations
Adding labels to get positive/negative affect- no improvement
Averaging labels to get positive/negative affect- no improvement
Just doing each of GSR, HR, and Motion alone. 

'''


# feature_tensors = []
# the_max = 0
# i = 0
# for element in feature_train:
#     if len(element) > the_max:
#         the_max = len(element.index)
#     element = element.astype('int')
#     df_final = element.reindex(range(35429), fill_value=0)
#     t = tf.convert_to_tensor(df_final)
#     t = tf.transpose(t)
#     feature_tensors.append(t)

# feature_tests = []
# for element in feature_test:
#     if len(element) > the_max:
#         the_max = len(element.index)
#     element = element.astype('int')
#     df_final = element.reindex(range(35429), fill_value=0)
#     t = tf.convert_to_tensor(df_final)
#     t = tf.transpose(t)
#     feature_tests.append(t)

# print(feature_tensors[30].shape)
# print(labels_train[30].shape)
# exit()

# normalize = layers.Normalization()
# normalize.adapt(abalone_features)

