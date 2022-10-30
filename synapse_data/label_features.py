from glob import glob
import csv,time,os
import dateutil.parser
import pandas as pd
import threading
import pickle

labels = "/data/MentalHealth/synapse_data/labels.csv"
label_df = pd.read_csv(labels)

features = []
labels = []

old_time = 0
new_time = 0
i = 0
old_id = 0
label_df = label_df.drop_duplicates()
for index, row in label_df.iterrows():
    if i % 500 == 0:
        print(i,"/",len(label_df),":",i/len(label_df))
    i += 1
    try:
        df = pd.read_csv("/data/MentalHealth/synapse_data/summary_data/" + str(row[0]) +"_summary.csv")
    except:
        continue
    if old_id == row[0]:
        pass
    else:
        old_time = 0
        old_id = row[0]
    new_time = row[1]
    feature_df = df.loc[(df['time'] > old_time) & (df['time'] <= new_time)]
    features.append(feature_df)
    labels.append(row[1:])
    old_time = new_time
    # if len(feature_df) != 0:
    #     mean_df = feature_df.median(axis=0)
    #     features.append(list(mean_df))
    #     labels.append(row[1:])
    #     old_time = new_time

# print(features[500],labels[500])

filehandler = open("features.pkl","wb")
pickle.dump(features,filehandler)
filehandler.close()
with open("features_final.csv","w") as csvfile: 
    writer = csv.writer(csvfile)
    for element in features:
        data = element.values.tolist()
        for element2 in data:
            # s = label_df.Series(list('time'))
            if element2[4] in label_df.time.values:
                real = element2
                real.append(1)
            else:
                real = element2
                real.append(0)
            new = False
            writer.writerow(real)

filehandler = open("labels.pkl","wb")
pickle.dump(labels,filehandler)
filehandler.close()
with open("labels_final.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    for element in labels:
        real = element.values.tolist()
        writer.writerow(real)
