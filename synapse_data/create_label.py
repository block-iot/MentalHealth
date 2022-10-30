
drm = "/data/MentalHealth/synapse_data/mood_data/DRM_edit.csv"
esm = "/data/MentalHealth/synapse_data/mood_data/ESM.csv"

from glob import glob
import csv,time,os
import dateutil.parser
import pandas as pd
import threading

headers = "Participant ID,StartTime,StartTimeStamp,Page1-Time (milliseconds),Page2-Time (milliseconds),Place of the event,Participating people,Activity type,The true-self degree,TIPI-C_1,TIPI-C_2,TIPI-C_3,TIPI-C_4,TIPI-C_5,PANAS_1,PANAS_2,PANAS_3,PANAS_4,PANAS_5,PANAS_6,PANAS_7,PANAS_8,PANAS_9,PANAS_10,Valence,Arousal"

def convert_time(x):
    some_time = dateutil.parser.parse(str(x))
    res = int(time.mktime(some_time.timetuple()))
    return res

def get_panas():
    #Choose labels for ESM
    df = pd.read_csv(esm)
    # ppg['csv_time_PPG'] = ppg['csv_time_PPG'].apply(lambda x: str(int(time.mktime(dateutil.parser.parse(x).timetuple()))))
    esm_panas = df[['Participant ID','StartTime','PANAS_1','PANAS_2','PANAS_3','PANAS_4','PANAS_5','PANAS_6','PANAS_7','PANAS_8','PANAS_9','PANAS_10','Valence','Arousal']].copy()
    # esm_panas = pd.to_datetime(esm_panas['StartTime'])
    esm_panas['StartTime'] = esm_panas['StartTime'].apply(lambda x: int(time.mktime(dateutil.parser.parse(x).timetuple())))
    # print(len(esm_panas))
    df1 = pd.read_csv(drm)
    drm_panas = df1[['Participant ID','StartTime','PANAS_1','PANAS_2','PANAS_3','PANAS_4','PANAS_5','PANAS_6','PANAS_7','PANAS_8','PANAS_9','PANAS_10','Valence','Arousal']].copy()
    # print(drm_panas['Submission time'])
    drm_panas['StartTime'] = drm_panas['StartTime'].apply(lambda x: convert_time(x))
    #.apply(lambda x: str(int(time.mktime(dateutil.parser.parse(x).timetuple()))))
    total_panas=pd.concat([esm_panas,drm_panas])
    total_panas = total_panas.sort_values(['Participant ID','StartTime'])
    return total_panas

def fix_data(file):
    fixed_data = []
    with open(file, mode="r") as employee_file:
        read = csv.reader(employee_file, delimiter=',')
        i = 0
        for row in read:
            if i == 0:
                header = row
            i += 1
            some_time = row[4]
            start = some_time.split("|")[0].strip()
            actual_time = row[2].split(" ")[0]
            res = ' '.join([actual_time,start])
            new_row = row
            new_row[2] = res
            fixed_data.append(new_row)
        employee_file.close()
        with open(file, mode="w") as employee_file:
            employee_writer = csv.writer(employee_file, delimiter=',')
            employee_writer.writerow(header)
            for element in fixed_data:
                employee_writer.writerow(element)
        employee_file.close()

# fix_data("/data/MentalHealth/synapse_data/mood_data/DRM_edit.csv")
total_panas = get_panas()
for col in total_panas.columns.values:
    total_panas[col] = total_panas[col].astype('int64')
total_panas.to_csv("/data/MentalHealth/synapse_data/labels.csv", encoding='utf-8', index=False,columns=['Participant ID','StartTime','PANAS_1','PANAS_2','PANAS_3','PANAS_4','PANAS_5','PANAS_6','PANAS_7','PANAS_8','PANAS_9','PANAS_10','Valence','Arousal'])
# print(total_panas)