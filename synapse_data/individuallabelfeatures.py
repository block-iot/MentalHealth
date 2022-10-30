from glob import glob
import csv,time,os
import dateutil.parser
import pandas as pd
import threading


all_patients = glob("/data/MentalHealth/synapse_data/new_simplified_data/*/", recursive = True)
print(len(all_patients))
header = []
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

parts= list(split(all_patients,8))
def make_intermediate(all_patients):
    for directory in all_patients:
        files = glob(directory+"*", recursive = True)
        acc = pd.DataFrame()
        ppg = pd.DataFrame()
        gsr = pd.DataFrame()
        path1 = ""
        for file in files:
            print(file)
            path1 = file.split("/")[-2]
            if "ACC" in file:
                acc = pd.read_csv(file)
                acc['csv_time_motion'] = acc['csv_time_motion'].apply(lambda x: str(int(time.mktime(dateutil.parser.parse(x).timetuple()))))
                acc = acc.sort_values('csv_time_motion')
                print("ACC")
            # if "PPG" in file:
            #     ppg = pd.read_csv(file)
            #     ppg['csv_time_PPG'] = ppg['csv_time_PPG'].apply(lambda x: str(int(time.mktime(dateutil.parser.parse(x).timetuple()))))
            #     ppg = ppg.sort_values('csv_time_PPG')
            if "GSR" in file:
                gsr = pd.read_csv(file)
                print(gsr.head())
                exit(0)
                # gsr = pd.to_datetime(gsr['csv_time_GSR'])
                # gsr = gsr[[str(int(time.mktime(dateutil.parser.parse(x).timetuple()))) for x in zip(gsr['csv_time_GSR'])]]
                # gsr['csv_time_GSR'] = gsr['csv_time_GSR'].apply(lambda x: str(int(time.mktime(dateutil.parser.parse(x).timetuple()))))
                gsr = gsr.sort_values('csv_time_GSR')
                print("GSR")
        df_cd = pd.merge(acc,gsr, how='outer',left_on="csv_time_motion",right_on="csv_time_GSR")
        display(df_cd)
        exit(0)
            
        # print(acc.head())
        # exit(0)
        path = "/data/MentalHealth/synapse_data/new_simplified_data/"+str(path1)+"/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        t1 = threading.Thread(target=lambda: gsr.to_csv(path+"GSR.csv", encoding='utf-8', index=False))
        t1.start()
        t2 = threading.Thread(target=lambda: acc.to_csv(path+"ACC.csv", encoding='utf-8', index=False))
        t2.start()
        t3 = threading.Thread(target=lambda: ppg.to_csv(path+"PPG.csv", encoding='utf-8', index=False))
        t3.start()
        # t1.join()
        # t2.join()
        # t3.join()
make_intermediate(all_patients)
exit(0)
t4 = threading.Thread(target=lambda:make_intermediate(parts[0]))
t4.start()
t5 = threading.Thread(target=lambda:make_intermediate(parts[1]))
t5.start()
t6 = threading.Thread(target=lambda:make_intermediate(parts[2]))
t6.start()
t7 = threading.Thread(target=lambda:make_intermediate(parts[3]))
t7.start()
t8 = threading.Thread(target=lambda:make_intermediate(parts[4]))
t8.start()
t9 = threading.Thread(target=lambda:make_intermediate(parts[5]))
t9.start()
t10 = threading.Thread(target=lambda:make_intermediate(parts[6]))
t10.start()
t11 = threading.Thread(target=lambda:make_intermediate(parts[7]))
t11.start()