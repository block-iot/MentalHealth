from glob import glob
import csv,time,os
import dateutil.parser
import pandas as pd
import threading

def write_csv(file,data,header):
    all_files = glob("/data/MentalHealth/synapse_data/simplified_data/*/*.csv", recursive = True)
    if file in all_files:
        pass
    else:
        path = '/'.join(file.split("/")[:-1])+"/"
        isExist = os.path.exists(path)
        if not isExist:
            os.makedirs(path)
        with open(file, mode="w") as employee_file:
            employee_writer = csv.writer(employee_file, delimiter=',')
            employee_writer.writerow(header)
    with open(file, mode="a") as employee_file:
        employee_writer = csv.writer(employee_file, delimiter=',')
        employee_writer.writerow(data)

def sum1forline(filename):
    with open(filename) as f:
        return sum(1 for line in f)

all_patients = glob("/data/MentalHealth/synapse_data/Physiol_Rec/*/", recursive = True)
header = []
def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))

parts= list(split(all_patients,8))
def make_intermediate(all_patients):
    for directory in all_patients:
        files = glob(directory+"*", recursive = True)
        summary = pd.DataFrame()
        path1 = ""
        for file in files:
            path1 = file.split("/")[-2]
            if "ACC" not in file and "PPG" not in file and "GSR" not in file:
                df = pd.read_csv(file)
                df['time'] = df['time'].apply(lambda x: int(time.mktime(dateutil.parser.parse(x).timetuple())))
                summary = pd.concat([df,summary])
        # summary['time'] = summary['time'].apply(lambda x: int(time.mktime(dateutil.parser.parse(x).timetuple())))
        print(path1)
        # ppg['csv_time_PPG'] = ppg['csv_time_PPG'].apply(lambda x: str(int(time.mktime(dateutil.parser.parse(x).timetuple()))))
        # gsr['csv_time_GSR'] = gsr['csv_time_GSR'].apply(lambda x: str(int(time.mktime(dateutil.parser.parse(x).timetuple()))))
        # print(acc.head())
        # exit(0)
        path = "/data/MentalHealth/synapse_data/summary_data/"
        # isExist = os.path.exists(path)
        # if not isExist:
        #     os.makedirs(path)
        t1 = threading.Thread(target=lambda: summary.to_csv(path+path1+"_summary.csv", encoding='utf-8', index=False))
        t1.start()
        t1.join()
        # exit(0)
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
