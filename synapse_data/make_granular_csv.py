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
                summary = pd.concat([df,summary])
        summary['time'] = summary['time'].apply(lambda x: int(time.mktime(dateutil.parser.parse(x).timetuple())))
        # ppg['csv_time_PPG'] = ppg['csv_time_PPG'].apply(lambda x: str(int(time.mktime(dateutil.parser.parse(x).timetuple()))))
        # gsr['csv_time_GSR'] = gsr['csv_time_GSR'].apply(lambda x: str(int(time.mktime(dateutil.parser.parse(x).timetuple()))))
        # print(acc.head())
        # exit(0)
        path = "/data/MentalHealth/synapse_data/summary_data/"
        # isExist = os.path.exists(path)
        # if not isExist:
        #     os.makedirs(path)
        t1 = threading.Thread(target=lambda: gsr.to_csv(path+path1+"_summary.csv", encoding='utf-8', index=False))
        t1.start()
        t1.join()
        # t2.join()
        # t3.join()
# make_intermediate(all_patients)
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

# t4.join()
# t5.join()
# t6.join()
# t7.join()
# t8.join()
# t9.join()
# t10.join()
# t11.join()



    # acc.to_csv()
    # gsr.to_csv(path+"GSR.csv", encoding='utf-8', index=False)
    # ppg.to_csv(path+"PPG.csv", encoding='utf-8', index=False)
    # exit(0)
        #ACC!
        # if "ACC" in file:
        #     with open(file) as csv_file:
        #         csv_reader = csv.reader(csv_file, delimiter=',')
        #         row_count = sum1forline(file)
        #         print("New File",file,"Total Rows:",row_count)
        #         line_count = 0
        #         for row in csv_reader:
        #             if line_count == 0:
        #                 if row == ["Motion_dataX","Motion_dataY","Motion_dataZ","csv_time_motion"]:
        #                     header = ["Motion_dataX","Motion_dataY","Motion_dataZ","csv_time_motion"]
        #                     pass
        #                 else:
        #                     print("error in acc")
        #                     exit(0)
        #                 line_count += 1
        #             else:
        #                 the_time = dateutil.parser.parse(row[3])
        #                 time_unix = str(int(time.mktime(the_time.timetuple())))
        #                 directory_name = str(directory.split("/")[-2])
        #                 write_csv("/data/MentalHealth/synapse_data/simplified_data/"+directory_name+"/ACC.csv",[row[0],row[1],row[2],time_unix],header)
        #                 line_count += 1
        #             if line_count % 100000 == 0:
        #                 print("Current:",line_count,"Total:",row_count,"Percentage:",line_count/row_count) 
        #             if line_count == row_count -1:
        #                 print("Finished File")
        # if "GSR" in file:
        #     with open(file) as csv_file:
        #         csv_reader = csv.reader(csv_file, delimiter=',')
        #         row_count = sum1forline(file)
        #         print("New File",file,"Total Rows:",row_count)
        #         line_count = 0
        #         for row in csv_reader:
        #             if line_count == 0:
        #                 if row == ["GSR","csv_time_GSR"]:
        #                     header = ["GSR","csv_time_GSR"]
        #                     pass
        #                 else:
        #                     print("error in gsr")
        #                     exit(0)
        #                 line_count += 1
        #             else:
        #                 the_time = dateutil.parser.parse(row[1])
        #                 time_unix = str(int(time.mktime(the_time.timetuple())))
        #                 directory_name = str(directory.split("/")[-2])
        #                 write_csv("/data/MentalHealth/synapse_data/simplified_data/"+directory_name+"/GSR.csv",[row[0],time_unix],header)
        #                 line_count += 1
        #             if line_count % 100000 == 0:
        #                 print("Current:",line_count,"Total:",row_count,"Percentage:",line_count/row_count) 
        #             if line_count == row_count -1:
        #                 print("Finished File")
        # if "PPG" in file:
        #     with open(file) as csv_file:
        #         csv_reader = csv.reader(csv_file, delimiter=',')
        #         row_count = sum1forline(file)
        #         print("New File",file,"Total Rows:",row_count)
        #         line_count = 0
        #         for row in csv_reader:
        #             if line_count == 0:
        #                 if row == ["PPG","csv_time_PPG"]:
        #                     header = ["PPG","csv_time_PPG"]
        #                     pass
        #                 else:
        #                     print("error in ppg")
        #                     exit(0)
        #                 line_count += 1
        #             else:
        #                 the_time = dateutil.parser.parse(row[1])
        #                 time_unix = str(int(time.mktime(the_time.timetuple())))
        #                 directory_name = str(directory.split("/")[-2])
        #                 write_csv("/data/MentalHealth/synapse_data/simplified_data/"+directory_name+"/PPG.csv",[row[0],time_unix],header)
        #                 line_count += 1
        #             if line_count % 100000 == 0:
        #                 print("Current:",line_count,"Total:",row_count,"Percentage:",line_count/row_count) 
        #             if line_count == row_count -1:
        #                 print("Finished File")
        # if "PPG" not in file and "GSR" not in file and "ACC" not in file:
        #     with open(file) as csv_file:
        #         csv_reader = csv.reader(csv_file, delimiter=',')
        #         row_count = sum1forline(file)
        #         print("New File",file,"Total Rows:",row_count)
        #         line_count = 0
        #         for row in csv_reader:
        #             if line_count == 0:
        #                 if row == ["PPG","csv_time_PPG"]:
        #                     header = ["PPG","csv_time_PPG"]
        #                     pass
        #                 else:
        #                     print("error in ppg")
        #                     exit(0)
        #                 line_count += 1
        #             else:
        #                 the_time = dateutil.parser.parse(row[1])
        #                 time_unix = str(int(time.mktime(the_time.timetuple())))
        #                 directory_name = str(directory.split("/")[-2])
        #                 write_csv("/data/MentalHealth/synapse_data/simplified_data/"+directory_name+"/PPG.csv",[row[0],time_unix],header)
        #                 line_count += 1
        #             if line_count % 100000 == 0:
        #                 print("Current:",line_count,"Total:",row_count,"Percentage:",line_count/row_count) 
        #             if line_count == row_count -1:
        #                 print("Finished File")
        