import pandas as pd
import numpy as py
import time
from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score

start = time.process_time()

print("Sistem Klasifikasi SVM")
print("----------------------")
print("")
print("Masukan metode yang diinginkan: ")
print("1. Klasifikasi full fitur")
print("2. Klasifikasi dengan fitur seleksi chi square")
opt = int(input("Pilihan: "))
if (opt == 1):
    dataset=pd.read_csv('DDoS.csv')
elif (opt == 2):
    dataset=pd.read_csv('DDoS_2.csv') 
else:
    print("Input salah")

x=dataset.drop('Label', axis=1)
y=dataset['Label']
min_max_scaler = preprocessing.MinMaxScaler()
x_data = min_max_scaler.fit_transform(x)
data = pd.DataFrame(x_data)
y=y.replace({'Benign' : 0, 'DoS attacks-Hulk' : 1, 'DoS attacks-SlowHTTPTest' : 1})
x_train, x_test, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=0)
print ("Program Klasifikasi SVM")
gam = float(input("Masukan nilai Gamma: "))
lam = float(input("Masukan nilai Lambda: "))

rms_svm = SVC(kernel='linear', gamma = gam, C= 1/lam)
rms_svm.fit(x_train,y_train)
y_pred=rms_svm.predict(x_test)
accuracy=cross_val_score(estimator=rms_svm,X=x_train,y=y_train,cv=10,scoring='accuracy')
precision=cross_val_score(estimator=rms_svm,X=x_train,y=y_train,cv=10,scoring='precision')
recall=cross_val_score(estimator=rms_svm,X=x_train,y=y_train,cv=10,scoring='recall')
f1=cross_val_score(estimator=rms_svm,X=x_train,y=y_train,cv=10,scoring='f1')

print("========================================")
for index, x in enumerate(accuracy):
    print ("K-Fold ", index, ": ", x)
print("Akurasi Rata-rata :",py.mean(accuracy))
print("Precision :",py.mean(precision))
print("Recall :",py.mean(recall))
print("F1-score :",py.mean(f1))

stop = time.process_time()
waktu = (stop-start)/60
print('Waktu proses sistem :', waktu, 'menit')
