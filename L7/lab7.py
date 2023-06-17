import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as mt
def SagmanPERmontg(d, index):
    global F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12
    n = d[index].size
    mounthedoun = ['Январь', 'Февраль', 'Март', 'Апрель', 'Май', 'Июнь', 'Июль', 'Август', 'Сентябрь', 'Октябрь',
                   'Декабрь', 'Ноябрь']
    print('n=', n)
    F0=np.zeros((n))
    i=0; j=0; l=0
    for g in range(len(mounthedoun)):
        while d['Месяц'][i]==mounthedoun[g]:
            F0[j]=d[index][i]
            i=i+1; j=j+1; l=l+1
        globals()['F' + str(g + 1)] = np.zeros((l))
        for l1 in range (0,l):
            globals()['F' + str(g + 1)][l1]=F0[l1]
        print('F1=',globals()['F' + str(g + 1)])
        j=0; l=0
    return F1, F2, F3, F4, F5, F6, F7, F8, F9, F10, F11, F12

def Sum_SagmanPERmontg(d, index, F_in):
    global F_SagmanPERmontg
    n = d[index].size
    F0=np.zeros((n)); F_SagmanPERmontg=np.zeros((12))
    i=0; j=0; l=0
    mounthedoun =['Январь','Февраль','Март','Апрель','Май','Июнь','Июль','Август','Сентябрь','Октябрь','Декабрь','Ноябрь']
    for g in range(len(mounthedoun)):
        while d['Месяц'][i]==mounthedoun[g]:
            F0[j] = F_in[i]
            i=i+1; j=j+1; l=l+1
        for l1 in range (0,l):
            F_SagmanPERmontg[g]=F_SagmanPERmontg[g]+F0[l1]
        j=0; l=0
    print('F_SagmanPERmontg=', F_SagmanPERmontg)
    return  F_SagmanPERmontg

def StatusaA(S, Y_coord, title):
    iter = Y_coord.size
    S0 = np.zeros(iter)
    for i in range(iter):
        S0[i] = abs(S[i] - 0)
    mS=np.mean(S0)
    dS=np.var(S0)
    scvS=mt.sqrt(dS)
    print('----- статистичны характеристики  -----')
    print('матиматичне сподівання ВВ=', mS)
    print('дисперсія ВВ =', dS)
    print('СКВ ВВ=', scvS)
    plt.title(title)
    plt.hist(S,  bins=20, facecolor="blue", alpha=0.5)
    plt.show()
    return StatusaA

def MNK (Y_coord):

    iter = Y_coord.size
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 5))
    for i in range(iter):
        Yin[i, 0] = Y_coord[i]
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
        F[i, 3] = float(i * i * i)
        F[i, 4] = float(i * i * i * i)

    FT=F.T
    FFT = FT.dot(F)
    FFTI=np.linalg.inv(FFT)
    FFTIFT=FFTI.dot(FT)
    C=FFTIFT.dot(Yin)
    Yout=F.dot(C)
    return Yout

def MNK_EXT (Y_coord, koef):

    iter = Y_coord.size
    Yin = np.zeros((iter, 1))
    F = np.ones((iter, 5))
    for i in range(iter):
        Yin[i, 0] = Y_coord[i]
        F[i, 1] = float(i)
        F[i, 2] = float(i * i)
        F[i, 3] = float(i * i * i)
        F[i, 4] = float(i * i * i * i)

    FT=F.T
    FFT = FT.dot(F)
    FFTI=np.linalg.inv(FFT)
    FFTIFT=FFTI.dot(FT)
    C=FFTIFT.dot(Yin)
    Yout=F.dot(C)
    j=koef
    for i in range(0, koef):
        Yout[i, 0] = C[0, 0] + C[1, 0]*j + (C[2, 0]*j*j) + (C[3, 0]*j*j*j) + (C[4, 0]*j*j*j*j)
        j = j+1

    return Yout

def Prodaj(d):
    global F_Prodaj
    n = d['КолВо реализации'].size
    F_Prodaj = np.zeros((n))
    for i in range(0, n):
        F_Prodaj[i] = d['КолВо реализации'][i]*d['Цена реализации'][i]
    print('F_Продаж=', F_Prodaj)
    return F_Prodaj

def Pribel(d):
    global F_Pribel
    n = d['КолВо реализации'].size
    F_Pribel = np.zeros((n))
    for i in range(0, n):
        F_Pribel[i] = d['КолВо реализации'][i]*(d['Цена реализации'][i]-d['Себестоимость единицы'][i])
    print('F_Прибуток=', F_Pribel)
    return F_Pribel


d = pd.read_excel('Pr12.xls', parse_dates=['Дата'])

dd = pd.read_excel('Pr12.xls', parse_dates=['Дата'], index_col='Дата')
print('d=', d)
print('dd=', dd)
print('---------------- Цена реализации -------------------')
print(d['Цена реализации'])
print(type(float(d['Цена реализации'][0])))


index='Цена реализации'
SagmanPERmontg(d, index)
plt.title(index); d[index].plot()
plt.show()
plt.title(index); dd[index].plot()
plt.show()
plt.title('Month_1-6')
plt.plot(F1); plt.plot(F2); plt.plot(F3)
plt.plot(F4); plt.plot(F5); plt.plot(F6)
plt.show()

Prodaj(d)
Sum_SagmanPERmontg(d, index, F_Prodaj)
plt.title('Продаж'); plt.plot(F_Prodaj)
plt.show()
s=pd.Series(F_SagmanPERmontg)
plt.title('Продаж'); s.plot(kind='bar')
plt.show()
F_SagmanPERmontg_Prodaj=F_SagmanPERmontg

Pribel(d)
Sum_SagmanPERmontg(d, index, F_Pribel)
plt.title('Прибуток'); plt.plot(F_Pribel)
plt.show()
s=pd.Series(F_SagmanPERmontg)
plt.title('Прибуток'); s.plot(kind='bar')
plt.show()
F_SagmanPERmontg_Pribel=F_SagmanPERmontg

plt.title('Продаж + Прибуток')
plt.plot(F_Prodaj)
plt.plot(F_Pribel)
plt.show()
s1=pd.Series(F_SagmanPERmontg_Pribel)
s2=pd.Series(F_SagmanPERmontg_Prodaj)
plt.title('Продаж + Прибуток')
s2.plot(kind='bar', color='b')
s1.plot(kind='bar', color='g')

plt.show()


Yout0 = MNK (F_Prodaj)
print('------------ вхідна вибірка  ----------')
StatusaA(F_Prodaj, Yout0, 'вхідна вибірка за рік')
print('-------------- МНК оцінка  ------------')
StatusaA(Yout0, Yout0, 'МНК оцінка за рік')
plt.title('MNK_Продаж')
plt.plot(F_Prodaj)
plt.plot(Yout0)
plt.show()

Yout1 = MNK (F_SagmanPERmontg_Prodaj)
print('------------ вхідна вибірка  ----------')
StatusaA(F_SagmanPERmontg_Prodaj, Yout1, 'вхідна вибірка за місяцями')
print('-------------- МНК оцінка  ------------')
StatusaA(Yout1, Yout1, 'МНК оцінка за місяцями')
plt.title('MNK_SagmanPERmontg_Prodaj')
plt.plot(F_SagmanPERmontg_Prodaj)
plt.plot(Yout1)
plt.show()

prognoz=6
Graf_Yout1=np.zeros(prognoz)
Yout1 = MNK_EXT  (F_SagmanPERmontg_Prodaj, prognoz)
print('------ MNK_EXT_SagmanPERmontg_Prodaj --------')
for i in range(prognoz):
    print ('Yout1[', i, ',0]=', Yout1[i,0])
plt.title('MNK_EXT_SagmanPERmontg_Prodaj')
for i in range(0, prognoz):
    Graf_Yout1[i] = Yout1[i, 0]
plt.plot(Graf_Yout1)
plt.show()
