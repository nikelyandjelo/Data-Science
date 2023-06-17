import pandas as pd
import numpy as np
import pylab
from mpl_toolkits.mplot3d import Axes3D
class ExelPars:

    def __init__(self, file, mass_length, count):
        self.file = pd.read_excel(file)
        self.mass_length = mass_length
        self.count = count
        print(self.file)


    def OutExel(self, ):
        names = self.file.columns[1:-1]

        for i in range(self.count):
            globals()['F' + str(i + 1)] = [0 for _ in range(self.mass_length)]
            for j in range(self.mass_length):
                globals()['F' + str(i + 1)][j] = float(str(self.file[names[j]][i]).replace(',', '.'))
            yield globals()['F' + str(i + 1)]

def Al_Voronin(mass_G, mass_F, mass_length):
    Integro = np.zeros((mass_length))

    for i in range(len(mass_F)):
        globals()['sum_F' + str(i + 1)] = 0
        for j in range(len(mass_F[0])):
            if i == 5:
                globals()['sum_F' + str(i + 1)] += 1 / mass_F[i][j]
            else:
                globals()['sum_F' + str(i + 1)] += mass_F[i][j]

    for j in range(len(mass_F)):
        globals()['F' + str(j + 1) + "0"] = [0 for _ in range(mass_length)]

        for i in range(len(mass_F[0])):
            if j == 5:
                globals()['F' + str(j + 1) + "0"][i] = (1 / mass_F[j][i]) / globals()['sum_F' + str(j + 1)]
            else:
                globals()['F' + str(j + 1) + "0"][i] = mass_F[j][i] / globals()['sum_F' + str(j + 1)]

    for j in range(len(mass_F[0])):

        for i in range(len(mass_F)):
            Integro[j] += mass_G[i] * (1 - globals()['F' + str(i + 1) + "0"][j]) ** (-1)

    min = 10000
    opt = 0
    for i in range(len(Integro)):
        if min > Integro[i]:
            min = Integro[i]
            opt = i;

    print('Integro', Integro)
    print('Оптимальний час роботи у колонці -', opt)
    return Integro

def OLAP():
    xmas = []
    for i in range(c + 1):
        xmas.append(np.ones((c)))
    for i in range(len(parsing[0])):
        xmas[0][i] = i
        if i != 0:
            j = 0
            while j != len(parsing[0]):
                xmas[i][j] = i + 1
                j = j + 1
    for i in range(len(parsing[0])):
        xmas[12][i] = 13
    xs1 = xmas[0];
    ys1 = np.ones((12));
    zs1 = parsing[0]
    xs = []
    ys = []
    zs = []
    for i in range(len(parsing[0])):
        xs.append(xmas[0])
        ys.append(xmas[i])
        zs.append(parsing[i])
    xs.append(xmas[0])
    ys.append(xmas[12])
    zs.append(Integro)
    fig = pylab.figure()
    ax = Axes3D(fig)
    ax.scatter(xs1, ys1, zs1)
    for i in range(len(xmas) - 1):
        if i % 2 == 1:
            a = 'r'
            b = '^'
        else:
            a = 'g'
            b = '*'
        ax.scatter(xs[i], ys[i], zs[i], c=a, marker=b)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_xlabel('X Label')
    pylab.show()

    fig = pylab.figure()
    ax = Axes3D(fig)
    clr = ['#4bb2c5', '#c5b47f', '#EAA228', '#579575', '#839557', '#958c12', '#953579', '#4b5de4', '#4bb2c5']
    ax.bar(xs1, ys1, 1, zdir='y', color=clr)
    for i in range(len(xmas) - 1):
        ax.bar(xs[i], zs[i], i, zdir='y', color=clr)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_xlabel('X Label')
    pylab.show()
r = 15
c = 12
strt = ExelPars("Lab4.xls", c, r)
parsing = list(strt.OutExel())
for i in parsing:
    print((str(i).replace("[", "")).replace("]", ""))
Integro = Al_Voronin([1 / sum([1 for i in range(15 + 1)]) for j in range(15 + 1)], parsing, 12)
OLAP()


