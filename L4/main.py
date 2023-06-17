import pandas as pd
import numpy as np

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


r = 15
c = 12
strt = ExelPars("Lab4.xls", c, r)
parsing = list(strt.OutExel())
for i in parsing:
    print((str(i).replace("[", "")).replace("]", ""))
Al_Voronin([1 / sum([1 for i in range(15 + 1)]) for j in range(15 + 1)], parsing, 12)


