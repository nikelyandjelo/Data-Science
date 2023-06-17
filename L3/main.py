import numpy as np
import math as mt
import matplotlib.pyplot as plt
from typing import Union, List
class lab2 ():
    def __init__(self):
        self.n = 10000;
        self.namber = int(self.n)
        VidAnomalii= 40;
        self.nAV = int((self.namber * VidAnomalii) / 100)
        self.dm = 0;
        self.dsig = 5
        self.SecAnomV = np.zeros((self.nAV));
        self.SSAV = np.zeros((self.nAV))

        self.Normal()
        self.anomalii()
        self.static()
        self.MNKD()
        self.Filter()

    def outliers_tietjen(self,
            x: Union[List, np.ndarray],
            k: int,
            hypo: bool = False,
            alpha: float = 0.05) -> Union[np.ndarray, bool]:
        arr = np.copy(x)
        n = arr.size

        def tietjen(x_, k_):
            x_mean = x_.mean()
            r = np.abs(x_ - x_mean)
            z = x_[r.argsort()]
            E = np.sum((z[:-k_] - z[:-k_].mean()) ** 2) / np.sum((z - x_mean) ** 2)
            return E

        e_x = tietjen(arr, k)
        e_norm = np.zeros(10000)

        for i in np.arange(10000):
            norm = np.random.normal(size=n)
            e_norm[i] = tietjen(norm, k)

        CV = np.percentile(e_norm, alpha * 100)
        result = e_x < CV

        if hypo:
            return result
        else:
            if result:
                ind = np.argpartition(np.abs(arr - arr.mean()), -k)[-k:]
                return np.delete(arr, ind)
            else:
                return arr

    def staticP(self,mS, dS, scvS):
        print('математичне сподівання ВВ=', mS)
        print('дисперсія ВВ =', dS)
        print('СКВ ВВ=', scvS)

    def MNK(self,Yin, F):
        FT = F.T
        FFT = FT.dot(F)
        FFTI = np.linalg.inv(FFT)
        FFTIFT = FFTI.dot(FT)
        C = FFTIFT.dot(Yin)
        Yout = F.dot(C)
        return Yout

    def Normal(self):
        for i in range(self.nAV):
            self.SecAnomV[i] = mt.ceil(np.random.randint(1, self.namber))
        self.S = np.random.normal(self.dm, 3 * self.dsig,self.namber)
        mS = np.median(self.S)
        dS = np.var(self.S)
        scvS = mt.sqrt(dS)
        print('------- статистичні характеристики НОРМАЛЬНОЇ похибки вимірів -----')
        print('матриця реалізацій ВВ=', self.S)
        print('математичне сподівання ВВ=', mS)
        print('дисперсія ВВ =', dS)
        print('СКВ ВВ=', scvS)
        print('------------------------------------------------------------------')
        # гістограма закону розподілу ВВ
        plt.hist(self.S, bins=20, facecolor="blue", alpha=0.5)
        plt.show()

    def anomalii(self):
        self.SV = np.zeros((self.n));
        self.S0 = np.zeros((self.n));
        self.SV0 = np.zeros((self.n));
        self.SV_AV = np.zeros((self.n))
        for i in range(self.n):
            self.S0 [i] = (0.0000005 * i * i)
            self.SV[i] = self.S0 [i] + self.S[i]
            self.SV0[i] = abs(self.SV[i] - self.S0 [i])
            self.SV_AV[i] = self.SV[i]
        SSAV = np.random.normal(self.dm, (10 * self.dsig), self.nAV)
        for i in range(self.nAV):
            k = int(self.SecAnomV[i])
            self.SV_AV[k] = self.S0 [k] + SSAV[i]

        plt.axis([0, 10000, -150, 200])
        plt.plot(self.SV)
        plt.plot(self.S0 )
        plt.ylabel('графіки тренда, вимірів з нормальним шумом')
        plt.show()

        plt.axis([0, 10000, -150, 200])
        plt.plot(self.SV_AV)
        plt.plot(self.S0 )
        plt.ylabel('графіки тренда, вимірів  аномаліями ')
        plt.show()

        self.a = self.outliers_tietjen(self.SV_AV, 1000, hypo=False, alpha=0.5)
        print('-------- Кількість даних без аномалій ----------')
        print(len(self.a))
        print('-------- Аномальні дані будуть відкинуті ----------')
        plt.axis([0, 9000, -150, 200])
        plt.plot(self.a)
        plt.plot(self.S0)
        plt.ylabel('видалені аномаліі')
        plt.show()

    def static(self):
        mSV0 = np.median(self.SV0)
        dSV0 = np.var(self.SV0)
        scvSV0 = mt.sqrt(dSV0)
        print('-------- статистичні характеристики виміряної вибірки без АВ ----------')
        self.staticP(mSV0, dSV0, scvSV0)

        SV_AV0 = np.zeros((self.n))
        for i in range(self.n):
            SV_AV0[i] = abs(self.SV_AV[i] - self.S0[i])

        print('-- статистичні характеристики виміряної вибірки за НАЯВНОСТІ АВ -------')
        mSV_AS = np.median(SV_AV0)
        dSV_AV = np.var(self.SV_AV)
        scvSV_AV = mt.sqrt(dSV_AV)
        self.staticP(mSV_AS, dSV_AV, scvSV_AV)

        plt.hist(self.S, bins=20, alpha=0.5, label='S')
        plt.hist(self.SV0, bins=20, alpha=0.5, label='S1')
        plt.hist(self.SV, bins=20, alpha=0.5, label='S3')
        plt.hist(self.SV_AV, bins=20, alpha=0.5, label='S3')
        plt.show()

    def Filte_A_B(self,Yin):
        YoutAB = np.zeros((9000, 1))
        T0 = 1
        Yspeed_retro = (Yin[1, 0] - Yin[0, 0]) / T0
        Yextra = Yin[0, 0] + Yspeed_retro
        alfa = 2 * (2 * 1 - 1) / (1 * (1 + 1))
        beta = (6 / 1) * (1 + 1)
        YoutAB[0, 0] = Yin[0, 0] + alfa * (Yin[0, 0])
        for i in range(1, 9000):
            YoutAB[i, 0] = Yextra + alfa * (Yin[i, 0] - Yextra)
            Yspeed = Yspeed_retro + (beta / T0) * (Yin[i, 0] - Yextra)
            Yspeed_retro = Yspeed
            Yextra = YoutAB[i, 0] + Yspeed_retro
            alfa = (2 * (2 * i - 1)) / (i * (i + 1))
            beta = 6 / (i * (i + 1))
        print('Yin=', Yin, 'YoutAB=', YoutAB)
        return YoutAB

    def MNKD(self):
        self.Yin = np.zeros((self.namber, 1));
        F = np.ones((self.namber, 3))
        for i in range(self.namber):
            self.Yin[i, 0] = float(self.S0[i])
            F[i, 1] = float(i);
            F[i, 2] = float(i * i)

        Yout0 = self.MNK(self.Yin, F)
        for i in range(self.namber):
            self.Yin[i, 0] = float(self.SV[i])
        Yout1 = self.MNK(self.Yin, F)
        for i in range(self.namber):
            self.Yin[i, 0] = float(self.SV_AV[i])
        Yout2 = self.MNK(self.Yin, F)
        Yout00 = np.zeros((self.n));
        Yout10 = np.zeros((self.n));
        self.Yout20 = np.zeros((self.n));
        for i in range(self.n):
            Yout00[i] = abs(Yout0[i] - self.S0[i])
            Yout10[i] = abs(Yout1[i] - self.S0[i])
            self.Yout20[i] = abs(Yout2[i] - self.S0[i])

        print('--------------------------- статистичні характеристики згладженої вибірки  ----------------------------')
        mYout00 = np.median(Yout00);
        mYout10 = np.median(Yout10);
        mYout20 = np.median(self.Yout20)
        dYout00 = np.var(Yout00);
        dYout10 = np.var(Yout10);
        dYout20 = np.var(self.Yout20)
        scvYout00 = mt.sqrt(dYout00);
        scvYout10 = mt.sqrt(dYout10);
        scvYout20 = mt.sqrt(dYout20)
        print(
            '--------------------------- за відсутності похибок ----- похибки нормальні ------- похибки аномальні ---')

        print('математичне сподівання ВВ3=', mYout00, '----', mYout10, '----', mYout20)
        print('дисперсія ВВ3 =            ', dYout00, '----', dYout10, '----', dYout20)
        print('СКВ ВВ3=                   ', scvYout00, '----', scvYout10, '----', scvYout20)
        print('-------------------------------------------------------------------------------------------------------')

        plt.plot(self.Yin)
        plt.plot(Yout0)
        plt.plot(Yout1)
        plt.plot(Yout2)
        plt.ylabel('статистичні характеристики згладженої вибірки')
        plt.show()

        plt.hist(self.S, bins=20, alpha=0.5, label='SV0')
        plt.hist(self.Yout20, bins=20, alpha=0.5, label='Yout20')
        plt.hist(Yout10, bins=20, alpha=0.5, label='Yout10')
        plt.show()

    def Filter(self):
        self.Yin = np.zeros((9000, 1));
        for i in range(9000):
            self.Yin[i, 0] = float(self.a[i])
        YoutABG = self.Filte_A_B(self.Yin)

        plt.plot(self.Yin)
        plt.plot(YoutABG)
        plt.show()

        Yout0AB = np.zeros((9000))
        for i in range(9000):
            Yout0AB[i] = abs(YoutABG[i] - self.S0[i])

        plt.hist(self.Yout20, bins=20, alpha=0.5, label='Yout20')
        plt.hist(Yout0AB, bins=20, alpha=0.5, label='Yout0AB')
        plt.show()


lab2()
import random
n = 10
arr = [random.randint(0,10) for _ in range(n)]
print(arr)
print(sum(arr))




