import pandas as pd
import numpy as np
import math
import time
import matplotlib.pyplot as plt


class SDE:
    '''
    pricing for derivatives with SDE defined by
    S_j_i1=(r-rf)*S_j_i0*unit_step+sigma*S_j_i0*pow(unit_step,0.5)*normal_rand
    '''
    def __init__(self,r,rf,T,St0,sigma,K,N=100):
        self.r=r
        self.rf=rf
        self.T=T
        self.St0=St0
        self.sigma=sigma
        self.K=K
        self.N=N

        np.random.seed(0)
        self.rn_mat = np.random.normal(size=(N, T))
        self.simulation()

    def trajactory(self,S_j_i0,rand):
        S_j_i1 = (1 + (self.r - self.rf) * 1.0 / 365) * S_j_i0 + self.sigma * S_j_i0 * math.sqrt(1.0 / 365) * rand
        return S_j_i1

    def simulation(self):

        self.S_df=pd.DataFrame(index=['trajactory_%d'%i for i in range(1,self.N+1)], \
                          columns=['day_%d'%j for j in range(self.T+1)])

        self.S_df['day_0']=self.St0

        for i in range(self.rn_mat.shape[0]):
            for j in range(self.rn_mat.shape[1]):
                self.S_df.iat[i,j+1]=self.trajactory(self.S_df.iat[i,j],self.rn_mat[i,j])

#example1
def example1():
    r=0.02
    rf=0.03
    T=5
    St0=1.09
    sigma=0.1
    K=1.095

    N=10
    example1=SDE(r,rf,T,St0,sigma,K,N)
    df=example1.S_df.copy()
    df['C_T']=df.iloc[:,-1]-example1.K
    df['C_T'][df['C_T']<0]=0
    C_t=np.exp(-r*T*1.0/365)*df['C_T'].mean()
    print C_t

#digital option
def digital_option(N=20):
    r = 0.02
    rf = 0.03
    T = 5
    St0 = 1.09
    sigma = 0.1
    K = 1.091
    H=1.08

    digi = SDE(r, rf, T, St0, sigma, K,N)
    df = digi.S_df.copy()
    df['C_T1'] = df.iloc[:, -1] - digi.K
    df['C_T1'][df['C_T1']<=0]=0
    df['C_T1'][df['C_T1']>0]=100
    C_t1 = np.exp(-r * T * 1.0 / 365) * df['C_T1'].mean()

    df['C_T2']=df['C_T1'].copy()
    min = df.iloc[:-1].min(axis=1)
    df['C_T2'][min[min < H].index] = 0

    C_t2 = np.exp(-r * T * 1.0 / 365) * df['C_T2'].mean()

    return df,C_t1,C_t2

def plot(df):
    df, C_t1, C_t2 = digital_option()
    df1=df[df['C_T1']+df['C_T2']>0]
    df2=df[(df['C_T1']+df['C_T2'])==0]

    df1=df1.T
    df1=df1[:-2]
    df1=df1.reset_index()
    del df1['index']
    df1.columns=range(1,df1.shape[1]+1)


    df2=df2.T
    df2=df2[:-2]
    df2=df2.reset_index()
    del df2['index']
    df2.columns=range(1,df2.shape[1]+1)

    plt.plot(df1.index,df1,color='red')
    plt.plot(df2.index,df2,color='darkblue')

    tmp_df=df.copy()
    tmp_df=tmp_df.iloc[:,:-2]
    value_max=tmp_df.max().max()
    value_min=tmp_df.min().min()
    ymin = value_min - 0.3 * (value_max-value_min)
    ymax = value_min + 1.3 * (value_max-value_min)
    plt.ylim(ymin, ymax)

    H = 1.08
    K = 1.091
    plt.plot(df1.index, [K] * len(df1.index), linewidth=4, color='green')
    plt.fill_between(df1.index, ymin, H, color='lightblue')

df,C_t1,C_t2=digital_option(20)
plot(df)

for N in [10,100,1000,10000,100000]:
    t1=time.time()
    df,C_t1,C_t2=digital_option(N)
    print N,C_t1,C_t2,time.time()-t1



