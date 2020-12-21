import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import random as rd
from sklearn.neighbors import KNeighborsRegressor as nnr
import random
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import copy
from sklearn.metrics import max_error

EPSILON = 1e-10

def mape(y,yp):
    #z=np.array([v!=0 and v or vt for v,vt in zip(y,yp)])
    r=np.abs((y-yp)/(y))
    r=r/len(r)
    return r.sum()

def smape(y,yp):
    c=(np.abs(y)+np.abs(yp))/2
    r= np.abs(y-yp)/c
    #print(y.shape,yp.shape)
    return np.mean(r)



def rmse(y,yp):
    r= mean_squared_error(y,yp)
    return np.sqrt(r)

class RandomSearchKNNR:
    def __init__(self,data=None, opt_func=mape,sample_size=64,prediction_size=1):
        self.data=data
        self.sample_size=sample_size
        self.prediction_size=prediction_size
        self.opt_func=opt_func

    def _wsplit(self,data,m=3,tau=1):
        l,i=len(data),0
        n=l-m*tau
        x,y=[[] for k in range(n)],[0 for k in range(n)]
        while i+m*tau<l:
            ind=[0 for k in range(m)]
            for j in range(m):
                val= i+j*tau
                ind[j]=val
            x[i]=data[ind]
            y[i]=data[ind[-1]+tau]
            i+=1
        return np.array(x),np.array(y)
    
    def _forecast(self,regressor,s,conf,n=50):
        pred=[]
        m,tau,nn,d,w=conf
        ind=[-i for i in range(m*tau,0,-tau)]
        while len(pred)<n:
            window=list(s[ind])
            val=regressor.predict([window])
            pred.append(val[0])
            s=np.append(s,val[0])
        return np.array(pred)

    def _create_search_space(self,k,distances=['minkowski']):
        weights=['uniform','distance']
        n=len(self.data)
        lim=int(np.sqrt(n))
        grid=[(m,tau,nn,distance,weight) for m in range(1,lim) 
        for tau in range(1,lim) for distance in distances 
        for weight in weights for nn in range(1,n>25 and 25 or lim)]
        return random.sample(grid,k)

    def fit(self,data=None):
        k=self.sample_size
        p=self.prediction_size
        if data is None and self.data is None:
            assert("An input time serie must be provide at initialization or a fit time")
        if data is not None:
            self.data=data
        space=self._create_search_space(k)
        #print(space)
        order,knns=[],[]
        iteration=0
        best_fitness=np.inf
        for m,tau,nn,d,w in space:
            x,y=self._wsplit(self.data,m,tau)
            #xt,yt=x[:-2*p,:],y[:-2*p]
            #xv,yv=x[-2*p:-p,:],y[-2*p:-p]
            xt,yt=x[:-p,:],y[:-p]
            xv,yv=x[-p:,:],y[-p:]
            if len(xt)==0 or len(xv)<p//2 or 2*k>len(xt):
                print("Invalid configuration",len(x) , m,tau)
                continue
            #print(len(xt),-p, m,tau)
            a = d == 'cosine' and 'brute' or 'auto'
            knnr=nnr(n_neighbors=nn, weights=w, 
                     algorithm=a, metric=d)
            knnr.fit(xt,yt)
            if p>1:
                #yp=self._forecast(knnr,self.data[(-m*tau-1)-2*p:-2*p],(m,tau,nn,d,w),n=p)
                yp=self._forecast(knnr,self.data[(-m*tau-1)-p:-p],(m,tau,nn,d,w),n=p)
            else:
                yp=knnr.predict(xv)
            knns.append((knnr,(m,tau,nn,d,w),x[:-p,:],y[:-p],x[-p:,:],y[-p:]))
            fitness= self.opt_func(yv,yp)
            #fitness=max_error(yv,yp)
            if best_fitness>fitness: best_fitness=fitness 
            if iteration%10==0:
                print(f"iteration: {iteration+1}, fitness: {fitness}, best fitness: {best_fitness}")
            iteration=iteration+1
            order.append(fitness)
        order=np.array(order)
        ind=np.argsort(order)
        self.regressor,self.conf,xt,yt,xv,yv=knns[ind[0]]
        xf=np.concatenate((xt, xv))
        yf=np.concatenate((yt, yv))
        self.regressor.fit(xf,yf)
        self.opv=order[ind[0]]
        return self

    def predict(self,n=None):
        if n is None:
            n=self.prediction_size
        return self._forecast(self.regressor,self.data,self.conf,n)

