import math

def local_forward(s0,u,T):
    return s0*math.exp(u*T)

def quanto_forward(s0,u,T,rho,sigma1,sigma2):
    return s0*math.exp((u-rho*sigma1*sigma2)*T)