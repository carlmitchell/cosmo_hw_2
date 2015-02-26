import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt

#Parameters with which to fiddle
mn = 939.565378 #Neutron mass in MeV
mp = 938.272046 #Proton mass in MeV
eta_b = (0.02266/0.020)*5.5*10**(-10) #Baryon fraction, WMAP 9 year

#Physical constants
G = 0.01548408 #G in MeV^(-4) s^(-2) (This is G/(hbar^3 c^5))
gstar = 10.75
zet3 = 1.202056903159594 #RiemannZeta(3), comes up because of Planck spectrum
Bd = 2.22452 #Deuterium binding energy in MeV

#Calculate important masses and differences
Q = mn - mp #Neutron-proton mass difference
md = mn + mp - Bd #Mass of deuterium
tau_n = 881.5*(939.565378-938.272046)/Q #Neutron lifetime in seconds

def lambda_np(x):
    """
    Returns the neutron-to-proton production rate as a function of the
    parameter x = Q/T (with Q = mn-mp)
    """
    return 255*(12+6*x+x**2)/(tau_n*x**5)

def npnn_eq(x):
    """
    Returns the ratio n_p^(0) / n_n^(0) as a function of the parameter
    x = Q/T (with Q = mn-mp)
    
    This ratio is the ratio of protons to neutrons in nuclear statistical 
    equilibrium.
    """
    return (mp/mn)**(1.5)*np.exp(x)

def dXdx(x,X):
    """
    Returns the derivative of X_n (the neutron abundance ratio) w.r.t.
    the parameter x = Q/T (with Q = mn-mp)
    """
    return x*lambda_np(x)/(4*np.pi**3*G*gstar*Q**4/45)*(1/npnn_eq(x) - X*(1+1/npnn_eq(x)))

if __name__ == '__main__':
    
    #Create an instance of the numerical integration class
    obj = integrate.ode(dXdx).set_integrator('lsoda')
    obj.set_initial_value((1/(1+(mp/mn)**1.5)), 10**(-8))
    
    #Create logarithmically spaced grid of steps
    steps = np.diff(np.logspace(-10,2,10000))
    
    #Initial conditions
    x = [10**(-10)] #Start at extremely high temperature
    X = [1/(1+(mp/mn)**1.5)]
    
    #Numerically integrate to find X_n(x)
    for i in range(len(steps)):
        if obj.successful():
            obj.integrate(obj.t+steps[i])
            x.append(obj.t)
            X.append(obj.y)
    T = Q/np.array(x)
    X = np.array(X)
    
    #Find T_Nuc
    weirdarray = np.abs(np.log(3)+np.log(zet3)-np.log(2)-2*np.log(np.pi)+np.log(X)+np.log(1-X)+1.5*np.log(2)+1.5*np.log(np.pi)+1.5*np.log(md*T/(mn*mp))+np.log(eta_b)+Bd/T)
    #weirdarray = np.abs(1.5*np.log(T/mp) + np.log(eta_b) +Bd/T) #Dodelson's version
    Tnuc = T[weirdarray == np.min(weirdarray)][0]
    print 'Tnuc=',Tnuc
    print 'X[Tnuc]=',X[T==Tnuc][0]

    #Plot the results
    plt.loglog(T,2*X)
    plt.xlim(np.max(T),np.min(T))
    plt.ylim(2*np.min(X),2*np.max(X))
    plt.show()
    