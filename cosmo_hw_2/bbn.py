import numpy as np
from scipy import integrate
from scipy import optimize
import matplotlib.pyplot as plt

plt.rc('font', family='serif') #Changes all plotting fonts.

#Parameters with which to fiddle
mn = 939.565378 #Neutron mass in MeV
mp = 938.272046 #Proton mass in MeV
eta_b = (0.02266/0.020)*5.5*10**(-10) #Baryon fraction, WMAP 9 year

#Double neutron mass
mn=2*mn

#Eta_B Big
#eta_b = 10**(-5)

#Eta_B HUGE
#eta_b = 1

#Physical constants
G = 0.01548408 #G in MeV^(-4) s^(-2) (This is G/(hbar^3 c^5))
gstar = 10.75
zet3 = 1.202056903159594 #RiemannZeta(3), comes up because of Planck spectrum
Bd = 2.22452 #Deuterium binding energy in MeV
Bhe = 28.3007 #Helium binding energy in MeV

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
    
    #Lower and upper bounds for x, effectively upper and lower bounds for T
    x_min = 10**(-1.5)
    x_max = 10**(6)
    
    #Array of values of x for integration
    x = np.logspace(np.log10(x_min),np.log10(x_max),1000000,dtype='float128')
    steps = np.diff(x)
    
    #Empty array of X values
    X = np.zeros_like(x,dtype='float128')
    
    #Initial conditions
    X[0] = (1/(1+npnn_eq(x[0])))
    
    #Numerically integrate to find X_n(x) (the boltzmann curve)
    for i in range(len(steps)):
        X[i+1] = X[i]+steps[i]*dXdx(x[i],X[i])
    
    #Calculate the equilibrium curve
    X_eq = 1/(1+npnn_eq(x))
    
    #Convert x's to T's
    T = Q/x
    
    #Find T_Nuc
    #weirdarray = np.abs(np.log(3)+np.log(zet3)-np.log(2)-2*np.log(np.pi)+np.log(X)+np.log(1-X)+1.5*np.log(2)+1.5*np.log(np.pi)+1.5*np.log(md*T/(mn*mp))+np.log(eta_b)+Bd/T)
    #weirdarray = np.abs(1.5*np.log(T/mp) + np.log(eta_b) +Bd/T) #Dodelson's version
    Tnuc = optimize.root(lambda T: 1.5*np.log(T*md/(mp*mn)) + np.log(eta_b) +Bd/T,0.01).x
    #Tnuc = T[weirdarray == np.min(weirdarray)][0]
    
    #Find neutron abundance directly before BBN
    X_neut = X[abs(T-Tnuc)==np.min(abs(T-Tnuc))][0]*np.exp(-132/tau_n*(0.1/Tnuc)**2)
    F_neut = X_neut*mn/(X_neut*mn+(1-X_neut)*mp)
    
    #Find Helium abundance directly after BBN
    X_he = 0.5*np.min([X_neut,1-X_neut])
    F_he = X_he*(2*mn+2*mp-Bhe)/(X_neut*mn+(1-X_neut)*mp)
    
    #Print relevant values
    print "Neutron Lifetime [s] =", tau_n
    print "T_nuc [MeV] =", Tnuc
    print "Neutron # Frac = ", X_neut
    print "Neutron Mass Frac = ", F_neut
    print "Helium # Frac = ", X_he
    print "Helium Mass Frac = ", F_he
    
    #Plot the results
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    fontsize = 20

    #Normal BBN
#     ax1.set_xlim(2,0.02)
#     ax1.set_ylim(10**(-5),1)
#     ax1.loglog(T,2*X,label='$2X_{n,Boltzmann}$',color='black',ls='--',linewidth=2)
#     ax1.loglog(T,2*X_eq,label='$2X_{n,EQ}$',color='black',linewidth=2)
#     ax1.scatter(.0644,2*.111,label='$\eta_b=5*10^{-10}$',color='black',marker='o',s=fontsize*2)
#     ax1.scatter(.0914,2*.134,label='$\eta_b=10^{-5}$',color='black',marker='v',s=fontsize*2)
#     ax1.scatter(.1898,2*.159,label='$\eta_b=1$',color='black',marker='s',s=fontsize*2)
    
    #Double Neutron Mass BBN
    ax1.set_xlim(10**4,10**1)
    ax1.set_ylim(10**(-5),3)
    ax1.loglog(T,3*(X),label='$3X_{n,Boltzmann}$',color='black',ls='--',linewidth=2)
    ax1.loglog(T,3*(X_eq),label='$3X_{n,EQ}$',color='black',linewidth=2)
    #ax1.scatter(Tnuc,2*(X_neut),label='$2X_{n,nuc}(T_{nuc})$',color='black')

    #Flipped masses
#     ax1.set_xlim(2,0.02)
#     ax1.set_ylim(10**(-5),1)
#     ax1.loglog(T,2*X,label='$2X_{p,Boltzmann}$',color='black',ls='--',linewidth=2)
#     ax1.loglog(T,2*X_eq,label='$2X_{p,EQ}$',color='black',linewidth=2)
#     ax1.scatter(Tnuc,2*X_neut,label='$2X_p(T_{nuc})$',color='black',marker='o',s=fontsize*2)
    
    ax1.set_xlabel('T [MeV]',fontsize=fontsize)
    ax1.set_ylabel('Fractional Abundance',fontsize=fontsize)
    plt.tick_params(axis='both', which='major', labelsize=fontsize)
    ax1.xaxis.set_tick_params(length=12,which='major',width=2)
    ax1.yaxis.set_tick_params(length=12,which='major',width=2)
    ax1.xaxis.set_tick_params(length=6,which='minor',width=2)
    ax1.yaxis.set_tick_params(length=6,which='minor',width=2)
    plt.legend(loc=3,scatterpoints=1,prop={'size':fontsize})
        
    plt.tight_layout()
    plt.show()
    
