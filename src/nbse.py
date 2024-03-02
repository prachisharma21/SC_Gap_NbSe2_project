import numpy as np
import scipy as sp
from scipy.integrate import quad
from numpy.linalg import eig
import csv
import time
import cmath
import os 

class CoupledGapEquation:
    """ This class object handles all the calculation needed to solve the self-consistent multi-pocket gap equation 
    Important: the variables are kept same as the paper draft to avoid confusion and readibility with the paper.
    """
    def __init__(self, m, r, us, g, n):
        # mass 
        self.m = m
        # ratio of the singlet of triplet coupling   
        self.r = r
        # inter-pocket coupling, i.e., between Gamma and K pocket
        self.us = us
        # interaction coupling. 
        # It has to be less than \mu, i.e. chemical potential. 
        # Therefore, it rescaled with \mu
        self.g = g
        # Density of states in 2D parabolic spectrum
        self.n = n
        # Jacobian of conversion between angles to relative angle 
        self.jacob = 0.5
        self.options = {'limit': 200}
        self.start_time = time.time()
   
    def vs(self, x, ld):
        """This method calculates density of states (by units) which is prefactor of the s-wave (singlet) pairing term in the SC gap equation for A' irrep 
        input: 
        x:  angle 
        ld: the lambda, coefficient denoting the spin-orbit coupling 
        """
        return (1 / (2 * np.pi) ** 2) * (self.m / (1 - ld * np.cos(3*x)))

    def vc(self, x,ld):
        """This method calculates density of states (by units) which is prefactor of the non-diagnol term (mixed-parity term) in SC gap equation for A' irrep 
        x:  angle 
        ld: the lambda, coefficient denoting the spin-orbit coupling
        """
        return (1/(2*np.pi)**2)*((self.m/(1- ld*np.cos(3*x)))*np.cos(3*x))

    def vt(self,x,ld):
        """This is density of states (by units) which is prefactor of the f-wave (triplet) term in SC gap equation for A' irrep 
        x:  angle 
        ld: the lambda, coefficient denoting the spin-orbit coupling
        """
        return  (1/(2*np.pi)**2)*((self.m/(1- ld*np.cos(3*x)))*(np.cos(3*x))**2)

    def vcd(self,x,ld):
        """This is density of states (by units) which is prefactor of the d-wave (singlet) term in SC gap equation for E' irrep
        x:  angle 
        ld: the lambda, coefficient denoting the spin-orbit coupling
        """
        return  (1/(2*np.pi)**2)*((self.m/(1- ld*np.cos(3*x)))*(np.cos(2*x))**2)

    def vcp(self,x,ld):
        """This method calculates density of states (by units) which is prefactor of the non-diagnol term (mixed-parity term) in SC gap equation for E' irrep 
        x:  angle 
        ld: the lambda, coefficient denoting the spin-orbit coupling
        """
        return  (1/(2*np.pi)**2)*((self.m/(1- ld*np.cos(3*x)))*np.cos(x)*np.cos(2*x))

    def vcp2(self,x,ld):
        """This is density of states (by units) which is prefactor of the p-wave (triplet) term in SC gap equation for E' irrep 
        x:  angle 
        ld: the lambda, coefficient denoting the spin-orbit coupling
        """
        return  (1/(2*np.pi)**2)*((self.m/(1- ld*np.cos(3*x)))*(np.cos(x))**2)
    
    def Vs(self, t, u, a):
        """
        Interaction harmonic term for the s-wave term (singlet) in the gap matrix. 
        Input
        t,u : angles
        a: an interaction parameter, which is defined as the ratio of inverse correlation length and Fermi momentum: \xi^{-1}/2k_F 
        """
        # Changing angles into relative angle basis
        x1 = (t+u)/2
        y1 = (u-t)/2
        return (1/(2*np.pi)**2)*self.jacob*(3*self.g/(4*4*2*self.m))* ((1+ 2*a**2)/((a**2 +(np.cos((x1-y1)/2))**2)*(a**2 +(np.sin((x1-y1)/2))**2)))
    
    def Vt3(self, t, u, a):
        """
        Interaction harmonic term for the f-wave term (triplet) in the gap matrix. 
        Input
        t,u : angles
        a: an interaction parameter, which is defined as the ratio of inverse correlation length and Fermi momentum: \xi^{-1}/2k_F 
        """
        # Changing angles into relative angle basis
        x1 = (t+u)/2
        y1 = (u-t)/2
        prefac = 4*(1/(2*np.pi)**2)*self.jacob*((1*self.g*(1+ 2*a**2))/(4*4*2*self.m))
        cosfac = np.cos(x1-y1)*np.cos(3*x1)*np.cos(3*y1)
        num = prefac*cosfac
        deno = ((a**2 +(np.cos((x1-y1)/2))**2)*(a**2 +(np.sin((x1-y1)/2))**2))
        return num/deno

    def Vs2(self,t, u, a):
        """
        Interaction harmonic term for the d-wave term (singlet) in the gap matrix. 
        Input
        t,u : angles
        a: an interaction parameter, which is defined as the ratio of inverse correlation length and Fermi momentum: \xi^{-1}/2k_F 
        """
        # Changing angles into relative angle basis
        x1 = (t+u)/2
        y1 = (u-t)/2
        prefac = 4*(1/(2*np.pi)**2)*self.jacob*((3*self.g*(1+ 2*a**2))/(4*4*2*self.m))
        cosfac = np.cos(2*x1)*np.cos(2*y1)
        num = prefac*cosfac
        deno = ((a**2 +(np.cos((x1-y1)/2))**2)*(a**2 +(np.sin((x1-y1)/2))**2))
        return num/deno

    def Vt1(self, t, u, a):
        """
        Interaction harmonic term for the p-wave term (triplet) in the gap matrix. 
        Input
        t,u : angles
        a: an interaction parameter, which is defined as the ratio of inverse correlation length and Fermi momentum: \xi^{-1}/2k_F 
        """
        # Changing angles into relative angle basis
        x1 = (t+u)/2
        y1 = (u-t)/2
        prefac = 4*(1/(2*np.pi)**2)*self.jacob*((1*self.g*(1+ 2*a**2))/(4*4*2*self.m))
        cosfac = np.cos(x1-y1)*np.cos(x1)*np.cos(y1)
        num = prefac*cosfac
        deno = ((a**2 +(np.cos((x1-y1)/2))**2)*(a**2 +(np.sin((x1-y1)/2))**2))
        return num/deno

    def interaction_harmonics_sf(self, a):
        """This method calculates the interaction harmonics needed to solve the coupled SC gap equation for (s+f) wave pairing state
        Input
        a: an interaction parameter, which is defined as the ratio of inverse correlation length and Fermi momentum: \xi^{-1}/2k_F
        """
        resVs01, errVs0 = sp.integrate.nquad(self.Vs, [[0, 2 * np.pi], [0, 4 * np.pi]], args=[a],
                                             opts=[self.options, self.options])
        resVt31, errVt3 = sp.integrate.nquad(self.Vt3, [[0, 2 * np.pi], [0, 4 * np.pi]], args=[a],
                                             opts=[self.options, self.options])
        return resVs01, resVt31
    
    def interaction_harmonics_pd(self, a):
        """This method calculates the interaction harmonics needed to solve the coupled SC gap equation for (p+d) wave pairing state
        Input
        a: an interaction parameter, which is defined as the ratio of inverse correlation length and Fermi momentum: \xi^{-1}/2k_F
        """
        resVs21, errVs2 =  sp.integrate.nquad(self.Vs2, [[0,2*np.pi],[0,4*np.pi]],args=[a],
                                              opts =[self.options,self.options])
        resVt11, errVt1 =  sp.integrate.nquad(self.Vt1, [[0,2*np.pi],[0,4*np.pi]],args=[a],
                                              opts =[self.options,self.options])
        return  resVs21, resVt11

    
    def qp(self, ld):
        return np.sqrt(1+self.r*(ld))

    def qm(self, ld):
        return np.sqrt(1-self.r*(ld))

    def Ap(self, a,ld):
        """Angle integrated scaled interaction term in K-pocket SC gap equation: V_{++}"""
        qpp = self.qp(ld)
        num = self.g
        deno = (4*2*self.m)*(a*np.sqrt(a**2 + qpp**2))
        return num/deno 

    def Am(self,a,ld):
        """Angle integrated scaled interaction term in K-pocket SC gap equation: V_{--}"""
        qmm = self.qm(ld)
        num = self.g
        deno = (4*2*self.m)*(a*np.sqrt(a**2 + qmm**2))
        return num/deno 

    def B(self,a,ld):
        """Angle integrated scaled interaction term in K-pocket SC gap equation: V_{\pm\mp}"""
        qpp = self.qp(ld)
        qmm = self.qm(ld)
        num = (4*self.g)
        deno = (4*2*self.m)*(np.sqrt(16*a**4 + (qpp**2-qmm**2)**2 + 8*(a**2)*(qpp**2+qmm**2)))
        return num/deno 
    
    def DOS_func_sf(self, ld):
        """
        This method calculated the integrated density of states prefactor in the (s+f) wave coupled SC gap equation
        Input
        ld: Spin-orbit coupling. 
        """
        resvs1, errvs = quad(self.vs,0, 2*np.pi, args=(ld))
        resvc1, errvc = quad(self.vc,0, 2*np.pi, args=(ld))
        resvt1, errvt = quad(self.vt,0, 2*np.pi, args=(ld))
        return resvs1,resvc1,resvt1

    def DOS_func_pd(self, ld):
        """
        This method calculated the integrated density of states prefactor in the (p+d) wave coupled SC gap equation
        Input
        ld: Spin-orbit coupling. 
        """
        resvcd1, errvcd  = quad(self.vcd,0, 2*np.pi, args=(ld))
        resvcp1, errvcp = quad(self.vcp,0, 2*np.pi, args=(ld))
        resvcp21, errvcp2 = quad(self.vcp2,0, 2*np.pi, args=(ld))
        return resvcd1,resvcp1,resvcp21


    def Eig_sf(self, a, ld):
        """
        This method diagonalize the coupled SC gap matrix to find the solution of (s+f) wave pairing state
        Input
        a: an interaction parameter, which is defined as the ratio of inverse correlation length and Fermi momentum: \xi^{-1}/2k_F
        ld: Spin-orbit coupling. 
        """
        resVs0,resVt3 = self.interaction_harmonics_sf(a)
        resvs,resvc,resvt = self.DOS_func_sf(ld)
        return eig(np.array([[-resVs0*resvs,-resVs0*resvc],[resVt3*resvc,resVt3*resvt]]))

    def Eig_pd(self,a, ld):
        """
        This method diagonalize the coupled SC gap matrix to find the solution of (p+d) wave pairing state
        Input
        a: an interaction parameter, which is defined as the ratio of inverse correlation length and Fermi momentum: \xi^{-1}/2k_F
        ld: Spin-orbit coupling. 
        """
        resVs2,resVt1 = self.interaction_harmonics_pd(a)
        resvcd,resvcp,resvcp2 = self.DOS_func_pd(ld)
        return eig(np.array([[-resVs2*resvcd,-resVs2*resvcp],[resVt1*resvcp,resVt1*resvcp2]]))

    def Eig_allpkts(self,a,ld):
        """
        This method diagonalize the multi-pocket SC gap matrix to find the SC pairing state in coupled Gamma and  pocket
        Input
        a: an interaction parameter, which is defined as the ratio of inverse correlation length and Fermi momentum: \xi^{-1}/2k_F
        ld: Spin-orbit coupling. 
        """
        #qpp = qp(r,lbd)
        #qmm = qm(r,lbd)
        resVs0,resVt3 = self.interaction_harmonics_sf(a)
        resvs,resvc,resvt = self.DOS_func_sf(ld)    
        App = self.Ap(a, ld)
        Bb = self.B(a,ld)
        Amm = self.Am(a,ld)
        E = eig(np.array([[-resVs0*resvs, -resVs0*resvc, -3*self.us*self.n, 3*self.us*self.n],\
                          [resVt3*resvc, resVt3*resvt, 0, 0],\
                          [-3*self.us*resvs, -3*self.us*resvc, -(App*self.n)/2, Bb*self.n ],\
                            [3*self.us*resvs, 3*self.us*resvc , Bb*self.n,-(Amm*self.n)/2]]\
                                ))
        return E    

    

    def sorted_eigvals(self, eigval, eigvec):
        """
        This helper method sorts the eigenvalues and find the eigenvector corresponding to the largest eigenvalue. 
        """
        sorted_idx = np.argsort(eigval)
        eval_sorted = eigval[sorted_idx]
        evec_sorted = eigvec[:, sorted_idx].T
        # putting numerically zero eigenvalues to be exactly zero. 
        evec_sorted[np.abs(evec_sorted) < 10 ** (-10)] = 0 
        return eval_sorted, evec_sorted
    
    def critical_temp(self, eigv):
        """
        This methods calculates the SC critical temperature.
        """
        return 1.13*np.exp(-1/eigv[-1])
    
    def run(self):
        """This method basically run all the functions and writes the data generated in csv files."""

        path = './data'
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            print("Folder %s already exists"% path)

        with open(path+'/Gpkt.csv', 'w', newline='') as gpkt, \
                open(path+'/sfphasegammapkt.csv', 'w', newline='') as sf, \
                open(path+'/pdphasegammapkt.csv', 'w', newline='') as pd, \
                open(path+'/allpkt_wrt_pd.csv','w',newline='') as allpdbdd,\
                open(path+'/allpkt_wrt_sf.csv','w',newline='') as allsfbdd,\
                open(path+'/allpkt.csv', 'w', newline='') as allpkt:
            writer1 = csv.writer(gpkt, dialect='excel')
            writer2 = csv.writer(sf, dialect='excel')
            writer3 = csv.writer(pd, dialect='excel')
            writer4 = csv.writer(allpdbdd,dialect='excel')
            writer5 = csv.writer(allsfbdd,dialect='excel')
            writer6 = csv.writer(allpkt, dialect='excel')
            # loop over lambda and a to find the solutions
            for lbd in np.arange(0.38, 0.8, 0.01):
                for a in np.arange(0.0026, 0.01, 0.00005):

                    # Solving of (s+f) wave SC gap matrix at the Gamma pocket
                    eigval, eigvec = self.Eig_sf(a, lbd)
                    eigval, eigvec = self.sorted_eigvals(eigval, eigvec)
                    Tcsf = self.critical_temp(eigval)

                    # Solving of (p+d) wave SC gap matrix at the Gamma pocket
                    eigval2, eigvec2 = self.Eig_pd(a, lbd)
                    eigval2, eigvec2 = self.sorted_eigvals(eigval2, eigvec2)
                    Tcpd = self.critical_temp(eigval2)

                    writer1.writerow([a, lbd, eigval[-1], Tcsf, eigvec[-1][0],
                                     eigvec[-1][1], eigval2[-1], Tcpd, eigvec2[-1][0], eigvec2[-1][1]])
                    
                    # To find region of the phase space where (s+f) states wins over (p+d) state at the Gamma pocket
                    if Tcsf > Tcpd:
                        writer2.writerow([a, lbd, Tcsf])
                    else:
                        writer3.writerow([a, lbd, Tcpd])

                    # Below we solve for the a coupled multi-pocket gap equation included the SC gap at the Gamma and K pocket
                    eigallpkt, eigvecallpkt = self.Eig_allpkts(a, lbd)
                    eigallpkt, eigvecallpkt = self.sorted_eigvals(eigallpkt, eigvecallpkt)
                    Tcall = self.critical_temp(eigallpkt)
                    
                    # To find the comparison of (p+d) phase boundary at Gamma pocket with the all-pocket coupled solution
                    if Tcall>Tcpd:
                        writer4.writerow([a,lbd,Tcall])

                    # To find the comparison of (s+f) phase boundary at Gamma pocket with the all-pocket coupled solution
                    if Tcsf>Tcall:
                        writer5.writerow([a,lbd,Tcsf])

                    # writing the data of critical temperature, phases, etc in csv files. 
                    writer6.writerow([a, lbd, eigallpkt[-1], Tcall, eigvecallpkt[-1][0], eigvecallpkt[-1][1],
                                      eigvecallpkt[-1][2], eigvecallpkt[-1][3], cmath.phase(eigvecallpkt[-1][0]),
                                      cmath.phase(eigvecallpkt[-1][1]), cmath.phase(eigvecallpkt[-1][2]),
                                      cmath.phase(eigvecallpkt[-1][3])])

        end_time = time.time()
        total_time = end_time - self.start_time
        print("total time", total_time)

