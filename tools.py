"""
Funções para cálculos de estrutura atômica de sistemas atômicos, fundamentados
na Teoria do Funcional da Densidade (DFT).

Funcionalidades disponíveis:

    * Grid de raios exponencial ajustado para amostragem eficiente de orbitais atômicos.
    * Métodos numéricos para cálculo de derivadas, integrais e derivadas funcionais sobre grids uniformes.
    * Energia de troca por elétron do gás uniforme de elétrons pela formulação de Wigner.
    * Energia de correlação por elétron do gás uniforme de elétrons pela formulação VWN.
    * Funcionais de troca e correlação LDA e GGA-PBE sem polarização de spin, assumindo simetria esférica.
    * Funções para resolução da equação de Kohn-Sham (ou de Schrödinger, no caso de potenciais não auto-consistentes)
      assumindo simetria esférica.
    * Funções para resolução da equação de Kohn-Sham (ou de Schrödinger, no caso de potenciais não auto-consistentes)
      pela expansão dos orbitais na base dos harmônicos esféricos.
"""

import numpy as np
from numba import njit
from scipy.sparse import block_diag, bmat, diags, linalg
from sympy.physics.wigner import gaunt
pi=np.pi



def exp_grid(Z,rmax=120,a=6,h=1/80):
    """Exponential grid obtained from the atm_cGuima3.f code.

    Args:
        Z (int): atomic number
        rmax (float, optional): maximum radius value. Defaults to 120.
        a (float, optional): grid gradation adjustment parameter. Defaults to 6.
        h (float, optional): auxiliar uniform grid spacing. Defaults to 1/80.

    Returns:
        float: auxiliar uniform grid spacing.
        1D-array: radius grid r.
        1D-array: derivative of r with respect to t.
        1D-array: second derivative of r with respect to t.
    """
    N=int(np.log(rmax*Z*np.exp(a)+1)/h)
    if N%2==0: #even
        N+=1 #odd
    i=np.arange(1,N+1) #1,2,...,N
    t=h*i
    r=np.exp(-a)/Z*(np.exp(t)-1)
    rt=np.exp(-a)/Z*np.exp(t)
    rtt=np.exp(-a)/Z*np.exp(t)
    return h,r,rt,rtt



"""
* Métodos numéricos para cálculo de derivadas, integrais e derivadas funcionais.
"""

@njit
def integrate(y,dx):
    """Numerical integration using the Simpson's method. Assumes an odd number of samples.

    Args:
        y (1D-array): the function to be integrated.
        dx (float): spacing of the uniform grid.

    Returns:
        1D-array: the integrated function.
    """
    return (dx/3)*(y[0]+y[-1]+np.sum(4*y[1::2])+np.sum(2*y[2:-1:2]))

@njit
def derivative(y,dx):
    """Numerical derivatives using the finite differences method of second order.

    Args:
        y (1D-array): the function to be differentiated.
        dx (float): spacing of the uniform grid.

    Returns:
        1D-array: the differentiated function dy/dx.
    """    
    f1=np.zeros(y.size)
    f1[0]=-3*y[0]+4*y[1]-y[2]
    f1[-1]=3*y[-1]-4*y[-2]+y[-3]
    f1[1:-1]=y[2::]-y[0:-2]
    return f1/(2*dx)

@njit
def functional_derivative(dt,rt,x,xt,xr,xrt,f):
    """Numerical evaluation of functional derivatives of semi-local functionals.

        - delta f(x,xr) / delta x = partial f / partial x - (d/dr) partial f / partial xr.
        - r is a non-uniform grid sampled over the uniform grid t (r = r(t)).

    Args:
        dt (float): spacing of the uniform grid t.
        rt (1D-array): derivative of r with respect to t.
        x (1D-array): the function x = x(r).
        xt (1D-array): derivative of x with respect to t.
        xr (1D-array): derivative of x with respect to r.
        xrt (1D-array): derivative of x with respect to rt.
        f (function): function f(x,xr) that returns an array with the same size of x and xr.

    Returns:
        1D-array: the functional derivative of f(x,xr).
    """
    N=x.size
    fx=np.zeros(N)
    fxr=np.zeros(N)
    d0=f(x,xr)
    d1=f(x[0:-1],xr[1::])
    d_1=f(x[1::],xr[0:-1])
    d20=f(x[0],xr[2])
    d21=f(x[-3],xr[-1])
    d_20=f(x[2],xr[0])
    d_21=f(x[-1],xr[-3])
    fx[1:-1]=d_1[1::]-d1[0:-1]
    fxr[1:-1]=d1[1::]-d_1[0:-1]
    fx[0]=-3*d0[0]+4*d_1[0]-d_20
    fxr[0]=-3*d0[0]+4*d1[0]-d20
    fx[-1]=3*d0[-1]-4*d1[-1]+d21
    fxr[-1]=3*d0[-1]-4*d_1[-1]+d_21
    fx/=(2*dt*xt)
    fxr/=(2*dt*xrt)
    return fx-derivative(fxr,dt)/rt



"""Energias de troca e correlação por elétron do gás uniforme de elétrons."""

@njit
def HEGx(n):
    """Homogeneous Electron Gas (HEG) exchange energy per electron.

    Args:
        n (1D-array): the electronic density.

    Returns:
        1D-array: the exchange energy e_x(n).
    """
    kF=(3*pi**2*n)**(1/3) #fermi wavevector
    return -3*kF/(4*pi)

@njit
def HEGc(n):
    """Homogeneous Electron Gas (HEG) correlation energy per electron.

    Args:
        n (1D-array): the electronic density.

    Returns:
        1D-array: the correlation energy e_c(n).
    """
    x0=-0.10498
    b=3.72744
    c=12.9352
    gamma=(1-np.log(2))/pi**2
    rs=(3/(4*pi*n))**(1/3)
    x=np.sqrt(rs)
    X=lambda x: x**2+b*x+c
    return gamma*(np.log(x**2/X(x)) + 2*b/np.sqrt(4*c-b**2)*np.arctan(np.sqrt(4*c-b**2)/(2*x+b))
             -b*x0/X(x0)*(np.log((x-x0)**2/X(x))+2*(2*x0+b)/np.sqrt(4*c-b**2)*np.arctan(np.sqrt(4*c-b**2)/(2*x+b))))    



"""Funcional de troca e correlação LDA, assumindo simetria esférica."""

def LDA(dt,r,rt,n):
    """The Local Density Approximation (LDA) functional.

    Args:
        dt (float): spacing of the uniform grid t.
        r (1D-array): radius grid r.
        rt (1D-array): derivative of r with respect to t.
        n (1D-array): the electronic density.

    Returns:
        float: exchange-correlation energy.
        1D-array: exchange-correlation potential.
    """
    ex=HEGx(n)
    ec=HEGc(n)
    exc=ex+ec
    Exc=4*np.pi*integrate(rt*(r**2)*n*exc,dt)
    Vxc=derivative(n*exc,dt)/derivative(n,dt)
    return Exc,Vxc



"""Funcional de troca e correlação GGA-PBE sem polarização de spin, assumindo simetria esférica."""

@njit
def PBE_energy_density(n,nr):
    """Energy density of the GGA-PBE functional.

    Args:
        n (array): the electronic density.
        nr (array): the derivative of n with respect to the radius grid r. Must have the same shape as n.

    Returns:
        1D-array: the energy density.
    """    
    beta=0.066725
    gamma=0.031091
    mu=0.21951
    kappa=0.804
    ex=HEGx(n)
    ec=HEGc(n)
    kF=(3*np.pi**2*n)**(1/3)
    kS=np.sqrt(4*kF/np.pi)
    t=np.abs(nr)/(2*kS*n)
    s=np.abs(nr)/(2*kF*n)
    Fx=1+kappa-kappa/(1+mu*s**2/kappa)
    A=(beta/gamma)/(np.exp(-ec/gamma)-1)
    H=gamma*np.log(1+(beta/gamma)*t**2*(1+A*t**2)/(1+A*t**2+A**2*t**4))
    exc=ex*Fx+ec+H #energy per electron
    return n*exc

def GGA_PBE(dt,r,rt,n):
    """The Generalized Gradient Approximation (GGA) functional proposed by Perdew-Burke-Ernzenhof (PBE).

    Args:
        dt (float): uniform grid t spacing.
        r (1D-array): radius grid r.
        rt (1D-array): derivative of r with respect to t.
        n (1D-array): the electronic density.

    Returns:
        float: exchange-correlation energy.
        1D-array: exchange-correlation potential.
    """
    nt=derivative(n,dt)
    nr=nt/rt
    nrt=derivative(nr,dt)
    nexc=PBE_energy_density(n,nr)
    Exc=4*np.pi*integrate(rt*r**2*nexc,dt)
    Vxc=functional_derivative(dt,rt,n,nt,nr,nrt,PBE_energy_density)
    return Exc,Vxc



"""
* Funções para resolução da equação de Kohn-Sham (ou de Schrödinger, no caso de potenciais não auto-consistentes)
  assumindo simetria esférica.
"""

def kinetic_energy_matrix(dt,r,rt,rtt):
    """The kinetic energy operator numerical representation as a matrix.

    Args:
        dt (float): spacing of the uniform grid t.
        r (1D-array): radius grid r.
        rt (1D-array): derivative of r with respect to t.
        rtt (1D-array): second derivative of r with respect to t.

    Returns:
        sparse matrix: the kinetic energy matrix.
    """    
    N=r.size
    Dt1=diags((-1/2,1/2),(-1,1),shape=(N,N)) / dt
    Dt2=diags((1,-2,1),(-1,0,1),shape=(N,N)) / dt**2
    Dr1=diags(1/rt)@Dt1
    Dr2=diags(1/rt**2)@(Dt2-diags(rtt)@Dr1)
    return (-1/2)*Dr2

def eigenstates_sphsym(Z,ls,r,K,V):
    """Use scipy to calculate eigenvalues and eigenvectors of the Kohn-Sham Hamiltonian.

    Args:
        Z (int): atomic number.
        ls (1D-array): orbital quantum numbers in ascending order.
        r (1D-array): radius grid r.
        K (sparse matrix): matrix representing the operation -0.5 * d²/dr².
        V (1D-array): Kohn-Sham potential.

    Returns:
        1D-array: eigenenergies e[i] of the matrix K+diag(l*(l+1)/(2*r**2)+V).
        2D-array: eigenvectors v[i,:] of the matrix K+diag(l*(l+1)/(2*r**2)+V).
    """
    e,psi=[],[]
    for l in np.unique(ls):
        H=K+diags(l*(l+1)/(2*r**2)+V)
        w,v=linalg.eigs(H,sigma=-Z**2/2,k=np.sum(ls==l))
        w=np.real(w)
        for i in range(w.size):
            e.append(w[i])
            psi.append(v[:,i])
    e=np.array(e)
    psi=np.array(psi)
    return e,psi

def normalize_eigenfunctions(dt,rt,psi):
    """Normalization of the Kohn-Sham orbitals.

    Args:
        dt (float): uniform grid t spacing.
        rt (1D-array): derivative of r with respect to t.
        psi (2D-array): Kohn-Sham orbitals.

    Returns:
        2D-array: normalized Kohn-Sham orbitals.
    """    
    for i in range(psi.shape[0]):
        N2=integrate(rt*np.abs(psi[i])**2,dt)
        N=np.sqrt(N2)
        psi[i]/=N
    return psi

def electronic_density_sphsym(r,psi,fs):
    """Angular average of the electronic density.

    Args:
        r (1D-array): radius grid r 
        psi (2D-array): Kohn-Sham orbitals.
        fs (1D-array_like): the occupation factors.

    Returns:
        1D-array: electronic-density.
    """
    n=np.zeros(r.size)
    for fi,psii in zip(fs,psi):
        n+=fi*np.abs(psii/r)**2
    n/=(4*np.pi)
    return n

@njit
def hartree_pot_sphsym(dt,r,rt,n):
    """Hartree potential by expansion of the 1/|r-r'| term using the Legendre expansion, assuming spherical symmetry.

    Args:
        dt (float): uniform grid t spacing.
        r (1D-array): radius grid r sampled over t.
        rt (1D-array): derivative of r with respect to t.
        n (1D-array): electronic density.

    Returns:
        1D-array: hartree potential.
    """
    VH=np.zeros(r.size)
    for i in range(r.size):
        rg=np.copy(r)
        rg[0:i]=r[i]
        VH[i]=4*np.pi*integrate(rt*r**2*n/rg,dt)
    return VH

def get_quantum_numbers_and_occupation_factors(econf):
    """The name is self explanatory.

    Args:
        econf (str): the electronic configuration.

    Returns:
        array_like: orbital quantum numbers in ascending order.
        array_like: occupation factors.
    """    
    ns,ls,fs=[],[],[]
    for orb in econf.split():
        n=int(orb[0])
        l={'s':0, 'p':1, 'd':2, 'f':3}[orb[1]]
        f=float(orb[2::])
        ns.append(n)
        ls.append(l)
        fs.append(f)
    ns=np.array(ns)
    ls=np.array(ls)
    fs=np.array(fs)
    ii=np.argsort(ls,kind='stable')
    ns=ns[ii]
    ls=ls[ii]
    fs=fs[ii]
    return ns,ls,fs



"""
* Funções para resolução da equação de Kohn-Sham (ou de Schrödinger, no caso de potenciais não auto-consistentes)
  pela expansão dos orbitais na base dos harmônicos esféricos.
"""

def hydrogenoid_hamiltonian_matrix(Z,lmax,dt,r,rt,rtt):
    """Numerical representation of the hydrogenoid hamiltonian.

    Args:
        Z (int): atomic number.
        lmax (int): basis cutoff.
        dt (float): uniform grid t spacing.
        r (1D-array): radius grid r sampled over t.
        rt (1D-array): derivative of r with respect to t.
        rtt (1D-array): second derivative of r with respect to t.

    Returns:
        sparse matrix: the hydrogenoid hamiltonian matrix.
    """
    Nr=r.size
    Dt1=diags((-1/2,1/2),(-1,1),shape=(Nr,Nr)) / dt
    Dt2=diags((1,-2,1),(-1,0,1),shape=(Nr,Nr)) / dt**2
    Dr1=diags(1/rt)@Dt1
    Dr2=diags(1/rt**2)@(Dt2-diags(rtt)@Dr1)
    blocks=[]
    for l in range(lmax+1):
        blocks+=[-0.5*Dr2+diags(-Z/r+l*(l+1)/(2*r**2))]*(2*l+1)
    return block_diag(blocks)

def eigenstates_coupled_eqs(Z,Nj,Nlm,Nr,H):
    """Use scipy to calculate the eigenenergies and eigenfunctions of the Hamiltonian.

    Args:
        Z (int): atomic number.
        Nj (int): number of states.
        Nlm (int): number of basis functions.
        Nr (int): number of grid points.
        H (sparse matrix): Hamiltonian matrix.

    Returns:
        1D-array: eigenenergies e[j].
        3D-array: eigenfunctions/ Kohn-Sham orbitals psi[j, lm, :].
    """
    w,v=linalg.eigs(H,sigma=-Z**2/2,k=Nj)
    w=np.real(w)
    ii=np.argsort(np.real(w))
    w=w[ii]
    v=v[:,ii]
    psi=np.zeros((Nj,Nlm,Nr),dtype=complex)
    for j in range(v.shape[1]):
        psi[j,:,:]=v[:,j].reshape(-1,Nr)
    return w,psi

def normalize_eigenfunctions_coupled_eqs(dt,rt,psi):
    """Normalize the Kohn-sham orbitals without losing information about the angular momentum composition.

    Args:
        dt (float): uniform grid t spacing.
        rt (1D-array): derivative of r with respect to t.
        psi (3D-array): Kohn-Sham orbitals.

    Returns:
        3D-array: normalized Kohn-Sham orbitals.
    """
    for j in range(psi.shape[0]):
        normj2=0.
        for lm in range(psi.shape[1]):
            normj2+=integrate(rt*np.abs(psi[j,lm])**2,dt)
        psi[j,:,:]/=np.sqrt(normj2)
    return psi

def filter_eigenfunctions(psi):
    """Filter the (relatively few) non-zero orbital radial components of the Kohn-Sham orbitals.

    Args:
        psi (3D-array): Kohn-Sham orbitals.

    Returns:
        2D-array: filtered Kohn-Sham orbitals.
    """
    jlm=[]
    orbs=[]
    for j in range(psi.shape[0]):
        for lm in range(psi.shape[1]):
            if np.max(np.abs(psi[j,lm]))>1e-10:
                jlm.append((j,lm))
                orbs.append(psi[j,lm])
    return np.array(orbs),jlm

def electronic_density_coupled_eqs(r,psi,jlm,occs):
    """Angular average of the electronic density.

    Args:
        r (1D-array): radius grid r.
        psi (2D-array): filtered Kohn-Sham orbitals. psi[i] = psi_jlm
        jlm (list of lists): jlm[i] = [j, lm].
        occs (1D-array): occupation factors.

    Returns:
        1D-array: electronic density.
    """    
    n=np.zeros(r.size)
    for (j,lm),psijlm in zip(jlm,psi):
        n+=occs[j]*np.abs(psijlm/r)**2
    n/=(4*pi)
    return n

@njit
def hartree_radial_integral(dt,r,rt,l1,psi_jlm):
    """Hartree potential radial integrals necessary to the calculate the projections onto the spherical harmonics basis.

    Args:
        dt (float): uniform grid t spacing.
        r (1D-array): radius grid r sampled over t.
        rt (1D-array): derivative of r with respect to t.
        l1 (int): orbital angular momentum quantum number.
        psi_jlm (1D-array): Kohn-Sham orbital

    Returns:
        1D-array: hartree radial integral.
    """
    N=r.size
    v=np.zeros(N)
    for i in range(N):
        rmin=np.copy(r)
        rmax=np.copy(r)
        rmax[0:i]=r[i] #r[i],r[i],...,r[i],r[i+1],...
        rmin[i+1::]=r[i] #r[0],r[1],...,r[i],r[i],...
        v[i]=integrate(rt*rmin**l1/rmax**(l1+1)*np.abs(psi_jlm)**2,dt)
    return v * 4*pi / (2*l1+1)

def hartree_radial(dt,r,rt,lmax,psi,jlm):
    """Collect non-zero Hartree radial integrals.

    Args:
        dt (float): uniform grid t spacing.
        r (1D-array): radius grid r sampled over t.
        rt (1D-array): derivative of r with respect to t.
        lmax (int): basis cutoff.
        psi (2D-array): filtered Kohn-Sham orbitals. psi[i] = psi_jlm
        jlm (list of lists): jlm[i] = [j, lm]

    Returns:
        2D-array: non-zero Hartree radial integrals.
    """
    Lambda=[]
    for (j,lm),psijlm in zip(jlm,psi):
        _Lambda=[]
        for l1 in range(lmax+1):
            _Lambda.append(hartree_radial_integral(dt,r,rt,l1,psijlm))
        Lambda.append(_Lambda)
    Lambda=np.array(Lambda)
    return Lambda

def decompose(lm):
    """Decompose index lm -> l,m

    Args:
        lm (int): compound basis index.

    Returns:
        int: orbital angular momentum quantum number (l=0,1,2,...).
        int: magnetic momentum quantum number (-l <= m <= l).
    """    
    l=int(np.sqrt(lm))
    m=lm-l**2-l
    return l,m

def hartree_potential_projections(Lambda,jlm,occs,Nlm,Nr):
    """Hartree potential by projection over the spherical harmonics basis.
    Radial integrals are numerical, angular integrals are analytical (Gaunt Symbols).

    Args:
        Lambda (2D-array): non-zero Hartree radial integrals.
        jlm (list of lists): jlm[i] = [j, lm]
        occs (1D-array): occupation factors.
        Nlm (int): number of basis functions.
        Nr (int): number of grid points.

    Returns:
        dict: non-zero Hartree potential projections.
    """
    VH={}
    for i in range(len(jlm)):
        j,l2m2=jlm[i]
        l2,m2=decompose(l2m2)
        for l1m1 in range(Nlm):
            l1,m1=decompose(l1m1)
            G1=float(gaunt(l1,l2,l2,-m1,-m2,m2))
            if G1!=0:
                for lm in range(Nlm):
                    l,m=decompose(lm)
                    for lpmp in range(Nlm):
                        lp,mp=decompose(lpmp)
                        G2=float(gaunt(l,l1,lp,-m,m1,mp))
                        if G2!=0:
                            if (lm,lpmp) not in VH.keys():
                                VH[lm,lpmp]=np.zeros(Nr)
                            VH[lm,lpmp]+=occs[j]*Lambda[i,l1]*(-1)**(m+m1+m2)*G1*G2
    return VH

def hartree_energy(dt,rt,psi,jlm,VH,occs):
    """Calculate Hartree energy using the potential projections.

    Args:
        dt (float): uniform grid t spaicng.
        rt (1D-array): derivative of r with respect to t.
        psi (2D-array): filtered Kohn-Sham orbitals.
        jlm (list of lists): jlm[i] = [j, lm]
        VH (dict): non-zero Hartree potential projections.
        occs (array_like): occupation factors.

    Returns:
        float: Hartree energy.
    """    
    U=0.
    for (j,lm),psijlm in zip(jlm,psi):
        if (lm,lm) in VH.keys():
            U+=occs[j]*integrate(rt*np.conj(psijlm)*VH[lm,lm]*psijlm,dt)
    return np.real(U/2)

def screening_potential_matrix(VH,Vxc,Nlm,Nr):
    """Matrix representation of the screening potential projections over the spherical harmonics basis.

    Args:
        VH (dict): non-zero Hartree potential projections.
        Vxc (1D-array): exchange-correlation potential, by spherical symmetry approximation.
        Nlm (int): number of basis functions.
        Nr (int): number of grid points.

    Returns:
        sparse matrix: screening potential matrix.
    """
    zeros=diags([0],shape=(Nr,Nr))
    blocks=[]
    for lm in range(Nlm):
        row=[]
        for lpmp in range(Nlm):
            if (lm,lpmp) in VH.keys():
                row.append(diags(VH[lm,lpmp]+Vxc*(lm==lpmp)))
            else:
                row.append(zeros)
        blocks.append(row)
    return bmat(blocks)
