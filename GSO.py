"""
Solução da equação de Kohn-Sham para átomos isolados, pelo método GSO.

Inputs obrigatórios:
    Z (int): número atômico (exemplo: 6).
    occs (list): fatores de ocupação (exemplo: 1s² 2s² 2p² -> [2,2,2/3,2/3,2/3]).
    xc (function): função do tipo xc(dt,r,rt,n) (exemplo: xc=LDA ou xc=GGA_PBE). Os argumentos são:
        dt (float): espaçamento de um grid uniforme auxiliar t.
        r (1D-array): grid de raios r = r(t).
        rt (1D-array): derivada de r em relação a t.
        n (1D-array): densidade eletrônica n = n(r).

Inputs opcionais:
    lmax (int): cutoff da base (padrão: 5)
    mix (float): parâmetro de mistura do potencial (padrão: 0.7).
    dE (float): tolerância sobre a convergência da energia total (padrão: 1e-4).
"""


if 'lmax' not in globals():
    lmax=5
if 'mix' not in globals():
    mix=0.7
if 'dE' not in globals():
    dE=1e-4


dt,r,rt,rtt=exp_grid(Z)
H0=hydrogenoid_hamiltonian_matrix(Z,lmax,dt,r,rt,rtt)
Nj=len(occs)
Nlm=(lmax+1)**2
Nr=r.size


vscr_prev=screening_potential_matrix({},np.zeros(Nr),Nlm,Nr) #zeros
vscr=screening_potential_matrix({},np.zeros(Nr),Nlm,Nr) #zeros
E=0


evolution=[]
self_consistent=False
print('Total energy [Hartree]:')
while not self_consistent:
    Vscr=(1-mix)*vscr_prev+mix*vscr
    e,psi=eigenstates_coupled_eqs(Z,Nj,Nlm,Nr,H0+Vscr)
    psi=normalize_eigenfunctions_coupled_eqs(dt,rt,psi)
    psi,jlm=filter_eigenfunctions(psi)
    Lambda=hartree_radial(dt,r,rt,lmax,psi,jlm)
    VH=hartree_potential_projections(Lambda,jlm,occs,Nlm,Nr)
    n=electronic_density_coupled_eqs(r,psi,jlm,occs)
    Exc,Vxc=xc(dt,r,rt,n)
    vscr_prev=vscr; vscr=screening_potential_matrix(VH,Vxc,Nlm,Nr)
    #energies
    Ecoul=-4*np.pi*Z*integrate(rt*r*n,dt)
    EH=hartree_energy(dt,rt,psi,jlm,VH,occs)
    Epot=Ecoul+EH+Exc
    Ekin=np.sum(occs*e)-Ecoul-2*EH-4*np.pi*integrate(rt*r**2*n*Vxc,dt)
    Eprev=E; E=Ecoul+EH+Exc+Ekin
    print(E)
    #convergence
    evolution.append(E)
    self_consistent=np.abs(E-Eprev)<dE
