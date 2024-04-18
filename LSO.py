"""
Solução da equação de Kohn-Sham para átomos isolados, pelo método LSO.

Inputs obrigatórios:
    Z (int): número atômico (exemplo: 6).
    econf (str): configuração eletrônica (exemplo: '1s2 2s2 2p2').
    xc (function): função do tipo xc(dt,r,rt,n), em que os argumentos são
        dt (float): espaçamento de um grid uniforme auxiliar t.
        r (1D-array): grid de raios r = r(t).
        rt (1D-array): derivada de r em relação a t.
        n (1D-array): densidade eletrônica n = n(r).

Inputs opcionais:
    mix (float): parâmetro de mistura do potencial (padrão: 0.7).
    dE (float): tolerância sobre a convergência da energia total (padrão: 1e-4).

Obs.: para obter os auto-estados do átomo hidrogenoide, use dE=np.inf (infinito) ou dE=número muito grande.
"""


if 'mix' not in globals():
    mix=0.7
if 'dE' not in globals():
    dE=1e-4


ns,ls,fs=get_quantum_numbers_and_occupation_factors(econf)
# dt,r,rt,rtt=exp_grid(Z)
dt,r,rt,rtt=exp_grid(sum(fs))
K=kinetic_energy_matrix(dt,r,rt,rtt)


vh_prev=np.zeros(r.size)
vxc_prev=np.zeros(r.size)
vh=np.zeros(r.size)
vxc=np.zeros(r.size)
E=0


evolution=[]
self_consistent=False
niter=1
print('Total energy [Hartree]')
while not self_consistent:
    vks=-Z/r+(1-mix)*(vh_prev+vxc_prev)+mix*(vh+vxc)
    e,psi=eigenstates_sphsym(Z,ls,r,K,vks)
    psi=normalize_eigenfunctions(dt,rt,psi)
    n=electronic_density_sphsym(r,psi,fs)
    vh_prev=vh; vh=hartree_pot_sphsym(dt,r,rt,n)
    vxc_prev=vxc; Exc,vxc=xc(dt,r,rt,n)
    #energies
    Ecoul=-4*np.pi*Z*integrate(rt*r*n,dt)
    Ehartree=2*np.pi*integrate(rt*r**2*n*vh,dt)
    Ekin=np.sum(fs*e)-Ecoul-2*Ehartree-4*np.pi*integrate(rt*r**2*n*vxc,dt)
    Eprev=E; E=Ecoul+Ekin+Ehartree+Exc
    print(E)
    #convergence
    evolution.append(E)
    self_consistent=np.abs(E-Eprev)<dE
