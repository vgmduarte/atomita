"""
Soluções da equação de Schrödinger para átomos hidrogenoides submetidos a um campo elétrico fraco uniforme
de intensidade variável.

Inputs obrigatórios:
    n (int): número quântico principal.
    Egrid (1D-array): grid de intensidades de campo elétrico.

Inputs opcionais:
    Z (int): número atômico (padrão: 1).
    lmax (int): cutoff da base (padrão: 5).
    nvals: quantidade de auto-energias por valor de campo elétrico (padrão: n**2).
    e0: chute inicial para as auto-energias (padrão -Z**2/(2*n**2)).
"""


if 'Z' not in globals():
    Z=1
if 'lmax' not in globals():
    lmax=5
if 'nvals' not in globals():
    nvals=n**2
if 'e0' not in globals():
    e0=-Z**2/(2*n**2)
    

Nlm=(lmax+1)**2
dt,r,rt,rtt=exp_grid(90)
Nr=r.size


Ve={}
for lm in range(Nlm):
    l,m=decompose(lm)
    for lpmp in range(Nlm):
        lp,mp=decompose(lpmp)
        G=float(gaunt(1,l,lp,0,-m,mp))
        if G!=0:
            Ve[lm,lpmp] = - (-1)**m * r * np.sqrt(4*np.pi/3) * G
print(f'Non-zero components: {Ve.keys()}')


H0=hydrogenoid_hamiltonian_matrix(Z,lmax,dt,r,rt,rtt)
V=screening_potential_matrix(Ve,np.zeros(Nr),Nlm,Nr)


eigenvalues=[]
for E in Egrid:
    print(f'{E = } Hartree/(e*a)')
    e=linalg.eigs(H0+E*V,k=nvals,sigma=e0,return_eigenvectors=False)
#     assert np.sum(np.abs(np.imag(e)))<1e-5
    e=np.sort(np.real(e))
    eigenvalues.append(e)
eigenvalues=np.array(eigenvalues)
