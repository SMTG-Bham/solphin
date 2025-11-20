# BELOW USES THE BLANK-METRIC CODE FROM DAN DAVIS AND CHRIS SAVORY, NEEDS STRIPPING BACK TO THEORY 


def blank_flat(alpha, n, length): 

    #Absorptance for flat scatterer, from Matlab script
    the_c = np.arcsin(1/n[0])
    theta = np.linspace(0.0, the_c, num=200)
    a_len = len(alpha)
    t_len = len(theta)
    toft = np.zeros(a_len*t_len)
    toft = toft.reshape(a_len, t_len)
    absorb_tr = np.zeros(a_len)

    for i, a_pt in enumerate(alpha):
        for j, t_pt in enumerate(theta):
            toft_pt = np.exp(((2*np.multiply((-a_pt), length))/np.cos(t_pt)))
            toft[i, j] = toft_pt
        a_t_1 = np.trapz((toft[i,:]*np.cos(theta)*np.sin(theta)), theta)
        a_t_2 = np.trapz((np.cos(theta)*np.sin(theta)), theta)
        absorb_tr[i] = 1 - (a_t_1/a_t_2)

    absorb = absorb_tr.conjugate()
    return(absorb)


def blank_lambert(alpha, n, length):

    #Absorptance for Lambertian scatterer coating, from Matlab script
    x = np.multiply(2,np.multiply(alpha,length))
    T = np.exp((-x)) - np.multiply(x, np.exp((-x))) + (x**2*sc.exp1(x))
    R = 1.0/(np.multiply(n,n))
    Abs = (1-T)/(1-T+(R*T))
    Abs = np.nan_to_num(Abs)
    Emi = (R*T)/(1-T+(R*T))

    return(Abs)

def blank_eta(spectrum, E, alpha, n, length, Qi, trap):
    #For given scatterer, calculates Blank et al. eta
    dE = E[1]-E[0]
    if trap == 1:
        Abs = blank_flat(alpha, n, length)
    elif trap == 2:
        Abs = blank_lambert(alpha, n, length)
    
    #np divide necessary to have array divide here?
    phibb = 2*np.divide(np.divide(np.multiply(E,E),((h**3)*(c**2))),(np.exp(E/kT)-1))
    phibb = np.nan_to_num(phibb) # NaNs to 0, as in Matlab
    
    phi_sun = []
    ps_E = []
    #opens spectrum file correctly, alter path if necessary
    with open(spectrum) as f:
                am = f.readlines()
      
    # reads in columns of results, strips newline chars,converts to float
    for i in am:
        ps_E.append(float(i.rstrip().split(' ')[0]))
        phi_sun.append(float(i.rstrip().split(' ')[1]))
    
    # works for all E values above 0.03? won't extrapolate for 0
    # values < gap shouldn't be relevant?
    phisun = 10000*np.interp(E, ps_E, phi_sun)
        
    Jsc = q*np.sum(Abs*phisun)*dE
    J0rad = q*np.sum(Abs*phibb)*dE
    
    Rrad = 4*np.pi*np.sum(alpha*(n**2)*phibb)*dE
    Rnrad = (Rrad-Qi*Rrad)/Qi
    
    pe = J0rad/(q*Rrad*length)
    J0 = q*length*(Rnrad + pe*Rrad)
    #Looks like this scans over only voltages between 0 and 2 V
    #need testing for band gaps>2 eV?
    V = np.linspace(0, 2, 1001)
    Pmax = np.max(V*(Jsc - J0*(np.exp(V/kT)-1)))
    eta = Pmax/1000

    return(eta)

def blank_parse(folder):
    
    # Parses outputs from current directory

    abs_data = pd.read_table(f'{folder}/absorption.dat', delim_whitespace=True,
                                skiprows=1, header=None)
    n_data = pd.read_table(f'{folder}/n_real.dat', delim_whitespace=True,
                    skiprows=1, header=None)
    
    E_p = list(abs_data[0])
    alpha_p = list(abs_data[1])
    n_p = list(n_data[1])

    return{"E": E_p, "alpha": alpha_p, "n": n_p}

def blank_calculate(spectrum, folder):

    data = blank_parse(folder)

    E = np.asarray(data["E"])
    alpha = np.asarray(data["alpha"])
    alpha = np.multiply(alpha, 100)
    n = np.asarray(data["n"])

    #Remove data for E>5eV, necessary for speed! Also done in Matlab

    E = np.asarray([o for o in E if o <= 5])
    alpha = alpha[0:(len(E))]
    n = n[0:(len(E))]

    #main, looping over lengths and Qi, outputs eta table
    length_arr = np.logspace(-8.0, -3.0, num=36)
    Qi_arr = np.logspace(0, -6, num=4)
    trap = [1, 2]

    for tr_pt in trap:
        eta_arr = np.zeros(len(length_arr)*(len(Qi_arr)+1))
        eta_arr = eta_arr.reshape(len(length_arr), (len(Qi_arr)+1))
        for k, l_pt in enumerate(length_arr):
            for l, q_pt in enumerate(Qi_arr):
                eta_max = blank_eta(spectrum, E, alpha, n, l_pt, q_pt, tr_pt)
                eta_arr[k, 0] = l_pt
                eta_arr[k, l+1] = eta_max
        if tr_pt == 1:
            head="Thickness[m] \t Eta as fraction for Flat scatterer with Qi = 1.0, 0.01, 1E-4, 1E-6"
            np.savetxt('flat_eta_out', eta_arr, header=head)
        elif tr_pt == 2:
            head="Thickness[m] \t Eta as fraction for Lambertian scatterer with Qi = 1.0, 0.01, 1E-4, 1E-6"
            np.savetxt('lamb_eta_out', eta_arr, header=head)
        