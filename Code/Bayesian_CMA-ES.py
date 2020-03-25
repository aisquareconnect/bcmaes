"""
                        Supplementary code for the paper:
                            Bayesian CMA-ES
                                      
"""

import numpy as np
import pandas as pd
import random
import math
import sys
import seaborn as sns; sns.set()
import cma.purecma as pcma
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
#%%
        
'''
g : objective function g
G : cdf of g
f : distribution of the min of g
'''

'''
prior : mu,sigma^2 -> NIG(mu_0,v,a,b)
==> E(mu)=mu_0 and E(sigma^2)=b/(a-1) (for a>1)
1) simulate X -> N(E(mu),E(sigma^2))
2) update mu and sigma^2 and get new E(mu) and E(sigma^2):
    E(mu) = (v*mu_0+n*x_bar)/(v+n)
    E(sigma^2) = B/(a+n2-1) where B = b+1/2*n*sigma_bar^2+(n*v)/(v+n)/2*(x_bar-mu_0)**2
3) repeat 1)
'''

#%%
def plot_function(function, colorbar):
        
        if function.__name__ in ['Rastrigin','Sphere', 'Schwefel2'] :
            if function.__name__ == 'Sphere' :
                m = 100
            else :    
                m = 10
        
        if function.__name__ == 'Eggholder':
            m = 550
        
        if function.__name__ == 'Schwefel1':
            m = 500
            
        func_name = function.__name__
        fig = plt.figure(figsize=(6,4.5))
        ax = fig.gca(projection='3d')
        
        # Make data
        X = np.arange(-m, m, 0.25)
        Y = np.arange(-m, m, 0.25)
        X, Y = np.meshgrid(X, Y)
        
        if func_name == 'Rastrigin' :
            Z = (X**2 - 10 * np.cos(2 * np.pi * X)) + \
            (Y**2 - 10 * np.cos(2 * np.pi * Y)) + 20
        if func_name == 'Sphere' :
            Z = (X**2 + Y**2)
        if func_name == 'Schwefel1' :
            Z = 418.9829*2 - X*np.sin(abs(X)**0.5) - Y*np.sin(abs(Y)**0.5)
        
        if func_name == 'Schwefel2' :
            Z = abs(X) + abs(Y) + abs(X) * abs(Y) 
        
        if func_name == 'Eggholder':
            Z = -(Y+47)*np.sin(abs(X/2+Y+47)**0.5)-X*np.sin(abs(X-Y-47)**0.5)
        
        # Plot the surface.
        ax.plot_surface(X, Y, Z, cmap=cm.viridis,
                               linewidth=0, antialiased=True)
        ax.grid(True)
        # Customize the z axis.
        ax.zaxis.set_major_locator(LinearLocator(5))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        plt.title(func_name + ' in dimension 2')
        if colorbar:
            # Add a color bar which maps values to colors.
            surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis,
                               linewidth=0, antialiased=False)        
            fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()
#%%
'''
Test Functions  :
'''

'''
Rastrigin
f* = 0 x*=(0 , 0)
'''
def Rastrigin(X):
    A = 10
    if type(X) == list:
        X = np.array(X)
        n = X.shape[0]
        X = X.reshape([1,n])
    else:
        try:
            n = X.shape[1]
        except:
            n = len(X)
            X = np.array(X).reshape([1,n])
        
    return np.sum(X**2 - A * np.cos(2 * np.pi * X), 1) + A * n

'''
Sphere :
f* = 0 x*=(0 , 0)
'''
def Sphere(X):
    if type(X) == list:
        X = np.array(X)
        n = X.shape[0]
        X = X.reshape([1,n])
    else:
        try:
            n = X.shape[1]
        except:
            n = len(X)
            X = np.array(X).reshape([1,n])
        
    return (np.sum(X**2,1))


'''
Schwefel1
f* = 0 x*=(420.9687 , 420.9687)
'''
def Schwefel1(X):
    if type(X) == list:
        X = np.array(X)
        n = X.shape[0]
        X = X.reshape([1,n])
    else:
        try:
            n = X.shape[1]
        except:
            n = len(X)
            X = np.array(X).reshape([1,n])
    if np.max(X)>500 or np.min(X)<-500:
        return 500
    A = 418.9829*X.shape[1]
    return A - np.sum(X*np.sin(np.abs(X)**(1/2)),1)

'''
Schwefel2
f* = 0 x*=(0 , 0)
'''
def Schwefel2(X):
    if type(X) == list:
        X = np.array(X)
        n = X.shape[0]
        X = X.reshape([1,n])
    else:
        try:
            n = X.shape[1]
        except:
            n = len(X)
            X = np.array(X).reshape([1,n])
    return np.sum(np.abs(X),1)+np.prod(np.abs(X),1)


#%%
def compute_min(list_param,n,tol,multivariate,func) :
    
    list_x_star = []
    list_fx_star = []
   
    
    variance = np.ones([n,n])
    mu_0, k_0, v_0, psi, factor = list_param[0], list_param[1], \
        list_param[2], list_param[3], list_param[4]
    x_star_min = mu_0
    var_star_min = psi
    fx_star_min = sys.float_info.max
    count = 0
    step = 0
        
    while np.linalg.norm(variance) > tol:
        E_mu = mu_0
        E_sigma = psi/(v_0-n-1)
        step += 1
        
        # prior           
        sample_X = np.random.multivariate_normal(mean=E_mu, cov=E_sigma, size=(n))

        g_x = func(sample_X)
        df = pd.DataFrame(sample_X)
        df['g_x'] = g_x
        df['d'] = multivariate_normal.pdf(sample_X,mean=E_mu, cov=E_sigma)
        x_order_d = df.sort_values( by=['d'],ascending  = False)
        x_order_f = x_order_d.sort_values( by=['g_x'],kind='mergesort')

        d_ordered = x_order_d[x_order_d.columns[-1:]].values/sum(df['d'])
        x_order_f = x_order_f[x_order_f.columns[:-2]].values
        x_order_d = x_order_d[x_order_d.columns[:-2]].values
        
        x_all = df[df.columns[:-2]].values
        d = df['d'][:, None]/sum(df['d'])
        x_bar_f = np.sum(x_order_f*d_ordered,axis=0)
        x_bar = np.sum(x_all*d,axis=0)
        mean = x_order_f[0]
        
        # sigma part :
        sigma_emp = np.dot((x_all-x_bar).T,(x_all-x_bar)*d)
        sigma_ordered = np.dot((x_order_f-x_bar_f).T, (x_order_f-x_bar_f)
                            * d_ordered)
        # other : 
        variance = (sigma_ordered - (sigma_emp-E_sigma))*factor
        
        variance_norm = np.linalg.norm(variance)
        fx_star = func(mean)

        if fx_star < fx_star_min:
            if fx_star_min - fx_star < 1e-5:
                return list_x_star, list_fx_star
            count = 0
            factor = 1
            fx_star_min = fx_star
            x_star_min = mean
            if variance_norm < 100 * n:
                var_star_min = variance
            list_x_star.append(mean)
            list_fx_star.append(func(mean))
        else:
            list_x_star.append(x_star_min)
            list_fx_star.append(func(x_star_min))
            count = count + 1
            
            if variance_norm > 100 * n:
                mean = x_star_min
                variance = var_star_min 
                mu_0 = mean
                
            if count > 5:
                factor = 1.5
            if count == 20:
                mean = x_star_min
                variance = var_star_min 
                mu_0 = mean
            if count > 20:
                factor = 0.9
            if count > 30:
                factor = 0.7
            if count > 40:
                factor = 0.5
            if count > 50:
                return list_x_star, list_fx_star
        
        mu_0 = (mu_0 * k_0 + n * mean)/(k_0+n)
        k_0 = k_0 + n
        v_0 = v_0 + n
        psi = psi + (k_0*n)/(k_0+n)*np.dot(np.mat(mean-mu_0),np.mat(mean-mu_0).T) + variance*(n-1)
        psi = psi * factor
    return list_x_star, list_fx_star
        
#%%
def compare(func, mu_0, min_f, nb_seed, save_fig):
    multivariate = True
    tol = 0.005
    # CMAES standard population size
    n = 4 + int(3 * math.log(len(mu_0)))
    k_0 = 4
    v_0 = n + 2
    factor = 1
    psi = np.eye(2)
    list_param = [mu_0, k_0, v_0, psi, factor]
    
    Function_evaluations = np.zeros(0)
    bcma_v = np.zeros(0)
    cma_v = np.zeros(0)
    iteration_max = 500
    
    for s in range(nb_seed):
        np.random.seed(s*3)
        list_x_star, values = compute_min(list_param,n,tol,
                                          multivariate,func)
        bcma_bestvalues = []
        for i in range(0,len(values),n):
            bcma_bestvalues.append(values[i]-min_f)
        if i < len(values)-1:
            bcma_bestvalues.append(values[-1]-min_f)
        
        random.seed(s)
        res = pcma.fmin(func, [mu_0[0],mu_0[1]], 1, verb_disp=0)
        data = res[1].logger._data
        cma_esvalues = [f for f in data['fit']]
        cma_esvalues = cma_esvalues[1:]
        best_seen = cma_esvalues[0]
        cma_bestvalues = []
        for elem in cma_esvalues:
            if elem < best_seen:
                cma_bestvalues.append(elem)
                best_seen = elem
            else:
                cma_bestvalues.append(best_seen)
        cma_bestvalues = np.abs([i - min_f for i in cma_bestvalues])
        max_length = max(min(len(bcma_bestvalues),len(cma_bestvalues)),30)
        for i in range(max_length):
            Function_evaluations = np.append(Function_evaluations, i)
            cma_v = np.append(cma_v, cma_bestvalues[i] \
                if i < len(cma_bestvalues) else cma_v[-1])
            bcma_v = np.append(bcma_v, bcma_bestvalues[i] \
                if i < len(bcma_bestvalues) else bcma_v[-1])
        if max_length<iteration_max: iteration_max=max_length

                
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    df1['Function_evaluations']= Function_evaluations
    df1['Method'] = ['CMA-ES']*len(Function_evaluations)
    df2['Function_evaluations']= Function_evaluations
    df2['Method'] = ['BCMA-ES']*len(Function_evaluations)
    df1['Error']= cma_v
    df2['Error']= bcma_v
    df = pd.concat([df1, df2], axis = 0)
    df = df[df['Function_evaluations']<= iteration_max]
    f, ax = plt.subplots(figsize=(8, 5.2))
    ax.set(yscale="log")
    if not save_fig: plt.title('Convergence comparison between CMA-ES and BCMA-ES\n'+func.__name__)
    sns.lineplot(x="Function_evaluations", y="Error", data=df, hue="Method")
    if save_fig: 
        print(f'saved figure {func.__name__}_convergence.png')
        plt.savefig(f'{func.__name__}_convergence.png')
    plt.show()
    return df

#%%
"""
Optional functions to test: 
- Rastigrin
- Sphere
- Schwefel1
- Schwefel2

To have a look on the function shape in dimension two, call plot_ function:
    -1st argument is the function we want to plot
    -2nd is the option to have the colorbar (if TRUE) of the plot or no (if FALSE)
    

To have a plot of convergence for a given starting point, call compare function:
    -1st argument is the function we want to study
    -2nd argument is the starting point
    -3rd argument is the global minimum of the function
    -4th argument is the umber of seed for our simulations
    -5th argument is flag to save figure
"""
function =  [Rastrigin, Sphere, Schwefel1, Schwefel2]
starting_points = [np.array([10,10]), np.array([10,10]),  \
    np.array([400,400]), np.array([10,10])]
save_fig = True
nb_seed = 30
sns.set(font_scale = 1.3)
for i, func in enumerate(function):
    compare(function[i], starting_points[i], 0, nb_seed, save_fig)
    
  
