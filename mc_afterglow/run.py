import pymc3 as pm
import math 
import matplotlib.pyplot as plt
import numpy as np
import afterglowpy
import arviz as az
import astropy.units as units 
import astropy.constants as const 
import corner
import theano 
import theano.tensor as tensor
from .helpers import band2freq, calc_luminosity_distance

plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'Times New Roman',
})

def run_analysis(observation_data: np.ndarray, args, given_tdomain: np.array = None):
    Z = {}
    Z['z']        = args.z
    Z['specType'] = 0
    Z['spread']   = args.spread
    Z['d_L']      = calc_luminosity_distance(args.z).value
    Z['jetType']  = afterglowpy.jet.TopHat
    if args.ring:
        Z['jetType'] = afterglowpy.jet.Cone
    
    tdomain  = np.asanyarray(given_tdomain)
    ts, te   = tdomain * afterglowpy.day2sec
    nt       = observation_data.size
    nus      = []
    if args.band_passes:
        nus = [band2freq(band).value for band in args.band_passes]
        
    for f in args.nus:
        if f not in nus:
            nus += [f]
    
    t     = np.geomspace(ts, te, nt)
    nu    = np.ones_like(t) * nus[0]

    @theano.compile.ops.as_op(itypes=[
        tensor.dscalar, 
        tensor.dscalar, 
        tensor.dscalar, 
        tensor.dscalar, 
        tensor.dscalar, 
        tensor.dscalar, 
        tensor.dscalar, 
        tensor.dscalar], otypes=[tensor.dvector])
    def fluxDensity(e_iso, theta_obs, theta_core, n_0, xi_n, epsilon_e, epsilon_b, p):
        """
        Because pymc3 only works with Tensor variables, we must
        use theanos to translate the Tensor variables into floats
        so that afterglowpy can compute the light curve

        Args:
            e_iso (Tensor[float64, scale]):        Isotropic-equivalent energy in erg
            theta_obs (Tensor[float64, scale]):    Viewing angle in radians
            theta_core (Tensor[float64, scale]):   Half-opening angle in radians
            n_0 (Tensor[float64, scale]):          circumburst density in cm^{-3}
            xi_n (Tensor[float64, scale]):         Fraction of electrons accelerated
            epsilon_e (Tensor[float64, scale]):    fraction of energy density due to electron
            epsilon_b (Tensor[float64, scale]):    fraction of energy density due to magnetic fields
            p (Tensor[float64, scale]):            electron power law index

        Returns:
            Tensor[vector, scalar]: The flux density array converted into a Tensor variable again
        """
        Z['thetaObs']  = float(theta_obs)     
        Z['E0']        = float(e_iso)
        Z['n0']        = float(n_0)
        Z['p']         = float(p)
        Z['epsilon_e'] = float(epsilon_e)
        Z['epsilon_B'] = float(epsilon_b)     
        Z['xi_N']      = float(xi_n)
        if args.jet:
            Z['thetaCore'] = float(theta_core)
        else:
            Z['thetaCore']  = np.pi * 0.5 - float(theta_core)
            Z['counterjet'] = True
            
        res = afterglowpy.fluxDensity(t, nu, **Z)
        return res
    
    print("Generating pymc3 model...")
    with pm.Model() as afterglow_model:
        # priors
        e_iso      = pm.Uniform(r'$E_{\rm iso}$', lower=1e51, upper=1e54)
        theta_obs  = pm.Uniform(r'$\theta_{\rm obs}$', lower=0.0, upper=math.pi * 0.5)
        theta_core = pm.Uniform(r'$\theta_0$', lower=0.01, upper=math.pi * 0.5)
        n_0        = pm.Uniform(r'$n_0$', lower=1.0, upper=2.0)
        xi_n       = pm.Uniform(r'$\xi_N$', lower=0.01, upper=1.0)
        epsilon_e  = pm.Uniform(r'$\epsilon_e$', lower=1e-2, upper=1.0)
        epsilon_b  = pm.Uniform(r'$\epsilon_B$', lower=1e-2, upper=1.0)
        p          = pm.Uniform(r'$p$', lower=2.01, upper=2.5)
        
        # The deterministic
        fnu      = fluxDensity(e_iso, theta_obs, theta_core, n_0, xi_n, epsilon_e, epsilon_b, p)
        fnu_true = pm.Deterministic('fnu_true', fnu)

        # log_likelihood
        logl = pm.Normal('logl', mu=fnu_true, sigma=args.sigma, observed=observation_data)
        
    gv = pm.model_graph.model_to_graphviz(afterglow_model)
    gv.format = 'pdf'
    gv.render(filename='afterglow_model_graph')
    
    with afterglow_model:
        print('Sampling from distribution...')
        step  = pm.Metropolis()
        trace = pm.sample(draws=args.draws, step=step, chains = args.chains, return_inferencedata=False)
        
    # axes = az.plot_trace(
    #     trace, 
    #     var_names = [
    #         r'$E_{\rm iso}$', 
    #         r'$\theta_{\rm obs}$', 
    #         r'$\theta_0$',
    #         r'$n_0$', 
    #         r'$\xi_N$', 
    #         r'$\epsilon_e$', 
    #         r'$\epsilon_B$', 
    #         r'$p$'], 
    #     figsize=(7,4)
    # )
    # fig = axes.ravel()[0].figure
    # fig.savefig("tracer_plot.pdf")
    
    # plt.tight_layout()
    fig = corner.corner(
        trace, 
        var_names = [
            r'$E_{\rm iso}$', 
            r'$\theta_{\rm obs}$', 
            r'$\theta_0$', 
            r'$n_0$', 
            r'$\xi_N$', 
            r'$\epsilon_e$',
            r'$\epsilon_B$',
            r'$p$']
    )
    
    fig.set_size_inches(8, 8)
    print(f"Saving cornerplot as {args.out_file}.pdf")
    fig.savefig(f'{args.out_file}.pdf')
    plt.show()