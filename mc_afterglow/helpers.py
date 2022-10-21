import astropy.constants as const 
import astropy.units as units 
from astropy.cosmology import FlatLambdaCDM

def band2freq(band: str) -> units.quantity.Quantity:
    if band == 'U':
        res = const.c.cgs / 3600 /units.Angstron.cgs
    elif band == 'B':
        res = const.c.cgs / 4400 /units.Angstrom.cgs 
    elif band == 'G':
        res = const.c.cgs / 4640 /units.Angstrom.cgs 
    elif band == 'V':
        res = const.c.cgs / 5510 /units.Angstrom.cgs 
    elif band == 'R':
        res = const.c.cgs / 6580 /units.Angstrom.cgs 
    elif band == 'R_c':
        res = const.c.cgs / 6400 /units.Angstrom.cgs 
    elif band == 'R_j':
        res = const.c.cgs / 6940 /units.Angstrom.cgs 
    elif band == 'r_s':
        res = const.c.cgs / 6890 /units.Angstrom.cgs 
    elif band == 'I':
        res = const.c.cgs / 8060 /units.Angstrom.cgs 
    elif band == 'I_c':
        res = const.c.cgs / 7900 /units.Angstrom.cgs 
    elif band == 'I_j':
        res = const.c.cgs / 8700 /units.Angstrom.cgs 
    elif band == 'J':
        res = const.c.cgs / 1.215 /units.micron.cgs 
    elif band == 'H':
        res = const.c.cgs / 1.654 /units.micron.cgs 
    elif band == 'K':
        res = const.c.cgs / 2.179 /units.micron.cgs 
    elif band == 'K_s':
        res = const.c.cgs / 2.157 /units.micron.cgs 
    elif band == 'K\'':
        res = const.c.cgs / 2.11 /units.micron.cgs 
    else:
        raise ValueError("Please provide a valid band pass")
    
    return res.to(units.Hz)

cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Tcmb0=2.725)
def calc_luminosity_distance(z: float) -> float:
    if z > 0 :
        # assuming matter dominated universe with zero curvature
        return cosmo.luminosity_distance(z).cgs
        # return z * (const.c.cgs / cosmo.H(0).cgs) 
    else:
        return 1e28 * units.cm