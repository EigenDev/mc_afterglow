import numpy as np 
import astropy.units as units
from mc_afterglow import Reader

class VanEertenFile(Reader):
    """
    Class that reads data file provided by Hendrik van Eerten
    This data was used to produce the nice fits shown in 
    van Eerten et al. (2012)
    link: https://iopscience.iop.org/article/10.1088/0004-637X/749/1/44/pdf
    """
    def read_file(args) -> dict, tuple:
        data = np.loadtxt(args.obs_file, delimiter=args.sep)
        data_dict = {}
        # Build dictionary
        for val in data[:,1]:
            if val not in data_dict:
                data_dict[val] = {}
                data_dict[val]['time'] = []
                data_dict[val]['flux']  = []
                data_dict[val]['dflux'] = []
        
        for row in data:
            freq = row[1]
            data_dict[freq]['time'] += [row[0]]
            data_dict[freq]['flux']  += [row[2]]
            data_dict[freq]['dflux'] += [row[3]]
        
        for key in data_dict.keys():
            data_dict[key]['time']  = np.asarray(data_dict[key]['time']) * units.s
            data_dict[key]['flux']  = np.asarray(data_dict[key]['flux'])  * units.mJy
            data_dict[key]['dflux'] = np.asarray(data_dict[key]['dflux']) * units.mJy
        
        # return the 
        return data_dict, [1e-1, 15]