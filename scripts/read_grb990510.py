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
    def read_file(args) -> dict:
        data = np.loadtxt(args.data_file, delimiter=',')
        data_dict = {}
        data_dict['990510'] = {}
        # Build dictionary
        for val in data[:,1]:
            if val not in data_dict['990510']:
                data_dict['990510'][val] = {}
                data_dict['990510'][val]['time'] = []
                data_dict['990510'][val]['flux']  = []
                data_dict['990510'][val]['dflux'] = []
        
        for row in data:
            freq = row[1]
            data_dict['990510'][freq]['time'] += [row[0]]
            data_dict['990510'][freq]['flux']  += [row[2]]
            data_dict['990510'][freq]['dflux'] += [row[3]]
        
        for key in data_dict['990510'].keys():
            data_dict['990510'][key]['time']  = np.asarray(data_dict['990510'][key]['time'])  * units.s
            data_dict['990510'][key]['flux']  = np.asarray(data_dict['990510'][key]['flux'])  * units.mJy
            data_dict['990510'][key]['dflux'] = np.asarray(data_dict['990510'][key]['dflux']) * units.mJy
        
        # return the 
        return data_dict, [1e-1, 15]