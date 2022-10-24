import argparse 
import ast 
import importlib 
import astropy.units as units
import sys 
import os 
from pathlib import Path
from .run import run_analysis 
from .helpers import band2freq, find_nearest

def main():
    parser = argparse.ArgumentParser("Data file to read from")
    event_type = parser.add_mutually_exclusive_group()
    event_type.add_argument('--ring', action='store_true', default=False)
    event_type.add_argument('--jet', action='store_true', default=True)
    parser.add_argument('script', help='script that contains the user-defined class instance that reads specific file')
    parser.add_argument('data_file', help="data file to read")
    parser.add_argument('--nus', type=float, help='list of frequencies', nargs='+', default=[])
    parser.add_argument('--tdomain', help='time domain for afterglow in days', default=[1e-2,1e3], type=float, nargs='+')
    parser.add_argument('--fname', help='name of plot file to be saved', default='some_lc', type=str)
    parser.add_argument('--grbs', help='list of grbs you want to plot against in given data', default=['all'], nargs='+')
    parser.add_argument('--z', help='redshift', default=0.0, type=float)
    parser.add_argument('--zero_time', help='the time of the burst in UT', type=float, default=0)
    parser.add_argument('--time_columns', help='the columns of time in the dataset', default=[0], type=int, nargs='+')
    parser.add_argument('--obs_columns', help='the columns of observables in the dataset', default=[1], type=int, nargs='+')
    parser.add_argument('--convert_mag2flux', default=True, help='set if need to convert the magnitude to mJy', action=argparse.BooleanOptionalAction)
    parser.add_argument('--band_passes', help='the band passes for the observables', default=None, type=str, nargs='+')
    parser.add_argument('--cmap', help='colormap for data', type=str, default='viridis')
    parser.add_argument('--scale_factors', default=None, type=float, help='factors to scale y-axis by', nargs='+')
    parser.add_argument('--sim_legend', default=True, help='unset if don\'t want legend for sim data', action=argparse.BooleanOptionalAction)
    parser.add_argument('--chains', default=8, help='number of Markov chains to sample from', type=int)
    parser.add_argument('--draws', help='number of draws to take for each free parameter', default=1000, type=int)
    parser.add_argument('--out_file', default='cornerplot', help='name of output file for corner plot', type=str)
    parser.add_argument('--spread', default=False, help='flag to activate spreading dynamics of blast waves', action='store_true')
    parser.add_argument('--sigma', help='spread in log_likelihood', default=1.0, type=float)
    args = parser.parse_args()
    
    # Temporarily add script to python path for easy import
    script_dirname = os.path.dirname(args.script)
    base_script    = Path(os.path.abspath(args.script)).stem
    sys.path.insert(0, f'{script_dirname}')
    
    user_class: str = None
    with open(args.script) as reader_file:
        root = ast.parse(reader_file.read())
    
    for node in root.body:
        if isinstance(node, ast.ClassDef):
            for base in node.bases:
                if base.id == 'Reader':
                    user_class = node.name 
                    break 
            
    reader_class          = getattr(importlib.import_module(f'{base_script}'), f'{user_class}')
    data_object, tdomain  = reader_class.read_file(args)
    for grb in args.grbs:
        try:
            data = data_object[grb][args.band_passes[0]]['flux']
        except KeyError:
            _, freq_key = find_nearest([*data_object[grb]], band2freq(args.band_passes[0]).value)
            data = data_object[grb][freq_key]['flux']
            if isinstance(data, units.Quantity):
                data = data.value

    run_analysis(data, args, tdomain)
    
if __name__ == "__main__":
    main()