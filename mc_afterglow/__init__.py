from pathlib import Path
from .reader import Reader
script_path = Path(__file__, '..').resolve()
with open(script_path.joinpath('VERSION')) as vfile:
    __version__ = vfile.readline()