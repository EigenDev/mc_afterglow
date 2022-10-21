from abc import ABC, abstractmethod

class Reader(ABC):
    @abstractmethod
    def read_file(*args) -> dict:
        """
        Required file read that returns the afterglow data 
        as a dictionary
        """
        pass