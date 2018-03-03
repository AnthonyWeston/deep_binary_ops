from abc import ABC

class AbstractModel(ABC):
    
    def __init__(self, filename_list: list):
        self.filename_list = filename_list