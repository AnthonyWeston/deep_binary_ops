from abc import ABC

class AbstractModelBuilder(ABC):
    
    def __init__(self, filename_list: list, training_size: int):
        self.filename_list = filename_list
        self.training_size = training_size