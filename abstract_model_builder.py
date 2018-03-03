from abc import ABC

class AbstractModelBuilder(ABC):
    
    def __init__(self, filename_list: list, training_size: int, batch_size: int, seed: int,
                learning_rate: int):
        self.filename_list = filename_list
        self.training_size = training_size
        self.batch_size = batch_size
        self.seed = seed
        self.learning_rate = learning_rate