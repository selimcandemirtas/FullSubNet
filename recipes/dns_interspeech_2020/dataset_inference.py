#from pathlib import Path

from audio_zen.dataset.base_dataset import BaseDataset
from audio_zen.utils import basename

class Dataset(BaseDataset):

    def __init__(self,
                 noisyWaves
                 ):
        
        super().__init__()
        
        self.noisyWaves = noisyWaves
        self.length = len(self.noisyWaves)


    def __len__(self):
        return self.length


    def __getitem__(self, item):

        noisy_y = self.noisyWaves[item]

        return noisy_y
