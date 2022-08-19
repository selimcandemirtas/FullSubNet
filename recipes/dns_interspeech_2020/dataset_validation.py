from audio_zen.dataset.base_dataset import BaseDataset

class Dataset(BaseDataset):

    def __init__(
            self,
            noisyWaves
    ):

        super(Dataset, self).__init__()

        self.noisyWaves = noisyWaves
        self.length = len(self.noisyWaves)


    def __len__(self):
        return self.length

    def __getitem__(self, item):
        
        noisy = self.noisyWaves
        clean = []
        
        return noisy, clean