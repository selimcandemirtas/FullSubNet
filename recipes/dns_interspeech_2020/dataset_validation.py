import os
from pathlib import Path

import librosa

from audio_zen.dataset.base_dataset import BaseDataset
from audio_zen.acoustics.feature import load_wav
from audio_zen.utils import basename


class Dataset(BaseDataset):
    def __init__(
            self,
            noisy_txt_path,
            clean_txt_path,
            sr,
    ):
        """
        Construct DNS validation set

        synthetic/
            with_reverb/
                noisy/
                clean_y/
            no_reverb/
                noisy/
                clean_y/
        """
        super(Dataset, self).__init__()
        #noisy_files_list = []
        
        noisy_files_list = [line.rstrip('\n') for line in open(noisy_txt_path, "r")]
        clean_files_list = [line.rstrip('\n') for line in open(clean_txt_path, "r")]

        #dataset_dir = Path(dataset_directory).expanduser().absolute()
        dataset_dir = str(Path(noisy_txt_path).parent)
        #noisy_files_list += librosa.util.find_files((dataset_dir / "noisy").as_posix())

        self.length = len(noisy_files_list)
        self.noisy_files_list = noisy_files_list
        self.sr = sr

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        """
        use the absolute path of the noisy speech to find the corresponding clean speech.

        Notes
            with_reverb and no_reverb dirs have same-named files.
            If we use `basename`, the problem will be raised (cover) in visualization.

        Returns:
            noisy: [waveform...], clean: [waveform...], type: [reverb|no_reverb] + name
        """
        noisy_file_path = self.noisy_files_list[item]
        parent_dir = Path(noisy_file_path).parents[1].name
        noisy_filename, _ = basename(noisy_file_path)

        # Find the corresponding clean speech using "parent_dir" and "file_id"
        #file_id = noisy_filename.split("_")[-1]
        file_id = noisy_filename.split('_')[0]
        clean_filename = file_id + '_c.WAV'

        clean_file_path = noisy_file_path.replace(f"noisy/{noisy_filename}", f"clean/{clean_filename}")

        noisy = load_wav(os.path.abspath(os.path.expanduser(noisy_file_path)), sr=self.sr)
        clean = load_wav(os.path.abspath(os.path.expanduser(clean_file_path)), sr=self.sr)

        return noisy, clean, noisy_filename
