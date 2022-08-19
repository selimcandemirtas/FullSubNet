import os
import sys
from pathlib import Path

import toml

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from audio_zen.utils import initialize_module
from recipes.dns_interspeech_2020.inferencer import Inferencer

class FullSubNetInferencer:

    def __init__(self, pretrainedModel):
        
        configuration = os.path.join(os.path.dirname(__file__), "fullsubnet/inference.toml")
        configPath = Path(configuration).expanduser().absolute()
        config = toml.load(configPath.as_posix())
        self.config = config

        self.pretrainedModel = pretrainedModel


    def predict(self, wave):

        config = self.config
        pretrainedModel = self.pretrainedModel

        inferencer_class = initialize_module(config["inferencer"]["path"], initialize=False)
    
        outputDir = ""

        inferencer = inferencer_class(
            config,
            pretrainedModel,
            outputDir,
            wave
        )

        enhanced = inferencer.enhance()

        return enhanced



'''
def inf(config, checkpoint_path, noisyWaves):

    inferencer_class = initialize_module(config["inferencer"]["path"], initialize=False)
    output_dir = ""

    inferencer = inferencer_class(
        config,
        checkpoint_path,
        output_dir,
        noisyWaves
    )

    enhancedWaves = inferencer.enhance()

    return enhancedWaves


def main():
    parser = argparse.ArgumentParser("Inference")
    #parser.add_argument("-C", "--configuration", type=str, required=True, help="Config file.")
    parser.add_argument("-C", "--configuration", type=str, default="modules/FullSubNet/recipes/dns_interspeech_2020/fullsubnet/inference.toml", help="Config file.")
    #parser.add_argument("-M", "--model_checkpoint_path", type=str, required=True, help="The path of the model's checkpoint.")
    parser.add_argument("-M", "--model_checkpoint_path", type=str, default="modules/FullSubNet/recipes/dns_interspeech_2020/fullsubnet/fullsubnet_best_model_58epochs.tar", help="The path of the model's checkpoint.")
    #parser.add_argument("-O", "--output_dir", type=str, required=True, help="The path for saving enhanced speeches.")
    parser.add_argument("-O", "--output_dir", type=str, default="C:/Users/s.demirtas/Desktop/enhanced/FSN", help="The path for saving enhanced speeches.")
    args = parser.parse_args()

    config_path = Path(args.configuration).expanduser().absolute()
    configuration = toml.load(config_path.as_posix())

    # append the parent dir of the config path to python's context
    # /path/to/recipes/dns_interspeech_2020/exp/'
    sys.path.append(config_path.parent.as_posix())

    checkpoint_path = args.model_checkpoint_path
    output_dir = args.output_dir


    #EKLEME KISIM:

    noisyWaves = []

    path = "C:/Users/s.demirtas/Desktop/py/synthesizer/data/output/short/noisy"
    for file in os.listdir(path):
        wave, _ = sf.read(
            os.path.join(path, file),
            dtype="float32"
        )
        if file.endswith('m.WAV'):
            noisyWaves.append(wave)

    enhancedWaves = inf(configuration, checkpoint_path, output_dir, noisyWaves)
    return enhancedWaves

enhancedWaves = main()

targetPath = "C:/Users/s.demirtas/Desktop/enhanced/enhanced"

for idx, enhanced in enumerate(enhancedWaves):
    sf.write(
        os.path.join(targetPath, f'{idx:02}_e.WAV'),
        enhanced,
        samplerate=16000
    )

#print("Tamamlandı.")
'''