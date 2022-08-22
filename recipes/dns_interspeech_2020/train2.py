import os
import sys
import argparse
import toml
import random
import torch
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

cwd = os.path.dirname(__file__)
sys.path.append(os.path.join(cwd, '../..'))
from audio_zen.model.module.sequence_model import SequenceModel
from audio_zen.utils import initialize_module
from fullsubnet.model import Model
#from torchsummary import summary

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

'''class FullSubNetFtModel(Model):

    def __init__(self,
                 num_freqs,
                 look_ahead,
                 sequence_model,
                 fb_num_neighbors,
                 sb_num_neighbors,
                 fb_output_activate_function,
                 sb_output_activate_function,
                 fb_model_hidden_size,
                 sb_model_hidden_size,
                 norm_type="offline_laplace_norm",
                 num_groups_in_drop_band=2,
                 weight_init=True
                 ):

        super().__init__(num_freqs, look_ahead, sequence_model, fb_num_neighbors, sb_num_neighbors, fb_output_activate_function, sb_output_activate_function, fb_model_hidden_size, sb_model_hidden_size, norm_type, num_groups_in_drop_band, weight_init)


    def load_checkpoint(self, checkpoint_path, device):
        model_checkpoint = torch.load(checkpoint_path, map_location=device)
        model_static_dict = model_checkpoint["model"]
        epoch = model_checkpoint["epoch"]
        #print(f"Loading model checkpoint (epoch == {epoch})...")

        model_static_dict = {key.replace("module.", ""): value for key, value in model_static_dict.items()}

        model.load_state_dict(model_static_dict)
        model.to(device)
        #model.eval()
        #return model, model_checkpoint["epoch"]'''

'''configPath = Path(os.path.join(cwd, 'fullsubnet/train.toml')).expanduser().absolute()
config = toml.load(configPath.as_posix())
modelArgs = config["model"]["args"]
model = FullSubNetFtModel(**modelArgs)

checkpointPath = os.path.join(cwd, 'fullsubnet/fullsubnet_best_model_58epochs.tar')
modelCheckpoint = torch.load(checkpointPath, map_location=DEVICE)
modelStateDict = modelCheckpoint["model"]
model.load_state_dict(modelStateDict)
model.to(DEVICE)'''


#sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..")))  # without installation, add /path/to/Audio-ZEN
import audio_zen.loss as loss
from audio_zen.utils import initialize_module


def entry(config, resume, only_validation):
    torch.manual_seed(config["meta"]["seed"])  # For both CPU and GPU
    np.random.seed(config["meta"]["seed"])
    random.seed(config["meta"]["seed"])
    torch.cuda.device('gpu') if torch.cuda.is_available() else torch.device('cpu')
    #print(f"{rank + 1} process initialized.")

    # The DistributedSampler will split the dataset into the several cross-process parts.
    # On the contrary, setting "Sampler=None, shuffle=True", each GPU will get all data in the whole dataset.
    
    train_dataset = initialize_module(config["train_dataset"]["path"], args=config["train_dataset"]["args"])
    train_dataset.clean_dataset_list = [os.path.join('/kaggle/input/fsn-data', sample) for sample in train_dataset.clean_dataset_list]
    train_dataset.noisy_dataset_list = [os.path.join('/kaggle/input/fsn-data', sample) for sample in train_dataset.noisy_dataset_list]
    
    valid_dataset = initialize_module(config["validation_dataset"]["path"], args=config["validation_dataset"]["args"])
    valid_dataset.clean_dataset_list = [os.path.join('/kaggle/input/fsn-data', sample) for sample in valid_dataset.clean_dataset_list]
    valid_dataset.noisy_dataset_list = [os.path.join('/kaggle/input/fsn-data', sample) for sample in valid_dataset.noisy_dataset_list]
    
    train_dataloader = DataLoader(
        dataset=train_dataset,
        shuffle=True,
        **config["train_dataset"]["dataloader"],
    )

    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        shuffle=True,
        **config["validation_dataset"]["dataloader"]
    )

    '''print('\nTrain dataset clean samples:')
    [print(f'{data}') for data in train_dataloader.dataset.clean_dataset_list[:10]]
    print('\nTrain dataset noisy samples:')
    [print(f'{data}') for data in train_dataloader.dataset.noisy_dataset_list[:10]]
    print('\nValidation dataset clean samples:')
    [print(f'{data}') for data in valid_dataloader.dataset.clean_dataset_list[:10]]
    print('\nValidation dataset noisy samples:')
    [print(f'{data}') for data in valid_dataloader.dataset.noisy_dataset_list[:10]]'''
    
    print(f'Train len: {len(train_dataloader)}')
    print(f'Val len: {len(valid_dataloader)}')
    
    #model = initialize_module(config["model"]["path"], args=config["model"]["args"])

    checkpointPath = os.path.join(cwd, 'fullsubnet/fullsubnet_best_model_58epochs.tar')
    model = initialize_module(config["model"]["path"], args=config["model"]["args"], initialize=True)
    modelCheckpoint = torch.load(checkpointPath, map_location=DEVICE)
    modelStateDict = modelCheckpoint["model"]
    model.load_state_dict(modelStateDict)
    model.to(DEVICE)
    model.train()

    modules = [module for module in model.modules() if not (isinstance(module, SequenceModel) or isinstance(module, Model))]
    modules = [module for module in modules if not isinstance(module, torch.nn.Linear)]

    for layer in modules:
        layer.requires_grad_(False)

    #summary(model)
    
    paramsFt = [{'params': module.to(DEVICE).parameters()} for module in modules]

    optimizer = torch.optim.Adam(
        #params=model.parameters(),
        params=paramsFt,
        lr=config["optimizer"]["lr"]/10,
        betas=(config["optimizer"]["beta1"], config["optimizer"]["beta2"])
    )

    loss_function = getattr(loss, config["loss_function"]["name"])(**config["loss_function"]["args"])

    trainer_class = initialize_module(config["trainer"]["path"], initialize=False)

    trainer = trainer_class(
        config=config,
        resume=resume,
        only_validation=only_validation,
        model=model,
        loss_function=loss_function,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        validation_dataloader=valid_dataloader
    )

    trainer.train()

'''
class CustomArguments:

    def __init__(self):

        self.configuration = os.path.join(cwd, 'fullsubnet/train.toml')
        self.resume = False
        self.only_validation = False
        self.preloaded_model_path = os.path.join(cwd, 'fullsubnet/fullsubnet_best_model_58epochs.tar')
'''

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='FullSubNet')
    parser.add_argument("-C", "--configuration", required=True, type=str, help="Configuration (*.toml).")
    parser.add_argument("-R", "--resume", action="store_true", help="Resume the experiment from latest checkpoint.")
    parser.add_argument("-V", "--only_validation", action="store_true", help="Only run validation, which is used for debugging.")
    parser.add_argument("-P", "--preloaded_model_path", type=str, help="Path of the *.pth file of a model.")
    args = parser.parse_args()

    #args = CustomArguments()
    
    os.environ.setdefault('LOCAL_RANK', '0')
    local_rank = int(os.environ["LOCAL_RANK"])
    
    if args.preloaded_model_path:
        assert not args.resume, "The 'resume' conflicts with the 'preloaded_model_path'."

    config_path = Path(args.configuration).expanduser().absolute()
    configuration = toml.load(config_path.as_posix())

    # append the parent dir of the config path to python's context
    # /path/to/recipes/dns_interspeech_2020/exp/'
    sys.path.append(config_path.parent.as_posix())

    configuration["meta"]["experiment_name"], _ = os.path.splitext(os.path.basename(args.configuration))
    configuration["meta"]["config_path"] = args.configuration
    configuration["meta"]["preloaded_model_path"] = args.preloaded_model_path

    entry(configuration, args.resume, args.only_validation)
