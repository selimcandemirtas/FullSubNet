import matplotlib.pyplot as plt
import torch
from torch.cuda.amp import autocast
from tqdm import tqdm

from audio_zen.acoustics.feature import drop_band
from audio_zen.trainer.base_trainer import BaseTrainer
from audio_zen.acoustics.mask import build_complex_ideal_ratio_mask, decompress_cIRM

plt.switch_backend('agg')


class Trainer(BaseTrainer):
    def __init__(self, config, resume, only_validation, model, loss_function, optimizer, train_dataloader, validation_dataloader):
        super().__init__(config, resume, only_validation, model, loss_function, optimizer)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = validation_dataloader

    def _train_epoch(self, epoch):
        loss_total = 0.0

        #for noisy, clean in tqdm(self.train_dataloader, desc="Training") if self.rank == 0 else self.train_dataloader:
        for noisy, clean in tqdm(self.train_dataloader, desc="Training"):
            self.optimizer.zero_grad()

            #noisy = noisy.to(self.rank)
            #clean = clean.to(self.rank)

            noisy_mag, noisy_phase, noisy_real, noisy_imag = self.torch_stft(noisy)
            _, _, clean_real, clean_imag = self.torch_stft(clean)
            cIRM = build_complex_ideal_ratio_mask(noisy_real, noisy_imag, clean_real, clean_imag)  # [B, F, T, 2]
            cIRM = drop_band(
                cIRM.permute(0, 3, 1, 2),  # [B, 2, F ,T]
                #self.model.module.num_groups_in_drop_band
                self.model.num_groups_in_drop_band
            ).permute(0, 2, 3, 1)

            with autocast(enabled=self.use_amp):
                # [B, F, T] => [B, 1, F, T] => model => [B, 2, F, T] => [B, F, T, 2]
                noisy_mag = noisy_mag.unsqueeze(1)
                cRM = self.model(noisy_mag)
                cRM = cRM.permute(0, 2, 3, 1)
                loss = self.loss_function(cIRM, cRM)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            loss_total += loss.item()

        #if self.rank == 0:
        self.writer.add_scalar(f"Loss/Train", loss_total / len(self.train_dataloader), epoch)

    @torch.no_grad()
    def _validation_epoch(self, epoch):
        visualization_n_samples = self.visualization_config["n_samples"]
        visualization_num_workers = self.visualization_config["num_workers"]
        visualization_metrics = self.visualization_config["metrics"]

        loss_total = 0.0
        #loss_list = {"With_reverb": 0.0, "No_reverb": 0.0, }
        #item_idx_list = {"With_reverb": 0, "No_reverb": 0, }
        item_idx = 0
        #noisy_y_list = {"With_reverb": [], "No_reverb": [], }
        noisy_y = []
        #clean_y_list = {"With_reverb": [], "No_reverb": [], }
        clean_y = []
        #enhanced_y_list = {"With_reverb": [], "No_reverb": [], }
        enhanced_y = []
        #validation_score_list = {"With_reverb": 0.0, "No_reverb": 0.0}
        validation_score = 0.0

        for i, (noisy, clean, name) in tqdm(enumerate(self.valid_dataloader), desc="Validation"):
            assert len(name) == 1, "The batch size for the validation stage must be one."
            name = name[0]

            #noisy = noisy.to(self.rank)
            #clean = clean.to(self.rank)

            noisy_mag, noisy_phase, noisy_real, noisy_imag = self.torch_stft(noisy)
            _, _, clean_real, clean_imag = self.torch_stft(clean)
            cIRM = build_complex_ideal_ratio_mask(noisy_real, noisy_imag, clean_real, clean_imag)  # [B, F, T, 2]

            noisy_mag = noisy_mag.unsqueeze(1)
            cRM = self.model(noisy_mag)
            cRM = cRM.permute(0, 2, 3, 1)

            loss = self.loss_function(cIRM, cRM)

            cRM = decompress_cIRM(cRM)

            enhanced_real = cRM[..., 0] * noisy_real - cRM[..., 1] * noisy_imag
            enhanced_imag = cRM[..., 1] * noisy_real + cRM[..., 0] * noisy_imag
            enhanced = self.torch_istft((enhanced_real, enhanced_imag), length=noisy.size(-1), input_type="real_imag")

            noisy = noisy.detach().squeeze(0).cpu().numpy()
            clean = clean.detach().squeeze(0).cpu().numpy()
            enhanced = enhanced.detach().squeeze(0).cpu().numpy()

            assert len(noisy) == len(clean) == len(enhanced)
            loss_total += loss

            # Separated loss
            #loss_list[speech_type] += loss
            item_idx += 1

            if item_idx <= visualization_n_samples:
                #self.spec_audio_visualization(noisy, enhanced, clean, name, epoch, mark=speech_type)
                self.spec_audio_visualization(noisy, enhanced, clean, name, epoch)

            noisy_y.append(noisy)
            clean_y.append(clean)
            enhanced_y.append(enhanced)

        self.writer.add_scalar(f"Loss/Validation_Total", loss_total / len(self.valid_dataloader), epoch)

        #for speech_type in ("With_reverb", "No_reverb"):
        #self.writer.add_scalar(f"Loss/Data", loss_total / len(self.valid_dataloader), epoch)

        validation_score = self.metrics_visualization(
            noisy_y, clean, enhanced_y,
            visualization_metrics, epoch, visualization_num_workers
        )

        #return validation_score_list["No_reverb"]
        return validation_score
