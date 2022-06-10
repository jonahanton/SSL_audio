"""
Adapted from https://github.com/nttcslab/msm-mae/blob/main/msm_mae/runtime.py
"""

# workaround for using heareval with `pip install -e .`
import sys
sys.path.append('..')

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio 
import torchaudio.transforms as AT

from einops import rearrange

from models.mst import get_mst_model


class Config:

    input_size = [80, 96]

    # FFT parameters
    sample_rate = 16000
    n_fft = 1024
    win_length = 400 
    hop_length = 160
    n_mels = 80
    f_min = 50
    f_max = 8000
    


def get_model(cfg, weight_file):

    sd = torch.load(weight_file, map_location='cpu')
    sd = sd['model']
    # remove `module.` prefix
    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    # remove `backbone.` prefix induced by BarlowTwins wrapper
    sd = {k.replace("backbone.", ""): v for k, v in sd.items()}
    # get patch size
    model_ps = sd['patch_embed.proj.weight'].shape[2:] 
    # get embedding dim 
    embed_dim = sd['pos_embed'].shape[2]

    model_size = 'tiny' if embed_dim == 192 else 'small' if embed_dim == 384 else 'base'
    model = get_mst_model(
        size=model_size,
        patch_size=(model_ps[0], model_ps[1]),
    )
    
    # load in weights
    model.load_state_dict(sd, strict=False)
    model.eval()
    del sd

    return model


def get_to_melspecgram(cfg):
    to_melspecgram = AT.MelSpectrogram(
        sample_rate=cfg.sample_rate,
		n_fft=cfg.n_fft,
		win_length=cfg.win_length,
		hop_length=cfg.hop_length,
		n_mels=cfg.n_mels,
        f_min=cfg.f_min,
        f_max=cfg.f_max,
		power=2,
		)
    return to_melspecgram


def get_timestamps(cfg, batch_audio, x):
    audio_len = len(batch_audio[0])
    sec = audio_len / cfg.sample_rate
    x_len = len(x[0])
    step = sec / x_len
    ts = torch.tensor([step * i for i in range(x_len)]).unsqueeze(0)
    ts = ts.repeat(len(batch_audio), 1)
    return ts


class RuntimeMST(nn.Module):

    def __init__(self, weight_file, cfg=Config()):
        super().__init__()

        self.cfg = cfg
        self.cfg.weight_file = weight_file
        self.encoder = get_model(self.cfg, self.cfg.weight_file)
        self.to_spec = get_to_melspecgram(self.cfg)

    
    def to_feature(self, batch_audio):
        # raw -> spectrogram
        x = self.to_spec(batch_audio)
        x = (x + torch.finfo().eps).log()
        x = x.unsqueeze(1)
        return x

    
    def normalize_batch(self, x):
        mu, sigma = x.mean(), x.std()
        x = (x - mu) / sigma
        return x

    
    def to_normalized_spec(self, batch_audio):
        # raw -> spectrogram 
        x = self.to_feature(batch_audio)
        # normalize among batch samples
        x = self.normalize_batch(x)
        return x 

    
    def encode_lms(self, x, cls_only):

        patch_fbins = self.encoder.grid_size()[0]
        embed_d = self.encoder.patch_embed.proj.out_channels
        unit_frames = self.cfg.input_size[1]  # number of time frames for inputs 
        # pad input's (x's) number of frames so that it's an integer multiple of unit_frames
        pad_frames = unit_frames - (x.shape[-1] % unit_frames)
        if pad_frames > 0:
            x = F.pad(x, (0, pad_frames))

        embeddings = []
        if cls_only:
            # [CLS] embeddings only
            for i in range(x.shape[-1] // unit_frames):
                emb, _, _ = self.encoder(x[..., i*unit_frames:(i+1)*unit_frames], mask_ratio=0.)
                assert self.encoder.use_cls_token, '[CLS] NOT AVAILABLE'
                emb = emb[:, :1]  # [emb] = [b, 1, d], n.b. emb = emb[:, 0] -> [emb] = [b, d]
                embeddings.append(emb)

            # concat along the 2nd dimension (dim=1)
            x = torch.hstack(embeddings)  # [x] = [b, n_unit_frames, d], n_unit_frames = x.shape[-1] // unit_frames
        else:
            # stack embeddings
            for i in range(x.shape[-1] // unit_frames):
                emb, _, _ = self.encoder(x[..., i*unit_frames:(i+1)*unit_frames], mask_ratio=0.)
                if self.encoder.use_cls_token:
                    emb = emb[:, 1:, :]
                emb = rearrange(emb, ' b (f t) d -> b t (f d)', f=patch_fbins, d=embed_d)
                embeddings.append(emb)
            # concat along the 2nd dimension (dim=1)
            x = torch.hstack(embeddings)  # [x] = [b, n_unit_frames*patch_tbins, patch_fbins*d]
            pad_emb_frames = int(embeddings[0].shape[1] * pad_frames / unit_frames)
            if pad_emb_frames > 0:
                x = x[:, :-pad_emb_frames]  # remove padded tails
        return x 

    
    def encode(self, batch_audio, cls_only):
        x = self.to_normalized_spec(batch_audio)
        return self.encode_lms(x, cls_only)
    

    def get_scene_embeddings(self, audio):
        """
        audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.
        Returns:
            embedding: A float32 Tensor with shape (n_sounds, model.scene_embedding_size).
        """
        x = self.encode(audio, cls_only=True)
        x = torch.mean(x, dim=1)  # average [CLS] tokens from different audio segments (orig. clip split into lengths of cfg.input_size[1] = 96) 
        return x 

    
    def get_timestamp_embeddings(self, audio):
        """
        audio: n_sounds x n_samples of mono audio in the range [-1, 1]. All sounds in a batch will be padded/trimmed to the same length.
        Returns:
            embedding: A float32 Tensor with shape (n_sounds, n_timestamps, model.timestamp_embedding_size).
            timestamps: A float32 Tensor with shape (n_sounds, n_timestamps). Centered timestamps in milliseconds corresponding to each embedding in the output.
        """
        x = self.encode(audio, cls_only=False)
        ts = get_timestamps(self.cfg, audio, x)
        return x, ts




if __name__ == "__main__":
    
    cfg = Config()
    wav, sr = torchaudio.load('data/audioset/samples/--1yd6dcNOQ.wav')
    # if audio has 2 channels, convert to mono
    if wav.shape[0] == 2:
        wav = torch.mean(wav, dim=0).unsqueeze(0)
    # [wav] = [1, length]

    # print(wav.shape[1]) --> 160000 (10s clip, 10*sr, sr=16000)

    model_file_path = 'checkpoint/0606_16-25-model=transformer_base-ps=16x16-maskratio=0.75/models/epoch-1.pth.tar'
    model = RuntimeMST(weight_file=model_file_path, cfg=cfg) 
    model.eval()

    x = model.to_normalized_spec(wav)
    print(x.shape)  # --> torch.Size([1, 1, 80, 1001])

    scene_emb = torch.mean(model.encode_lms(x, cls_only=True), dim=1)
    timestamp_emb = model.encode_lms(x, cls_only=False)
    print('scene embed:', scene_emb.shape)  # --> scene embed: torch.Size([1, 768])
    print('timestamp embed:', timestamp_emb.shape)  # --> timestamp embed: torch.Size([1, 63, 3840])
    # Why timestamp_emb.shape[2] = 3840? As grid_size in freq. dim = 5, and 768*5 = 3840  
    # Why timestamp_emb.shape[1] = 63?
    # corresponds to 63 patches, as [10s*16000Hz(=sr) / 160(=hop)] + 1 = 1001 frames, length of patch = 16 frames, and ceil(1001 / 16) = 63  

    ts = get_timestamps(cfg, wav, timestamp_emb)
    print(ts)
    print(ts.shape)  # --> torch.Size([1, 63])
    # ts[0] = 0, ts[1] = 0.159. Why? ts 0 -> 1 <-> 16 frames, (16/1001(=total n frames)) * 10s = 0.159s 