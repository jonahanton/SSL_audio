"""
This program converts the original audio files recursively found in the source folder,
then stores them in the destination folder while holding the same relative path structure.

This converstion includes the following processes:
    - Stereo to mono
    - Resampling to a sampling rate 
    - Converting to a log-mel spectrogram

Code adapted from https://github.com/nttcslab/msm-mae/blob/main/wav_to_lms.py.

Example:
    python -m data_manager.wav_to_lms data/audioset data/audioset_lms
"""

import numpy as np 
from pathlib import Path
import librosa 
import soundfile
from multiprocessing import Pool
import torch.multiprocessing as mp 
import torch 
import torchaudio
import torchaudio.transforms as AT
import fire 
from tqdm import tqdm 
from pprint import pprint

class FFT_parameters:
    # We extract log-mel spectrograms with 64 features using a window size of 64 ms and a stride of 10 ms from a waveform sampled at 16kHz.
    sample_rate = 16000
    window_size = 1024
    n_fft       = 1024
    hop_size    = 160
    n_mels      = 64
    f_min       = 60
    f_max       = 7800


class ToLogMelSpec:
    def __init__(self, cfg):
        # Spectrogram extractor
        self.cfg = cfg
        self.to_spec = AT.MelSpectrogram(
			sample_rate=cfg.sample_rate,
			n_fft=cfg.n_fft,
			win_length=cfg.window_size,
			hop_length=cfg.hop_size,
			n_mels=cfg.n_mels,
			f_min=cfg.f_min,
			f_max=cfg.f_max,
			power=2,
		)

    def __call__(self, audio):
        x = self.to_spec(torch.tensor(audio))
        x = (x + torch.finfo().eps).log()
        return x


def _converter_worker(args):
    subpathname, from_dir, to_dir, prms, to_lms, suffix, verbose = args
    from_dir, to_dir = Path(from_dir), Path(to_dir)
    to_name = to_dir/(subpathname[:-len(suffix)]+'.npy')

    if to_name.exists():
        print('already exist', subpathname)
        return ''

    # load and convert to a log-mel spectrogram
    try:
        wav, org_sr = librosa.load(str(from_dir/subpathname), sr=prms.sample_rate)
        lms = to_lms(wav)
    except Exception as e:
        print('ERROR failed to open or convert', subpathname, '-', str(e))
        return ''

    to_name.parent.mkdir(parents=True, exist_ok=True)
    np.save(to_name, lms)

    if verbose:
        print(from_dir, '->', to_name, lms.shape)
    
    return to_name.name


def convert_wav(from_dir, to_dir, sample_rate=16000, suffix='.wav', skip=0, verbose=False):
    from_dir = str(from_dir)
    files = [str(f).replace(from_dir, '') for f in Path(from_dir).glob(f'**/*{suffix}')]
    files = [f[1:] if f[0] == '/' else f for f in files]
    files = sorted(files)
    if skip > 0:
        files = files[skip:]

    prms = FFT_parameters()
    to_lms = ToLogMelSpec(prms)

    print(f'Processing {len(files)} {suffix} files at a sampling rate of {sample_rate} Hz...')
    assert len(files) > 0

    with Pool() as p:
        args = [[f, from_dir, to_dir, prms, to_lms, suffix, verbose] for f in files]
        shapes = list(tqdm(p.imap(_converter_worker, args), total=len(args)))

    print('finished.')



if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    fire.Fire(convert_wav)