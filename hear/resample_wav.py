"""
This program converts the original audio files recursively found in the source folder,
then stores them in the destination folder while holding the same relative path structure.

This converstion includes the following processes:
    - Resampling to a sampling rate 

Code adapted from https://github.com/nttcslab/msm-mae/blob/main/wav_to_lms.py.
"""

import numpy as np 
from pathlib import Path
import librosa 
import soundfile
from multiprocessing import Pool
import torch.multiprocessing as mp 
import torch 
import fire 
from tqdm import tqdm 
from pprint import pprint


def _converter_worker(args):
    subpathname, from_dir, to_dir, sample_rate, suffix, verbose = args
    from_dir, to_dir = Path(from_dir), Path(to_dir)
    to_name = to_dir/subpathname

    if to_name.exists():
        print('already exist', subpathname)
        return ''

    # load and convert to 16kHz
    try:
        wav, org_sr = librosa.load(str(from_dir/subpathname), sr=sample_rate)
    except Exception as e:
        print('ERROR failed to open or convert', subpathname, '-', str(e))
        return ''

    to_name.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(
        file=to_name,
        data=wav,
        samplerate=sample_rate,
    )

    if verbose:
        print(from_dir, '->', to_name)
    
    return to_name.name


def convert_wav(from_dir, to_dir, sample_rate=16000, suffix='.wav', skip=0, verbose=False):
    from_dir = str(from_dir)
    files = [str(f).replace(from_dir, '') for f in Path(from_dir).glob(f'**/*{suffix}')]
    files = [f[1:] if f[0] == '/' else f for f in files]
    files = sorted(files)
    if skip > 0:
        files = files[skip:]

    print(f'Processing {len(files)} {suffix} files at a sampling rate of {sample_rate} Hz...')
    assert len(files) > 0

    with Pool() as p:
        args = [[f, from_dir, to_dir, sample_rate, suffix, verbose] for f in files]
        shapes = list(tqdm(p.imap(_converter_worker, args), total=len(args)))

    print('finished.')



if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    convert_wav(
        from_dir='data/mridangam_stroke-v1.5-full',
        to_dir='data/mridangam_stroke-v1.5-full-16kHz'
    )