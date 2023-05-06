# Self-Supervised Learning for Audio Inference
This repository contains the codebase for all experiments for the research project `Self-Supervised Learning for Audio Inference` (Imperial MSc AI 2022). <br />
Author: [Jonah Anton](https://github.com/jonahanton) <br />
Supervisors: [Harry Coppock](https://harrycoppock.com/), Bjorn Schuller

**Abstract**: <br />
This report presents Audio Barlow Twins, a novel self-supervised audio representation learning ap- proach. Audio Barlow Twins adapts the [Barlow Twins](https://arxiv.org/abs/2103.03230) [Zbontar _et al._, 2021] objective from Computer Vision (CV) to the audio domain. The Barlow Twins objective encourages the empirical cross-correlation matrix be- tween the embeddings of two distorted views of a data sample towards the identity matrix, thereby both enforcing invariance to the applied data augmentations as well as explicitly preventing repre- sentational collapse through decorrelation of the individual components of the embedding vectors (redundancy-reduction). Audio Barlow Twins utilises a combination of different data augmentations, all of which act directly on the audio data preprocessed as mel-spectrograms, and considers both convolutional and Transformer encoder architectures. We pre-train on the large-scale audio dataset AudioSet, and evaluate the quality of the learned representations on 18 tasks from the HEAR 2021 Challenge (https://arxiv.org/abs/2203.03022) [Turian _et al._, 2021], achieving results on a par with the current state-of-the-art for instance discrimination self- supervised approaches to audio representation learning. The impact of the individual components of the learning framework are analysed through extensive ablation studies.
<br />

This work has been adapted to a paper [Audio Barlow Twins: Self-Supervised Audio Representation Learning](https://arxiv.org/pdf/2209.14345.pdf)[Anton et al., 2022] and has been accepted at ICASSP 2023.
