# Self-Supervised Learning for Audio Inference
This repository contains the codebase for all experiments for the research project `Self-Supervised Learning for Audio Inference` (Imperial MSc AI 2022). <br />
Author: [Jonah Anton](https://github.com/jonahanton) <br />
Supervisors: [Harry Coppock](https://harrycoppock.com/), Bjorn Schuller

Research proposal: adaptation of [Barlow Twins](https://arxiv.org/abs/2103.03230) [Zbontar _et al._, 2021] to the audio domain, using a Transformer encoder with masking of one of the two views, inspired by the recent work of Assran _et al._ in [Masked Siamese Networks for Label-Efficient Learning](https://arxiv.org/abs/2204.07141). The quality of the learned representations will be evaluated on the [HEAR Benchmark](https://arxiv.org/abs/2203.03022) [Turian _et al._, 2021]. A possible later reserach direction (progress dependent) is introducing a curriculum learning technique for the masking.  
