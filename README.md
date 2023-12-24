# Wav2vec2-XLS-R-LCNN (19LA 0.6%EER)

This repository is a baseline model supported to the paper [FSD dataset](https://github.com/xieyuankun/FSD-Dataset)  .

The code sturcture is modified on the [ASVspoof2021_AIR](https://github.com/yzyouzhang/ASVspoof2021_AIR). 
Thanks to You Zhang for contribution to the field of audio deepfake detection!


### 1. Offline Data Extraction
```
python preprocess.py
```
Extract the last hidden states from the huggingface version of [wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m).

The raw wave files of the ASVspoof 2019 dataset is stored in `/home/chenghaonan/xieyuankun/data/asv2019`.
The last hidden states of wav2vec2 will be saved in `/home/chenghaonan/xieyuankun/data/asv2019/preprocess_xls-r`

It is worth noting that we perform pad or trim operations on the input before feeding it into wav2vec. 
Speech segments exceeding 4 seconds are trimmed, while those shorter than 4 seconds are repeated to fill 
the duration to 4 seconds. In my experiments, repeating yielded better results, 
possibly due to the characteristics of the 19 dataset where the artifacts of repeated silence 
better distinguish between genuine and fake instances. 
After offline processing with wav2vec, you will obtain features with a shape of (1, 201, 1024), 
where 201 corresponds to the temporal dimension.
## 2. Train Model

```
python main_train.py 
```
Before running the `main_train.py`, please change the `path_to_features` according to the files' location on your machine.
If training is slow, consider adjusting the num_worker parameter in conjunction with the number of CPU cores. 
The default is set to 8. If performance remains slow, you may explore multi-GPU training in args.

For loss, we employ weighted cross entropy, setting the weight for real to 10 and for fake to 1, in accordance with the imbalance observed in the 19LA dataset.
## 3. Test
```
python generate_score.py 
python evaluate_tDCF_asvspoof19.py
```

We also provided a pretrained model in `models_0.63/try/anti-spoofing_feat_model.pt`

The final test scores EER = 0.63572%,  min-tDCF = 0.01904
