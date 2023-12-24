import raw_dataset as dataset
from feature_extraction import *
import os
import torch
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor,Wav2Vec2Tokenizer
from transformers import Wav2Vec2Model,Wav2Vec2FeatureExtractor

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

cuda = torch.cuda.is_available()
print('Cuda device available: ', cuda)
device = torch.device("cuda" if cuda else "cpu")

def pad_dataset(wav):
    waveform = wav.squeeze(0)
    waveform_len = waveform.shape[0]
    cut = 64600
    if waveform_len >= cut:
        waveform = waveform[:cut]
        return waveform
    # need to pad
    num_repeats = int(cut / waveform_len) + 1
    padded_waveform = torch.tile(waveform, (1, num_repeats))[:, :cut][0]
    return padded_waveform
    
def normalization(orign_data):
    d_min = orign_data.min()
    if d_min < 0:
        orign_data += torch.abs(d_min)
        d_min = orign_data.min()
    d_max = orign_data.max()
    distance = d_max - d_min
    norm_data = (orign_data - d_min).true_divide(distance)
    return norm_data
    
for part_ in ["train","dev","eval"]:
    asvspoof_raw = dataset.ASVspoof2019Raw("LA", "/home/chenghaonan/xieyuankun/data/asv2019",
                                           "/home/chenghaonan/xieyuankun/data/asv2019/ASVspoof2019_LA_cm_protocols/", part=part_)
    target_dir = os.path.join("/home/chenghaonan/xieyuankun/data/asv2019/preprocess_xls-r", part_,
                              "xls-r")
    processor =  Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").cuda()
    #model.eval()
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for idx in tqdm(range(len(asvspoof_raw))):
        waveform, filename, tag, label = asvspoof_raw[idx]
        waveform = waveform.to(device)
        waveform = pad_dataset(waveform).to('cpu')
        input_values = processor(waveform, sampling_rate=16000,return_tensors="pt").input_values.cuda()
        with torch.no_grad():
          wav2vec2 = model(input_values).last_hidden_state.cuda()
        print(wav2vec2.shape)
        torch.save(wav2vec2, os.path.join(target_dir, "%05d_%s_%s_%s.pt" % (idx, filename, tag, label)))
    print("Done!")
    
    

