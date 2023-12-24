import soundfile as sf
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from transformers import Wav2Vec2Model
import torchaudio
import librosa
import numpy as np
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
# # audio_input, sample_rate = librosa.load('LA_T_1004631.flac',sr=16000)  # (31129,)
# audio_input = torch.randn((16,64600))

# model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")   
# # wav2vec2 = model(input_values).logits# torch.Size([1, 97, 768])
# wav2vec2 = model(audio_input).extract_features    # torch.Size([1, 97, 768])
# wav2vec2 = wav2vec2.transpose(1,2)
# print(wav2vec2.shape)# torch.Size([1, 97, 32])
def torchaudio_load(filepath):
    wave, sr = librosa.load(filepath,16000)
    waveform = torch.Tensor(np.expand_dims(wave, axis=0))
    return [waveform, sr]
    
def paddataset(wav):
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

wav,samplerate = torchaudio_load('LA_E_1000147.wav')
print(wav.size())
waveform = paddataset(wav)


input_values = processor(waveform, sampling_rate=16000, return_tensors="pt").input_values  # torch.Size([1, 31129])
print(input_values.size())
#
# class PadDataset(torch.utils.data.Dataset):
#
#     def __init__(self, dataset: torch.utils.data.Dataset, cut: int = 64600, label=None):
#         self.dataset = dataset
#         self.cut = cut  # max 4 sec (ASVSpoof default)
#         self.label = label
#
#     def __getitem__(self, index):
#         waveform, sample_rate = self.dataset[index]
#         waveform = self.apply_pad(waveform, self.cut)
#
#         if self.label is None:
#             return waveform, sample_rate
#         else:
#             return waveform, sample_rate, self.label
#
#     @staticmethod
#     def apply_pad(waveform, cut):
#
#         return padded_waveform

