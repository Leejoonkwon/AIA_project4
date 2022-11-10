# pip install scipy librosa unidecode inflect
# LJ Speech dataset에서 사전 학습된 Tacotron2와 WaveGlow 모델 로드
from ctypes import alignment
from sympy import sequence
import torch

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda:0' if USE_CUDA else 'cpu')
print('torch : ', torch.__version__, '사용 DEVICE : ', DEVICE)


waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', 
                          model_math='fp32', force_reload=True,
                          map_location=torch.device('cpu'))
waveglow = waveglow.remove_weightnorm(waveglow)
waveglow = waveglow.to('cuda')
waveglow.eval()

tacotron2 = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', 
                           model_math='fp32', force_reload=True,
                           map_location=torch.device('cpu')
                           )
tacotron2 = tacotron2.to('cuda')
tacotron2.eval()

# text = "hello world, I missed you so much"
# text = "As world leaders gathered at the United Nations in New York and condemned him, \
# Russian President Vladimir Putin was back home, scrambling to refill his depleted war machine."
text = "The awesome yellow planet of Tatooine emerges from a total eclipse, her two moons glowing against the darkness. A tiny silver spacecraft, a Rebel Blockade Runner firing lasers \
        from the back of the ship, races through space. It is pursed by a giant Imperial Stardestroyer. Hundreds of deadly \
        laserbolts streak from the Imperial Stardestroyer, causing the main solar fin of the Rebel craft to disintegrate."

utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')
sequences, lengths = utils.prepare_input_sequence([text])

with torch.no_grad():
    mel, _, _ = tacotron2.infer(sequences, lengths)
    audio = waveglow.infer(mel)
audio_numpy = audio[0].data.cpu().numpy()
rate = 22050


from scipy.io.wavfile import write
write('audio_test02.wav', rate, audio_numpy)

from IPython.display import Audio
Audio(audio_numpy, rate=rate)

'''
# Text to Speech(TTS)
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('seaborn-white')

def plot_data(data, figsize=(16, 4)):
    fig, axes = plt.subplots(1, len(data), figsize=figsize)
    for i in range(len(data)):
        axes[i].imshow(data[i], aspect='auto', origin='bottom',
                       interpolation='none', cmap='viridis')  
        
def TTS(text):
    
    sampling_rate = 22050
    
    sequence = np.array(tacotron2.text_to_sequence(text, ['english_cleaners']))[None, :]
    sequence = torch.from_numpy(sequence).to(device='cuda', dtype=torch.int64)
    
    with torch.no_grad():
        mel_outputs, mel_outputs_postnet, _, alignments = tacotron2.infer(sequence)
        audio = waveglow.infer(mel_outputs_postnet)
        
    mel_output = mel_outputs.data.cpu().numpy()[0]
    mel_outputs_postnet = mel_outputs_postnet.data.cpu().numpy()[0]
    alignment = alignments.data.cpu().numpy()[0].T
    audio_np = audio[0].data.cpu().numpy()
    
    return mel_output, mel_outputs_postnet, alignment, audio_np, sampling_rate

import librosa.display
from IPython.display import Audio

text = "hello world, I missed you so much"
mel_output, mel_outputs_postnet, alignment, audio_np, sampling_rate = TTS(text)

fig = plt.figure(figsize=(14, 4))
librosa.display.waveplot(audio_np, sr=sampling_rate)
plot_data(mel_output, mel_outputs_postnet, alignment)
Audio(audio_np, rate=sampling_rate)
'''    