import torch
import tools
import librosa
import numpy as np

net = tools.load_model()

y,sr = librosa.load(r"..\data\music\genres_original\blues\blues.00000.wav")
chroma = librosa.feature.chroma_stft(y=y)
rms = librosa.feature.rms(y=y)
sc = librosa.feature.spectral_centroid(y=y)
sb = librosa.feature.spectral_bandwidth(y=y)
rolloff = librosa.feature.spectral_rolloff(y=y)
zero_c = librosa.feature.zero_crossing_rate(y=y)
harmony ,perc = librosa.effects.hpss(y=y)
tempo = librosa.feature.tempo(y=y)
mfcc = librosa.feature.mfcc(y=y,n_mfcc=20)

fea = np.zeros(57)
fea[0] = np.mean(chroma)
fea[1] = np.var(chroma)
fea[2] = np.mean(rms)
fea[3] = np.var(rms)
fea[4] = np.mean(sc)
fea[5] = np.var(sc)
fea[6] = np.mean(sb)
fea[7] = np.var(sb)
fea[8] = np.mean(rolloff)
fea[9] = np.var(rolloff)
fea[10] = np.mean(zero_c)
fea[11] = np.var(zero_c)
fea[12] = np.mean(harmony)
fea[13] = np.var(harmony)
fea[14] = np.mean(perc)
fea[15] = np.var(perc)
fea[16] = tempo
mfcc_mean = np.mean(mfcc,axis = 1)
mfcc_var = np.var(mfcc,axis = 1)
j = 17
for i in range(20):
    fea[j] = mfcc_mean[i]
    j += 1
    fea[j] = mfcc_var[i]

fea = fea.astype(np.float32)
fea = torch.from_numpy(fea)
fea = torch.unsqueeze(fea,dim = 0)
fea = fea.cuda()
net.to('cuda')

print(net(fea))