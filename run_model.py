import torch
import tools
import librosa
import numpy as np

net = tools.load_model()
min_max = torch.load(r"..\models\min_max.pth")
classes = { 0 : 'blues',
            1 : 'classical',
            2 : 'country',
            3 : 'disco',
            4 : 'hiphop',
            5 : 'jazz',
            6 : 'metal',
            7 : 'pop',
            8 : 'reggae',
            9 : 'rock'}

y,sr = librosa.load(r"..\data\music\genres_original\jazz\jazz.00020.wav")
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
    j += 1

for i in range(57):
    fea[i] = (fea[i] - min_max[i][0]) / (min_max[i][1] - min_max[i][0])

fea = fea.astype(np.float32)
fea = torch.from_numpy(fea)
fea = torch.unsqueeze(fea,dim = 0)
fea = fea.cuda()
net.to('cuda')

res = net(fea)
pos_max = 0
res_max = torch.max(res)
res_min = torch.min(res)
res_max1 = res[0][0]
for i in range(10):
    if(res[0][i] > res_max1):
        res_max1 = res[0][i]
        pos_max = i
print(classes[pos_max])

for i in range(10):
    res[0][i] = (res[0][i] - res_min) / (res_max - res_min)

res_sum = torch.sum(res)
for i in range(10):
    print("%s : %f" %(classes[i],res[0][i] / res_sum))
#print(res.size())
