from numpy import np




Res50 = np.load('Res50_proba.npy')
Res50Mod = np.load('Res50Mod.npy')
IncV3 = np.load('incv3_proba.npz.npy')
IncV4 = np.load('incv4_proba.npz.npy')
Xcep = np.load('xcep_proba.npz.npy')

total = []
for i in range(774):
    total[i] = Res50[i] + Res50Mod[i] + IncV4[i] + IncV3[i] + Xcep[i]
    total[i] = total[i] / 5
total = np.array(total)
if total.shape[-1] > 1:
    total = total.argmax(axis=-1)
else:
    total = (total > 0.5).astype('int32')
