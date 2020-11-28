import torch
import torchaudio
import matplotlib.pyplot as plt
from pathlib import Path
from keras.models import Sequential
from keras.layers import Dense


# p = Path('Supercharger_Blockiergebuehr_Tesla_Fordert_Geld_V.mp3')
#
# print(p.exists())

use_cuda = torch.cuda.is_available()
print("cuda.is_available: {}".format(use_cuda))
device = torch.device("cuda" if use_cuda else "cpu")


filename = "Supercharger_Blockiergebuehr_Tesla_Fordert_Geld_V_15sec.mp3"
print(torchaudio.info(filename))

waveform, sample_rate = torchaudio.load(filename)

print("Shape of waveform before mel: {}".format(waveform.size()))
print("Sample rate of waveform before mel: {}".format(sample_rate))

plt.figure()
plt.plot(waveform.t().numpy())
plt.show()

# waveform = torchaudio.transforms.MelScale(n_mels=128, sample_rate=8000).forward(waveform.t())
# print("Shape of waveform before mel: {}".format(waveform.size()))
# plt.figure()
# plt.plot(waveform.t().numpy())
# plt.show()

specgram = torchaudio.transforms.MelSpectrogram()(waveform)

print("Shape of spectrogram: {}".format(specgram.size()))

plt.figure()
p = plt.imshow(specgram.log2()[0, :, :].detach().numpy(), cmap='gray')
plt.show()

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=100))
model.add(Dense(units=10, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

x_train = specgram

model.fit(x_train, x_train, epochs=5, batch_size=32)




