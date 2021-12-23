import numpy as np
from scipy.signal import chirp, spectrogram
import matplotlib.pyplot as plt
import sounddevice as sd


volume = 0.5     # range [0.0, 1.0]
fs = 44100       # sampling rate, Hz, must be integer
duration = 1.0   # in seconds, may be float
f = 440.0        # sine frequency, Hz, may be float

# generate samples, note conversion to float32 array
t = np.linspace(0, duration, int(np.floor(duration*fs)))
x1 = (np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)
x2 = chirp(t, f0=2000, f1=20000, t1=duration, method='linear')
myarray = x2

sd.default.samplerate = fs
sd.default.channels = 2

# sd.query_device( default_low_input_latency= 0.09,
#                  default_low_output_latency= 0.09,
#                  default_high_input_latency= 0.18,
#                  default_high_output_latency= 0.18,)

sd.default.latency = ('low', 'low')


myrecording = sd.playrec(myarray, fs, dtype='float32')
sd.wait()
sd.stop()



print('***  ', np.shape(myrecording))



plt.figure()
for i in range(myrecording.shape[1]):
    plt.subplot(3,1,i+1)
    plt.plot(t, myrecording[:,i]) 
    plt.title('Recorded Signal Channel Number: {channelNo}'.format(channelNo = i+1))
    plt.ylabel('Voltage (V)')
    plt.xlabel('Time (s)')

plt.subplot(3,1,3)
plt.plot(t, x2)
plt.title('Generated Sweep Signal')
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')
plt.show()

# plt.figure()
# plt.plot(t, x1)
# plt.show()


# sd.play(myrecording, fs)
# sd.wait()
# sd.stop()










