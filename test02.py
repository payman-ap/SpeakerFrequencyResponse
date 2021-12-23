
import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt
import sounddevice as sd


volume = 0.5     # range [0.0, 1.0]
fs = 44100       # sampling rate, Hz, must be integer
duration = 1.0   # in seconds, may be float
f = 1000        # sine frequency, Hz, may be float

# generate samples, note conversion to float32 array
t = np.linspace(0, duration, int(np.floor(duration*fs)))
x1 = 1*(np.sin(2*np.pi*np.arange(fs*duration)*f/fs)).astype(np.float32)
x2 = 0.05*sps.chirp(t, f0=100, f1=24000, t1=duration, method='logarithmic')   # method{‘linear’, ‘quadratic’, ‘logarithmic’, ‘hyperbolic’}
myarray = x1



def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sps.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sps.lfilter(b, a, data)
    return y



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

# myrecording[0:10000] = 0
# myrecording[-4410:] = 0

# myrecording_tmp = myrecording
# myrecording = butter_bandpass_filter(myrecording_tmp, 2000, 20000, fs, order=5)

print('***  ', np.shape(myrecording))



plt.figure()
for i in range(myrecording.shape[1]):
    plt.subplot(3,1,i+1)
    plt.plot(t, myrecording[:,i]) 
    plt.title('Recorded Signal Channel Number: {channelNo}'.format(channelNo = i+1))
    plt.ylabel('Voltage (V)')
    plt.xlabel('Time (s)')

plt.subplot(3,1,3)
plt.plot(t, myarray)
plt.title('Generated Sweep Signal')
plt.ylabel('Voltage (V)')
plt.xlabel('Time (s)')
plt.show()

dt = 1/fs # delta t
N = len(myrecording) # number of freq. bins
freq = (1/(dt*N))*np.arange(N/2)



xfft = np.fft.fft(myarray, N)
yfft = np.fft.fft(myrecording[:,1], N)

xfft2 = xfft[0:int(np.floor(len(xfft)/2))]
xfft2[1:-1] = (1/N)*2*xfft2[1:-1]
yfft2 = yfft[0:int(np.floor(len(yfft)/2))]
yfft2[1:-1] = (1/N)*2*yfft2[1:-1]
yfft2[0] = 0
xfft2[0] = 0








fig, axs = plt.subplots(2,1)
plt.sca(axs[0])
plt.plot(freq, np.abs(xfft2), color='c', linewidth=1.5, label='Played')
plt.xlim(freq[0], freq[-1])
plt.legend()
plt.sca(axs[1])
plt.plot(freq, np.abs(yfft2), color='k', linewidth=1.5, label='Recorded')
plt.xlim(freq[0], freq[-1])
plt.legend()
plt.show()
