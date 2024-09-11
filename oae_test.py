import pyaudio
import numpy as np
from scipy.fft import fft
import scipy.signal
import matplotlib.pyplot as plt

SAMPLE_RATE = 44100
CHUNK_SIZE = 1024  # Number of frames per buffer

# OAE Frequencies
f1 = 2000           # Hz, Modifiable
f2 = 1.22 * f1      # Hz
f_oae = 2*f1 - f2   # Hz, the OAE freq we are looking for

test_duration = 1  # Seconds
N = SAMPLE_RATE*test_duration


# Generate stimulus tones
t = np.linspace(0, test_duration, test_duration*SAMPLE_RATE, False)
f1_tone = np.sin(f1*np.pi*t).astype(np.float32)
f2_tone = np.sin(f2*np.pi*t).astype(np.float32)
tones = np.vstack((f1_tone, f2_tone)).T


pos = 0 # Position used in callback

def callback(in_data, frame_count, time_info, status):
    global pos
    # Calculate the end position
    end_pos = pos + frame_count
    # Return the next chunk of audio data
    data = tones[pos:end_pos, :]
    # Advance the position
    pos = end_pos
    return (data.tobytes(), pyaudio.paContinue)

# Open PyAudio stream in callback mode
p = pyaudio.PyAudio()
stream_out = p.open(format=pyaudio.paFloat32,
                channels=2,
                rate=SAMPLE_RATE,
                output=True,
                frames_per_buffer=CHUNK_SIZE,
                stream_callback=callback)

stream_in = p.open(format=pyaudio.paFloat32,
                channels=1,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE)



# Start the streams
stream_out.start_stream()
stream_in.start_stream()


# Record until the tones to finish playing
recorded_audio = np.empty((0),dtype=np.float32)
while stream_out.is_active():
    data_in = np.frombuffer(stream_in.read(CHUNK_SIZE), dtype=np.float32)
    data_fft = fft(data_in)
    #ax.cla()
    #ax.plot(np.abs(data_fft))
    #plt.show()
    # Perform real-time Fourier transform
    recorded_audio = np.hstack((recorded_audio,data_in))
    
recorded_audio = recorded_audio-recorded_audio.mean()
print(np.shape(recorded_audio))
print(recorded_audio)

#Better STFT approach, needs to be modified
fig, ax = plt.subplots()

win = scipy.signal.windows.hamming(CHUNK_SIZE)
SFT = scipy.signal.ShortTimeFFT(win, int(CHUNK_SIZE/2), fs=SAMPLE_RATE)
Sx = SFT.stft(recorded_audio)
Sx_dB = 20*np.log10(abs(Sx))
im1 = ax.imshow(Sx_dB, origin='lower', aspect='auto', extent=SFT.extent(N), cmap='viridis')
plt.show()


# Close stream and PyAudio
stream_out.stop_stream()
stream_out.close()
p.terminate()

