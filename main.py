import pyaudio
import wave
import numpy as np
import matplotlib.pyplot as plt
import struct
from pydub import AudioSegment
import threading
import time

# Convert MP3 to WAV
def convert_mp3_to_wav(mp3_file, wav_file):
	audio = AudioSegment.from_mp3(mp3_file)
	audio.export(wav_file, format="wav")

# Precompute FFT for the entire file
def precompute_fft(wav_file, chunk_size=4096):  # Increased frequency resolution
	wf = wave.open(wav_file, 'rb')
	frame_rate = wf.getframerate()
	total_frames = wf.getnframes()
	channels = wf.getnchannels()
	
	fft_results = []
	timestamps = []

	for i in range(0, total_frames, chunk_size):
		data = wf.readframes(chunk_size)
		if len(data) < chunk_size * wf.getsampwidth():
			break

		audio_data = np.frombuffer(data, dtype=np.int16)
		fft_data = np.abs(np.fft.fft(audio_data)[:chunk_size//2])

		fft_results.append(fft_data)
		timestamps.append(i / frame_rate)

	wf.close()
	return fft_results, timestamps, frame_rate

# Audio Playback + FFT Visualization in sync
def audio_visualizer(wav_file):
	CHUNK = 4096*2  # Increased frequency resolution
	MAX_Y = 10000000*10  # Fixed Y-axis max value

	# Precompute FFT
	fft_results, timestamps, frame_rate = precompute_fft(wav_file, CHUNK)
	wf = wave.open(wav_file, 'rb')

	p = pyaudio.PyAudio()
	stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
					channels=wf.getnchannels(),
					rate=wf.getframerate(),
					output=True)

	# Matplotlib Setup
	fig, ax = plt.subplots()
	x = np.fft.fftfreq(CHUNK, d=1/frame_rate)[:CHUNK//2]
	line, = ax.plot(x, np.random.rand(CHUNK//2), 'r')

	ax.set_xlim(20, frame_rate//2)  # Log scale
	ax.set_ylim(0, MAX_Y)  # Fixed max value
	ax.set_xscale('log')
	ax.set_xlabel('Frequency (Hz)')
	ax.set_ylabel('Magnitude')
	ax.set_title('Real-Time Frequency Spectrum')

	plt.ion()
	plt.show()

	# Start playback
	start_time = time.time()

	for i, fft_data in enumerate(fft_results):
		data = wf.readframes(CHUNK)
		if len(data) == 0:
			break
		stream.write(data)

		# Sync visualization with audio playback time
		expected_time = timestamps[i]
		current_time = time.time() - start_time
		sleep_time = expected_time - current_time
		if sleep_time > 0:
			time.sleep(sleep_time)

		line.set_ydata(fft_data)
		plt.pause(0.01)

	# Cleanup
	stream.stop_stream()
	stream.close()
	p.terminate()
	plt.ioff()
	plt.show()

# Main function
def main(mp3_file):
	wav_file = "temp.wav"
	convert_mp3_to_wav(mp3_file, wav_file)
	audio_visualizer(wav_file)

if __name__ == "__main__":
	mp3_filename = "pedro.mp3"  # Updated file name
	main(mp3_filename)
