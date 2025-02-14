import wave
import numpy as np
import argparse
import threading

# Precompute FFT for the entire WAV file
def precompute_fft(wav_file, chunk_size=2048, overlap=0.5):
    wf = wave.open(wav_file, 'rb')
    frame_rate = wf.getframerate()
    total_frames = wf.getnframes()
    step_size = int(chunk_size * (1 - overlap)) # Overlapping windows for more fft data
    fft_results = []
    timestamps = []
    
    for i in range(0, total_frames - chunk_size, step_size):
        wf.setpos(i)
        data = wf.readframes(chunk_size)
        if len(data) < chunk_size * wf.getsampwidth():
            break
        audio_data = np.frombuffer(data, dtype=np.int16)
        fft_data = np.abs(np.fft.fft(audio_data)[:chunk_size//2])
        fft_results.append(fft_data)
        timestamps.append(i / frame_rate)

    wf.close()
    print(f"FFT Computation: Frequency range: 0Hz to {frame_rate/2:.2f}Hz, Resolution: {len(fft_results[0])} Chunk duration: {chunk_size/frame_rate:.3f} seconds, Data length: {(chunk_size/frame_rate)*len(fft_results):.3f}")
    return fft_results, timestamps, frame_rate, chunk_size

def bar(level, entries, genarr=False):
    ret = ""
    retarr = []
    span = 4095/entries
    for i in range(entries):
        if i != 0 and not genarr:
            ret += ","
        if level > span*(i + 1):
            if genarr:
                retarr += [255]
            else:
                ret += "4095"
        elif genarr:
            retarr += [0]
        else:
            ret += "0"
    if genarr:
        return retarr
    return ret

def bands_levels_gen(fft_results, bands):
    bands_levels = []
    max_levels = []
    # Make sure we have the correct amount of bands in the bands_levels
    for band in bands:
        if band[0] == band[1]:
            bands_levels.append(None)
            max_levels.append(0)
        else:
            bands_levels.append([])
            ## TODO remove the one that does not work, np is more efficiant if the conversion to np array is fast enough
            max_levels.append(np.max(np.array(fft_results)[:, band[0]:band[1]]))
            #max_levels.append(max(max(x[band[0]:band[1]]) for x in fft_results))

    # iterate over fft_results and for each frame, add maximum value in each band to each band
    for fft_frame in fft_results:
        for i in range(len(bands)):
            if bands_levels[i] is not None:
                bands_levels[i].append(int((np.max(fft_frame[bands[i][0]:bands[i][1]]) / max_levels[i]) * 4095))

    return bands_levels

def band_compute(bands, fft_results, timestamps, frame_rate, chunk_size, simulation):
    for i in range(len(bands)):
        # Extract frequencies in the specified range
        low_index = int(bands[i][0] / (frame_rate / chunk_size))
        high_index = int(bands[i][1] / (frame_rate / chunk_size))
        bands[i] = (low_index, high_index, bands[i][2])

    # Generate levels
    bands_levels = bands_levels_gen(fft_results, bands)

    if not simulation:
        expected_frames = int((timestamps[-1] - timestamps[0]) * 60)  # Total number of frames needed
        for i in range(len(bands_levels)):
            if bands_levels[i] is None:
                continue
            # Ensure each frame corresponds to 1/60th of a second
            # and ensure smooth brightness updates per 1/60s
            levels = bands_levels[i]
            levels = np.interp(np.linspace(0, len(levels)-1, expected_frames), np.arange(len(levels)), levels)
            levels = np.round(levels).astype(int)
            bands_levels[i] = levels

    max_length = 0
    if not all(x is None for x in bands_levels):
        max_length = max(len(x) for x in bands_levels if x is not None) # bands_levels and max length of an arr
    return bands_levels, max_length

def glyph_compute(bands, fft_results, timestamps, frame_rate, chunk_size, simulation=False):
    ret = []
    # Will return an array of exactly the same numbers of bands, even if they are not used, the not used ones will have None as element, the others will have an array of the levels
    bands_levels, length = band_compute(bands.copy(), fft_results, timestamps, frame_rate, chunk_size, simulation)

    # Combine levels from each band and put them into the correct led section of the glyph data
    # Iterate over all levels
    for i in range(length):
        glyph_data = []
        # For each level on each band, add them to the correct glyph index
        # Iterate over all bands and push them to the correct element in the glyph data
        for b in range(len(bands)):
            # Band not used
            if bands_levels[b] is None:
                # Fill unused leds with zeroes, number of leds covered by band is third index:
                for j in range(bands[b][2]):
                    glyph_data.append(0)
            # Band used
            else:
                match bands[b][3]:
                    case "bar":
                        glyph_data += bar(bands_levels[b][i], bands[b][2], genarr=True)
                    case "lin":
                        v = bands_levels[b][i] if not simulation else int((bands_levels[b][i]/4095)*255)
                        for j in range(bands[b][2]):
                            glyph_data.append(v)
                    case "tht":
                        v = (4095 if not simulation else 255) if bands_levels[b][i] > bands[b][4] else 0
                        for j in range(bands[b][2]):
                            glyph_data.append(v)
        if simulation:
            ret.append(glyph_data)
        else:
            ret.append(",".join(map(str, glyph_data)) + ",\r\n")
        glyph_data = []
    return ret
