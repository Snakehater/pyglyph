import argparse
import os
from pydub import AudioSegment
from generator import generate_nothing_ogg
from visualizer import audio_visualizer

# Convert MP3 to WAV
def convert_mp3_to_wav(mp3_file, wav_file):
    audio = AudioSegment.from_mp3(mp3_file)
    audio.export(wav_file, format="wav")

def main():
    parser = argparse.ArgumentParser(description="Process MP3 and generate Nothing-compatible OGG files or visualize audio.")
    parser.add_argument("-c", action="store_true", help="Create an OGG file for Nothing Glyph Composer")
    parser.add_argument("-i", action="store_true", help="Inspect CUSTOM1 data when used with -c")
    parser.add_argument("filename")
    parser.add_argument("-b", help="Bands save file")
    args = parser.parse_args()
    if (len(args.filename.split(".")) != 2):
        print(args.filename.split("."))
        print("Error file")
        exit(1)
    
    wav_file = "temp.wav"
    output_file = f"{args.filename.split('.')[0]}.ogg"
    bands = []

    if args.b != None and os.path.isfile(args.b):
        with open(args.b, "r") as f:
            for line in f.readlines():
                # Strip newline characters and split by ','
                parts = line.strip().split(',')
                
                # Remove empty strings from trailing commas
                parts = [p for p in parts if p]
                
                # Convert numerical values to integers, keep strings as is
                processed = tuple(int(p) if p.isdigit() else p for p in parts)
                
                # Append to bands list
                bands.append(processed)
    if len(bands) == 0:
        bands = None

    convert_mp3_to_wav(args.filename, wav_file)
    
    if args.c:
        generate_nothing_ogg(wav_file, output_file, inspect=args.i, bands=bands)
    else:
        audio_visualizer(wav_file, bands=bands, bands_file=args.b)

if __name__ == "__main__":
    main()
