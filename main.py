import argparse
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
    args = parser.parse_args()
    if (len(args.filename.split(".")) != 2):
        print(args.filename.split("."))
        print("Error file")
        exit(1)
    
    wav_file = "temp.wav"
    output_file = f"{args.filename.split('.')[0]}.ogg"
    
    convert_mp3_to_wav(args.filename, wav_file)
    
    if args.c:
        generate_nothing_ogg(wav_file, output_file, inspect=args.i)
    else:
        audio_visualizer(wav_file)

if __name__ == "__main__":
    main()
