from mutagen.oggopus import OggOpus
import sys
import base64
import zlib

def decode_nothing_glyph(encoded_str):
	try:
		decoded_bytes = base64.b64decode(encoded_str)
		decompressed_bytes = zlib.decompress(decoded_bytes)
		try:
			return decompressed_bytes.decode("utf-8")  # Try UTF-8 first
		except UnicodeDecodeError:
			try:
				return decompressed_bytes.decode("iso-8859-1")  # Fallback to Latin-1
			except UnicodeDecodeError:
				return decompressed_bytes.hex()  # Return hex if decoding fails
	except Exception:
		return "(Decoding error)"

def print_metadata(file_path):
	try:
		audio = OggOpus(file_path)
		metadata_keys = ["TITLE", "ALBUM", "AUTHOR", "COMPOSER", "CUSTOM1", "CUSTOM2"]
		
		for key in metadata_keys:
			value = audio.tags.get(key.lower(), ["Unknown"])[0]
			if key in ["AUTHOR", "CUSTOM1"]:
				value = decode_nothing_glyph(value)
			print(f"{key}: {value}")
	except Exception as e:
		print(f"Error reading file: {e}")

if __name__ == "__main__":
	if len(sys.argv) < 2:
		print("Usage: python script.py <path_to_ogg_file>")
	else:
		print_metadata(sys.argv[1])
