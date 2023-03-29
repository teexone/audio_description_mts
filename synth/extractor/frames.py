from argparse import ArgumentParser
from ffmpegio import ffmpeg
import os

def get_frames(input_stream, output_stream, step=1):
    """Extracts frames from input stream

    Args:
        input_stream: any input format compatible with ffmpeg
        output_stream: any output format compatible with ffmpeg
        step (float): number of frames per second to extract
    """
    input_stream = input_stream.replace("\\", "/")
    output_stream = output_stream.replace("\\", "/")
    ffmpeg(f'-i {input_stream} -vf fps={step} -v quiet {output_stream}')

if __name__ == "__main__":
    parser = ArgumentParser("Frames extractor")
    parser.add_argument("-i", required=True)
    parser.add_argument("-f", default=1)
    parser.add_argument("-d", default='./imgs')
    parser.add_argument("--prefix", default='')
    parser.add_argument("--format", default="jpg")
    args = parser.parse_args()

    format_string = f"{args.d}/{args.prefix}%d.{args.format}"
    os.makedirs(args.d, exist_ok=True)

    get_frames(args.i, format_string, args.f)
