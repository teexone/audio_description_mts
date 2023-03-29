from pyAudioAnalysis import audioSegmentation as sage
from ffmpegio import ffmpeg
import os, sys

class HiddenPrint():
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

def get_segments(input_file, output_file='output.mp3', min_duration=1, interval=4):
    input_file = input_file.replace("\\", "/")
    output_file = output_file.replace("\\", "/")
    ffmpeg(f'-i {input_file} -v quiet -y {output_file}')
    with HiddenPrint():
        flags_indices, _, _, _ = sage.mid_term_file_classification(output_file, os.path.join("pyAudioAnalysis","pyAudioAnalysis","data","models","svm_rbf_4class"), "svm")
    start = -1
    end = -1
    silences = []
    for i, value in enumerate(flags_indices):
        if value == 0:
           if start >= 0:
                end = i
                silences.append(((start, end), end - start-1))
                start = -1
        else:
            if start < 0:
                start = i
    if start >= 0 and start <= len(flags_indices):
        silences.append(((start, len(flags_indices)), len(flags_indices) - start-1))
    results = []
    i = 0
    start = -1
    max_length = len(flags_indices)//interval
    current_duration = min_duration
    for time, dur in silences:
        while time[0] - i >= interval:
            if start == -1:
                results.append((-1,-1))
            elif len(results) < max_length:
                results.append((start, current_duration))
                start = -1
                current_duration = min_duration
            i += interval
        if dur >= current_duration:
            start = time[0]
            current_duration = dur
    if start != -1 and len(results) < max_length:
        results.append((start, current_duration))
    while len(results) < max_length:
        results.append((-1,-1))
    return results
