from ffmpegio import ffmpeg
from ..segmentor.audio_segmenting import get_segments
from ..synthesizer.speech import sound

def merge(input_video_filepath, output_video_filename, captions:list=[], wave_output_filepath="result.wav", speaker_gender='female', segments=[]):
    seg = list(filter(lambda x: x[0] != -1, segments))
    synthesized_captions = []
    for i, segment in enumerate(seg):
        synthesized_captions.append(captions[i])
    sound(synthesized_captions, wave_output_filepath, seg, speaker_gender=speaker_gender)    
    input_video_filepath = input_video_filepath.replace("\\", "/")
    output_video_filename = output_video_filename.replace("\\", "/")
    ffmpeg( 
        f'-i {input_video_filepath} -i {wave_output_filepath} -c:v copy -filter_complex "[0:a][1:a] amix=inputs=2 [audio_out]" -map 0:v -map "[audio_out]" -v quiet -y {output_video_filename}'
        )
    
if __name__ == '__main__':
    merge(input_video_filepath='data/videos/bbth_tst.mp4', output_video_filename='result.mp4', captions=['Заставка', 'Две девушки и мужчина поднимаются по лестнице', 'Две девушки и мужчина поднимаются по лестнице', 'Две девушки и мужчина поднимаются по лестнице', 'Две девушки и мужчина стоят возле двери', 'Две девушки и мужчина входят в дверь'])