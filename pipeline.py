import os
import secrets
from connect_person import connect_person_no_attention
from synth.extractor import frames
from synth.merge.merger import merge
from synth.segmentor.audio_segmenting import get_segments
from synth.translator.translate import translate
from image2text.driver.driver import Engine 


def process_video(
        token: str, 
        video_file_path: str, 
        processing_dir: str,
        engine: Engine,
        embeddings,
        fps=1,
    ):
    """
    Выстраивает весь пайплайн, который занимается обработкой полученных сегментов
    Args:
        token: токен пользователя
        video_file_path: путь к полученному сегменту
        processing_dir: папка - временное хранилище файлов на этам 
        engine: класс отвечающий за подпроцессы
        embeddings: числовое представление персонажей
        fps: количество кадров которые берутся за одну секунду сегмента
    Returns:
        fpath: Путь к видео с наложенной озвучкой
    """
    os.makedirs(processing_dir, exist_ok=True)

    seed = secrets.token_hex(16)
    output_stream = os.path.join(processing_dir, f"{token}{seed}__%d.png")
    frames.get_frames(video_file_path, output_stream, fps)
    
    produced_images = [
        os.path.join(processing_dir, filename) for filename in os.listdir(processing_dir)
    ]

    segments = get_segments(video_file_path, os.path.join(processing_dir, f"{token}{seed}.mp3"), min_duration=2, interval=1/fps)
    inds = [x[0] for x in segments if x[0] != -1]
    image_paths = [x for x in produced_images if f"{token}{seed}" in x]
    image_paths = [image_paths[i] for i in inds]

    captions = engine.process_images(image_paths)
    captions = connect_person_no_attention(image_paths, captions, embeddings)
    captions = translate(captions)

    fpath = os.path.join(processing_dir, f"{token}__output.mp4")
    merge(video_file_path, fpath, captions, segments=segments)
    return fpath
