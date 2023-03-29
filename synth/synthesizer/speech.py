from silero import silero_tts
import torch
import torchaudio

speakers = {
    'female': 'xenia',
    'male': 'eugene'
}

def sound(texts, output_file_path="result.wav", segmen_list=[], speaker_gender='female'):
    model, _ = silero_tts(language='ru', speaker='v3_1_ru')
    model.to(torch.device('cpu'))
    sample_rate = 48000  
    audio=[]
    for text in texts:
        audio.append(model.apply_tts(text=text, speaker=speakers[speaker_gender], sample_rate=sample_rate, put_accent=False, put_yo=True))
    if len(segmen_list) == 0:
        output_tensor = torch.zeros(sample_rate)
        output_tensor = torch.stack((output_tensor, output_tensor))
        torchaudio.save(output_file_path, output_tensor, sample_rate)
        return
    output_tensor = torch.zeros(segmen_list[0][0] * sample_rate)
    for caption, segment in zip(audio, segmen_list):
        if output_tensor.size(dim=0) < segment[0]*sample_rate:
            delta = segment[0]*sample_rate - output_tensor.size(dim=0)
            output_tensor = torch.cat((output_tensor, torch.zeros(delta)))
        output_tensor = torch.cat((output_tensor, caption))
    output_tensor = torch.stack((output_tensor, output_tensor))
    torchaudio.save(output_file_path, output_tensor, sample_rate)

if __name__ == '__main__':
    sound(['Привет', 'Меня зовут Наташа', 'Мне пятнадцать годиков'], [[2, 2],[5, 1],[6, 3]])