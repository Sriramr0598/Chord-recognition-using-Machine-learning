# -*- coding: utf-8 -*-


import pyaudio
import wave
import time


def record_and_save_audio(chunk_size, channels, sample_rate, record_seconds, audio_save_file_path):
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)

    for t in range(3):
        print("recording starts in " + str(3 - t) + " seconds...")
        time.sleep(1)

    print("Start playing the chord. recording now...")

    frames = []

    for i in range(0, int(sample_rate / chunk_size * record_seconds)):
        data = stream.read(chunk_size)
        frames.append(data)

    print("Recording completed. Recording save path : " + str(audio_save_file_path))

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(audio_save_file_path, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()
