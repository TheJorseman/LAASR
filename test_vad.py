from LAASR.models.vad import VAD

model = VAD()
speech_timestamps = model.from_file('en_example.wav')
print(speech_timestamps)
