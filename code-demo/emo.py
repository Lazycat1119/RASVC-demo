from funasr import AutoModel

'''
Using the emotion representation model
rec_result only contains {'feats'}
	granularity="utterance": {'feats': [*768]}
	granularity="frame": {feats: [T*768]}
'''
model = AutoModel(model="iic/emotion2vec_base", model_revision="v2.0.4")
wav_file = f"/home/wl/qkc/dataset/xiaode/auto1/Alto-1#newboy/0000.wav"
rec_result = model.generate(wav_file, output_dir="./outputs", granularity="utterance")
print(rec_result)