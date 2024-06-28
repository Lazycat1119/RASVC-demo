import os
import argparse

import numpy
import torch
import os
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler

from funasr import AutoModel
from torchaudio.functional import resample
from scipy.io import wavfile
from scipy.io.wavfile import write
from tqdm import tqdm

import utils
import dac
from dataset.f0_pred import CrepeF0Predictor
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
import logging
logging.getLogger('numba').setLevel(logging.WARNING)
import torch.autograd.profiler as profiler
import time

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile", type=str, default="logs/quickvc/config.json", help="path to json config file")
    parser.add_argument("--ptfile", type=str, default="logs/quickvc/G_5109000.pth", help="path to pth file")
    parser.add_argument("--txtpath", type=str, default="convert.txt", help="path to txt file")
    parser.add_argument("--outdir", type=str, default="output/quickvc", help="path to output dir")
    parser.add_argument("--use_timestamp", default=False, action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    hps = utils.get_hparams_from_file(args.hpfile)

    glist=[]

    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    total = sum([param.nelement() for param in net_g.parameters()])
 
    print("Number of parameter: %.2fM" % (total/1e6))
    print("Loading checkpoint...")
    _ = utils.load_checkpoint(args.ptfile, net_g, None)

    print(f"Loading hubert_soft checkpoint")
    hubert_soft = torch.hub.load("bshall/hubert:main", f"hubert_soft").cuda()
    model = AutoModel(model="iic/emotion2vec_base", model_revision="v2.0.4")
    #model = dac.DAC.load('/home/wl/descript-audio-codec-main/weights_16khz.pth').to('cuda')

    print("Loaded soft hubert.")
    
    print("Processing text...")
    titles, srcs, tgts = [], [], []
    with open(args.txtpath, "r") as f:
        for rawline in f.readlines():
            title, src, tgt = rawline.strip().split("|")
            titles.append(title)
            srcs.append(src)
            tgts.append(tgt)

    print("Synthesizing...")
    start=time.time()
    with torch.no_grad():
        for line in tqdm(zip(titles, srcs, tgts)):
            title, src, tgt = line
            print(title)
            print(src)
            print(tgt)
            # tgt
            wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
            wav_tgt1 = resample(wav_tgt, hps.data.sampling_rate, 16000)
            #wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
            wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
            print(wav_tgt.shape)
            # l = wav_tgt.shape[2] // 2
            # #c = c[:, :, :l * 2]
            # wav_tgt= wav_tgt[:, :, :2 * l]
            mel_tgt = mel_spectrogram_torch(
                wav_tgt, 
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
            # src
            wav_src, _ = librosa.load(src, sr=hps.data.sampling_rate)
            print(wav_src.shape)

            wav_src = resample(wav_src, hps.data.sampling_rate, 16000)
            f0_predictor = CrepeF0Predictor(hop_length=320, sampling_rate=16000,threshold=0.05)
            f0,uv = f0_predictor.compute_f0_uv(
                wav_src
            )
            f0_t, uv_t = f0_predictor.compute_f0_uv(
                wav_tgt1
            )
            # print("9999988888")
            # print(f0.shape)
            # print(uv.shape)
            # print(f0_t.shape)
            # print(uv_t.shape)
            f0_t=torch.tensor(f0_t)
            f0=torch.tensor(f0)
            print("8888888888777777777")
            print(torch.mean(f0_t))
            print(torch.mean(f0))
            mean=torch.mean(f0_t)-torch.mean(f0)
            mean_f0=torch.full_like(torch.tensor(f0),mean)
            f0=f0+mean_f0
            # wav1 = wav_src.unsqueeze(0).numpy()
            out_path = str(src).replace(".wav", ".convert.wav")
            wavfile.write(
                out_path,
                16000,
                (wav_src* np.iinfo(np.int16).max).astype(np.int16)
            )

            out_path1 = str(tgt).replace(".wav", ".convert.wav")
            wavfile.write(
                out_path1,
                16000,
                (wav_tgt1* np.iinfo(np.int16).max).astype(np.int16)
            )

            rec_result = model.generate(out_path, output_dir="./outputs", granularity="utterance")
            emotion = rec_result[0]['feats']
            print("99999999999")
            print(emotion.shape)

            print(f0.shape)
            wav_src = torch.from_numpy(wav_src).unsqueeze(0).cuda()
            print(wav_src.shape)
            mel_src = mel_spectrogram_torch(
                wav_src,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
            print(wav_src.size())
            wav_src=wav_src.unsqueeze(0)
            #long running
            #do something other
            c = hubert_soft.units(wav_src)
            #c, codes, latents, _, _ = model.encode(wav_src)



            c=c.transpose(2,1)
            #print(c.size())
            ll = c.shape[2] // 4
            lll=mel_tgt.shape[2]//4
            llll=mel_src.shape[2]//4
            mel_src=mel_src[:,:,:llll*4]
            mel_tgt = mel_tgt[:, :, :lll * 4]
            c = c[:, :, :ll * 4]
            f0 = torch.tensor(f0[:ll*4]).unsqueeze(0)
            uv= torch.tensor(uv[:ll*4]).unsqueeze(0)
            f0 = torch.FloatTensor(numpy.array(f0, dtype=float)).cuda()
            uv = torch.FloatTensor(numpy.array(uv, dtype=float)).cuda()


            emotion = torch.FloatTensor(emotion).cuda()
            emotion = emotion.unsqueeze(0)
            # if ll>lll:
            #     mel_tgt = mel_tgt[:, :, :lll * 4]
            #     c = c[:, :, :lll * 4]
            # else:
            #     mel_tgt = mel_tgt[:, :, :ll * 2]
            #     c = c[:, :, :ll * 2]

            audio = net_g.infer(c, f0,uv,emotion=emotion,mel1=mel_src,mel=mel_tgt)
            audio = audio[0][0].data.cpu().float().numpy()
            if args.use_timestamp:
                timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
                write(os.path.join(args.outdir, "{}.wav".format(timestamp+"_"+title)), hps.data.sampling_rate, audio)
            else:
                write(os.path.join(args.outdir, f"{title}.wav"), hps.data.sampling_rate, audio)
    end=time.time()
    print("运行时间")
    print(str(end-start))

