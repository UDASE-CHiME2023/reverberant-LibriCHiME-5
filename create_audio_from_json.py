#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script to generate the reverberant LibriCHiME-5 dataset
"""

import json
import numpy as np
import os
import soundfile as sf
from constants import SR_AUDIO, MAX_AMP
from paths import (udase_chime_5_audio_path, librispeech_path, 
                   voicehome_path, reverberant_librichime_5_json_path, 
                   reverberant_librichime_5_audio_path)
from tqdm import tqdm
import scipy as sp
import argparse

def compute_loudness(x):
    return 10*np.log10(np.sum((x - np.mean(x))**2))

def create_reverberant_speech(mix_infos, dtype, voicehome_path, librispeech_path):
    
    mix_len = mix_infos['length']
    speakers = [x for x in list(mix_infos.keys()) if 'speaker_' in x]
    
    speech_sigs = np.zeros((mix_len, len(speakers)), dtype=dtype)
       
    for spk_ind, spk in enumerate(speakers):
        
        # get speaker info
        spk_infos = mix_infos[spk]
        spk_utts = spk_infos['utterances']
        
        # get RIR info
        rir_infos = spk_infos['RIR']
        rir_file = rir_infos['file']
        rir_channel = rir_infos['channel']
        
        # read RIR file
        rir_path = os.path.join(voicehome_path, rir_file)
        rir_sig, sr = sf.read(rir_path)
        rir_sig = rir_sig[:, rir_channel]
        assert sr == SR_AUDIO
        
        # for each speaker's utterance
        for utt in spk_utts:
            
            # read utterance info
            utt_file = utt['file']
            start_librispeech = utt['start_librispeech']
            end_librispeech = utt['end_librispeech']
            start_mix = utt['start_mix']
            end_mix = utt['end_mix']
            
            utt_len = end_mix - start_mix
            rir_len = rir_sig.shape[0]
            
            # read speech file
            speech_path = os.path.join(librispeech_path, utt_file)
            speech_sig, sr = sf.read(speech_path)
            assert sr == SR_AUDIO
            
            # add reverberation
            if start_mix==0 and end_mix==mix_len:
                # utterance spans the entire mix:
                # clip the end of the wet utterance so that it fits in 
                # in the mixture
                speech_sig_cut = speech_sig[start_librispeech:end_librispeech]
                rev_speech_sig = sp.signal.fftconvolve(speech_sig_cut, rir_sig, 
                                                       mode='full')
                rev_speech_sig = rev_speech_sig[:utt_len]
                speech_sigs[start_mix:end_mix, spk_ind] = rev_speech_sig
                
            elif start_mix==0 and end_mix!=mix_len:
                # utterance is at the beginning of the mix:
                # clip the beginning of the wet utterance so that the 
                # reverberant tail is preserved and it fits in 
                # [start_mix, end_mix]
                speech_sig_cut = speech_sig[start_librispeech:end_librispeech]
                rev_speech_sig = sp.signal.fftconvolve(speech_sig_cut, rir_sig, 
                                                       mode='full')
                
                rev_speech_sig = rev_speech_sig[-utt_len:]
                speech_sigs[start_mix:end_mix, spk_ind] = rev_speech_sig
                
            elif start_mix!= 0 and end_mix==mix_len:
                # utterance is at the end of the mix
                # clip the end of the wet utterance so that it fits in 
                # [start_mix, end_mix]
                speech_sig_cut = speech_sig[start_librispeech:end_librispeech]
                rev_speech_sig = sp.signal.fftconvolve(speech_sig_cut, rir_sig, 
                                                       mode='full')
                rev_speech_sig = rev_speech_sig[:utt_len]
                speech_sigs[start_mix:end_mix, spk_ind] = rev_speech_sig
            else:
                # utterance in the middle of the mix
                # we do not clip and allow the unintelligible reverberant 
                # tail to extend beyond initial utterance length
                speech_sig_cut = speech_sig[start_librispeech:end_librispeech]
                rev_speech_sig = sp.signal.fftconvolve(speech_sig_cut, rir_sig, 
                                                       mode='full')
                
                new_end_mix = end_mix + rir_len - 1
                if new_end_mix <= mix_len:
                    speech_sigs[start_mix:new_end_mix, spk_ind] = rev_speech_sig
                else:
                    # clip wet utterance if it extends beyond the mixture length
                    speech_sigs[start_mix:mix_len, spk_ind] = rev_speech_sig[:mix_len-start_mix]

    return speech_sigs

    
if __name__ == "__main__":
    
    # parse arguments
    parser = argparse.ArgumentParser(description="Revereberant LibriCHiME-5 generation", 
                                     add_help=False)
    parser.add_argument("--subset", type=str, default="dev", 
                        help="Subset (`dev`/`eval`)")
    args = parser.parse_args()
    
    print("Creating " + args.subset + " set")
    
    # paths
    dataset_json_path = output_path = os.path.join(reverberant_librichime_5_json_path, 
                                                   args.subset + '.json')
    
    output_path = os.path.join(reverberant_librichime_5_audio_path, args.subset)
    
    # create output dir if necessary
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    
    # load metadata
    with open(dataset_json_path) as f:  
        dataset = json.load(f)
    
    # create audio mixtures
    for mix_infos in tqdm(dataset, total=len(dataset)):
        
        # get mixture info
        mix_name = mix_infos['name']
        mix_len = mix_infos['length']
        mix_max_n_spk = mix_infos['max_num_sim_active_speakers']
        speakers = [x for x in list(mix_infos.keys()) if 'speaker_' in x]
        
        # read noise file
        noise_file = mix_infos['noise']['filename']
        noise_path = os.path.join(udase_chime_5_audio_path, args.subset, 
                                  '0', noise_file+ '.wav')
        noise_sig, sr = sf.read(noise_path)
        if len(noise_sig.shape) == 2:
            noise_sig = noise_sig[:,1]
                
        assert noise_sig.shape[0] == mix_len
        assert sr == SR_AUDIO
        
        # compute noise loudness
        noise_loudness = compute_loudness(noise_sig)
        assert not(np.isinf(noise_loudness))
        
        # create reverberant speech signals for all speakers 
        speech_sigs = create_reverberant_speech(mix_infos, noise_sig.dtype, 
                                                voicehome_path, 
                                                librispeech_path)
        
        # mix reverberant speech signals
        speech_mix_sig = np.zeros(mix_len, dtype=noise_sig.dtype)
        
        for spk_ind, spk in enumerate(speakers):
            
            # get infos
            spk_infos = mix_infos[spk]
            
            # get speech signal
            speech_sig = speech_sigs[:, spk_ind]
            
            # compute speech loudness
            speech_loudness = compute_loudness(speech_sig)
            assert not(np.isinf(speech_loudness))
            
            # compute original SNR
            orig_snr = speech_loudness - noise_loudness
                            
            # get per-speaker snr
            snr_spk = spk_infos['SNR']
            
            # compute speech gain
            # we scale the speech signal and not the noise signal
            # to keep the original loudness of the CHiME data
            speech_gain = 10**( (snr_spk - orig_snr)/20.0)

            # scale speech
            scaled_speech_sig = speech_sig * speech_gain
            
            # check new snr
            speech_loudness_new = compute_loudness(scaled_speech_sig)
            new_snr = speech_loudness_new - noise_loudness
            assert np.isclose(snr_spk, new_snr)

            # add scaled speech to mixture
            speech_mix_sig += scaled_speech_sig
         
        # mix speech and noise
        mix_sig = noise_sig + speech_mix_sig
        
        # handle clipping
        if np.max(np.abs(mix_sig)) > 1.0 or np.max(np.abs(speech_mix_sig)) > 1.0:
            
            scale_clipping = MAX_AMP/max(np.max(np.abs(mix_sig)), 
                                         np.max(np.abs(speech_mix_sig)))
            
            mix_sig = mix_sig*scale_clipping
            speech_mix_sig = speech_mix_sig*scale_clipping
            noise_sig = noise_sig*scale_clipping
        
        # save audio files
        if not os.path.isdir(os.path.join(output_path, str(mix_max_n_spk))):
            os.makedirs(os.path.join(output_path, str(mix_max_n_spk)))
        
        output_mix_file = os.path.join(output_path, str(mix_max_n_spk), 
                                       mix_name + '_mix.wav')
        sf.write(output_mix_file, mix_sig, SR_AUDIO, 'PCM_16')
        
        output_speech_file = os.path.join(output_path, str(mix_max_n_spk), 
                                          mix_name + '_speech.wav')
        sf.write(output_speech_file, speech_mix_sig, SR_AUDIO, 'PCM_16')
        
        output_noise_file = os.path.join(output_path, str(mix_max_n_spk), 
                                         mix_name + '_noise.wav')
        sf.write(output_noise_file, noise_sig, SR_AUDIO, 'PCM_16')