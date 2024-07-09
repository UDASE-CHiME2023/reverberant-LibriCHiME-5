# Reveberant LibriCHiME-5

This repository contains instructions for creating the reverberant LibriCHiME-5 dataset (audio files) for the UDASE task of the CHiME-7 challenge.

The code to generate the metadata files is available in [this repository](https://github.com/UDASE-CHiME2023/reverberant-LibriCHiME-5-metadata).

If you use this code in your research, please cite:
> Leglaive, S., Borne, L., Tzinis, E., Sadeghi, M., Fraticelli, M., Wisdom, S., Pariente, M., Pressnitzer, D., & Hershey, J. R. (2023). The CHiME-7 UDASE task: Unsupervised domain adaptation for conversational speech enhancement. In 7th International Workshop on Speech Processing in Everyday Environments (CHiME).

## Preparation

You must first:

 - Follow the instructions in [this repository](https://github.com/UDASE-CHiME2023/CHiME-5) to extract the audio segments from the original CHiME-5 data. 
    
 - Download the LibriSpeech [dev-clean](https://www.openslr.org/resources/12/dev-clean.tar.gz) and [test-clean](https://www.openslr.org/resources/12/test-clean.tar.gz) data. Put the data in a folder with the following structure:

    ```
    ├── dev-clean
    │   ├── 1272
    │   │   ├── 128104
    │   │   │   ├── 1272-128104-0000.flac
    │   │   │   ├── ...
    ├── test-clean
    │   ├── 1089
    │   │   ├── 134686
    │   │   │   ├── 1089-134686-0000.flac
    │   │   │   ├── ...
    ```

- Download the [VoiceHome](https://zenodo.org/record/1314196) dataset.

## Installation

```
# clone repository
git clone https://github.com/UDASE-CHiME2023/reverberant-LibriCHiME-5.git
cd reverberant-LibriCHiME-5

# activate CHiME environment
conda activate CHiME
```

## Dataset creation

- Set the paths in `paths.py`:
    - `udase_chime_5_audio_path` is the path to the audio segments that you should have previously extracted from the CHiME-5 data (see preparation section above).
    - `librispeech_path` is the path to the LibriSpeech dataset that you should have previously downloaded (see preparation section above).
    - `voicehome_path` is the path to the VoiceHome dataset that you should have previously downloaded (see preparation section above).
    - `reverberant_librichime_5_json_path` is the path to the metadata of the reverberant LibriCHiME-5 dataset, you do not need to change it.
    - `reverberant_librichime_5_audio_path` is the path where you want to store the reverberant LibriCHiME-5 dataset.
- Run 
    
    `python create_audio_from_json.py --subset dev`

    `python create_audio_from_json.py --subset eval`

For the dev and eval sets of the reverberant LibriCHiME-5 dataset, we have three subsets dependending on the maximum number of simultaneously-active speakers (1, 2 or 3). These subsets are stored in separate subfolders whose name indicates the maximum number of simultaneously-active speakers.

At the path defined by the variable `reverberant_librichime_5_audio_path` in `path.py`, you should obtain: 

```
├── dev
│   ├── 1 (3 561 files)
│   │   ├── [...]_mix.wav
│   │   ├── [...]_noise.wav
│   │   ├── [...]_speech.wav
│   ├── 2 (1 695 files)
│   │   ├── [...]_mix.wav
│   │   ├── [...]_noise.wav
│   │   ├── [...]_speech.wav
│   ├── 3 (195 files)
│   │   ├── [...]_mix.wav
│   │   ├── [...]_noise.wav
│   │   ├── [...]_speech.wav
├── eval
│   ├── 1 (4 182 files)
│   │   ├── [...]_mix.wav
│   │   ├── [...]_noise.wav
│   │   ├── [...]_speech.wav
│   ├── 2 (1 482 files)
│   │   ├── [...]_mix.wav
│   │   ├── [...]_noise.wav
│   │   ├── [...]_speech.wav
│   ├── 3 (192 files)
│   │   ├── [...]_mix.wav
│   │   ├── [...]_noise.wav
│   │   ├── [...]_speech.wav
```

For each mixture in the dataset:
- `[...]_mix.wav` is the speech+noise mixture;
- `[...]_speech.wav` is the reference speech signal;
- `[...]_noise.wav` is the reference noise signal.

Summing the reference speech and noise signals gives the mixture signal.

As the original CHiME-5 recordings, these audio signals are not normalized.
