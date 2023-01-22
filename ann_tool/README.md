# BattleSound - Annotation Tool
To annotate 5-s segments of audio, you can use this tool.

## Requirements
* OS: Windows
* Download `AnnTool.exe` via https://drive.google.com/file/d/1WWZw2j6UKKIIZnDK3pb3GSQTGp8Xtj7j/view?usp=share_link
* Create `audio` folder in current directory.

## How to use

1. Copy and paste audio files to `audio` folder. The foramt should be `.wav`.
2. Execute `AnnTool.exe` by double-click.
3. If you are ready to annotate, press `[Enter]` key.
4. After 3 seconds, 0.5 seconds of audio is played.
5. After hearing the audio, you can do one actions of the followings:
    * Press `1` key: Label with `VOICE` label.
    * Press `2` key: Label with `GUN (EFFECT)` label.
    * Press `3` key: Label with `MIXED` label.
    * Press `4` key: Label with `UNKOWN (NOISE)` label.
    * Press `R` key: Play the current audio once again.
    * Press `B` key: Cacel the last annoated sample.
    * Press `Q` key: Treminate the program. Before the terminate, the programm will ask you once again.
6. If you annoate the 0.5 seconds of audio with one of the four labels, the next 0.5 seconds of audio will be played.
7. Enjoy labeling!

## How to resume
* The program will detect the pre-exist anntation file, then ask you whether resume or not.
* If you wnat to resume, press `Y` key.
* If you dont't want to resume, press `N` key.



