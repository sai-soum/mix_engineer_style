import os
import pydub
from pydub import AudioSegment
import sys
import glob
from tqdm import tqdm



source_path_mp3 = "/import/c4dm-datasets-ext/mtg-jamendo/"
target_path_wav = "/import/c4dm-datasets-ext/mtg-jamendo_wav/"



engineers_playlist = glob.glob(os.path.join(source_path_mp3, "*"))
print(f"Number of eras: {len(engineers_playlist)}")
for engineer in engineers_playlist:
    print(f"Era: {os.path.basename(engineer)}")
    songs_mp3 = glob.glob(os.path.join(engineer, "*.mp3"))
    print(f"Number of songs: {len(songs_mp3)}")
    if not os.path.exists(os.path.join(target_path_wav,os.path.basename(engineer))):
        os.makedirs(os.path.join(target_path_wav, os.path.basename(engineer) ))
    for song in tqdm(songs_mp3):
        print(f"Song: {os.path.basename(song)}")
        sound = AudioSegment.from_mp3(song)
        destination = os.path.join(target_path_wav,os.path.basename(engineer), os.path.basename(song)[:-3] + "wav")
        print(f"Destination: {destination}")
        if not os.path.exists(destination):
            
            sound.export(destination, format="wav")
        
    


                               