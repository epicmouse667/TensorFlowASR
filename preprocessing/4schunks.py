# utilize the train_dataset tsv file ver 1.0 which is /content/drive/MyDrive/conformer/phoneme_train_corpus.tsv
import pandas as pd
import math
import librosa
from tqdm import tqdm
df = pd.read_csv("/content/drive/MyDrive/conformer/phoneme_train_corpus.tsv", sep="\t")
new_paths = []
transcripts = []
new_durations = []
for index,_ in tqdm(df.iterrows()):
  path = "/content/LibriSpeech4schunks_train"+df.at[index,"PATH"][36:] #the url of original url
  duration = df.at[index,"DURATION"]
  trans = df.at[index,"TRANSCRIPT"].split()
  win_size = 400
  stride = 200
  offset = 0
  i = 0
  if(len(trans)<win_size):
    new_trans =trans
    transcript = " ".join(new_trans)
    new_path = f"{path[:-5]}/chunk00{i}.flac"
    new_paths.append(new_path)
    new_durations.append(librosa.get_duration(filename=new_path))
    transcripts.append(transcript)
    continue
  while(offset+win_size<len(trans)):
    new_trans = trans[offset:offset+win_size]
    new_path = f"{path[:-5]}/chunk00{i}.flac"
    i=i+1
    new_paths.append(new_path)
    transcript = " ".join(new_trans)
    transcripts.append(transcript)
    new_durations.append(librosa.get_duration(filename=new_path))
    offset = offset+stride
  new_trans=trans[-win_size:]
  new_path = f"{path[:-5]}/chunk00{i}.flac"
  new_paths.append(new_path)
  transcript = " ".join(new_trans)
  new_durations.append(librosa.get_duration(filename=new_path))
  transcripts.append(transcript)
d = {"PATH":new_paths,"DURATION":new_durations,"TRANSCRIPT":transcripts}
df_new = pd.DataFrame(data=d)
df_new