#change grapheme to corresponding phoneme
import pandas as pd
import textgrid
import glob
from tqdm import tqdm
from decimal import Decimal
import math
tg_dir = "/content/tg/my_corpus_aligned_dev_new"
corpus_dir = "/content/corpus/dev"
libri_dir = "/content/LibriSpeech/dev-clean"
# iterate through all the textgrid under my_corpus_aligned dir
filenames=glob.glob(corpus_dir+"/*/*/*.tsv")
dfs=[]
for filename in tqdm(filenames):
  #load dataframe
  df=pd.read_csv(filename,sep="\t")
  for i in range(df.shape[0]):
    text_grid_filename=tg_dir+df.at[i,"PATH"][len(libri_dir):-5]+".TextGrid"
    tg=textgrid.TextGrid.fromFile(text_grid_filename)
    # str is the phoneme sequence correspond to sentence
    s=""
    for j in range(len(tg[1])):
      duration=Decimal(str(tg[1][j].maxTime))-Decimal(str(tg[1][j].minTime))
      frame_number = math.ceil(duration/Decimal("0.01"))
      if tg[1][j].mark!="":
        # 10ms is the default mfcc window stride for conformer model speech config
        for k in range(frame_number):
          s+=(tg[1][j].mark+" ")
      else:
        # $ symbol represents silent space
        for k in range(frame_number):
          s+="$ "
    #change the sentence with the phoneme sequence
    df.at[i,"TRANSCRIPT"]=s
  dfs.append(df)
  # load dataframe to "test.tsv"
Df=pd.concat(dfs,ignore_index=True)
Df.to_csv("/content/phoneme_dev_corpus.tsv",sep="\t",index=None)