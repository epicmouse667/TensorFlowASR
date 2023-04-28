'''
calculating PER(phoneme error rate)
the tail and head parameters enables you to calculate the error rate of 
the middle part of the phoneme sequence with length of [effective_len]
Args:
    df:dataframe that containss greedy_inference and groundtruth of phoneme sequences.
    head:head_redundancy
    tail:tail_redundancy
Return:
    PER
'''
def per(df,head=0,tail=0):
  count=0
  error=0
  for index,_ in df.iterrows():
    if(len(df.at[index,"GROUNDTRUTH"]))<400:
      continue
    groundtruth=df.at[index,"GROUNDTRUTH"].split()[head:-tail-1]
    greedy=df.at[index,"GREEDY"].split()[head:-tail-1]
    # print(len(groundtruth))
    for i in range(len(greedy)):
      if(greedy[i]!=groundtruth[i]):
        error=error+1
      count=count+1
  return error/count
