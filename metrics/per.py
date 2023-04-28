def per(head,tail,df):
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
