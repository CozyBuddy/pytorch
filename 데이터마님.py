import pandas as pd


channel =pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/youtube/channelInfo.csv')
video =pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/youtube/videoInfo.csv')
print(channel.head())
print(video.head())


channel['ct'] = pd.to_datetime(channel['ct'])
video['ct'] = pd.to_datetime(video['ct'])


print(channel.info())
print(video.info())

print(video['videoname'].value_counts())


print(video.sort_values('ct', ascending=False).drop_duplicates('videoname' , keep='first')[['viewcnt' , 'ct','videoname']])

cond = channel['ct'] > '2021-10-03'

print(channel[cond].sort_values('ct').drop_duplicates('channelname' , keep='first')[['channelname' , 'subcnt']])

cond = channel['ct'] > '2021-10-03 03:00:00' 
cond2 = channel['ct'] < '2021-11-01 15:00:00'

print(channel[cond].sort_values('ct').drop_duplicates('channelname' , keep='first')[['channelname','subcnt' ,'ct']])
df = channel[cond].sort_values('ct').drop_duplicates('channelname' , keep='first')[['channelname','subcnt' ,'ct']]

df2 = channel[cond2].sort_values('ct', ascending=False).drop_duplicates('channelname' , keep='first')[['channelname','subcnt' ,'ct']]

print(df)
print(df2)

df3 = pd.merge(df,df2 ,on='channelname')

print(df3)

df3['증가수'] = df3['subcnt_y'] - df3['subcnt_x']

print(df3[['channelname' , '증가수']])

cond = video['likecnt'] == 0
print(video[~cond])
video2 = video.sort_values('ct',ascending=False).drop_duplicates('videoname' ,keep='first')

video2['ratio'] = video2['dislikecnt'] / video2['likecnt']
print(video2.sort_values('ratio')[['videoname' , 'ratio']])

print(video)

answer  = video[video.index.isin(set(video.index) -  set(video.drop_duplicates().index))]
result = answer[['videoname','ct']]
print(result)


import pandas as pd

df= pd.read_csv('https://raw.githubusercontent.com/Datamanim/datarepo/main/worldcup/worldcupgoals.csv')
print(df.head())
print(df.info())
print(df.groupby('Country').sum(numeric_only=True).sort_values( 'Goals',ascending=False).iloc[:5])
print(df.value_counts('Country').iloc[:5])

#df['Years'] = len(df['Years'].str.replace('-','')) % 4 
df['split'] = df['Years'].str.split('-')
def check(e):
    if(len(e) == 4):
        return True
    elif(len(e) % 5 == 4):
        return True
    else :
        return False
df['check'] = df['Years'].apply(check)

cond = df['check'] == False
print(len(df[cond]))
print(df.head())