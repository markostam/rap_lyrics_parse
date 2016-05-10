# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 01:04:55 2016

@author: markostamenovic
"""

import pandas as pd
from scipy import stats
import numpy as np
import seaborn as sns
import re
from compiler.ast import flatten
from collections import Counter
from datetime import datetime
from langdetect import detect # language detector
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

rg = '/Users/markostamenovic/GoogleDrive/Education/UR Spring 2016/Linguistics for Data Science - CSC/Final Project/rg_local_final.json'
az = '/Users/markostamenovic/GoogleDrive/Education/UR Spring 2016/Linguistics for Data Science - CSC/Final Project/lyrics_az.json'

rg = pd.read_json(rg)
#clear out nonrap and nonelnglish songs
rg_rap = rg[rg['genre']=='rap'] #get rid of the nonrap
rg_rap = rg_rap[rg_rap['year'] != 'N/A'] #clean out undated rap
rg_rap['year']=rg_rap['year'].apply(lambda x: x.replace('}','')) #clean it up
rg_rap['datetime']=rg_rap['year'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d')) #convert string date to dateime
rg_rap['year']=rg_rap['datetime'].apply(lambda x: x.year) #put year in it's own column for simplicity
rg_rap = rg_rap[rg_rap['year'] > 1979] #stuff before 1979 isn't rap it's all james brown etc
rg_rap['lang'] = rg_rap['lyrics'].apply(lambda x: langDetect(x)) #get english songs
rg_rap = rg_rap[rg_rap['lang']=='en'] #banish nonenglish songs
#downcase
rg_rap['artist'] = rg_rap['artist'].apply(lambda x: x.lower())
rg_rap['title'] = rg_rap['title'].apply(lambda x: x.lower())
rg_rap['album'] = rg_rap['album'].apply(lambda x: x.lower())
rg_rap['artist'] = rg_rap['artist'].apply(lambda x: x.lower())
rg_rap['producer'] = rg_rap['producer'].apply(lambda x: [i.lower() for i in x])
rg_rap['producer'] = rg_rap['producer'].apply(lambda x: [i.lower() for i in x])
rg_rap['featured_artist'] = rg_rap['featured_artist'].apply(lambda x: [i.lower() for i in x])
rg_rap['writer'] = rg_rap['writer'].apply(lambda x: [i.lower() for i in x])
rg_rap['lyrics'] = rg_rap['lyrics'].apply(lambda x: [i.lower() for i in x])
#remove ' lyrics' from song title
rg_rap['title'] = rg_rap['title'].apply(lambda x: x.replace(' lyrics',''))
#parse all collaborators from lyric data
[re.findall(pattern,i) for i in [j[0] for j in a] if re.findall(pattern,i)]
#set of all collaborators from lyric data
a=[i for i in rg_rap['lyrics']]
len(set([k[0] for k in [re.findall(pattern,i) for i in [j[0] for j in a] if re.findall(pattern,i)]]))
#TODO
#also have to remove '2X' and spaces at EOL
#convert &amp; to &
#get all collaborators who appear more than once. these are good to go for rearranging
b=Counter([k[0] for k in [re.findall(pattern,i) for i in [j[0] for j in a] if re.findall(pattern,i)]])
len([i for i in b.items() if i[1]>1])
ogartistVerseDf,guestVerseDf = parseCollabs(rg_rap)




az = pd.read_json(az)
#filter entries with no year
az = az[az.year != 'N/A']
#filter songs older than 1975
az.year = az.year.astype(int)
az = az[az['year'] > 1800]
az = az[az['year'] < 2017]



#rg_rap = [i for i in rg if i['genre'] == 'rap']

#filter nonrap
az_rap = az[az.genre == 'urban']
#filter entries with no year
az_rap = az_rap[az_rap.year != 'N/A']
#filter songs older than 1975
az_rap.year = az_rap.year.astype(int)
az_rap = az_rap[az_rap['year'] > 1975]
az_rap = az_rap[az_rap['year'] < 2020]
az_rap = az_rap[az_rap.lyrics != 'N/A']

'''how to get certain row/column value
az_rap['lyrics'].irow(1)
another example of a query
az_rap[az_rap['artist']=='kanye west']['lyrics'].irow(45)'''

#lowercase artist names, lyrics, song titles and albums
az_rap['artist'] = az_rap['artist'].apply(lambda x: x.lower())
az_rap['title'] = az_rap['title'].apply(lambda x: x.lower())
az_rap['album'] = az_rap['album'].apply(lambda x: x.lower())
az_rap['artist'] = az_rap['artist'].apply(lambda x: x.lower())
az_rap['lyrics'] = az_rap['lyrics'].apply(lambda x: [i.lower() for i in x])

pattern = r"_\[.*\]_"
#find all possible song breakpoints
breaks = az_rap['lyrics'].apply(lambda x: [re.findall(pattern,i) for i in x if re.findall(pattern,i)])
breaks2 = flatten([j for j in [i for i in breaks] if j])
breaksd = Counter(breaks2)

breaksd.most_common(20)

def disambiguateCollabs():
    #check for pattern
    pattern = r"_\[.*\]_"
        #check if pattern is a known nonintersting pattern
            #if not get artist name from pattern
            #get lyrics from between patterns
            #create new row with same information as current row
            #overwrite new row artist and lyrics from this pattern
            #delete lyrics and pattern title from lyrics 
            #(maybe just save indices to delete after iterating through list)
    
    
#create a frequency dict from a list    
def list2dict(list):
    d = {}
    for i in list:
        if i not in d:
            d[i] = 1
        else:
            d[i] += 1
    return d
    
#detect language based on lyrics
def langDetect(x):
    from langdetect import detect # language detector
    try:
        return detect(''.join(x))
    except:
        return 'error'
        
#downcase lyrics
def downcase(x):
    return [i.lower() for i in x]
        

#parse collaborations for the rg corpus        
def parseCollabs(originalDF):
    
    pattern = ": (.*)  " #guest verses are always denoted as "verse 1: ARTIST " so we regex to just get artist
    #define new df for featured verses
    guestVerseDf = pd.DataFrame(columns=originalDF.columns)
    ogartistVerseDf = pd.DataFrame(columns=originalDF.columns)
    featCount=[]

    for row in xrange((20)): #go through the dataframe
        curEntry = originalDF.iloc[row] #
       
        lyrics = curEntry['lyrics']
        lyricsOG = lyrics
        featured = curEntry['featured_artist']
        for i in xrange(len(lyrics)): #search each line of lyrics for pattern
            foundpat = re.findall(pattern,lyrics[i])
            if foundpat: #if pattern isn't empty
                featCount.append(i)
                for j in featured: #search featured artists for pattern
                    if fuzz.partial_ratio(j,foundpat) > 85: #if found partist is featured
                        featVerse=[]
                        featCount=[]
                        featEntry = curEntry.copy(deep=True)
                        try:
                            count=i+1; #get the first line of the actual guest verse (first one is just "verse 1: ARTIST")
                            k=lyrics[count]                    
                            while k != '  ':
                                featCount.append(count)
                                #del ogartistVerseDf.iloc[row]['lyrics'][count] #delete the featured line from the original song
                                featVerse.append(k)
                                count+=1
                                k=lyrics[count]
                            featArtist = j
                            featEntry['lyrics']=featVerse
                            featEntry['artist']=featArtist
                            guestVerseDf = guestVerseDf.append(featEntry)
                        except:
                            pass
        lyricsOG = [lyrics[i] for i in xrange(len(lyrics)) if i not in featCount]
        curEntry['lyrics'] = lyricsOG
        ogartistVerseDf = ogartistVerseDf.append(curEntry.copy(deep=True))

    return ogartistVerseDf,guestVerseDf
            
            