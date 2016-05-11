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
from fuzzywuzzy import 
from math import log

rg = '/Users/markostamenovic/GoogleDrive/Education/UR Spring 2016/Linguistics for Data Science - CSC/Final Project/rg_local_final.json'
az = '/Users/markostamenovic/GoogleDrive/Education/UR Spring 2016/Linguistics for Data Science - CSC/Final Project/lyrics_az.json'
def rg():
    rg = pd.read_json(rg)
    #clear out nonrap and nonelnglish songs
    rg_rap = rg[rg['genre']=='rap'] #get rid of the nonrap
    rg_rap = rg_rap[rg_rap['year'] != 'N/A'] #clean out undated rap
    rg_rap['year']=rg_rap['year'].apply(lambda x: x.replace('}','')) #clean it up
    rg_rap['datetime']=rg_rap['year'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d')) #convert string date to dateime
    rg_rap['year']=rg_rap['datetime'].apply(lambda x: x.year) #put year in it's own column for simplicity
    rg_rap['month']=rg_rap['datetime'].apply(lambda x: x.month) #same with month
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
    #disambiguate featured lyrics from original songs using parseCollabs function
    ogartistVerseDf,guestVerseDf = parseCollabs(rg_rap)
    rg_disam = pd.DataFrame(columns=originalDF.columns)
    #normalize all lyrics
    rg_disam['lyrics'] = rg_disam['lyrics'].apply(lambda x: normalizeLyrics(x))
    #get set of all artists
    artists = set(rg_disam['artist']) #get set of artists
    #get bigrams
    rg_disam['bigrams'] = rg_disam['lyrics'].apply(lambda x: findBigrams(x))

def az():

    az = pd.read_json(az)
    #filter entries with no year
    az = az[az.year != 'N/A']
    #filter songs older than 1975
    az.year = az.year.astype(int)
    az = az[az['year'] > 1800]
    az = az[az['year'] < 2017]

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

#convert a list to a frequency vector dictionary 
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
        
#parse collaborations for the rg corpus to separate guest verses out of original artist verse     
def parseCollabs(originalDF):
    
    pattern = ": (.*)  " #guest verses are always denoted as "verse 1: ARTIST " so we regex to just get artist
    #define new df for featured verses
    guestVerseDf = pd.DataFrame(columns=originalDF.columns)
    ogartistVerseDf = pd.DataFrame(columns=originalDF.columns)
    featCount=[]

    for row in xrange(len(originalDF)): #go through the dataframe
        curEntry = originalDF.iloc[row] #
        print row
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
                            while k != '  ': #guest verse always ends with '  '
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
        #remove featured lyrics from the original song
        lyricsOG = [lyrics[i] for i in xrange(len(lyrics)) if i not in featCount] 
        curEntry['lyrics'] = lyricsOG
        ogartistVerseDf = ogartistVerseDf.append(curEntry.copy(deep=True))

    return ogartistVerseDf,guestVerseDf
            
def findBigrams(input_list):
    input_list = input_list.split(' ')
    return zip(input_list, input_list[1:])

def normalizeLyrics(lyrics):
    #convert from list of lists to string
    lyrics = ' '.join(lyrics)
    # special cases (English...)
    lyrics = lyrics.replace("'m ", " am ")
    lyrics = lyrics.replace("'re ", " are ")
    lyrics = lyrics.replace("'ve ", " have ")
    lyrics = lyrics.replace("'d ", " would ")
    lyrics = lyrics.replace("'ll ", " will ")
    lyrics = lyrics.replace(" he's ", " he is ")
    lyrics = lyrics.replace(" she's ", " she is ")
    lyrics = lyrics.replace(" it's ", " it is ")
    lyrics = lyrics.replace(" ain't ", " is not ")
    lyrics = lyrics.replace("n't ", " not ")
    lyrics = lyrics.replace("'s ", " ")
    # remove boring punctuation and weird signs
    punctuation = (',', "'", '"', ",", ';', ':', '.', '?', '!', '(', ')',
                   '{', '}', '/', '\\', '_', '|', '-', '@', '#', '*')
    for p in punctuation:
        lyrics = lyrics.replace(p, '')
    #remove fake words 
    fake_words = ['verse [0-9]','verse[0-9]','intro','outro', 'hook', 'x[0-9]','[0-9]x','chorus','verse one',
    'verse two','verse three','verse four','verse five','verse six','verse seven','verse eight',
    'verse nine','verse ten','speaking','sample',]
    for f in fake_words:
        lyrics = re.sub(f,'',lyrics)
    #remove     duplicate     whitespace
    lyrics = re.sub(' +',' ',lyrics)

    return lyrics

def yearSnapshot(df):
    years = set(df['year'])
    #flat map
    yearSnaps = pd.DataFrame(columns = ('year','bigrams'))
    for curYear in range(int(min(years)),int(max(years)+1)):
        bigrams = [item for sublist in df[df['year']==curYear]['bigrams'] for item in sublist]
        yearSnaps = yearSnaps.append({'year':curYear,'bigrams':bigrams}, ignore_index=True)

    return yearSnaps

def monthSnapshot(df):
    '''
    returns a df with bigrams corresponding to each month of each year
    this is equivalent to a monthly snapshot but can also be used as a yearly snapshot
    '''
    years = set(df['year'])
    #flat map
    monthSnaps = pd.DataFrame(columns = ('year','month','bigrams'))
    for curYear in range(int(min(years)),int(max(years)+1)):
        bigrams=[]
        for month in range(1,13):
            for day in range(1,monthrange(curYear,month)[1]+1):
                bigram = [item for sublist in df[df['datetime']==datetime(curYear,month,day)]['bigrams'] for item in sublist]
                if bigram:
                    bigrams.append([item for sublist in df[df['datetime']==datetime(curYear,month,day)]['bigrams'] for item in sublist])
        monthSnaps = monthSnaps.append({'year':curYear,'month':month,'bigrams':bigrams}, ignore_index=True)

    return monthSnaps

def crossEntropy(df,snapshots):
    '''
    populates each song in the df with a cross-entropy value
    '''
    years = set(df['year'])
    artists = set(df['artist'])
       
    for artist in artists:
        artistYears = set(df[df['artist']==artist]['year'])

        for year in years:
            if year not in artistYears:
                HSLM = None
            else:
                songs = df[(df['artist']==artist) & (df['year']==year)]['title'] #title of all songs by artist that year
                for song in songs:
                        #get bigrams from song
                        bigrams = df[(df['artist']==artist) & (df['year']==year) & (df['title']==song)]['bigrams'] #set of bigrams in that song
                        bigrams = [item for sublist in bigrams for item in sublist] #flatmap
                        bigrams = set(bigrams)
                        #get bigrams from year
                        snapshot = snapshots[snapshots['year']==year]['bigrams']
                        snapshot = [item for sublist in snapshot for item in sublist] #flatmap
                        snapshot = [item for sublist in snapshot for item in sublist] #flatmap
                        snapshot = Counter(snapshot)
                        df[(df['artist']==artist) & (df['year']==year) & (df['title']==song)]['HSLMy'] = HpSLMy(bigrams,snapshot)
    return df

def HpSLMy(bigrams,snapshot):
    '''
    calculates the corss-entropy of one song with the snapshot model
    snapshot model = either month or year of songs
    with laplacian (plus one) smoothing
    '''
    HSLM=0.0
    for bigram in bigrams:
        logPSLM = log(snapshot[bigram]+1)
        HSLM += logPSLM
    if len(bigrams)==0:
        HSLM=0
    else:
        HSLM /= (-1*len(bigrams))

    return HSLM