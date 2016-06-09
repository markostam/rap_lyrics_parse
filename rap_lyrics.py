# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 01:04:55 2016

Code for cleaning, parsing and analyzing linguistic 
trends in rap music lyrical data and metadata from 
AZlyrics and RapGenius

@author: markostamenovic
"""

import pandas as pd
from scipy import stats
import numpy as np
import seaborn as sns
from calendar import monthrange
import re
from compiler.ast import flatten
from collections import Counter
from datetime import datetime
from langdetect import detect # language detector
from fuzzywuzzy import fuzz #fuzzy language detector
from fuzzywuzzy import process
from math import log
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
import matplotlib as mpl
import matplotlib.pyplot as plt
from operator import add
from itertools import combinations
import networkx as nx



rg = '/path/to/rg_local_final.json'
az = '/path/to/lyrics_az.json'
def rg():
    '''
    data cleaning for rap genius corpus
    '''
    rg = pd.read_json(rg)
    #clear out nonrap and nonelnglish songs
    rg_rap = rg[rg['genre']=='rap'] #get rid of the nonrap
    rg_rap = rg_rap[rg_rap['year'] != 'N/A'] #clean out undated rap
    rg_rap['year']=rg_rap['year'].apply(lambda x: x.replace('}','')) #clean it up
    rg_rap['datetime']=rg_rap['year'].apply(lambda x: datetime.strptime(x, '%Y-%m-%d')) #convert string date to dateime
    rg_rap['year']=rg_rap['datetime'].apply(lambda x: x.year) #put year in it's own column for simplicity
    rg_rap['month']=rg_rap['datetime'].apply(lambda x: x.month) #same with month
    rg_rap = rg_rap[rg_rap['year'] > 1978] #stuff before 1979 isn't rap it's all james brown etc
    rg_rap = rg_rap[rg_rap['year'] < 2017] #clear out mislabeled years
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
    rg_disam = pd.DataFrame(columns=rg_disam.columns)
    rg_disam = rg_disam.append(ogartistVerseDf)
    rg_disam = rg_disam.append(guestVerseDf)
    #normalize all lyrics
    rg_disam['lyrics'] = rg_disam['lyrics'].apply(lambda x: normalizeLyrics(x))
    #get set of all artists
    artists = set(rg_disam['artist']) #get set of artists
    #get bigrams and freq vecs
    rg_disam['bigrams'] = rg_disam['lyrics'].apply(lambda x: findBigrams(x))
    rg_disam['wordFreqs'] = rg_disam['lyrics'].apply(lambda x: wordFreqs(x))
    #get snapshot for months/years
    snap = snapshot(rg_disam)
    #save the df's as csv's
    rg_disam.to_csv('/Users/markostamenovic/GoogleDrive/Education/UR Spring 2016/Linguistics for Data Science - CSC/Final Project/rg_disam.csv',encoding='utf-8')
    snap.to_csv('/Users/markostamenovic/GoogleDrive/Education/UR Spring 2016/Linguistics for Data Science - CSC/Final Project/snap.csv',encoding='utf-8')
    #calculate cross-entropies for each song
    rg_disam = crossEntropyYearly(rg_disam,snap)

def az():
    '''
    data cleaning for az lyrics corpus
    '''
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
    '''
    disambiguates collaborators out of 
    a primary artists lyrics by deleting
    their verse from teh original song by finding a pattern
    and then putting the deleted part into a new row with 
    everything the same only lyrics from the verse that was
    cut from the og song
    '''
    pattern = ": (.*)  " #guest verses are always denoted as "verse 1: ARTIST " so we regex to just get artist
    #define new df for featured verses
    guestVerseDf = pd.DataFrame(columns=originalDF.columns)
    ogartistVerseDf = pd.DataFrame(columns=originalDF.columns)
    featCount=[]
    errors=[]

    for row in xrange(len(originalDF)): #go through the dataframe
        curEntry = originalDF.iloc[row] #
        #print row
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
                            print 'error'
                            errors.append(row)
        #remove featured lyrics from the original song
        lyricsOG = [lyrics[i] for i in xrange(len(lyrics)) if i not in featCount] 
        curEntry['lyrics'] = lyricsOG
        ogartistVerseDf = ogartistVerseDf.append(curEntry.copy(deep=True))
        print 'errors on rows ', errors

    return ogartistVerseDf,guestVerseDf
            
def findBigrams(input_list):
    '''
    list of bigrams from string input
    '''
    input_list = input_list.split(' ')
    return zip(input_list, input_list[1:])

def wordFreqs(input_list):
    '''
    function to create a freqvec inside a df
    based on a lambda func or something
    '''
    from nltk.corpus import stopwords
    stop = stopwords.words('english')
    input_list = [i for i in input_list if i not in stop]
    freqVec = Counter(input_list)
    return freqVec

def buildWordFreqsYear(df):
    '''
    builds a new df containing 
    word freq vectors over each year
    '''
    from nltk.corpus import stopwords
    yearFreqs=[]
    #yearFreqs = pd.DataFrame(columns = ('year','wordfreqs'))
    stop = stopwords.words('english')
    for year in set(df['year']):
        allwords = ''.join([i for i in df[df['year'] == year]['lyrics']]).split()
        allwords = [i for i in allwords if i not in stop]
        freqDist = Counter(allwords)
        #yearFreqs = yearFreqs.append({'year':year,'wordfreqs':freqDist},ignore_index=True)
        yearFreqs.append([year,freqDist])
    return yearFreqs

def buildWordFreqsMonth(df):
    '''
    builds a new df containing 
    word freq vectors over each month
    '''
    from nltk.corpus import stopwords
    #wordFreqs=[]
    wordFreqs = pd.DataFrame(columns = ('year','month','word_freqs'))
    stop = stopwords.words('english')
    for year in set(df['year']):
        for month in set(df['month']):
            allwords = ''.join([i for i in df[(df['month']==month) & (df['year']==year)]['lyrics']]).split()
            allwords = [i for i in allwords if i not in stop]
            freqDist = Counter(allwords)
            wordFreqs = wordFreqs.append({'year':year,'month':month,'word_freqs':freqDist},ignore_index=True)
            #wordFreqs.append([year,month,freqDist])
    return wordFreqs

def normalizeLyrics(lyrics):
    '''
    normalizes lyrics by downcasing,
    removing special cases, punctuation, spaces
    fake words like 'verse' etc etc etc
    some ideas from:
    https://github.com/tbertinmahieux/MSongsDB/tree/master/Tasks_Demos/Lyrics
    '''
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
    'verse nine','verse ten','speaking','sample','produced by ']
    for f in fake_words:
        lyrics = re.sub(f,'',lyrics)
    #remove     duplicate     whitespace
    lyrics = re.sub(' +',' ',lyrics)

    return lyrics

def snapshot(df):
    '''
    returns a df with bigrams corresponding to each month of each year
    and frequency 
    this is equivalent to a monthly snapshot but can also be used as a yearly snapshot
    '''
    years = set(df['year'])
    #flat map
    monthSnaps = pd.DataFrame(columns = ('year','month','bigrams','wordfreqs'))
    for curYear in range(int(min(years)),int(max(years)+1)):
        bigrams=[]
        for month in range(1,13):
            for day in range(1,monthrange(curYear,month)[1]+1):
                bigram = [item for sublist in df[df['datetime']==datetime(curYear,month,day)]['bigrams'] for item in sublist]
                if bigram:
                    bigrams.append(bigram)
            monthSnaps = monthSnaps.append({'year':curYear,'month':month,'wordfreqs':wordFreqs}, ignore_index=True)

    return monthSnaps

def crossEntropyMonthly(df,snapshots):
    '''
    populates each song in the df with a cross-entropy value
    based on comparison to all the language for that year
    '''
    df['xEntropyMonthly']=None
    colnum=df.columns.get_loc("xEntropyMonthly")
    years = set(df['year'])

    for i in xrange(len(snapshots)):
        year = snapshots.iloc[i]['year']
        month = snapshots.iloc[i]['month']
        #get bigrams from year AKA yearly snapshot
        snapshot = snapshots.iloc[i]['bigrams']
        snapshot = [item for sublist in snapshot for item in sublist] #flatmap
        snapshot = Counter(snapshot) #get bigram freqs
        #get artists active in that year
        artists = set(df[(df['year']==year) & (df['month']==month)]['artist'])
        for artist in artists:
                songs = df[(df['artist']==artist) & (df['year']==year) & (df['month']==month)]['title'] #title of all songs by artist that year
                for song in songs:
                    #try:
                        #get bigrams from song
                        bigrams = df[(df['artist']==artist) & (df['year']==year) & (df['month']==month) & (df['title']==song)]['bigrams'] #set of bigrams in that song
                        bigrams = [item for sublist in bigrams for item in sublist] #flatmap
                        bigrams = set(bigrams)
                        HSLMval=HpSLMy(bigrams,snapshot,250)
                        print HSLMval
                        rownum = df[(df['artist']==artist) & (df['year']==year) & (df['month']==month) & (df['title']==song)].index[0]
                        df.ix[rownum,colnum] = HSLMval
                    #except:
                    #    print "error!!"
    return df

def crossEntropyYearly(df,snapshots):
    '''
    populates each song in the df with a cross-entropy value
    based on comparison to all the language for that year
    '''
    df['xEntropyYearly']=None
    colnum=df.columns.get_loc("xEntropyYearly")
    years = set(df['year'])
    
    for year in years:
        #get bigrams from year AKA yearly snapshot
        snapshot = snapshots[snapshots['year']==year]['bigrams']
        snapshot = [item for sublist in snapshot for item in sublist] #flatmap
        snapshot = [item for sublist in snapshot for item in sublist] #flatmap
        snapshot = Counter(snapshot)   
        #get artists active in that year
        artists = set(df[df['year']==year]['artist'])
        for artist in artists:
                songs = df[(df['artist']==artist) & (df['year']==year)]['title'] #title of all songs by artist that year
                for song in songs:
                    try:
                        #get bigrams from song
                        bigrams = df[(df['artist']==artist) & (df['year']==year) & (df['title']==song)]['bigrams'] #set of bigrams in that song
                        bigrams = [item for sublist in bigrams for item in sublist] #flatmap
                        bigrams = set(bigrams)
                        HSLMval=HpSLMy(bigrams,snapshot,250)
                        rownum = df[(df['artist']==artist) & (df['year']==year) & (df['title']==song)].index[0]
                        df.ix[rownum,colnum] = HSLMval
                    except:
                        print "error!!"
    return df

def HpSLMy(bigrams,snapshot,k):
    '''
    calculates the cross-entropy of one song with the snapshot model
    with laplacian (plus one) smoothing
    snapshot model = either month or year of songs
    k = length to truncate analysis to (truncate to normalize for length effects)
    '''
    HSLM=0.0
    count = 0
    for bigram in bigrams:
        logPSLM = log(snapshot[bigram]+1)
        HSLM += logPSLM
        count += 1
        if count==k:
            break
    if len(bigrams)==0:
        HSLM=0
    else:
        HSLM /= count

    return HSLM

def plotArtistLMplot(artist):
    '''
    plots all points in df
    using scatterplot with regression
    '''
    entropy = [i if i > 0 else None for i in rg_disam[rg_disam['artist'] == artist]['xEntropyYearly']]
    dates = [i for i in rg_disam[rg_disam['artist'] == artist]['datetime']]
    year = [i.year for i in dates]
    month = [float(i.month)/12 for i in dates] #month in terms of base 10
    dates = [year[i]+month[i] for i in xrange(len(year))]
    #dates = matplotlib.dates.date2num(dates)
    #dates = dates-min(dates)
    #dates = 100*dates/max(dates)
    dates = [float("{:.1f}".format(float(i))) for i in dates]
    plotdf = pd.DataFrame({'dates':dates,'entropy':entropy})
    fig = sns.lmplot(x='dates', y='entropy', data=plotdf, x_jitter=1)
    fig.set(xlabel='Year', ylabel='Difference from Mothly Snapshot',xlim=[min(year),max(year)],ylim=[min(entropy)-.1,max(entropy)-.1])
    plt.show()

def plotArtist(artist):
    '''
    plots entropy over time for given artist
    using lineplot with error bars
    '''
    entropy = [i if i > 0 else None for i in rg_disam[rg_disam['artist'] == artist]['xEntropyYearly']]
    dates = [i for i in rg_disam[rg_disam['artist'] == artist]['datetime']]
    year = [i.year for i in dates]
    month = [float(i.month)/12 for i in dates] #month in terms of base 10
    dates = [year[i]+month[i] for i in xrange(len(year))]
    dates = [float("{:.1f}".format(float(i))) for i in dates]
    plotdf = pd.DataFrame({'Year':year,'Entropy':entropy})
    fig = sns.factorplot(x='Year', y='Entropy', data=plotdf, x_jitter=1, size=4, aspect=2)
    plt.title('Artist Distance from Community over Time: %s' %artist)
    #fig.set(xlabel='Year', ylabel='Difference from Mothly Snapshot',xlim=[min(year),max(year)],ylim=[min(entropy),max(entropy)])
    plt.show()

def plotAll():
    '''
    plots entropy over time for given artist
    using lineplot with error bars
    '''
    entropy = [i if i > 0 else None for i in rg_disam['xEntropyYearly']]
    dates = [i for i in rg_disam['datetime']]
    year = [i.year for i in dates]
    month = [float(i.month)/12 for i in dates] #month in terms of base 10
    dates = [year[i]+month[i] for i in xrange(len(year))]
    dates = [float("{:.1f}".format(float(i))) for i in dates]
    plotdf = pd.DataFrame({'Year':year,'Entropy':entropy})
    fig = sns.factorplot(x='Year', y='Entropy', data=plotdf, x_jitter=1, size=4, aspect=3)
    #fig.set(xlabel='Year', ylabel='Difference from Mothly Snapshot',xlim=[min(year)-1,max(year)+1])
    plt.title('Distance from Community over Time: Entire Set')
    plt.show()

def plotAllLMplot():
    '''
    plots all points in df
    using scatterplot with regression
    '''
    entropy = [i if i > 0 else None for i in rg_disam['xEntropyYearly']]
    dates = [i for i in rg_disam['datetime']]
    year = [i.year for i in dates]
    month = [float(i.month)/12 for i in dates] #month in terms of base 10
    dates = [year[i]+month[i] for i in xrange(len(year))]
    # dates = matplotlib.dates.date2num(dates)
    # dates = dates-min(dates)
    # dates = 100*dates/max(dates)
    dates = [float("{:.1f}".format(float(i))) for i in dates]
    plotdf = pd.DataFrame({'dates':dates,'entropy':entropy})
    fig = sns.regplot(x='dates', y='entropy', data=plotdf, x_jitter=1, truncate=True, order=3)
    fig.set(xlabel='Year', ylabel='Difference from Mothly Snapshot',xlim=[min(year)-1,max(year)+1])
    #axes[0,1].set_ylim(0,)
    #fig.set_xlim([0,100])
    plt.show()

def plotWordFreqMonth(df,wordlist):
    '''
    wordlist must be a list of lists for multiple words, 
    multiple words of the same meaning
    or even for only one word.
    eg: 
    plotWordFreqLong(rg_disam,[['swag']])
    plotWordFreqLong(rg_disam,[['swag'],['style']])
    plotWordFreqLong(rg_disam,[['bands','bandz','band'],['racks','rack']])
    
    '''
    ###################################
    if not wordFreqs:
        wordFreqs = buildWordFreqsMonth(df) 
    # calculating wordfreq vector for every query is very slow
    # precalculate this if you are doing a bunch of graphing
    ###################################
    freqList=[]
    #calculate freqdist by year for each word
    for words in wordlist:
        count=1
        for word in words:
            if count == 1:
                stem1 = [wordFreqs[yearindex][1][word]/float(sum(wordFreqs[yearindex][1].values())) for yearindex in xrange(len(wordFreqs))]
                count+=1
            else:
                stem2 = [wordFreqs[yearindex][1][word]/float(sum(wordFreqs[yearindex][1].values())) for yearindex in xrange(len(wordFreqs))]
                stem1 = map(add, stem1, stem2)
        freqList.append(stem1)
    for wordindex in xrange(len(wordlist)):
        plt.plot([i for i in (set(rg_rap['year']))],freqList[wordindex], label=wordlist[wordindex][0])
        #plt.plot(range(1995,2016),freqList[wordindex][11:-1], label=wordlist[wordindex][0])
        #plt.plot(label=wordlist[wordindex][0])    

    plt.title('Word Popularity vs. Year')
    plt.xlim([1979,2017])
    plt.xlabel('Year')
    plt.ylabel('Normalized Word Frequency')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()


def plotWordFreqLong(df,wordlist):
    '''
    wordlist must be a list of lists for multiple words, 
    multiple words of the same meaning
    or even for only one word.
    eg: 
    plotWordFreqLong(rg_disam,[['swag']])
    plotWordFreqLong(rg_disam,[['swag'],['style']])
    plotWordFreqLong(rg_disam,[['bands','bandz','band'],['racks','rack']])
    
    '''
    ###################################
    if not yearFreqs:
        yearFreqs = buildWordFreqsYear(df) 
    # calculating wordfreq vector for every query is very slow
    # precalculate this if you are doing a bunch of graphing
    ###################################
    freqList=[]
    #calculate freqdist by year for each word
    for words in wordlist:
        count=1
        for word in words:
            if count == 1:
                stem1 = [yearFreqs[yearindex][1][word]/float(sum(yearFreqs[yearindex][1].values())) for yearindex in xrange(len(yearFreqs))]
                count+=1
            else:
                stem2 = [yearFreqs[yearindex][1][word]/float(sum(yearFreqs[yearindex][1].values())) for yearindex in xrange(len(yearFreqs))]
                stem1 = map(add, stem1, stem2)
        freqList.append(stem1)
    for wordindex in xrange(len(wordlist)):
        plt.plot([i for i in (set(rg_rap['year']))],freqList[wordindex], label=wordlist[wordindex][0])
        #plt.plot(range(1995,2016),freqList[wordindex][11:-1], label=wordlist[wordindex][0])
        #plt.plot(label=wordlist[wordindex][0])    

    plt.title('Word Popularity vs. Year')
    plt.xlim([1979,2017])
    plt.xlabel('Year')
    plt.ylabel('Normalized Word Frequency')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()

    #return freqList

def plotHist(df,minyear,maxyear):
    '''
    plot histogram of dataset dist
    '''
    fig=sns.distplot(df[(df['year']>minyear)&(df['year']<maxyear)]['year'],bins=maxyear-minyear-1,hist=True)
    fig.set(xlim=([minyear,maxyear]))
    plt.title('Longitudinal Distribution of Dataset')
    plt.xlabel('Year')
    plt.ylabel('Percentage Songs')

def buildNetworkGraph(df):
    '''
    builds network graph based on 
    featured artists, producers and writers
    '''
    from itertools import combinations
    G=nx.Graph()
    #iterate through the rows of the df for columns you want
    #using zip is apparently the fastest way to do it
    for r in zip(df['artist'],df['featured_artist'],df['producer'],df['writer']):
        #artist is first value of the tuple
        artist = r[0]
        #flatmap the collaborators, producers, and writers and get rid of any empty cells
        collabors = [i for j in r[1:4] if i for i in j]
        #make sure artist isn't a producer or writer
        collabors = [i for i in collabors if i != artist]

        #update edges
        for c in collabors:
            if G.has_edge(artist,c):
                G[artist][c]['weight'] += 1
            else:
                G.add_edge(artist, c, weight=1)

    return G

def getMostConnected(G,df):
    '''
    returns most connected nodes
    based on degree
    '''
    import operator
    sorted_x = sorted(G.degree().items(), key=operator.itemgetter(1))
    mostConnect = [i for i in reversed(sorted_x)]
    return mostConnect

def getMostBetween(G,df):
    '''
    returns most connected nodes
    based on betweenness
    '''
    import operator
    betweenness = nx.betweenness_centrality(G)
    sorted_x = sorted(betweenness.items(), key=operator.itemgetter(1))
    mostConnect = [i for i in reversed(sorted_x)]
    return mostBetween

