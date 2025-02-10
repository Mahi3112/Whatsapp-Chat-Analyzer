from urlextract import URLExtract
from wordcloud import WordCloud
import pandas as pd
from collections import Counter
import emoji
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
from transformers import pipeline
def fetch_stats(selected_user,df):

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # fetch the number of messages
    num_messages = df.shape[0]

    # fetch the total number of words
    words = []
    for message in df['message']:
        words.extend(message.split())

    # fetch media
    media_count = 0
    for message in df['message']:
        if message == '<Media omitted>\n':
            media_count = media_count+1

    #fetch links
    extractor = URLExtract()
    links = []
    for message in df['message']:
        links.extend(extractor.find_urls(message))

    return num_messages,len(words),media_count,len(links)

def most_active_users(df):
    x = df['user'].value_counts().head()
    df = round((df['user'].value_counts()/df.shape[0])*100,2).reset_index().rename(columns={'index':'name','user':'percent'})
    return x,df

def create_wordcloud(selected_user,df):
    f=open('stop_hinglish.txt','r')
    stop_words=f.read()

    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp=df[df['user']!='group_notification']
    temp=temp[temp['message']!='<Media omitted>\n']
    temp=temp[temp['message']!='This message was deleted\n']
    def remove_stop_words(message):
        y = []
        for word in message.lower().split():
            if word not in stop_words:
                y.append(word)
        return " ".join(y)        
    wc = WordCloud(width=500,height=500,min_font_size=10,background_color='white')
    temp['message'] = temp['message'].apply(remove_stop_words)
    df_wc = wc.generate(temp['message'].str.cat(sep=" "))
    return df_wc

#most common words
def most_commonwords(selected_user,df):
    f=open('stop_hinglish.txt','r')
    stop_words=f.read()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    temp=df[df['user']!='group_notification']
    temp=temp[temp['message']!='<Media omitted>\n']
    temp=temp[temp['message']!='This message was deleted\n']

    words = []
    for message in temp['message']:
        for word in message.lower().split():
            if word not in stop_words:
                words.append(word)
    most_common_df = pd.DataFrame(Counter(words).most_common(20))
    return most_common_df

def emoji_helper(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    emojis = []
    for message in df['message']:
        emojis.extend([c for c in message if emoji.is_emoji(c)])
    emoji_df = pd.DataFrame(Counter(emojis).most_common(len(Counter(emojis))))
    return emoji_df

def monthly_timeline(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    timeline = df.groupby(['year','month_num','month']).count()['message'].reset_index()

    time = []
    for i in range(timeline.shape[0]):
        time.append(timeline['month'][i]+"-"+str(timeline['year'][i]))
    
    timeline['time'] = time

    return timeline

def daily_timeline(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df['only_date'] = df['date'].dt.date
    daily_timeline=df.groupby('only_date').count()['message'].reset_index()

    return daily_timeline

def week_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df['day_name'] = df['date'].dt.day_name()
    return df['day_name'].value_counts()

def month_activity_map(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    df['month_name'] = df['date'].dt.month_name()
    return df['month_name'].value_counts()

def heatmap(selected_user,df):
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    activity_heatmap = df.pivot_table(index='day_name',columns='period',values='message',aggfunc='count').fillna(0)
    
    return activity_heatmap

def sentiment_analysis(selected_user,df):
    sentiments=SentimentIntensityAnalyzer()
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp=df[df['user']!='group_notification']
    temp=temp[temp['message']!='<Media omitted>\n']
    temp=temp[temp['message']!='This message was deleted\n']
    temp['positive'] = [sentiments.polarity_scores(i)["pos"] for i in temp['message']]
    temp['negative'] = [sentiments.polarity_scores(i)["neg"] for i in temp['message']]
    temp['neutral'] = [sentiments.polarity_scores(i)["neu"] for i in temp['message']]

    return temp

def get_emotion(selected_user,df):
    emotion = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]
    temp=df[df['user']!='group_notification']
    temp=temp[temp['message']!='<Media omitted>\n']
    temp=temp[temp['message']!='This message was deleted\n']
    temp['emotion'] = [emotion(message)[0]['label'] for message in temp['message']]

    return temp