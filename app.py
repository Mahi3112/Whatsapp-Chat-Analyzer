import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
st.sidebar.title("Whatsapp Chat Analyzer")

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    
    user_list = df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,"Overall")

    selected_user = st.sidebar.selectbox("Show analysis wrt",user_list)

    if st.sidebar.button("Show Analysis"):

        num_messages, words, media_count, links = helper.fetch_stats(selected_user,df)
        st.title('Top Statistics')
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.header("Messages")
            st.title(num_messages)
        with col2:
            st.header("Words")
            st.title(words)
        with col3:
            st.header("Media")
            st.title(media_count)
        with col4:
            st.header("Links")
            st.title(links)
        
        #monthly timeline
        st.title('Monthly Timeline')
        timeline = helper.monthly_timeline(selected_user,df)
        fig, ax=plt.subplots()
        ax.plot(timeline['time'],timeline['message'])
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        #daily timeline
        st.title('Daily Timeline')
        daily_timeline = helper.daily_timeline(selected_user,df)
        fig, ax=plt.subplots()
        ax.plot(daily_timeline['only_date'],daily_timeline['message'])
        plt.xticks(rotation='vertical')
        st.pyplot(fig)

        #week activity map
        st.title('Weekly Activity')
        col1,col2=st.columns(2)
        with col1:
            st.header('Most busy day')
            busy_day=helper.week_activity_map(selected_user,df)
            fig, ax=plt.subplots()
            ax.bar(busy_day.index,busy_day.values)
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        #month activity map
        with col2:
            st.header('Most busy month')
            busy_month=helper.month_activity_map(selected_user,df)
            fig, ax=plt.subplots()
            ax.bar(busy_month.index,busy_month.values,color='orange')
            plt.xticks(rotation='vertical')
            st.pyplot(fig)

        st.title('Heat Map')
        user_heatmap = helper.heatmap(selected_user,df)
        fig,ax=plt.subplots()
        ax = sns.heatmap(user_heatmap)
        st.pyplot(fig)

        #find most active user
        if selected_user == 'Overall':
            st.title('Most Active Users')
            x,new_df=helper.most_active_users(df)
            top_3_users = x.nlargest(3) 
            fig, ax=plt.subplots()
            col1, col2 = st.columns(2)
            with col1:
                ax.bar(top_3_users.index,top_3_users.values)
                plt.xticks(rotation='vertical')
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df.iloc[:3, :])
        
        #Word Cloud
        st.title('Word Cloud')
        df_wc = helper.create_wordcloud(selected_user,df)
        fig, ax=plt.subplots()
        ax.imshow(df_wc)
        st.pyplot(fig)

        #most common words
        most_common_df=helper.most_commonwords(selected_user,df)
        st.title('Most Common Words')
        fig, ax=plt.subplots()
        ax.barh(most_common_df[0],most_common_df[1])
        st.pyplot(fig)
        
        #emoji analysis
        st.title('Emoji Analysis')
        emoji_df = helper.emoji_helper(selected_user,df)
        col1,col2=st.columns(2)
        with col1:
            st.dataframe(emoji_df)        
        with col2:
            fig,ax = plt.subplots()
            ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct="%0.2f")
            st.pyplot(fig)
        
        #sentiment analysis
        st.title('Sentiment Analysis')
        sen_df=helper.sentiment_analysis(selected_user,df)
        col1,col2=st.columns(2)
        with col1:
            st.dataframe(sen_df[['user', 'message', 'positive', 'negative', 'neutral']])
        with col2:
            sentiment_totals = sen_df[['positive', 'negative', 'neutral']].sum()
            fig, ax = plt.subplots()
            ax.bar(sentiment_totals.index, sentiment_totals.values, color=['green', 'red', 'gray'])
            ax.set_title('Sentiment Distribution')
            ax.set_ylabel('Scores')
            ax.set_xlabel('Sentiment')
            st.pyplot(fig)

        #emotion detection
        st.title('Emotion Detection')
        em_df=helper.get_emotion(selected_user,df)
        col1,col2=st.columns(2)
        with col1:
            st.dataframe(em_df[['user','message','emotion']])
        with col2:
            emotion_counts = em_df['emotion'].value_counts()
            fig, ax = plt.subplots()
            ax.bar(emotion_counts.index, emotion_counts.values, color='skyblue')
            plt.xticks(rotation='vertical')
            ax.set_title('Emotion Distribution')
            ax.set_ylabel('Count')
            ax.set_xlabel('Emotion')
            st.pyplot(fig)
