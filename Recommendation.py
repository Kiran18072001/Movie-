import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup as soup
from tmdbv3api import TMDb
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from PIL import Image
from tmdbv3api import Movie
from fuzzywuzzy import fuzz
import pickle

clf = pickle.load(open('nlp_model.pkl', 'rb'))
vectorizer = pickle.load(open('tranform.pkl','rb'))



tmdb_movie=Movie()
tmdb = TMDb()
tmdb.api_key = '281825c11d7e66ad7a6a8fb94fa92276'

df = pd.read_csv("main_data.csv")
    


def result():
    placeholder.empty()
    image_address = []
    over_view=[]
    name =[]
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(df['comb'])
    similarity = cosine_similarity(count_matrix) 
    i = df[df["movie_title"]==m].index[0]
    lst = list(enumerate(similarity[i]))
    lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
    lst = lst[1:6] # excluding first item since it is the requested movie itself
    l=[]
    for i in range(len(lst)):
        a = lst[i][0]
        l.append(df['movie_title'][a])



    result = tmdb_movie.search(str(m))
    movie_id = result[0].id
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id,tmdb.api_key))
    data_json = response.json()    
    st.title(str(m).capitalize())
    st.image("https://image.tmdb.org/t/p/original/"+data_json['poster_path'],width=300)
    st.write('**Overview : **.',str(data_json['overview']))


    st.title("Cast")
    response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}&append_to_response=credits'.format(movie_id,tmdb.api_key))
    data_json = response.json()
    images=[]
    names=[]
    for i in range(0,6):     
        r =data_json["credits"]["cast"][i]
        image = "https://image.tmdb.org/t/p/w600_and_h900_bestv2{}".format(r['profile_path'])
        images.append(image)
        names.append(r['name'])
        
    st.image(images,width =200,caption=names)
    
    html3 ="""
    <div style="background-color:black;padding:10px">
    <h2 style="color:white;text-align:center;">User Reviews </h2>
    </div>"""
    st.markdown(html3,unsafe_allow_html=True)
    
    imbd = data_json["imdb_id"]
    url = "https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt".format(imbd)
    page1 = requests.get(url)
    page_soup = soup(page1.content,"html.parser",)
    content = page_soup.find_all("div",{"class":"text show-more__control"})

    reviews_list = [] # list of reviews
    reviews_status = [] # list of comments (good or bad)
    for reviews in content:

        reviews_list.append(reviews.text)
        # passing the review to our model
        movie_review_list = np.array([reviews.text])
        movie_vector = vectorizer.transform(movie_review_list)
        pred = clf.predict(movie_vector)
        reviews_status.append('Good' if pred else 'Bad')

    df_review = pd.DataFrame(columns= ["Review","Sentiments"],data=list(zip(reviews_list, reviews_status)))
    st.dataframe(df_review[0:21].style.highlight_max(axis=0))

    
    
    for i in l:
        try:
        
            result = tmdb_movie.search(str(i))
            movie_id = result[0].id
            response = requests.get('https://api.themoviedb.org/3/movie/{}?api_key={}'.format(movie_id,tmdb.api_key))
            data_json = response.json()
            name.append(str(i))
            image_address.append("https://image.tmdb.org/t/p/original/"+data_json['poster_path'])
            over_view.append(data_json['overview'])      

        except:
            pass


    html2 ="""
    <div style="background-color:black;padding:10px">
    <h2 style="color:white;text-align:center;">Top 5 Recommendation </h2>
    </div>"""
    st.markdown(html2,unsafe_allow_html=True)
    
    for h in range(0,5):

        st.title(name[h].capitalize())
        st.image(image_address[h],width=200)
        st.write('**Overview : **.',str(over_view[h]))
    
    


st.title("Movie Recommendation System")
m = st.text_input("Enter the Movie Name ","")
m = str(m).lower()


if m:    
    names=[]
    q=dict()
    movie=df["movie_title"].to_list()
    for i in movie:
        fuzz1=fuzz.token_set_ratio(m, i)
        q.update({i:fuzz1})

    s = dict(sorted(q.items(), key=lambda x: x[1],reverse=True)[0:5])
    s = list(s.keys())
    placeholder = st.empty()
    placeholder.markdown('''Just to make sure there isn't any Spelling Mistake. Click on the **">"** sign on top left and choose your movie.''')
    b1 = st.sidebar.button(s[0].capitalize(), key="1")
    b2 = st.sidebar.button(s[1].capitalize(), key="2")
    b3 = st.sidebar.button(s[2].capitalize(), key="3")
    b4 = st.sidebar.button(s[3].capitalize(), key="4")
    b5 = st.sidebar.button(s[4].capitalize(), key="5")
    if b1:
        m=str(s[0]).lower()
        result()

    elif b2:
        m=str(s[1]).lower()
        result()

    elif b3:
        m=str(s[2]).lower()
        result()

    elif b4:
        m=str(s[3]).lower()
        result()

    elif b5:
        m=str(s[4]).lower()
        result()





    


