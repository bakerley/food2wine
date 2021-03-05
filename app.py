import streamlit as st
from food2wine.im2wine import process_from_url, clean_text, process_from_upload
from food2wine.list2wine import get_wine_from_ingredients
import matplotlib.pyplot as plt
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import time
from threading import Thread

def progress_bar():

    latest_iteration = st.empty()
    bar = st.progress(0)

    for i in range(100):
        # Update the progress bar with each iteration.
        latest_iteration.text(f'{i+1}%')
        bar.progress(i + 1)
        time.sleep(0.1)
    latest_iteration.empty()
    bar.empty()  # Remove the progress bar

st.markdown("""
    # ** What are we drinking tonight ?**
""")

option = st.selectbox('Tell me more about your food:', ['I\'ve got a url for you', 'I\'ve got an image for you', 'I\'ve got a list of ingredients for you'])

if 'url' in option:

    url = st.text_input("Please provide a url", 'https://circulairehttps-smisolutionsmark.netdna-ssl.com/wp-content/uploads/lasagne-classique.jpg')

    if st.button('Get my wine from Url'):
        # print is visible in server output, not in the page
        meal = process_from_url(url)
        recipe = meal[0]
        wine = meal[1]

        st.write('You will eat', recipe['title'])
        st.write('You will eat', recipe['ingrs'])

        st.pyplot(wine)

elif 'ingredients' in option:

    list_food = st.text_input("Please provide a list of ingredients", 'Whole wheat, wheat bran, sugar/glucose-fructose, salt, malt (corn flour, malted barley), vitamins (thiamine hydrochloride, pyridoxine hydrochloride, folic acid, d-calcium pantothenate), minerals (iron, zinc oxide).')


    if st.button('Get my wine from a list'):
        # print is visible in server output, not in the page
        cleaned_text = clean_text(list_food)
        wine_recommendation = get_wine_from_ingredients(cleaned_text)
        st.write('Your list was', cleaned_text)

        st.pyplot(wine_recommendation)

    st.set_option('deprecation.showfileUploaderEncoding', False)

else:

    uploaded_file = st.file_uploader("Choose a photo to upload", type=['png', 'jpg'])

    if st.button('Get my wine from my picture'):
        # print is visible in server output, not in the page

        meal = process_from_upload(uploaded_file)
        recipe = meal[0]
        wine = meal[1]

        st.write('You will eat', recipe['title'])
        st.write('You will eat', recipe['ingrs'])

        st.pyplot(wine)

CSS = """
h1 {
    color: rgb(109,7,26);
}
body {
    background-image: url(https://images.unsplash.com/photo-1576005623432-b77544a9e8b2?ixid=MXwxMjA3fDB8MHxzZWFyY2h8Mnx8d2luZXxlbnwwfHwwfGJsYWNrX2FuZF93aGl0ZQ%3D%3D&ixlib=rb-1.2.1&auto=format&fit=crop&w=800&q=60);
}
"""
st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)
