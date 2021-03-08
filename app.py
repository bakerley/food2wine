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
from PIL import Image

st.set_page_config(layout="wide")


# def progress_bar():

#     latest_iteration = st.empty()
#     bar = st.progress(0)

#     for i in range(100):
#         # Update the progress bar with each iteration.
#         latest_iteration.text(f'{i+1}%')
#         bar.progress(i + 1)
#         time.sleep(0.1)
#     latest_iteration.empty()
#     bar.empty()  # Remove the progress bar

def split_list(a_list):
    half = (len(a_list)+1)//2
    return a_list[:half], a_list[half:]

st.markdown("""
    # ** What are we drinking tonight ?**
""")


option = st.sidebar.selectbox('Tell me more about your food:', ['I\'ve got a url for you', 'I\'ve got an image for you', 'I\'ve got a list of ingredients for you'])

if 'url' in option:

    url = st.sidebar.text_input("Please provide a url", 'https://circulairehttps-smisolutionsmark.netdna-ssl.com/wp-content/uploads/lasagne-classique.jpg')

    if st.sidebar.button('Get my wine from Url'):
        # print is visible in server output, not in the page
        meal = process_from_url(url)
        st.balloons()
        recipe = meal[0]
        wine = meal[1]

        st.markdown(f'## You will eat ** {recipe["title"]} **')

        st.markdown(f"<hr><br>", unsafe_allow_html=True)

        A, B = split_list(recipe['ingrs'])

        col1, col2, col3 = st.beta_columns(3)

        with col1:
            st.image(
                f"{url}",
                width=400, # Manually Adjust the width of the image as per requirement
            )
        with col2:
            st.markdown(f"### The ingredients you need:")
            for i in A:
                st.markdown(f'- {i}' )
        with col3:
            st.markdown(f"<br><br>", unsafe_allow_html=True)
            for i in B:
                st.markdown(f'- {i}' )

        st.markdown(f"<hr>", unsafe_allow_html=True)

        st.markdown(f'## We recommand that you drink:')

        st.pyplot(wine)

        st.markdown(f"<hr><br>", unsafe_allow_html=True)

        st.markdown(f'## If you want to cook you can follow:')

        for i in recipe['recipe']:
            st.markdown(f'- {i}')


elif 'ingredients' in option:

    list_food = st.sidebar.text_input("Please provide a list of ingredients", 'Whole wheat, wheat bran, sugar/glucose-fructose, salt, malt (corn flour, malted barley), vitamins (thiamine hydrochloride, pyridoxine hydrochloride, folic acid, d-calcium pantothenate), minerals (iron, zinc oxide).')


    if st.sidebar.button('Get my wine from a list'):
        # print is visible in server output, not in the page
        cleaned_text = clean_text(list_food)
        wine_recommendation = get_wine_from_ingredients(cleaned_text)
        st.markdown(f'## From those ingredients, you could drink:')
        st.markdown(f"<hr><br>", unsafe_allow_html=True)

        st.pyplot(wine_recommendation)

    st.set_option('deprecation.showfileUploaderEncoding', False)

else:

    uploaded_file = st.sidebar.file_uploader("Choose a photo to upload", type=['png', 'jpg'])

    if st.sidebar.button('Get my wine from my picture'):
        # print is visible in server output, not in the page

        meal = process_from_upload(uploaded_file)
        recipe = meal[0]
        wine = meal[1]

        st.markdown(f'## You will eat ** {recipe["title"]} **')

        st.markdown(f"<hr><br>", unsafe_allow_html=True)

        A, B = split_list(recipe['ingrs'])

        col1, col2, col3 = st.beta_columns(3)

        with col1:
            image = Image.open(uploaded_file)
            st.image(image)
        with col2:
            st.markdown(f"### The ingredients you need:")
            for i in A:
                st.markdown(f'- {i}' )
        with col3:
            st.markdown(f"<br><br>", unsafe_allow_html=True)
            for i in B:
                st.markdown(f'- {i}' )

        st.markdown(f"<hr>", unsafe_allow_html=True)

        st.markdown(f'## We recommand that you drink:')

        st.pyplot(wine)

        st.markdown(f"<hr><br>", unsafe_allow_html=True)

        st.markdown(f'## If you want to cook you can follow:')

        for i in recipe['recipe']:
            st.markdown(f'- {i}')

CSS = """
h1 {
    color: rgb(109,7,26);
    font-size: 3em;
    text-align: center;
}

h2 {
    text-align: center;
}

body {
  height: 100vh;
  background-color: white; /* For browsers that do not support gradients */
  background-image: linear-gradient(to bottom, #F15454, #FCD373); /* Standard syntax (must be last) */
}

hr {
    color: black;
    opacity: 1;
}

.css-1aumxhk {
    background-color: #cee0f8;
    background-image: none;
    color: black;
    }

}
"""
st.write(f'<style>{CSS}</style>', unsafe_allow_html=True)
