import streamlit as st
from food2wine.im2wine import process_from_url
import matplotlib.pyplot as plt
import numpy as np

url = st.text_input("Please provide a url", 'https://circulairehttps-smisolutionsmark.netdna-ssl.com/wp-content/uploads/lasagne-classique.jpg')

if st.button('click me'):
    # print is visible in server output, not in the page
    meal = process_from_url(url)
    recipe = meal[0]
    wine = meal[1]

    st.write('You will eat', recipe['title'])
    st.write('You will eat', recipe['ingrs'])

    st.pyplot(wine)

#list_food = st.text_input("Please provide a url", 'Enter your URL')

# if st.button('click me'):
#     # print is visible in server output, not in the page
#     meal = process_from_url(url)
#     recipe = meal[0]
#     wine = meal[1]

#     st.write('You will eat', recipe['title'])
#     st.write('You will eat', recipe['ingrs'])

#     st.pyplot(wine)
