from food2wine.im2recipe import get_recipe_url, get_recipe_upload
from food2wine.list2wine import get_wine_from_ingredients
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import re
import time

# for i in range(1):
#     t = time.time()
#     demo_urls = 'https://circulairehttps-smisolutionsmark.netdna-ssl.com/wp-content/uploads/lasagne-classique.jpg'
#     recipe = get_recipe(demo_urls)
#     get_recipe_from_ingredients(recipe['ingrs'])

def process_from_url(url):
    recipe = get_recipe_url(url)
    wine = get_wine_from_ingredients(recipe['ingrs'])
    return recipe, wine

def process_from_upload(path):
    recipe = get_recipe_upload(path)
    wine = get_wine_from_ingredients(recipe['ingrs'])
    return recipe, wine

def clean_text(list_food):
    # 2. Retaining only alphabets.
    review_text = re.sub("[^a-zA-Z]"," ",list_food)
    # 3. Converting to lower case and splittingç
    word_tokens= review_text.lower().split()
    # 4. Remove stopwords
    le=WordNetLemmatizer()
    stop_words= set(stopwords.words("english"))
    word_tokens= [le.lemmatize(w) for w in word_tokens if not w in stop_words]
    return list(word_tokens)


if __name__ == "__main__":
    url = 'https://circulairehttps-smisolutionsmark.netdna-ssl.com/wp-content/uploads/lasagne-classique.jpg'
    wine = process_from_url(url)
    print(wine)
