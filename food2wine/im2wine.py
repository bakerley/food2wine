from im2recipe import get_recipe_url, get_recipe_upload
from list2wine import get_recipe_from_ingredients
import time

# for i in range(1):
#     t = time.time()
#     demo_urls = 'https://circulairehttps-smisolutionsmark.netdna-ssl.com/wp-content/uploads/lasagne-classique.jpg'
#     recipe = get_recipe(demo_urls)
#     get_recipe_from_ingredients(recipe['ingrs'])

def process_from_url(url):
    recipe = get_recipe_url(demo_urls)
    wine = get_recipe_from_ingredients(recipe['ingrs'])
    return recipe, wine

def process_from_upload(path):
    recipe = get_recipe_upload(path)
    wine = get_recipe_from_ingredients(recipe['ingrs'])
    return recipe, wine
