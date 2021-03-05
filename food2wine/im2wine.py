from food2wine.im2recipe import get_recipe_url, get_recipe_upload
from food2wine.list2wine import get_recipe_from_ingredients
import time

# for i in range(1):
#     t = time.time()
#     demo_urls = 'https://circulairehttps-smisolutionsmark.netdna-ssl.com/wp-content/uploads/lasagne-classique.jpg'
#     recipe = get_recipe(demo_urls)
#     get_recipe_from_ingredients(recipe['ingrs'])

def process_from_url(url):
    recipe = get_recipe_url(url)
    wine = get_recipe_from_ingredients(recipe['ingrs'])
    return recipe, wine

def process_from_upload(path):
    recipe = get_recipe_upload(path)
    wine = get_recipe_from_ingredients(recipe['ingrs'])
    return recipe, wine


if __name__ == "__main__":
    url = 'https://circulairehttps-smisolutionsmark.netdna-ssl.com/wp-content/uploads/lasagne-classique.jpg'
    wine = process_from_url(url)
    print(wine)
