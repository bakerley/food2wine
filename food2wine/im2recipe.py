import requests
from io import BytesIO
import random
from collections import Counter
import sys; sys.argv=['']; del sys
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import os
from food2wine.model_im2recipe.args import get_parser
import pickle
from food2wine.model_im2recipe.model import get_model
from torchvision import transforms
from food2wine.model_im2recipe.utils.output_utils import prepare_output
from PIL import Image
import time

data_dir = './food2wine/data'

use_gpu = False
device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
map_loc = None if torch.cuda.is_available() and use_gpu else 'cpu'

ingrs_vocab = pickle.load(open(os.path.join(data_dir, 'ingr_vocab.pkl'), 'rb'))
vocab = pickle.load(open(os.path.join(data_dir, 'instr_vocab.pkl'), 'rb'))

ingr_vocab_size = len(ingrs_vocab)
instrs_vocab_size = len(vocab)
output_dim = instrs_vocab_size

greedy = [True, False, False, False]
beam = [-1, -1, -1, -1]
temperature = 1.0
#numgens = len(greedy)

#use_urls = True # set to true to load images from demo_urls instead of those in test_imgs folder
#show_anyways = True #if True, it will show the recipe even if it's not valid


def model_create():
    args = get_parser()
    args.maxseqlen = 15
    args.ingrs_only=False
    model = get_model(args, ingr_vocab_size, instrs_vocab_size)
    # Load the trained model parameters
    model_path = os.path.join(data_dir, 'modelbest.ckpt')
    model.load_state_dict(torch.load(model_path, map_location=map_loc))
    model.to(device)
    model.eval()
    model.ingrs_only = False
    model.recipe_only = False
    return model

def transform():
    transf_list_batch = []
    transf_list_batch.append(transforms.ToTensor())
    transf_list_batch.append(transforms.Normalize((0.485, 0.456, 0.406),
                                                  (0.229, 0.224, 0.225)))
    return transforms.Compose(transf_list_batch)

def get_recipe_from_image(image):
    transf_list = []
    transf_list.append(transforms.Resize(256))
    transf_list.append(transforms.CenterCrop(224))
    transform = transforms.Compose(transf_list)

    image_transf = transform(image)
    image_tensor = to_input_transf(image_transf).unsqueeze(0).to(device)

    #for i in range(numgens):
    with torch.no_grad():
        outputs = model.sample(image_tensor, greedy=greedy[1],
                               temperature=temperature, beam=beam[1], true_ingrs=None)

    ingr_ids = outputs['ingr_ids'].cpu().numpy()
    recipe_ids = outputs['recipe_ids'].cpu().numpy()

    outs, valid = prepare_output(recipe_ids[0], ingr_ids[0], ingrs_vocab, vocab)
    #if valid['is_valid']:
    return outs
    #else:
       # pass

def get_recipe_url(demo_urls):
    response = requests.get(demo_urls)
    try:
        image = Image.open(BytesIO(response.content))
        return get_recipe_from_image(image)
    except:
        raise Exception("This is not an image url. Please provide a valid one.")


def get_recipe_upload(path):
    image = Image.open(path).convert('RGB')
    return get_recipe_from_image(image)


to_input_transf = transform()
model = model_create()

if __name__ == "__main__":
    demo_urls = 'https://circulairehttps-smisolutionsmark.netdna-ssl.com/wp-content/uploads/lasagne-classique.jpg'
    recipe = get_recipe_url(demo_urls)
    print(recipe)
































