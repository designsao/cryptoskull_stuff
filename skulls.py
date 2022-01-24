import os
import requests, json

import numpy as np
import pandas as pd

# import PIL
from PIL import Image, ImageChops, ImageStat

from matplotlib.pyplot import imshow

from itertools import product



# CONSTANTS ########################################################################################



special_tokens = [9, 19, 20, 24, 27, 36, 41, 42, 43, 70]



beard_groups = [
        [8], # NONE
        [7, 0, 1, 2], # thin
        [4, 5, 6, 3], # thick
    ]



eyes_groups = [
        [11, 14, 15, 16, 17], # HARDCODE THESE: squint (no color), alien, alien, alien, small
        [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43], # side look
        [0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 12], # small
        [13, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29], # wide
        [44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58], # sunk
        [59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73], # tall
        [74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88], # extra sunk
    ]



d_flipped_nose_map = {'#4':'#2', '#5':'#3', '#7':'#6'}



hair_groups = [
        [122], # NONE
        [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61], # horns
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], # triple (horns + mohawk)
        [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31], # short mohawk
        [32], # very deep mohawk
        [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47], # deep mohawk
        [62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76], # deep widows peak
        [77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92], # shallow widows peak
        [93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107], # emo
        [108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121], # side comb
    ]



# GETTING COLORS FOR PROPERTIES ####################################################################



def get_background_color_map():
    d_background = {}
    for background_id in range(132):
        token_ids = df_meta[df_meta['Background ID'] == '#%d' % background_id]['token_id'].values
        color = cropped_skulls[token_ids[0]].load()[0, 0]
        if color != cropped_skulls[token_ids[1]].load()[0, 0]:
            raise
        d_background.update({'#%d' % background_id:color})
    return d_background



def get_skull_color_map():
    d_skull = {}
    for skull_gene in range(15):
        token_ids = df_meta[df_meta['Skull Gene'] == '#%d' % skull_gene]['token_id'].values
        color = cropped_skulls[token_ids[0]].load()[11, 19]
        if color != cropped_skulls[token_ids[1]].load()[11, 19]:
            raise
        d_skull.update({'#%d' % skull_gene:color})
    return d_skull



def get_bones_color_map():
    d_bones = {}
    for bones_gene in range(15):
        token_ids = df_meta[df_meta['Bones Gene'] == '#%d' % bones_gene]['token_id'].values
        color = cropped_skulls[token_ids[0]].load()[4, 20]
        if color != cropped_skulls[token_ids[1]].load()[4, 20]:
            raise
        d_bones.update({'#%d' % bones_gene:color})
    return d_bones



def get_beard_color_map():
    d_beard = {}
    for i, beard_group in enumerate(beard_groups):
        if i == 0:
            continue
        y = 9 if i == 1 else 10
        for beard_gene in beard_group:
            token_ids = df_meta[df_meta['Beard Gene'] == '#%d' % beard_gene]['token_id'].values
            color = cropped_skulls[token_ids[0]].load()[11, 23]
            if color != cropped_skulls[token_ids[1]].load()[11, 23]:
                raise
            d_beard.update({'#%d' % beard_gene:color})
    return d_beard



def get_eyes_color_map():
    d_eyes = {}
    for i, eyes_group in enumerate(eyes_groups):
        if i == 0:
            continue
        y = 9 if i == 1 else 10
        for eyes_gene in eyes_group:
            token_ids = df_meta[df_meta['Eyes Gene'] == '#%d' % eyes_gene]['token_id'].values
            color = cropped_skulls[token_ids[0]].load()[9, y]
            if color != cropped_skulls[token_ids[1]].load()[9, y]:
                raise
            d_eyes.update({'#%d' % eyes_gene:color})
    return d_eyes



def get_hair_color_map():
    d_hair = {}
    for i, hair_group in enumerate(hair_groups):
        if i == 0:
            continue
        x = 4 if i == 1 else 11
        for hair_gene in hair_group:
            token_ids = df_meta[df_meta['Hair Gene'] == '#%d' % hair_gene]['token_id'].values
            color = cropped_skulls[token_ids[0]].load()[x, 3]
            if color != cropped_skulls[token_ids[1]].load()[x, 3]:
                raise
            d_hair.update({'#%d' % hair_gene:color})
    return d_hair



# MAPPING GENES TO GENE GROUPS #####################################################################



def get_group(gene_str, groups):
    gene_int = int(gene_str[1:])
    for i, group in enumerate(groups):
        if gene_int in group:
            return i
    raise



def get_beard_group(gene_str):
    return get_group(gene_str, beard_groups)

def get_eyes_group(gene_str):
    return get_group(gene_str, eyes_groups)

def get_hair_group(gene_str):
    return get_group(gene_str, hair_groups)



# RE-COLORING HELPER ###############################################################################



def replace_black(im, color):
    #https://stackoverflow.com/a/3753428

    data = np.array(im)   # "data" is a height x width x 4 numpy array
    red, green, blue, alpha = data.T # Temporarily unpack the bands for readability

    # Replace black with color... (leaves alpha values alone...)
    black_areas = (red == 0) & (blue == 0) & (green == 0)
    data[..., :-1][black_areas.T] = color # Transpose back needed

    return Image.fromarray(data)



# INTERACTING WITH ONLINE METADATA #################################################################



def get_raw_token_metadata(token_id):

    url = 'https://cryptoskulls.com/api/token/%d' % token_id
    return json.loads(requests.get(url).text)



def get_metadata(full_refresh=False):

    if full_refresh:
        l_d = []
        for token_id in range(10000):
            response = get_raw_token_metadata(token_id)
            try:
                d = {att.get('trait_type', 'feature'):att['value'] for att in response['attributes']}
                d.update({'token_id': token_id})
                l_d.append(d.copy())
            except:
                print(response)
        df_meta = pd.DataFrame.from_dict(l_d)
        df_meta.to_csv('metadata.csv', index=False)
        return df_meta
    else:
        return pd.read_csv('metadata.csv')



# EXTRACTING 24x24 IMAGES FROM MAIN SKULL IMAGE ####################################################



def get_boxes(img, box_size=24):
    
    lx = range(0, img.size[0] + box_size, box_size)
    ly = range(0, img.size[1] + box_size, box_size)
    
    return [(tx[0], ty[0], tx[1], ty[1]) for ty, tx in product(zip(ly[:-1], ly[1:]), zip(lx[:-1], lx[1:]), repeat=True)]



def get_skull_images():

    d = 24
    im_skulls = Image.open('cryptoskulls.png').convert('RGB')#.crop((0, 0, d*n, d*n))
    skull_boxes = get_boxes(im_skulls, d)
    cropped_skulls = [im_skulls.crop(skull_box) for skull_box in skull_boxes]
    return cropped_skulls



# CHEATY PRE-CALCULATIONS ##########################################################################



df_meta = get_metadata()
cropped_skulls = get_skull_images()

d_background = get_background_color_map()
d_skull = get_skull_color_map()
d_bones = get_bones_color_map()
d_beard = get_beard_color_map()
d_eyes = get_eyes_color_map()
d_hair = get_hair_color_map()



# ASSEMBLE! ########################################################################################



def assemble(token_id, project_name, dim, resize=None):

    d_meta = df_meta[df_meta['token_id'] == token_id].iloc[0].to_dict()
    # background
    color_background = d_background[d_meta['Background ID']]
    im_out = Image.new(mode='RGB', size=(dim, dim), color=color_background)
    # bones
    if d_meta['Bones Gene'] != '#15':
        color_bones = d_bones[d_meta['Bones Gene']]
        im_bones = Image.open('%s\%s\%s' % (project_name, 'bones', 'bones0.png'))
        im_out.paste(replace_black(im_bones, color_bones), im_bones)
    # skull
    color_skull = d_skull[d_meta['Skull Gene']]
    im_skull = Image.open('%s\%s\%s' % (project_name, 'skull', 'skull0.png'))
    im_out.paste(replace_black(im_skull, color_skull), im_skull)
    # beard
    if d_meta['Beard Gene'] != '#8':
        type_beard = get_beard_group(d_meta['Beard Gene'])
        color_beard = d_beard[d_meta['Beard Gene']]
        im_beard = Image.open('%s\%s\%s' % (project_name, 'beard', 'beard%d.png' % type_beard))
        im_out.paste(replace_black(im_beard, color_beard), im_beard)
    # eyes
    type_eyes = get_eyes_group(d_meta['Eyes Gene'])
    if type_eyes == 0:
        im_eyes = Image.open('%s\%s\%s' % (project_name, 'eyes', '%s.png' % d_meta['Eyes Gene']))
        im_out.paste(im_eyes, im_eyes)
    else:
        color_eyes = d_eyes[d_meta['Eyes Gene']]
        im_eyes = Image.open('%s\%s\%s' % (project_name, 'eyes', 'eyes%d.png' % type_eyes))
        im_out.paste(replace_black(im_eyes, color_eyes), im_eyes)
    # nose
    nose_gene = d_meta['Nose Gene']
    nose_gene_use = d_flipped_nose_map.get(nose_gene, nose_gene)
    im_nose = Image.open('%s\%s\%s' % (project_name, 'nose', '%s.png' % nose_gene_use))
    if nose_gene in d_flipped_nose_map.keys():
        im_nose = im_nose.transpose(Image.FLIP_LEFT_RIGHT)
    im_out.paste(im_nose, im_nose)
    # teeth
    im_teeth = Image.open('%s\%s\%s' % (project_name, 'teeth', '%s.png' % d_meta['Teeth Gene']))
    im_out.paste(im_teeth, im_teeth)
    # hair
    if d_meta['Hair Gene'] != '#122':
        type_hair = get_hair_group(d_meta['Hair Gene'])
        color_hair = d_hair[d_meta['Hair Gene']]
        if type_hair == 2:
            im_hair = Image.open('%s\%s\%s' % (project_name, 'hair', 'hair%d.png' % 1))
            im_hair3 = Image.open('%s\%s\%s' % (project_name, 'hair', 'hair%d.png' % 3))
            im_hair.paste(im_hair3, im_hair3)
        else:
            im_hair = Image.open('%s\%s\%s' % (project_name, 'hair', 'hair%d.png' % type_hair))
        im_out.paste(replace_black(im_hair, color_hair), im_hair)

    if resize is not None:
        im_out = im_out.resize((resize, resize), Image.NEAREST)

    return im_out



# EVALUATING ASSEMBLY RESULTS ######################################################################



def calc_diff(im1, im2):
    return sum(ImageStat.Stat(ImageChops.difference(im1, im2)).rms)



def validate():

    for token_id in range(0, 10000):
        if token_id in special_tokens:
            continue
        im_new = assemble(token_id, 'cryptoskulls_backup', 24)
        im_old = cropped_skulls[token_id]
        diff = calc_diff(im_new, im_old)
        if diff > 0:
            print(token_id)
            print(diff)
            imshow(im_new)
            raise
    print('success')



# CLONING PROJECT BASE IMAGES WITH RESIZE OPTION ###################################################



def clone_project(project_from, project_to, dimensions):

    # Example: skulls.clone_project('cryptoskulls_24', 'cryptoskulls_96', 96)

    for trait in os.listdir(project_from):
        dir_from = os.path.join(project_from, trait)
        dir_to = os.path.join(project_to, trait)
        if not os.path.exists(dir_to):
            os.makedirs(dir_to)
        for im_filename in os.listdir(dir_from):
            im = Image.open(os.path.join(dir_from, im_filename))
            im = im.resize((dimensions, dimensions), Image.NEAREST)
            im.save(os.path.join(dir_to, im_filename))



# SKULL MOSAICS ####################################################################################



def create_skull_mosaic(skull_id, *args, **kwargs):

    # handle lists of skulls
    if type(skull_id) == list:
        filenames = [create_skull_mosaic(skull_id_i, *args, **kwargs) for skull_id_i in skull_id]
        return filenames

    im_raw = cropped_skulls[skull_id]
    im_scaled = im_raw.copy().resize((im_raw.size[0]*4, im_raw.size[1]*4), Image.NEAREST)

    im_padded = Image.new('RGB', (100, 100), im_scaled.load()[0, 0])
    im_padded.paste(im_scaled, (2, 2))

    im_og = im_padded.copy()

    output_file_base_name = 'cryptoskull mosaic #%d' % skull_id

    return create_mosaic(im_og, output_file_base_name, *args, **kwargs)



def create_file_mosaic(filename, *args, **kwargs):

    im_og = Image.open(filename).convert('RGB')

    output_file_base_name = '%s mosaic' % filename.split('.')[0]

    return create_mosaic(im_og, output_file_base_name, *args, **kwargs)



def create_mosaic(im_og, output_file_base_name, gif_mode=False, verbose=False):

    d = 24

    # getting background colors which are used to decide what skulls to use
    skull_bg_pixels = np.array([cropped_skull.load()[0, 0] for cropped_skull in cropped_skulls])

    # breaking down into pixels, counts and prioritizing

    pixel_access = im_og.load()
    og_pixels = np.array([pixel_access[x, y] for y in range(im_og.size[1]) for x in range(im_og.size[0])])

    u_og_pixels = np.unique(og_pixels.copy(), axis=0)

    counts = [np.equal(og_pixels, u_og_pixel).all(axis=1).sum() for u_og_pixel in u_og_pixels]

    counts_sort_index = np.argsort(counts)
    u_og_pixels = u_og_pixels[counts_sort_index]
    counts = [counts[i] for i in counts_sort_index]

    if verbose:
        print(u_og_pixels)
        print(counts)

    # assembling
    im_out_blank = Image.new(mode='RGBA', size=(im_og.size[0] * d, im_og.size[1] * d))
    boxes_out = get_boxes(im_out_blank, d)
    
    n = 5 if gif_mode else 1
    l_im_out = [im_out_blank.copy() for i in range(n)]
    
    np.random.seed(500)

    used_rep_box_ids = np.array([])
    for u_og_pixel, count in zip(u_og_pixels, counts):
        dists = np.linalg.norm(u_og_pixel - skull_bg_pixels, axis=1) # distance
        sorted_box_ids = np.argsort(dists)
        sorted_box_ids = sorted_box_ids[~np.isin(sorted_box_ids, used_rep_box_ids)]
        rep_box_ids = sorted_box_ids[:count]
        used_rep_box_ids = np.append(used_rep_box_ids, rep_box_ids.copy())

        for im_out in l_im_out:
        
            np.random.shuffle(rep_box_ids)

            rep_box_id_id = 0
            for og_pixel, box_out in zip(og_pixels, boxes_out):
                if np.equal(og_pixel, u_og_pixel).all():
                    im_out.paste(cropped_skulls[rep_box_ids[rep_box_id_id]], box_out)
                    rep_box_id_id += 1

    # saving            
    if gif_mode:
        filename = '%s.gif' % output_file_base_name
        l_im_out[0].save(filename, save_all=True, append_images=l_im_out[1:], optimize=False, duration=250, loop=0)
    else:
        filename = '%s.png' % output_file_base_name
        l_im_out[0].save(filename)
    
    return filename



