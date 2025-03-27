import os
import cv2 
import easyocr
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from importlib import reload
import matplotlib.pyplot as plt

import functions; reload(functions)
from functions import *

############## EASYOCR

path = 'photos/best.jpg'
img = cv2.imread(path)
reader = easyocr.Reader(['en', 'nl'])

results = reader.readtext(img)
strs, anno = get_stringlist(path, reader)
rows = get_rows(strs)
df = rows_to_df(rows)

points = results_df(results)
points = remove_jezelf(points)
points['col'] = classify_columns(points, 6)


plt.imshow(anno)
cv2.imwrite('annotated.jpg', anno)
det = img_details(img, results)
cv2.imwrite('annotated.jpg', det)

#################

def imgpath_to_words(path, reader, ncol=6):
    results = reader.readtext(path)
    df = results_df(results)
    df = remove_jezelf(df)
    df['col'] = classify_columns(df, ncol)
    df = single_file(df)
    rows = get_rows(df['text'].values)

    return pd.DataFrame(rows)

def parse_results(results, ncol=6):
    df = results_df(results)
    df['col'] = classify_columns(df, ncol)
    # df = single_file(df)

    return df
################
img_paths = ['get/'+path for path in os.listdir('get')]
reader = easyocr.Reader(['en', 'nl'])


path_results = get_results(reader)
img_results = get_results(reader, bypath=False)
########

dfs = [parse_results(r) for r in img_results]

dfs[0]

df = dfs[0]

df.sort_values('y', inplace=True)
df['i'] = df.groupby('col').cumcount()
left = df[df.col.isin([0,1,2])].copy()

lpiv = left.pivot(columns='col', values='text', index='i')
lpiv.columns=['n', 'in', 'out']
lpiv.reset_index(drop=True, inplace=True)

nums = left[left.col==0].copy()
difs = (nums.y.shift(-1) - nums.y).reset_index(drop=True)
nums['dif'] = nums.y.shift(-1) - nums.y 
lpiv['dif'] = difs

left.head()

# def adjust(n, in, out):
nums = left[left.col==0].copy()
n = nums['text'].values.copy()
difs = (nums.y.shift(-1) - nums.y).values.copy()
dutch = left[left.col==1]['text'].values.copy()
english = left[left.col==2]['text'].values.copy()

lpiv

def get_cols(df):
    norm_y = (df['y'] - df['y'].min()) / (df['y'].max() - df['y'].min())
    norm_x = (df['x'] - df['x'].min()) / (df['x'].max() - df['x'].min())
    df['sort_key'] = norm_y * factor + norm_x

    df = df.sort_values('sort_key')
    # number column
    nums = df[df['col']==0].copy()
    num_dif = (nums['y'].shift(-1) - nums['y']).values.copy()
    n = nums['text'].values.copy()
    # left text column
    left = df[df['col']==1].copy()
    word_diff = (left['y'] - left['y'].shift(1)).values.copy()
    dutch = left['text'].values.copy()
    # right text column
    english = df[df['col']==2]['text'].values.copy()

    return n, dutch, english, num_dif, word_diff

df = dfs[1]
df['col'] = classify_columns(df, 6)
left = df[df.col.isin([0,1,2])]

a, b, c, d, e = get_cols(df)

n, nl, eng, nd, wd = get_cols(df)

nl = line_clean(nl, wd)

test = adjust(n, nd, nl, eng)

test


# n = left[left.col==0]['text'].values
# t = l['text'].values
# d = l['dif'].values
# m = np.nanmedian(l['dif'])

df = left.copy()
factor=100

def line_clean(text, difs):
    new = [text[0]]
    offset = 0
    m = np.nanmedian(difs)

    for i in range(1, len(t)):
        if difs[i] < 0.2*m:
            new[-1] = new[-1] + ' ' + text[i]
            offset += 1
        else:
            new.append(text[i])

    return new



test = adjust(a, b, c, d)
d[:10]
test
   
dfs[0]

def adjust(n, dif, dutch, english):
    output = {'n': [], 'nl': [], 'eng': []}
    offset = 0
    med_dif = np.nanmedian(dif)

    for idx in range(len(n) - 1):
        output['n'].append(n[idx])
        output['eng'].append(english[idx])

        if dif[idx] < med_dif * 1.5:
            output['nl'].append(dutch[idx+offset])
        else:
            output['nl'].append(dutch[idx+offset]+' '+dutch[idx+offset+1])
            offset += 1

    idx = len(n) - 1
    output['n'].append(n[idx])
    output['eng'].append(english[idx])
    if len(dutch) == len(output['nl']) + offset + 2:
        output['nl'].append(dutch[idx+offset]+' '+dutch[idx+offset+1])
    else:
        output['nl'].append(dutch[idx+offset])

    return pd.DataFrame(output)
    



lpiv.head()
lpiv.tail()


left.sort_values('y')

nums.dif.median()
nums.head()

nums.iloc[9:]

nums.dif.max()

img_results[0][0]

nums.head()

nums.dif.hist()
t = results_df(img_results[0])
t.head()

lpiv.tail()


# for i, row in lpiv.iterrows():
    





plt.imshow(cv2.imread(img_paths[2]))

img_paths[2]



r = reader.readtext(cv2.imread(img_paths[5]))


img_paths[5]
r

img_paths[5]
dfs = []
issues = []





for path in tqdm(img_paths):
    try:
        df = imgpath_to_words(path, reader)
        dfs.append(df)
    except Exception as e:
        issues.append(e)


# reading from path
results = path_results[1]



df = results_df(results)
df = remove_jezelf(df)
df['col'] = classify_columns(df, 6)

sf = single_file(df)
rows = get_rows(sf.text.values)
rows[::-1]
pd.DataFrame(rows)


bla = imgpath_to_words(img_paths[1], reader)

sf.head(20)



bla

pd.DataFrame(rows)

len(dfs[1])

from PIL import Image

image = Image.open('image_file.jpeg')
    
# next 3 lines strip exif
data = list(image.getdata())
image_without_exif = Image.new(image.mode, image.size)
image_without_exif.putdata(data)
    
image_without_exif.save('image_file_without_exif.jpeg')

# as a good practice, close the file handler after saving the image.
image_without_exif.close()

bla = imgpath_to_words(path, reader)



test = single_file(points)

# test = get_rows(sorted.text.values)


