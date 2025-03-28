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


################
img_paths = ['get/'+path for path in os.listdir('get')]
reader = easyocr.Reader(['en', 'nl'])


path_results = get_results(reader)
img_results = get_results(reader, bypath=False)
########

def page_to_output(results, nfiles, ncol):
    df = parse_results(results)


test = [df for r in img_results for df in single_file(results_df(r))]



dfs = [results_df(r) for r in img_results]


out = []
for i, item in enumerate(test):
    try:
        out.append(parse_df(item))
    except Exception as e:
        out.append({
            'i': i,
            'error': e
        })


for i, item in enumerate(out):
    item.to_excel(f'output/out_{i}.xlsx', index=False)



t = test[1]
t.shape
t

n, nl, eng, nd, wd = get_cols(t)
nl


[df for i in range(10) for df in bla(i)]


n, nl, eng, ndiff, wdiff = get_cols(dfs[5])
nl = line_clean(nl, wdiff)
test = adjust(n, nl, eng, ndiff)

test
test.to_excel('example6.xlsx', index=False)

dfs[4]
df = dfs[4]
left = df[df.col.isin([0,1,2])].copy()

left
n, nl, eng, nd, wd = get_cols(left)
nl = line_clean(nl, wd)
nl
eng

a = pd.DataFrame({'nl':nl, 'd':wd})
left
a
img_paths[4]
img_details(cv2.imread(img_paths[4]), img_results[4], 'bla.jpg')

len(nl)
len(eng)
pd.DataFrame({'a':nl, 'b':eng})

img = cv2.imread(img_paths[4])

imgg = cv2.cvtColor(img, cv2.IMREAD_GRAYSCALE)
# imggg = cv2.cvtColor(imgg, cv2.COLOR_GRAY2RGB)

im = img
r = reader.readtext(im, detail=1, contrast_ths=0.03, link_threshold=0.6, adjust_contrast=0.5)
img_details(im, r, 'bla2.jpg')

d = results_df(r)

d['col'] = classify_columns(d, 6)

nums = d[d.col==0]

nums

plt.imshow(bin)


#############

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

# imgb = cv2.GaussianBlur(imgg, (5,5), 0)
# thresh = cv2.adaptiveThreshold(imgg, 55, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                cv2.THRESH_BINARY, 11, 2)
# thresh2 = cv2.adaptiveThreshold(imgg, 200, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
#                                cv2.THRESH_BINARY, 11, 2)
# _, bin = cv2.threshold(imgg, 110, 255, cv2.THRESH_BINARY)
# inv = cv2.bitwise_not(imgg)
# kernel = np.ones((3,3), np.uint8)
# cleaned = cv2.morphologyEx(imgg, cv2.MORPH_OPEN, kernel)
#
