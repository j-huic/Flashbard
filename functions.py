import os
import cv2
import easyocr
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans

def get_rows(strs):
    isnum = [string.isdigit() for string in strs]
    nums = np.where(isnum)[0]

    rows = []
    for i in range(sum(isnum) - 1):
        rows.append(strs[nums[i]:nums[i+1]])

    return rows

def get_stringlist(path, reader=None):
    if reader is None:
        reader = easyocr.Reader(['en', 'nl'])

    img = cv2.imread(path)
    results = reader.readtext(img)
    strs = [item[1] for item in results]

    annotated = img_details(img, results)

    return strs, annotated

def rows_to_df(rows):
    df = pd.DataFrame(rows)
    df.columns = ['n'] + [str(i) for i in range(df.shape[1]-1)]
    df.sort_values('n', ascending=True, inplace=True)
    
    return df

def healthy_row(dfrow):
    if "(" in dfrow.iloc[1] and ")" not in dfrow.iloc[1]:
        return False
    # elif 

def img_details(img, results, path):
    img = img.copy()
    for (bbox, text, confidence) in results:
        bbox = [(int(x), int(y)) for x, y in bbox]
        cv2.polylines(img, [np.array(bbox)], isClosed=True, color=(0,255,0), thickness=2)
        cv2.putText(img, f'{text} ({confidence:.2f}', (bbox[0][0], bbox[0][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,115,0), 2)

    cv2.imwrite('annotated/'+path, img)

def results_df(results):
    df = {
        'text': [],
        'x': [],
        'y': [],
        'prob': []
    }

    for item in results:
        df['text'].append(item[1])
        df['x'].append(int(item[0][3][0]))
        df['y'].append(int(item[0][3][1]))
        df['prob'].append(float(item[2]))

    return pd.DataFrame(df)

def classify_columns(df, ncol=6):
    df = df.copy()
    points = np.array(df[['x', 'y']])
    kmeans = KMeans(n_clusters=ncol, n_init=10)
    df['col_ass'] = kmeans.fit_predict(points[:, [0]])

    ranks = df.groupby('col_ass')['x'].apply(np.mean)
    index = ranks.sort_values().index
    legend = dict(zip(index, list(range(len(index)))))
    newcol = df['col_ass'].apply(lambda x: legend[x])

    return newcol

def remove_jezelf(df):
    if 'TEST JEZELF' not in df.text.values:
        return df
    else:
        y = df[df.text=='TEST JEZELF']['y'].values[0]

    return df[df.y < y].copy()

def fix_col_order(df, col='col'):
    ranks = df.groupby(col)['x'].apply(np.mean)
    index = ranks.sort_values().index
    legend = dict(zip(index, list(range(len(index)))))
    newcol = df[col].apply(lambda x: legend[x])

    return newcol

def single_file(df, idcol='col', nfiles=2):
    file_map = np.split(np.sort(df[idcol].unique()), nfiles)
    files = []

    for file_idx in file_map:
        file = df[df[idcol].isin(file_idx)]
        files.append(file)

    output = pd.concat(files)

    return output

def wipe_metadata(path):
    image = Image.open(path)
    data = list(image.getdata())
    iwe = Image.new(image.mode, image.size)
    iwe.putdata(data)
    iwe.save(path)
    iwe.close()

def get_results(reader, bypath=True):
    paths = ['get/' + path for path in  os.listdir('get')]
    results = []

    for path in tqdm(paths):
        tqdm.write(path)
        try:
            if bypath:
                results.append(reader.readtext(path))
            else:
                img = cv2.imread(path)
                results.append(reader.readtext(img))

        except Exception as e:
            raise Exception(f'error in {path}: ', e)

    return results

