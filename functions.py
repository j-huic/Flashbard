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

def results_df(results, ncol=6):
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

    df = pd.DataFrame(df)
    df['col'] = classify_columns(df, ncol)

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

    return files

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

def get_cols(df, factor=100):
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

def make_df(d):
    lengths = [len(d[key]) for key in d.keys()]
    max = np.max(lengths)

    if len(np.unique(lengths)) == 1:
        return pd.DataFrame(dict)
    else:
        for key in d.keys():
            l = list(d[key])
            length = len(l)
            len_diff = max - length

            if length < max:
                l += [None] * len_diff
                d[key] = l

    return pd.DataFrame(d)

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

def parse_df(df):
    df['col'] = df['col'] % 3
    n, nl, eng, ndiff, wdiff = get_cols(df)
    nl = line_clean(nl, wdiff)
    out = adjust(n, nl, eng, ndiff)
    empty_row = pd.DataFrame([[None]*len(out.columns)], columns=out.columns)

    return pd.concat([out, empty_row])

def line_clean(text, wdiff):
    new = [text[0]]
    offset = 0
    m = np.nanmedian(wdiff)

    for i in range(1, len(text)):
        if wdiff[i] < 0.2*m:
            new[-1] = new[-1] + ' ' + text[i]
            offset += 1
        else:
            new.append(text[i])

    return new

def adjust(n, dutch, english, ndiff):
    output = {'n': [], 'nl': [], 'eng': []}
    offset = 0
    med_dif = np.nanmedian(ndiff)

    for idx in range(len(n) - 1):
        output['n'].append(n[idx])
        output['eng'].append(english[idx])

        if ndiff[idx] < med_dif * 1.5:
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

