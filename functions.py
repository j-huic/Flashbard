import os
import cv2
import json
import math
import pickle
import easyocr
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from pathlib import Path
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

    annotated = img_details(img, results, Path(path).stem)

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
    # img = cv2.imread(imgpath)
    for (bbox, text, confidence) in results:
        bbox = [(int(x), int(y)) for x, y in bbox]
        cv2.polylines(img, [np.array(bbox)], isClosed=True, color=(0,255,0), thickness=2)
        cv2.putText(img, f'{text} ({confidence:.2f}', (bbox[0][0], bbox[0][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,115,0), 2)

    os.makedirs('annotated', exist_ok=True)
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
        df['x'].append((int(item[0][3][0]) + int(item[0][0][0])) / 2)
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
        file = df[df[idcol].isin(file_idx)].copy()
        file[idcol] = file[idcol] % 3
        # file.loc[idcol] = file[idcol] % len(file_idx)
        # file[idcol] = file[idcol].astype(int)
        files.append(file)

    return files

def wipe_metadata(path):
    image = Image.open(path)
    data = list(image.getdata())
    iwe = Image.new(image.mode, image.size)
    iwe.putdata(data)
    iwe.save(path)
    iwe.close()

def get_results(reader, bypath=False):
    paths = ['get/' + path for path in  os.listdir('get')]
    results = []

    for path in tqdm(paths):
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
    df = df.copy()
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
    ldiff = (left['y'] - left['y'].shift(1)).values.copy()
    dutch = left['text'].values.copy()

    # right text column
    right = df[df['col']==2].copy()
    rdiff = (right['y'] - right['y'].shift(1)).values.copy()
    english = right['text'].values.copy()

    return n, dutch, english, num_dif, ldiff, rdiff

def make_df(d):
    lengths = [len(d[key]) for key in d.keys()]
    max = np.max(lengths)

    if len(np.unique(lengths)) == 1:
        return pd.DataFrame(d)
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

    return df

def parse_df(df):
    df.loc[:,'col'] = df.loc[:,'col'] % 3
    n, nl, eng, ndiff, ldiff, rdiff = get_cols(df)
    nl = line_clean(nl, ldiff)
    eng = line_clean(eng, rdiff)
    out = adjust(n, nl, eng, ndiff)
    empty_row = pd.DataFrame([[None]*len(out.columns)], columns=out.columns)

    return pd.concat([out, empty_row])

def line_clean(text, wdiff, factor=0.3):
    new = [text[0]]
    offset = 0
    m = np.nanmedian(wdiff)

    for i in range(1, len(text)):
        # if wdiff[i] > 1.5*m:
        #     # new.append('')
        #     new.append(text[i])
        if wdiff[i] < factor*m:
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

def unique_filename(proposed):
    if not os.path.isfile(proposed):
        return proposed
    else:
        base, ext = os.path.splitext(proposed)
        suffix = 2

        while True:
            newname = base + '_' + str(suffix) + ext
            if not os.path.isfile(newname):
                return newname
            else:
                suffix += 1
                continue

def get_remaining_paths(input_dir):
    if input_dir[-1] == '/':
        input_dir = input_dir[:-1]

    input_paths = [item for item in os.listdir(input_dir)]

    existing = [Path(item).stem for item in os.listdir('results')]
    remaining = []

    for path in input_paths:
        if Path(path).stem not in existing:
            remaining.append(input_dir + '/' + path)
        else:
            continue

    return remaining

def update_log(changed):
    with open('.log.json', 'r') as file:
        log = json.load(file)

    newhist = list(set(log['hist'].copy() + changed))
    log['hist'] = newhist

    with open('.log.json', 'w') as file:
        json.dump(log, file)

def sep_cols(df):
    dfs = []
    for col in df['col'].unique():
        dfs.append(df[df['col']==col].copy())

    return dfs

def get_dims(df):
    ncol = []
    for col in df['col'].unique():
        n = find_n_rows(df[df['col']==col]['y'].values)
        ncol.append(n)

    return ncol

def find_n_rows(y):
    df = pd.DataFrame({'y':y})
    df.sort_values('y', inplace=True)
    df['d'] = df['y'] - df['y'].shift(1)
    df['mult'] = np.round(df['d'] / df['d'].median())

    return df['mult'].sum() + 1

def points(df):
    output = []

    for col in df['col'].unique():
        points = df[df['col'] == col]
        pointlist = []
        for _, row in points.iterrows():
            point = {
                'text': row['text'],
                'x': int(row['x']),
                'y': int(row['y'])
            }
            pointlist.append(point)

        output.append(pointlist)

    return output

def same_row(points, rowheight, factor=0.3):
    min = np.min(points)
    max = np.max(points)

    if (max - min) > rowheight*factor:
        return False
    else:
        return True

def load_results(dir='results', subcat=None, filetype='pkl'):
    parent = os.path.dirname(os.path.abspath(__file__))
    full_dir = parent + '/' + dir
    item_paths = [full_dir + '/' + path for path in os.listdir(full_dir)]
    result_paths = [path for path in item_paths if path.endswith(filetype)]

    if subcat is not None:
        stems_to_get = [Path(path).stem for path in os.listdir(parent + '/' + subcat)]
        result_paths = [str(Path(parent) / dir / f"{stem}.{filetype}") 
            for stem in stems_to_get]
        

    result_paths.sort()
    results = []

    for path in result_paths:
        if filetype == 'csv':
            with open(path, 'r') as file:
                results.append(pd.read_csv(file))
        else:
            with open(path, 'rb') as file:
                results.append(pickle.load(file))

    return results, result_paths

def get_row_multiplier(distance, median_distance, threshold=0.3):
    if np.isnan(distance):
        return np.nan

    ratio = distance / median_distance
    nearest_int = round(ratio)

    if abs(ratio - nearest_int) <= threshold:
        return nearest_int
    
    return math.ceil(ratio)

def index_by_row(df, m_thresh=0.3, diag=False):
    df = df.sort_values('y')
    df.reset_index(inplace=True)
    df['d'] = df['y'] - df['y'].shift(1)
    med = df['d'].median()
    df['mult'] = df.apply(lambda x: get_row_multiplier(x['d'], med, m_thresh), axis=1)
    df.loc[0, 'mult'] = 0
    df['i'] = df['mult'].cumsum().astype(int)

    if diag:
        return df

    dfx = df[['text', 'i', 'x']].copy()
    dfx.set_index('i', inplace=True)
    dfx.sort_values(['i', 'x'], inplace=True)
    dfx = dfx.groupby('i').agg({'text': ' '.join})

    return dfx

def assign_grid_positions(df, m_thresh=0.3):
    dfxs = []
    for col in df['col'].unique():
        dfx = index_by_row(df[df['col']==col].copy(), m_thresh=m_thresh)
        dfx.columns = [col]
        dfxs.append(dfx)

    all = pd.concat(dfxs, axis=1)
    all = all.sort_values('i')

    return all.reset_index(drop=True)

def concatenate_multirow_cells(df):
    df = df[sorted(df.columns)].copy()
    df.replace('', np.nan, inplace=True)
    
    for i in range(1, len(df)):
        nas = df.iloc[i].isna()
        if nas.any():
            if nas.loc[0]:
                for col in df.columns:
                    if not pd.isna(df.iloc[i][col]):
                        df.loc[i-1, col] = df.loc[i-1, col] + ' ' + df.loc[i, col]
                
                df.iloc[i, :] = np.nan
            else:
                df.iloc[i, nas] = ''
    
    return df.dropna(how='all').reset_index(drop=True)

def concatenate_multirow_cells2(df):
    df = df[sorted(df.columns)].copy()
    df.replace('', np.nan, inplace=True)
    
    # Process rows bottom to top to properly handle consecutive NaN rows
    for i in range(len(df) - 1, 0, -1):
        nas = df.iloc[i].isna()
        
        # If the entire row is NaN, skip it
        if nas.all():
            continue
            
        # If we find a row with some NaN values
        if nas.any():
            # If the first column is NaN, this is a continuation row
            if nas.loc[0]:
                # Find the nearest non-NaN row above
                target_row = i - 1
                while target_row >= 0 and df.iloc[target_row].isna().all():
                    target_row -= 1
                
                # If we found a valid target row
                if target_row >= 0:
                    # Merge non-NaN values from current row to target row
                    for col in df.columns:
                        if not pd.isna(df.iloc[i][col]):
                            current_val = df.loc[target_row, col]
                            new_val = df.iloc[i][col]
                            # Concatenate with space only if current value isn't NaN
                            if pd.isna(current_val):
                                df.loc[target_row, col] = new_val
                            else:
                                df.loc[target_row, col] = f"{current_val} {new_val}"
                
                # Mark this row to be dropped
                df.iloc[i, :] = np.nan
            else:
                # Just replace NaN with empty string in this row
                df.iloc[i, nas] = ''
    
    return df.dropna(how='all').reset_index(drop=True)

def get_best_fit(df, tries=None, diag=False):
    if tries is None:
        # tries = [(i*0.05 + 0.25) for i in range(7)]
        tries = [0.4, 0.35, 0.45, 0.3, 0.5, 0.25, 0.55]

    dfs = []
    scores = []

    for thresh in tries:
        grid = assign_grid_positions(df, m_thresh=thresh)
        output = concatenate_multirow_cells2(grid)
        score = (output.isna() | (output=='')).sum().sum()
        dfs.append(output)
        scores.append(score)
        
    best = dfs[scores.index(np.min(scores))]

    if diag:
        return dfs, scores
    else:
        return best


def single_file_alt(df):
    left = df.iloc[:,:2]
    right = df.iloc[:,2:4]

    right.columns = left.columns
    
    return pd.concat([left, right])


def headershift(df):
    df.iloc[1:,1] = df.iloc[:-1,1]
    df.iloc[0,1] = np.nan
    return df