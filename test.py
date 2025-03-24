import cv2
import easyocr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

############## EASYOCR
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
    # if 'TEST JEZELF' in strs:
    #     strs = strs[:strs.index('TEST JEZELF')]

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

def img_details(img, results):
    img = img.copy()
    for (bbox, text, confidence) in results:
        bbox = [(int(x), int(y)) for x, y in bbox]
        cv2.polylines(img, [np.array(bbox)], isClosed=True, color=(0,255,0), thickness=2)
        cv2.putText(img, f'{text} ({confidence:.2f}', (bbox[0][0], bbox[0][1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,115,0), 2)

    return img

path = 'photos/best.jpg'
img = cv2.imread(path)

reader = easyocr.Reader(['en', 'nl'])
results = reader.readtext(img)
strs, anno = get_stringlist('photos/best.jpg', reader)
rows = get_rows(strs)
df = rows_to_df(rows)
df.head()

test = pd.DataFrame(results)

test.shape
test.head()

def results_df(results):
    df = {
        'text': [],
        'x': [],
        'y': [],
        'prob': []
    }
    for item in results:
        df['text'].append(item[1])
        df['x'].append(int(item[0][0][0]))
        df['y'].append(int(item[0][0][1]))
        # df['coords'].append([int(num) for num in item[0][0]])
        df['prob'].append(int(item[2]))

    return pd.DataFrame(df)

results[2][2]

bla = results_df(results)

bla.head()

bla.sort_values(['y', 'x'], inplace=True)
bla[bla.text=='TEST JEZELF']
bla.iloc[160:190,:]

def remove_jezelf(df):
    if 'TEST JEZELF' not in df.text.values:
        print('not here')
        return df
    else:
        y = df[df.text=='TEST JEZELF']['y'].values[0]

    return df[df.y < y].copy()
        
from sklearn.cluster import KMeans

def classify_columns(df, n):
    points = np.array(df[['x', 'y']])

    return

def classify_columns(df, ncol):
    points = np.array(df[['x', 'y']])
    kmeans = KMeans(n_clusters=6, n_init=10)
    col_ass = kmeans.fit_predict(points[:, [0]])

    return col_ass


df.head(20)

df = remove_jezelf(bla)
df.y.min()
df.tail(20)

'TEST JEZELF' in df.text.values


plt.imshow(anno)
cv2.imwrite('annotated.jpg', anno)

det = img_details(img, results)
cv2.imwrite('annotated.jpg', det)




plt.imshow(bla)

plt.imshow(img)
cv2.imwrite('bla.jpg', img)

df.sort_index()
df[df.n=='1048']

isna = df.apply(lambda x: sum(x.isna()), axis=1)

df[isna<5]

isna.hist()

sum(isna<8)
isna.min()
isna.unique()
row = df.loc[1,:]

row.isna()



df.shape

i = df.loc[:,0].values
len(i)

df.head()
df.sort_values('0')

df.columns = [str(i) for i in range(20)]



strs.index('TEST JEZELF')

bla = strs[150:170].copy()

bla[:bla.index('TEST JEZELF')]

bla

rows[::-1]

strs[int(np.where(np.array(strs)=="1057")[0][0]):]
    


gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


contour_img = img2.copy()
cv2.drawContours(contour_img, contours, -1, (0, 255, 0), 2)

plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))









#############


