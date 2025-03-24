import cv2
import easyocr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

path1 = 'photos/zoomed.jpg'
path2 = 'photos/photo1.jpg'

img = cv2.imread(path1)
img2 = cv2.imread(path2)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


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

strs[20:]

results[-1]

'1083' in strs

plt.imshow(anno)
cv2.imwrite('annotated.jpg', anno)

det = img_details(img, results)
cv2.imwrite('annotated.jpg', det)



bla = img_details(img, result)

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


