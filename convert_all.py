import os
import cv2
import json
import easyocr
import pandas as pd
from tqdm import tqdm
from functions import *

def main():
    # config load
    with open('config.json', 'r') as file:
        config = json.load(file)

    # loading and reading input images
    get = config['input_folder'] + '/'
    img_paths = [get + path for path in os.listdir(get)]
    img_list = [cv2.imread(path) for path in img_paths]
    reader = easyocr.Reader(['en', 'nl'])

    # running OCR and parsing into dataframe
    results = [reader.readtext(img) for img in tqdm(img_list)]
    dfs = [df for r in results for df in single_file(results_df(r))]

    # reconsiling inconsistencies
    parsed = []
    for i, df in enumerate(dfs):
        try:
            parsed.append(parse_df(df))
        except Exception as e:
            print(f'Error in {img_paths[int(i/2)]}: ', e)

    # concatenating all output files into one dataframe
    all = pd.concat(parsed)
    
    # writing to output
    output_filetype = config['output_filetype']
    output_filename = unique_filename(config['output_folder'] + '/output' + output_filetype)
    output_filename

    if output_filetype == '.csv':
        all.to_csv(output_filename, index=False)
    elif output_filetype == '.xlsx':
        all.to_excel(output_filename, index=False)

if __name__ == '__main__':
    main()
