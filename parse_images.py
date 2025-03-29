import cv2
import json
import pickle
import easyocr
import pandas as pd
from tqdm import tqdm
from functions import *
from pathlib import Path

def main():
    with open('config.json', 'r') as file:
        config = json.load(file)

    # loading and reading input images
    get = config['input_dir']
    # img_paths = [get + '/' + path for path in os.listdir(get)]
    img_paths = get_remaining_paths(get)
    img_list = [cv2.imread(path) for path in img_paths]
    reader = easyocr.Reader(config['langs'])

    results = [reader.readtext(img) for img in tqdm(img_list)]

    for img, result in zip(img_paths, results):
        img_id = Path(img).stem
        filename = 'results/' + img_id + '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(result, file)

        # df = results_df(result)
        # filename = 'results/' + img_id + '.csv'
        # df.to_csv(filename, index=False)

if __name__ == '__main__':
    main()
