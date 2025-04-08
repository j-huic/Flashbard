import cv2
import json
import pickle
import easyocr
from tqdm import tqdm
from functions import *
from pathlib import Path

def main():
    with open('config.json', 'r') as file:
        config = json.load(file)

    # loading and reading input images
    get = config['input_dir']
    img_paths = get_remaining_paths(get)
    reader = easyocr.Reader(config['langs'])

    results = []

    for path in tqdm(img_paths):
        try:
            img = cv2.imread(path)
            result = reader.readtext(img)
            results.append(result)
            img_details(path, result, Path(path).stem+'.jpg')
        except:
            print(f'Error parsing {path}, skipping file')

    for img, result in zip(img_paths, results):
        img_id = Path(img).stem
        os.makedirs('results', exist_ok=True)
        filename = 'results/' + img_id + '.pkl'
        with open(filename, 'wb') as file:
            pickle.dump(result, file)

if __name__ == '__main__':
    main()
