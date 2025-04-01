import argparse
import pandas as pd
from functions import *

parser = argparse.ArgumentParser()
# source and destination are directory names relative to root folder
# must be without leading or trailing slashes
parser.add_argument('--source', default='results')
parser.add_argument('--imgsource', default=None)
parser.add_argument('--outputdir', default='output')
parser.add_argument('--outputname', default='words')
# number of total columns and sets of columns in each image
parser.add_argument('--ncol', type=int, default=6)
parser.add_argument('--nfiles', type=int, default=2)

args = parser.parse_args()

def main(args):
    # get paths of results to parse
    if args.imgsource is None:
        results, paths = load_results(dir=args.source)
    else:
        results, paths = load_results(dir=args.source, subcat=args.imgsource)

        
        final_dfs = []
     
    for result, path in zip(results, paths):
        try:
            df = results_df(results=result, ncol=args.ncol)
            files = single_file(df=df, nfiles=args.nfiles)
            grids = [assign_grid_positions(df) for df in files]

            try:
                adjusted = [concatenate_multirow_cells(df) for df in grids]
            except:
                print(
                    f'Erro parsing {path} -> incomplete formatting adjustments\n'
                    f'(Likely due to a character recognition issue)'
                )
                adjusted = grids

            for item in adjusted:
                final_dfs.append(item)
            
        except Exception as e:
            print(f'Error processing results from {path}; skipping file completely', e)

    all = pd.concat(final_dfs)
    all = all[sorted(all.columns)]

    filename = unique_filename(f'{args.outputdir}/{args.outputname}.csv')
    all.to_csv(filename, index=False, header=False)
    print(f'Saved to {filename}')

if __name__ == '__main__':
    main(args)

