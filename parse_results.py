import argparse
import pandas as pd
from functions import *

parser = argparse.ArgumentParser()

# source and destination are directory names relative to root folder
parser.add_argument('--source', default=None)
parser.add_argument('--outputdir', default='output')
# number of total columns and sets of columns in each image
parser.add_argument('--ncol', type=int, default=6)
parser.add_argument('--nfiles', type=int, default=2)
parser.add_argument('--sort', type=int, default=0)

args = parser.parse_args()

def main(args):
    # get paths of results to parse
    results, paths = load_results(dir='results', subcat=args.source)
    final_dfs = []
     
    for result, path in zip(results, paths):
        try:
            df = results_df(results=result, ncol=args.ncol)
            files = single_file(df=df, nfiles=args.nfiles)

            try:
                best = [get_best_fit(df) for df in files]
            except:
                print(
                    f'Erro parsing {path} -> incomplete formatting adjustments\n'
                    f'(Likely due to a character recognition issue)'
                )
                best = [assign_grid_positions(df) for df in files]
               
            for item in best:
                final_dfs.append(item)
            
        except Exception as e:
            print(f'Error processing results from {path}; skipping file', e)

    all = pd.concat(final_dfs)
    all = all[sorted(all.columns)]
    all.sort_values(by=all.columns[args.sort], inplace=True)

    outputname = args.source + '.csv'

    filename = unique_filename(f'{args.outputdir}/{outputname}')
    all.to_csv(filename, index=False, header=False)
    print(f'Saved to {args.outputdir}/{filename}')

if __name__ == '__main__':
    main(args)

