import json

print('Welcome to config setup\n')

langs = input(
    f'Input the languages included in the screenshot separated with a space\n'
    f'For example: <en es> for english and spanish (without <>)\n'
    f'Scroll down at jaided.ai/easyocr to see supported languages\n'
)
var_langs = langs.split(' ')

var_input_dir = input(
    f'\nInput the name of your desired input directory (required)\n'
    f'When running mass convert, Flashbard will evaluate every photo within this folder\n'
)

var_output_dir = input(
    f'\nInput the name of your desired output directory (empty string for root)\n'
    f'Leave empty for default ("output") or "root" to output to root\n'
)
if var_output_dir == '':
    var_output_dir = 'output'

temp = input(
    f'\nChoose the preferred output type\n'
    f'(CSV is recommended as it is a valid file type for Anki decks)\n'
    f'1) .csv\n'
    f'2) .xlsx\n'
)
if temp == 1:
    var_output_filetype = 'csv'
elif temp == 2:
    var_output_filetype = 'xlsx'
else:
    var_output_filetype = 'csv'

n_files = input(
    f'Word tables may include several sets of columns (files)\n'
    f'Explicitly setting the format of these tables is crucial for correct formatting\n'
    f'Indicate how many sets of columns there are in each photo (typically 1 or 2)\n'
)
var_n_files = int(n_files)

n_columns = input(
    f'\nIndicate how many columns each set contains\n'
    f'For instance, at least two columns for the input and output languages, and '
    f'often a third column for tags or extra info\n'
)
var_n_columns = var_n_files * int(n_columns)

variable_dict = {name.split('var_')[1]: value for name, value in globals().items() 
    if name.startswith('var_')}

print('\nConfig done:\n')
print(variable_dict)

with open('configtest.json', 'w') as file:
    json.dump(variable_dict, file)

# var_dict = {k: v for k, v in locals().items() if not k.startswith("__") and k != "temp"}


# config = {
#     "langs": langs,
#     "input_dir"
# }






