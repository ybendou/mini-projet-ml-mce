import pandas as pd
import numpy as np



def clean_noisy_data(dataset,classes = 2):
    
    if 'id' in dataset.columns :
        dataset = dataset.drop(columns = ['id']) # Drop id column as it's not relevent for predictions

    #Changing numerical data into float types 
    output_column = dataset.columns[-1]

    string_columns = []
    output_is_string = False
    for c in dataset.columns : 
        try : 
            dataset[c] = dataset[c].astype(float)
        except ValueError :
            if c == output_column:
                output_is_string = True
            string_columns.append(c)
            
    for c in string_columns :
        dataset[c] = dataset[c].str.replace('\t','')
        dataset[c] = dataset[c].replace('?',np.nan)
    
    #Ordinal encoding of the output variable 
    if output_is_string:
        outputs = dataset[output_column].unique()
        assert len(outputs) == classes, 'Error the number of output classes should be the same as the one in the dataset'
        dataset[output_column] = 1*(dataset[output_column] == outputs[0]) #return an ordinal encoding of the output variable
    dataset[output_column] = dataset[output_column].astype(int)
    return dataset