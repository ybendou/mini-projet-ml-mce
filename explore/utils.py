import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder



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

def detect_type(data):
    data = data.iloc[:,:-1]
    num_variables = []
    categ_variables = []
    columns = list(data.columns)
    n = len(columns)
    for i in range(n):
        if data[columns[i]].dtype == 'int' or data[columns[i]].dtype == 'float':
            num_variables.append(columns[i])
        else :
            categ_variables.append(columns[i])
    return num_variables, categ_variables

def replace_missing(data, num_variables, categ_variables, num_strategy = 'mean', categ_strategy = 'most_frequent'):
    data = data.iloc[:,:-1]
    ct = ColumnTransformer([("categ_imput", SimpleImputer(missing_values = np.nan, strategy = categ_strategy), categ_variables),
                            ("num_imput", SimpleImputer(missing_values = np.nan, strategy = num_strategy), num_variables)])
    data_transformed = ct.fit_transform(data)
    columns = categ_variables + num_variables
    data_tr_table = pd.DataFrame(data_transformed, columns = columns)
    return data_tr_table

def center_encode(data, num_variables, categ_variables):
    cat_enc = OneHotEncoder()
    center_norm = StandardScaler()
    ct = ColumnTransformer([("categ_encod", cat_enc, categ_variables),
                            ("norm", center_norm, num_variables)])
    data_transformed = ct.fit_transform(data)
    columns = categ_variables + num_variables
    data_tr_table = pd.DataFrame(data_transformed, columns = columns)
    return data_tr_table


