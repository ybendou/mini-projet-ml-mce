import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder



def clean_noisy_data(dataset,classes = 2):
    """
        Cleans noisy datasets, removes id column and clean noisy strings, transformrs numerical data in to float.
    """
    if 'id' in dataset.columns :
        dataset = dataset.drop(columns = ['id']) # Drop id column as it's not relevent for predictions
    
    
    for c in dataset.columns :
        if dataset[c].dtype == 'float' or dataset[c].dtype == 'int' or dataset[c].dtype == 'int64' or dataset[c].dtype == 'float64' : #clean columns that are objects or strings
            pass
        else : 
            dataset[c] = dataset[c].str.replace('\t','')
            dataset[c] = dataset[c].replace('?',np.nan)
            dataset[c] = dataset[c].str.replace(' ','')

    #Changing numerical data into float types 
    output_column = dataset.columns[-1] # don't process the output column
    string_columns = []
    numerical_columns = []
    output_is_string = False 
    
    for c in dataset.columns : 
        try : 
            dataset[c] = dataset[c].astype(float)
            numerical_columns.append(c)
        except ValueError :
            if c == output_column:
                output_is_string = True # if output is a string, process it later
            string_columns.append(c)
            
 
    #Ordinal encoding of the output variable if it's a string variable
    if output_is_string:
        outputs = dataset[output_column].unique()
        assert len(outputs) == classes, 'Error the number of output classes should be the same as the one in the dataset'
        dataset[output_column] = 1*(dataset[output_column] == outputs[0]) #return an ordinal encoding of the output variable
    dataset[output_column] = dataset[output_column].astype(int)
    
    dataset[numerical_columns] = dataset[numerical_columns].astype(float)
    return dataset

def detect_type(data):
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
    ct = ColumnTransformer([("categ_imput", SimpleImputer(missing_values = np.nan, strategy = categ_strategy), categ_variables),
                            ("num_imput", SimpleImputer(missing_values = np.nan, strategy = num_strategy), num_variables)])
    data_transformed = ct.fit_transform(data)
    columns = categ_variables + num_variables
    data_tr_table = pd.DataFrame(data_transformed, columns = columns)
    return data_tr_table

def center_encode(data, num_variables, categ_variables):
    cat_enc = OrdinalEncoder()
    center_norm = StandardScaler()
    categ_data = data[categ_variables]
    num_data = data[num_variables]
    data_transformed_cat = cat_enc.fit_transform(categ_data) #.toarray() #ordinal encode catageorical data 
    data_transformed_num = center_norm.fit_transform(num_data) # center and normalize numerical data
    categ_columns = categ_variables #cat_enc.get_feature_names(categ_variables)
    columns = list(categ_columns) + num_variables
    data_transformed = np.concatenate((data_transformed_cat, data_transformed_num), axis = 1)
    data_tr_table = pd.DataFrame(data_transformed, columns = columns)
    return data_tr_table