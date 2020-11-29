import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import itertools
from sklearn.metrics import f1_score

def clean_noisy_data(dataset,classes = 2):
    """
    Clean the data with replacing the miss-filled values by the correct ones and defining the right types for the dataset
    variables.
    
    Input : The dataset to clean
    Output : Cleaned dataset
    
    """
    
    if 'id' in dataset.columns :
        dataset = dataset.drop(columns = ['id']) # Drop id column as it's not relevent for predictions
    
    
    for c in dataset.columns :
        if dataset[c].dtype != 'float32' and dataset[c].dtype != 'float64' and dataset[c].dtype != 'int32'  and dataset[c].dtype != 'int64' :
            dataset[c] = dataset[c].str.replace('\t','')
            dataset[c] = dataset[c].replace('?',np.nan)
            dataset[c] = dataset[c].str.replace(' ','')

        
    #Changing numerical data into float types 
    output_column = dataset.columns[-1]

    string_columns = []
    numerical_columns = []
    output_is_string = False
    for c in dataset.columns : 
        try : 
            dataset[c] = dataset[c].astype(float)
            numerical_columns.append(c)
        except ValueError :
            if c == output_column:
                output_is_string = True
            string_columns.append(c)
            
 
    #Ordinal encoding of the output variable 
    if output_is_string:
        outputs = dataset[output_column].unique()
        assert len(outputs) == classes, 'Error the number of output classes should be the same as the one in the dataset'
        dataset[output_column] = 1*(dataset[output_column] == outputs[0]) #return an ordinal encoding of the output variable
    dataset[output_column] = dataset[output_column].astype(int)
    
    dataset[numerical_columns] = dataset[numerical_columns].astype(float)
    return dataset


def detect_type(data):
    """
    Detecting the type of the variables numerical or categorical
    
    Input : Dataset 
    Output : Lists of numerical variables and categorial ones
    """
    
    num_variables = []
    categ_variables = []
    columns = list(data.columns)
    n = len(columns)
    for i in range(n):
        if data[columns[i]].dtype == 'int' or data[columns[i]].dtype == 'float': #Checking if the variable is numerical
            num_variables.append(columns[i])
        else : 
            categ_variables.append(columns[i])
    return num_variables, categ_variables


def replace_missing(data, num_variables, categ_variables, num_strategy = 'mean', categ_strategy = 'most_frequent'):
    """
    Replacing the missing values in categorical variables and numerical variables by 2 corresponding strategies
    (mean for numerical variables and the most frequent value for categorical varibles for example)
    
    Input : Dataset, numerical variables of the data, categorial variables of the data and the defined strategies
    Output : A transformed dataset with missing values filled
    """
    
    ct = ColumnTransformer([("categ_imput", SimpleImputer(missing_values = np.nan, strategy = categ_strategy), categ_variables),
                        ("num_imput", SimpleImputer(missing_values = np.nan, strategy = num_strategy), num_variables)])
    data_transformed = ct.fit_transform(data) #Recuperate the transformed array
    columns = categ_variables + num_variables
    data_tr_table = pd.DataFrame(data_transformed, columns = columns)#Putting the transformation into a dataframe
    return data_tr_table

def center_encode(data, num_variables, categ_variables):
    """
    Centring and normalizing the data. Transforming the categorical variables.
    
    Input : Dataset, numerical variables of the data and categorical variables of the data.
    Output : Transformed dataset
    
    """
    cat_enc = OrdinalEncoder()
    center_norm = StandardScaler()
    if categ_variables != [] and categ_variables != []:
        categ_data = data[categ_variables]
        num_data = data[num_variables]
        data_transformed_cat = cat_enc.fit_transform(categ_data) #.toarray() #ordinal encode catageorical data 
        data_transformed_num = center_norm.fit_transform(num_data) # center and normalize numerical data
        categ_columns = categ_variables #cat_enc.get_feature_names(categ_variables)
        columns = list(categ_columns) + num_variables
        data_transformed = np.concatenate((data_transformed_cat, data_transformed_num), axis = 1)
        data_tr_table = pd.DataFrame(data_transformed, columns = columns)
    elif categ_variables != []:
        categ_data = data[categ_variables]
        data_transformed_cat = cat_enc.fit_transform(categ_data)
        columns = categ_variables
        data_tr_table = pd.DataFrame(data_transformed_cat, columns = columns)
    elif num_variables != []:
        num_data = data[num_variables]
        data_transformed_num = center_norm.fit_transform(num_data)
        columns = num_variables
        data_tr_table = pd.DataFrame(data_transformed_num, columns = columns)
    return data_tr_table


def feature_selection(dataset,cut_off_variance=0.95,keep_features=True):
    """
        Applies Feature selection using PCA. We only keep the features which have the highest absolute coefficient to the principal component vectors.
        We also fix the number of the remaining vectors based on the number of components which garantees 95% of the original variance of the dataset.
    """
    X,_ = dataset.values[:,:-1],dataset.values[:,-1]
    pca = PCA()
    pca.fit(X)
    y = np.cumsum(pca.explained_variance_ratio_)

    
    number_of_features = len(dataset.columns) - sum(y>=cut_off_variance)
    print(f'number of features selected : {number_of_features}')
    
    if keep_features : 
        pca = PCA(n_components=2)
        pca.fit(X)
        features_importance_sorted = np.argsort(pca.components_[0])
        features_remaining = features_importance_sorted[:number_of_features]
        data_compressed = dataset.iloc[:,list(features_remaining)+[len(dataset.columns)-1]]
    else :
        pca = PCA(n_components=number_of_features)
        new_X = pca.fit_transform(X)        
        data_compressed = pd.DataFrame(new_X)
        data_compressed[dataset.columns[-1]] = dataset.iloc[:,-1]
        
    return data_compressed

def split_data(X, y):
    """
    Split the data into training set and validation set
    
    Input : Data and output
    Output : Training set and validation set of the data, training set and validation set of the class
    
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
    return X_train, X_test, y_train, y_test

def determine_combinaisons(parameters):
    """
    Determine all possible combinaisons to have from a defined set of parameters of a model
    
    Input : Dictionary of parameters
    Output : List of all combinaisons, each combinaison of parameters defined as a dictionary 
    
    """
    parameters_values = list(parameters.values())
    combinations = list(itertools.product(*parameters_values))
    comb_parameters = []
    for c in combinations : 
        d = {}
        keys = list(parameters.keys())
        for k in range(len(keys)):
            d[keys[k]] = c[k]
        comb_parameters.append(d)
    return comb_parameters

def training(model, parameters, X, y):
    """
    Train the model with defined parameters and returns the cross validation score
    
    Input : Model, parameters of the model, data, class
    Output : the score of the cross validation
    
    """
    clf = model(**parameters)
    scores = cross_val_score(clf, X, y, cv=5, scoring='f1_macro')
    clf.fit(X,y)
    score = np.mean(scores)
    return score,clf

def select_params(model, parameters, X, y):
    """
    Selecting the best parameters to take for the model 
    
    Input : Model, dictionary of possible parameters, Data, class
    Output : Chosen combinaison of parameters, score of the cross validation with this combinaison
    
    """
    comb_parameters = determine_combinaisons(parameters)
    total_scores = []
    for i in range(len(comb_parameters)):
        total_scores.append(training(model, comb_parameters[i], X, y)[0])
    score_max = np.max(total_scores)
    ind_max = np.argmax(total_scores)
    return comb_parameters[ind_max], score_max
def test_evaluate(model,X,y):
    y_pred = model.predict(X)
    score = f1_score(y, y_pred,average='macro')
    return score






