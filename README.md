# mini-projet-ml-mce
Git repository for the mini-project in the Machine Learning course at IMT Atlantique.

To run the code with a Logistic Regression model on one the banknote dataset with the following parameters {kernel=rbf, C=10, gamma=auto}: 
 
``` 
$cd src
$python main.py -m=LogisticRegression -d=banknote --pca=True -v=0.98 -p kernel=rbf C=10 gamma=auto

```

For help and to check the different options run : 

```
python main.py -h
```