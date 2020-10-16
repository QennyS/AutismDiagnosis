# Internship Report

Yilei Li, UESTC, Summer 2019

## Abstract:

The purpose of the project is to apply dimensionality reduction on a data set that has approximately 140 thousands features with labels by **Principal Component Analysis (PCA)**. Afterwards, use **Random Forest** model to train part of the data and then predict the label. 

## Contents:

* **Principal Component Analysis (PCA)**:
  * Computing PCA using the covariance method
  * Computing PCA using the Singular Value Decomposition (SVD) method
* **Random Forest**:
  * Traversal multiple parameters to attempt to reach the highest accuracy of predicting the label 


## Principal Component Analysis (PCA)
The Principal Component Analysis (PCA) is realized in ```pca.py``` by three approaches: Covariance, SVD, and directly using sklearn library.

### Instructions:
First, scroll down to the function that you want to use, specify which data set to use, e.g. 

```
data = np.loadtxt('0.1_data.csv', delimiter=',')
```
Replace `0.1_data.csv` with ideal data set.

Then, enter python interpreter:

```
python3 -i pca.py
```
#### For covariance method:
```
pca(k)
```
Replace k with target dimension. 

**Note**: This method requires more time to run and may get killed by system due to running out of memory. 

#### For SVD method:
```
pca_svd(k)
```
Replace k with target dimension
	
**Note**: k must not exceed ```min(number of samples, number of features)```

#### Use Sklearn library:
```
sklearn_pca()
```
**Note**: k must not exceed ```min(number of samples, number of features)```

Either SVD method or Sklearn library function could be used on the dimensionality reduction, and they should output the same result. 

## Random Forest
The Random Forest is realized in ```rf.py``` . There are two versions of implementation: `ver_1` and `ver_2`. 

`ver_1`: First use `StratifiedKFold` function from sklearn library to split the data into 5 sets. Afterwards, do cross validation on 5 sets. 

`ver_2`: Instead of split the set at beginning,  `cross_val_score` function from sklearn library to directly do cross validation on each cases during traversal. 

### Instructions:
First specify which data set to use in main function, e.g. 

```
data = np.loadtxt('700.csv', delimiter=',')
```
Replace `700.csv` with ideal data set

In main function, comment out or uncomment one verion, e.g.

```
ver_1(X,Y)
#ver_2(X,Y) This line has been commented out
```


Then, run with

```
rf.py
```

## Personal Conclusion 
During the first few days in intership, I tried to understand the mathematical reasoning behind the `PCA`. After reading some articles online, I started to write the program by using covariance method. This approach is feasible when the data set is small enough, but the data from MRI is too large to process on my computer and the program always got killed by operating system due to running out of memory. So I tried from another angle: SVD method. After some work, my program could output correct numerical result, but there were some issues with the sign. After viewing the source code of sklearn library, I figured out the problem and added a helper funtion `svd_flip`. Luckily, issues were resolved immediately. 

Then I did  dimensionality reduction on the 140 thousands features data to 1000, 900, 700, etc., features. First I attempt to use `grid_Search` function to adjust the parameter of ramdom forest model. However, Chen Mingren pointed out to me that this approach may not work, and he suggested that traversal would be a better way. So I switched to traversal to adjust parameters. Fully traversal is extremely time-consuming. I had run multiple times of traversal, each traversal took about 20 minutes to 1 hour. Therefore, a good strategy is really necessary. My general idea is ajusting the most important parameters in random forest model like `n_estimators`, `max_features`, and `max_depth`. Moreover, I only tested a few wide-span numbers on each parameter at first to see the performance quickly. Then I traversal certain parameter in a more careful manner. 


## Outcome 

Generally, the outcome of cross validation is about 64%, with highest being 66%.