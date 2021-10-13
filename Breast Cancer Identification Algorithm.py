import scipy
import numpy
import matplotlib
import pandas
import sklearn
import sys
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load Datasets:

file_name = "C:\Users\Garv Gaur\OneDrive - Lakeview Academy\data.csv"
classes = ["diagnosis", "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean", "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean", "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se", "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se", "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst", "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"]
cancer_dataset = pandas.read_csv(file_name, names = classes)

# Dimensions of the Dataset

print(cancer_dataset.shape)

# Dataset sample and metrics

print(cancer_dataset.head(20))
print(cancer_dataset.describe())
print(cancer_dataset.groupby("diagnosis").size())

# Plots and Histograms of each DataType

cancer_dataset.plot(kind = 'box', subplots = True, layout = (6,5), sharex = False, sharey = False)
plt.show()
cancer_dataset.hist()
plt.show()

# The scatterplot shows many correlations, but it is so big that you need a huge monitor. Show at your own risk.

# scatter_matrix(cancer_dataset)
# plt.show()

# If you want to view the whole dataset, please uncomment the next line.

#print(cancer_dataset.values)

# Separation of data (numerical) and testing class (diagnosis)

array = cancer_dataset.values
numeric = array[: , 1 : ]
diag = array[: , 0]

# Separation of data into training set (70%) and testing set (30%) randomly

validation_size = 0.3
seed = 5
numeric_train, numeric_validation, diag_train, diag_validation = model_selection.train_test_split(numeric, diag, test_size = validation_size, random_state = seed)

# The testing process begins! The algorithms will be judged on accuracy, which is stored here.

scoring = "accuracy"

# Each model is added to the testing lineup

models = []
models.append(("LR", LogisticRegression(solver = "liblinear", multi_class = "ovr")))
models.append(("LDA", LinearDiscriminantAnalysis()))
models.append(("KNN", KNeighborsClassifier()))
models.append(("CART", DecisionTreeClassifier()))
models.append(("NB", GaussianNB()))
models.append(("SVM", SVC(gamma = "auto")))

# Evaluation of each model starts

results = []
model_names = []
for name, model in models:

    # Use the training set a bit at a time
    
    kfold = model_selection.KFold(n_splits = 10, random_state = seed)
    cv_results = model_selection.cross_val_score(model, numeric_train, diag_train, cv = kfold, scoring = scoring)
    results.append(cv_results)
    model_names.append(name)
    
    # Results of each algorithm on the testing set: will decide what algo to use on the final set
    
    res_string = "%s: %f (%f)\n" % (name, cv_results.mean(), cv_results.std())
    print(res_string, end = "")

# Graphing initial results

algo_comp_graph = plt.figure()
algo_comp_graph.suptitle("Algorithm Comparison")
axes = algo_comp_graph.add_subplot(111)
plt.boxplot(results)
axes.set_xticklabels(model_names)
plt.show()

# Final test with whatever algorithm you want: in my case LDA

lda = LinearDiscriminantAnalysis()
lda.fit(numeric_train, diag_train)

# How the algorithm will do on testing dataset

predictions = lda.predict(numeric_validation)
print(accuracy_score(diag_validation, predictions))
print(confusion_matrix(diag_validation, predictions))
print(classification_report(diag_validation, predictions))

# Final test on the whole dataset

# Again, use the training set a bit at a time

final_result = []
kfold = model_selection.KFold(n_splits = 10, random_state = seed)
results = model_selection.cross_val_score(lda, numeric, diag, cv = kfold, scoring = scoring)
final_result.append(results)

# Results!

fin_res_string = "LDA Results: %f (%f)" % (results.mean(), results.std())
print(fin_res_string)

cell_test_arr = input("Please input the metrics of the cell separated by commas").split(",")

for i in range(len(cell_test_arr)):
    cell_test_arr[i] = float(cell_test_arr[i])
cell_test_arr = [cell_test_arr]

predictions = lda.predict(cell_test_arr)
if int(accuracy_score(["B"], predictions)) == 0:
    print("The LDA algorithm picked malignant")
else:
    print("The algorithm picked benign")
"""
17.99,10.38,122.8,1001,0.1184,0.2776,0.3001,0.1471,0.2419,0.07871,1.095,0.9053,8.589,153.4,0.006399,0.04904,0.05373,0.01587,0.03003,0.00619,325.38,17.33,184.6,2019,0.1622,0.6656,0.7119,0.2654,0.4601,0.1189
"""