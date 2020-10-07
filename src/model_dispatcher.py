from sklearn import tree
from sklearn import ensemble
from sklearn.decomposition import TruncatedSVD

models= {
    "dt_gini":tree.DecisionTreeClassifier(
        criterion="gini",max_depth=10
    ),
    "dt_entropy":tree.DecisionTreeClassifier(
        criterion="entropy"
    ),

    "rf":ensemble.RandomForestClassifier( n_estimators = 100,
        max_depth = 8,
        min_samples_split = 4,
        n_jobs = -1,
        random_state = 1),    
}