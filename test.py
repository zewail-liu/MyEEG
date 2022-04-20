from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE

Data, label = make_classification(n_classes=2, weights=[0.1, 0.9], n_samples=1000,flip_y=0,)
print(Data.shape, label.shape)
print('Original: %s' % Counter(label))

Data, label = SMOTE().fit_resample(Data, label)
print(Data.shape, label.shape)
print('Resampled: %s' % Counter(label))
