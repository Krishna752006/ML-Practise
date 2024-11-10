import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('6. student performance.csv')
data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'Yes':1,'No':0})

'''label_encoder = LabelEncoder()
data['Location'] = label_encoder.fit_transform(data['Location'])'''

corr_matrix = data.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.show()