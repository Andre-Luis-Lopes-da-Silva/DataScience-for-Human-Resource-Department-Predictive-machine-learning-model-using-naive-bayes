import pandas as pd

df = pd.read_csv('/home/andre/Desktop/Naive - recursos humanos/WA_Fn-UseC_-HR-Employee-Attrition.csv')  

df['Attrition'] = df['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)
df['OverTime'] = df['OverTime'].apply(lambda x: 1 if x == 'Yes' else 0)
df['Over18'] = df['Over18'].apply(lambda x: 1 if x == 'Y' else 0)

df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber'], axis = 1, inplace=True)

previsores1 = df.drop(['Attrition'], axis = 1)
previsores = previsores1.iloc[:,0:30].values

classe = df.iloc[:,1].values 
        
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()  

lep = 0
total_colunas = len(previsores1.columns)
for lep in range(total_colunas):
    previsores[:,lep] = labelencoder.fit_transform(previsores[:,lep])  
     
from sklearn.naive_bayes import GaussianNB
classificador = GaussianNB()
classificador.fit(previsores, classe)  

# Samples to be analyzed 
employee1 = [23, 2, 624, 2, 0, 1, 1, 1, 0, 64, 2, 1, 7, 3, 2, 809, 999, 8, 1, 0, 0, 0, 0, 8,	0, 0, 6, 4, 0, 5]
employee2 = [31, 1, 113, 1, 7, 0, 1, 2, 1, 31, 1, 1, 6, 1, 1, 682, 1328, 1, 0, 12, 1, 3, 1, 10, 3, 2, 10, 7, 1, 7]

resultado = classificador.predict([employee1, employee2]) 

print('The classes are: ', classificador.classes_)  
print('Class 1 the employee will leave from company')
print('Class 0 the employee will stay in company')
print('Number of training samples observed in each class: ', classificador.class_count_) 
print('Probability of each class: ', classificador.class_prior_) 

def answer(result):
    if result == 1:
        print("This employee will leave from company (class 1)")
    else:
       print("This employee will stay in company (class 0)")

print('Testing the result:')
print('Characteristics of employee1: ', employee1)
print('Characteristics of employee2: ', employee1)
print('Submitting it to the classifier:')
print('Classification of employee1 :', resultado[0])
answer(resultado[0])
print('Classification of employee2 :', resultado[1])
answer(resultado[1])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(previsores, classe, test_size=0.5, random_state=0)  

from sklearn import metrics

y_pred = classificador.predict(X_test) 

print('Metrics of this algorithm:')
print('Accuracy: {:.2}'.format(metrics.accuracy_score(y_test, y_pred)))

from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

print('Precision: {:.2}'.format(metrics.precision_score(y_test, y_pred)))

print('Recall: {:.2}'.format(metrics.recall_score(y_test, y_pred)))

print('F1 Score: {:.2}'.format(metrics.f1_score(y_test, y_pred, average='macro')))

print(metrics.classification_report(y_test, y_pred))






