# DataScience-for-Human-Resource-Department-Predictive-machine-learning-model-using-naive-bayes
This study presents a predictive machine learning model using naïve bayes to classify the profile of employees who will leave the company.

This datascience solution was performed using dataset by “IBM HR Analytics Employee Attrition & Performance, Predict attrition of your valuable employees”. Available at: https:www.kaggle.com/pavansubhasht/ibm-hr-analytics-attrition-dataset

Hiring a new employee involves costs. Therefore, the identifying of a candidate who has a profile that will not stay with the company will be useful to the company optimize your decisions. 

After the clearing of the data and the data exploratory analysis (EDA), some insights were perfomed.

The columns: Attrition, OverTime and Over18 possessed “y” or “n” and they were replaced by “0” or “1”, because machine learning does not process variables like “string”.  

Variables that have a unique attribute for each employee are not useful for analysis. Therefore, these variables were removed, as the columns: EmployeeCount, StandardHours, Over18 and EmployeeNumber.

The classes of this algorithm are class 1 (the employee will leave from company) and class 0 (the employee will stay in company). These will be the answers. 

The metrics obtained in this study were: 
Accuracy: 0.81,
Precision: 0.43,
Recall: 0.63 and
F1 Score: 0.7.
