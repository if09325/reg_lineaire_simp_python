# -*- coding: utf-8 -*-
#library pandas to read files
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy import stats


#load my data inside a data frame 
startups_df = pd.read_csv(r"ds.csv")

#select X variables (independant)
X_RD_Spend = startups_df.iloc[0:len(startups_df),0]
X_Administration_Spend = startups_df.iloc[0:len(startups_df),1]            
X_MarketingSpend = startups_df.iloc[0:len(startups_df),2]
#select Y variables (dependent) 
Y_Profit = startups_df.iloc[0:len(startups_df),4]
                     

slope, intercept, r_value, p_value,std_err = stats.linregress(X_RD_Spend,Y_Profit)
def predict(x) : 
    return (slope * x + intercept)
#draw a graph using payplot including FitLine(ligne des prédictions)
fitLine = predict(X_RD_Spend)
plt.plot(X_RD_Spend,fitLine,c='red')
axes = plt.axes()
axes.grid()
plt.scatter(X_RD_Spend,Y_Profit)

#plt.scatter(X_Administration_Spend, Y_Profit)
#plt.scatter(X_MarketingSpend, Y_Profit)
plt.show()
                 
print("R² :--------->  ", r_value)
print("p-value :---->  ", p_value) 
print("std_err :---->  ", std_err)                   
                    

