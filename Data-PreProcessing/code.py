# --------------
#Importing header files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Code starts here
data = pd.read_csv(path)
data.Rating.plot(kind='hist')
data = data[data.Rating<=5]
data.Rating.plot(kind='hist')

#Code ends here


# --------------
# code starts here
total_null = data.isnull().sum()
percent_null = total_null/len(data)*100
missing_data = pd.concat([total_null,percent_null],keys=['Total','Percent'],axis=1)
print(missing_data)

data.dropna(inplace=True)
total_null_1 = data.isnull().sum()
percent_null_1 = total_null_1/len(data)*100
missing_data_1 = pd.concat([total_null_1,percent_null_1],keys=['Total','Percent'],axis=1)
print(missing_data_1)

# code ends here


# --------------
#Code starts here
fig = sns.catplot(x='Category', y='Rating', data=data, kind='box', height=10).set_xticklabels(rotation=90)
fig.set(title="Rating vs Category [BoxPlot]")

#Code ends here


# --------------
#Importing header files
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

#Code starts here
print(data.Installs.value_counts())
data.Installs = data['Installs'].apply(lambda x:int(x[:-1].replace(",","")))
le = LabelEncoder()
le.fit(data.Installs)
data.Installs = le.transform(data.Installs)
sns.regplot(x='Installs', y='Rating', data=data).set(title="Rating vs Installs [RegPlot]")

#Code ends here


# --------------
#Code starts here

data.Price = data.Price.apply(lambda x: float(x[1:]) if len(x)>1 else float(x))
sns.regplot(x="Price", y="Rating", data=data).set(title = "Rating vs Price [RegPlot]")

#Code ends here


# --------------
#Code starts here
data.Genres.unique()
data.Genres = data.Genres.apply(lambda x: x.split(";")[0])
gr_mean = data[["Genres", "Rating"]].groupby(by="Genres", as_index=False).mean()
gr_mean.describe()
gr_mean = gr_mean.sort_values(by='Rating')
print(gr_mean.iloc[0])
print(gr_mean.iloc[-1])

#Code ends here


# --------------
#Code starts here
data['Last Updated'].head()
data['Last Updated'] = data['Last Updated'].apply(lambda x: pd.to_datetime(x))
max_date = data['Last Updated'].max()
data['Last Updated Days'] = (max_date - data['Last Updated']).apply(lambda x: x.days)
sns.regplot(x="Last Updated Days", y="Rating", data=data).set(title="Rating vs Last Updated [RegPlot]")

#Code ends here


