# --------------
#Header files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#path of the data file- path
data = pd.read_csv(path)
data.Gender.replace('-', 'Agender', inplace = True)
gender_count = data.Gender.value_counts()
gender_count.plot(kind='bar')
#Code starts here 




# --------------
#Code starts here
alignment = data.Alignment.value_counts()
alignment.plot(kind='pie')
plt.title('Character Alignment')


# --------------
#Code starts here
sc_df = data[['Strength', 'Combat']]
sc_covariance = sc_df.cov().loc['Strength','Combat']
sc_strength = sc_df.Strength.std()
sc_combat = sc_df.Combat.std()
sc_pearson = sc_covariance/(sc_strength*sc_combat)
print(sc_covariance, sc_strength, sc_combat, sc_pearson)

ic_df = data[['Intelligence', 'Combat']]
ic_covariance = ic_df.cov().loc['Intelligence','Combat']
ic_intelligence = ic_df.Intelligence.std()
ic_combat = ic_df.Combat.std()
ic_pearson = ic_covariance/(ic_intelligence*ic_combat)
print(ic_covariance, ic_intelligence, ic_combat, ic_pearson)


# --------------
#Code starts here
total_high = data.Total.quantile(0.99)
super_best = data[data['Total']>total_high]
super_best_names = list(super_best.Name)
super_best_names


# --------------
#Code starts here
fig,(ax_1,ax_2,ax_3) = plt.subplots(1,3,figsize=[17,8])
super_best.Intelligence.plot(kind='box', ax=ax_1)
super_best.Speed.plot(kind='box', ax=ax_2)
super_best.Power.plot(kind='box', ax=ax_3)


