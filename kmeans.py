import pandas as pd
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

dataroot = pathlib.Path('./data')

train_columns = ['num','datetime','target','temperature','windspeed','humidity','precipitation','insolation','nelec_cool_flag','solar_flag']
test_columns = ['num','datetime','temperature','windspeed','humidity','precipitation','insolation','nelec_cool_flag','solar_flag']
train_df = pd.read_csv(dataroot/'train.csv', skiprows=[0], names=train_columns)
test_df = pd.read_csv(dataroot/'test.csv', skiprows=[0], names=test_columns)

# adding features related to datetime
eda_df = train_df.copy()
eda_df['datetime'] = pd.to_datetime(eda_df['datetime'])
eda_df['hour'] = eda_df['datetime'].dt.hour
eda_df['weekday'] = eda_df['datetime'].dt.weekday
eda_df['date'] = eda_df['datetime'].dt.date
eda_df['day'] = eda_df['datetime'].dt.day
eda_df['month'] = eda_df['datetime'].dt.month
eda_df['weekend'] = eda_df['weekday'].isin([5,6]).astype(int)

by_weekday = eda_df.groupby(['num','weekday'])['target'].median().reset_index().pivot('num', 'weekday', 'target').reset_index()
by_hour = eda_df.groupby(['num','hour'])['target'].median().reset_index().pivot('num','hour','target').reset_index().drop('num', axis=1)
by_weekday.columns = ['num'] + [f'day{c}' for c in by_weekday.columns[1:]]
by_hour.columns = [f'hour{c}' for c in by_hour.columns]

df = pd.concat([by_weekday, by_hour], axis= 1)
df.set_index('num', inplace=True)

df.iloc[:, :7] = StandardScaler().fit_transform(df.T.iloc[:7]).T
df.iloc[:, 7:] = StandardScaler().fit_transform(df.T.iloc[7:]).T

#df.iloc[:, :7].T.plot(alpha=0.5, linewidth=0.5, legend=False)
#df.iloc[:, 7:].T.plot(alpha=0.5, linewidth=0.5, legend=False)
#
#def change_n_clusters(n_clusters, data):
#    sum_of_squared_distances = []
#    for n in n_clusters:
#        kmeans = KMeans(n_clusters=n)
#        kmeans.fit(data)
#        sum_of_squared_distances.append(kmeans.inertia_)
#
#    plt.figure(1 , figsize=(12, 6))
#    plt.plot(n_clusters , sum_of_squared_distances, '-o')
#    plt.xlabel('Number of Clusters')
#    plt.ylabel('Inertia')
#
#change_n_clusters(list(range(2, 12)), df)

kmeans = KMeans(n_clusters=4, random_state=2)
df['km_cluster'] = kmeans.fit_predict(df)

for ix, g in df[['km_cluster']].groupby('km_cluster'):
    print(ix, list(g.index))
num_to_cluster = df['km_cluster'].to_dict()
print(num_to_cluster)
#print(num_to_cluster)
#import pprint
#pprint.pprint(num_to_cluster)


#for c in sorted(df['km_cluster'].unique()):
#    df[df.km_cluster==c].iloc[:, :7].T.plot(linewidth=0.7, legend=False)
#plt.show()
