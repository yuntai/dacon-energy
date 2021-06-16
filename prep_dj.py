
# function for feature engineering
def CDH(xs):
	ys = []
    for i in range(len(xs)):
        if i < 11:
            ys.append(np.sum(xs[:(i+1)]-26))
        else:
            ys.append(np.sum(xs[(i-11):(i+1)]-26))
    return np.array(ys)

# data preprocessing function for testset
def train_preprocess(train):
    X_train = train.copy()
    X_train['datetime'] = pd.to_datetime(X_train['datetime'])
    X_train['hour'] = X_train['datetime'].dt.hour
    X_train['month'] = X_train['datetime'].dt.month
    X_train['day'] = X_train['datetime'].dt.day
    X_train['date'] = X_train['datetime'].dt.date
    X_train['weekday'] = X_train['datetime'].dt.weekday
    ## one hot encoding for weekday and hour
    X_train = pd.concat([X_train, pd.get_dummies(X_train['weekday'], prefix ='weekday')], axis=1)
    X_train = pd.concat([X_train, pd.get_dummies(X_train['hour'], prefix ='hour')], axis=1)
    ## daily minimum temperature
    X_train = X_train.merge(X_train.groupby(['date'])['temperature'].min().reset_index().rename(columns = {'temperature':'min_temperature'}), on = ['date'], how = 'left')
    ## THI
    X_train['THI'] = 9/5*X_train['temperature'] - 0.55*(1-X_train['humidity']/100)*(9/5*X_train['temperature']-26)+32
    ## mean_THI
    X_train = X_train.merge(X_train.groupby(['num','date'])['THI'].mean().reset_index().rename(columns = {'THI':'mean_THI'}), on = ['num','date'], how = 'left')
    ## CDH
    cdhs = np.array([])
    for num in range(1,61,1):
        temp = X_train[X_train['num'] == num]
        cdh = CDH(temp['temperature'].values)
        cdhs = np.concatenate([cdhs, cdh])
    X_train['CDH'] = cdhs
    ## mean_CDH
    X_train = X_train.merge(X_train.groupby(['num','date'])['CDH'].mean().reset_index().rename(columns = {'CDH':'mean_CDH'}), on = ['num','date'], how = 'left')  
    ## date to numeric
    X_train['date_num'] = X_train['month'] + X_train['day']/31
    # split each building
    X_trains = [X_train[X_train.num == num] for num in range(1,61,1)]
    ## THI_category
    THI_nums = list(range(1,9))+list(range(10,61))
    for num in THI_nums:
        temp_df = X_trains[num-1]
        temp_df['THI_1'] = (temp_df['THI'] < 68).astype(int)
        temp_df['THI_2'] = ((temp_df['THI'] >= 68)&(temp_df['THI'] < 75)).astype(int)
        temp_df['THI_3'] = ((temp_df['THI'] >= 75)&(temp_df['THI'] < 80)).astype(int)
        temp_df['THI_4'] = (temp_df['THI'] >= 80).astype(int)
        X_trains[num-1] = temp_df
    ## feature engineering on each cluster
    for num in clust_to_num[0]:
        temp_df = X_trains[num-1]
        temp_df['working_time'] = ((temp_df['hour'].isin([8,9,10,11,12,13,14,15,16,17,18,19]))).astype(int)
        X_trains[num-1] = temp_df
    for num in clust_to_num[2]:
        temp_df = X_trains[num-1]
        temp_df['working_time'] = ((temp_df['hour'].isin([18,19,20,21,22]))).astype(int)
        X_trains[num-1] = temp_df
    for num in clust_to_num[3]:
        temp_df = X_trains[num-1]
        temp_df['working_time'] = ((temp_df['hour'].isin([8,9,10,11,12,13,14,15,16,17,18,19]))&(temp_df['weekday'].isin([0,1,2,3,4]))).astype(int)
        temp_df['lunch_time'] = ((temp_df['hour'].isin([11,12]))&(temp_df['weekday'].isin([0,1,2,3,4]))).astype(int)
        X_trains[num-1] = temp_df
    
    y_trains = [df['target'].values for df in X_trains]
    X_trains = [df.drop('target', axis = 1) for df in X_trains]
    # drop unnecessary columns
    X_trains = [df.drop(['num', 'datetime', 'hour', 'month', 'day', 'date', 'weekday','solar_flag','nelec_cool_flag'], axis=1).reset_index().drop('index', axis=1) for df in X_trains]
    # standard scaling on numerical features
    num_features = ['temperature', 'windspeed', 'humidity', 'precipitation', 'insolation','min_temperature','THI','mean_THI','CDH','mean_CDH','date_num']
    means = []
    stds = []
    for i, df in enumerate(X_trains):
        means.append(df.loc[:,num_features].mean(axis=0))
        stds.append(df.loc[:,num_features].std(axis=0))
        df.loc[:,num_features] = (df.loc[:,num_features] - df.loc[:,num_features].mean(axis=0))/df.loc[:,num_features].std(axis=0)
        X_trains[i] = df
    return X_trains, y_trains, means, stds

# data preprocessing function for testset
def test_preprocess(test, means, stds):
    X_train = test.copy()
    X_train = X_train.interpolate()
    
    X_train['datetime'] = pd.to_datetime(X_train['datetime'])
    X_train['hour'] = X_train['datetime'].dt.hour
    X_train['month'] = X_train['datetime'].dt.month
    X_train['day'] = X_train['datetime'].dt.day
    X_train['date'] = X_train['datetime'].dt.date
    X_train['weekday'] = X_train['datetime'].dt.weekday
    ## one hot encoding for weekday and hour
    X_train = pd.concat([X_train, pd.get_dummies(X_train['weekday'], prefix ='weekday')], axis=1)
    X_train = pd.concat([X_train, pd.get_dummies(X_train['hour'], prefix ='hour')], axis=1)
    ## daily minimum temperature
    X_train = X_train.merge(X_train.groupby(['num','date'])['temperature'].min().reset_index().rename(columns = {'temperature':'min_temperature'}), on = ['num','date'], how = 'left')
    ## THI
    X_train['THI'] = 9/5*X_train['temperature'] - 0.55*(1-X_train['humidity']/100)*(9/5*X_train['temperature']-26)+32
    ## mean_THI
    X_train = X_train.merge(X_train.groupby(['num','date'])['THI'].mean().reset_index().rename(columns = {'THI':'mean_THI'}), on = ['num','date'], how = 'left')
    ## CDH
    cdhs = np.array([])
    for num in range(1,61,1):
        temp = X_train[X_train['num'] == num]
        cdh = CDH(temp['temperature'].values)
        cdhs = np.concatenate([cdhs, cdh])
    X_train['CDH'] = cdhs
    ## mean_CDH
    X_train = X_train.merge(X_train.groupby(['num','date'])['CDH'].mean().reset_index().rename(columns = {'CDH':'mean_CDH'}), on = ['num','date'], how = 'left')  
    ## date to numeric
    X_train['date_num'] = X_train['month'] + X_train['day']/31
    
    X_trains = [X_train[X_train.num == num] for num in range(1,61,1)]
    ## THI_category
    THI_nums = list(range(1,9))+list(range(10,61))
    for num in THI_nums:
        temp_df = X_trains[num-1]
        temp_df['THI_1'] = (temp_df['THI'] < 68).astype(int)
        temp_df['THI_2'] = ((temp_df['THI'] >= 68)&(temp_df['THI'] < 75)).astype(int)
        temp_df['THI_3'] = ((temp_df['THI'] >= 75)&(temp_df['THI'] < 80)).astype(int)
        temp_df['THI_4'] = (temp_df['THI'] >= 80).astype(int)
        X_trains[num-1] = temp_df
    ## feature engineering on each cluster
    for num in clust_to_num[0]:
        temp_df = X_trains[num-1]
        temp_df['working_time'] = ((temp_df['hour'].isin([8,9,10,11,12,13,14,15,16,17,18,19]))).astype(int)
        X_trains[num-1] = temp_df
    for num in clust_to_num[2]:
        temp_df = X_trains[num-1]
        temp_df['working_time'] = ((temp_df['hour'].isin([18,19,20,21,22]))).astype(int)
        X_trains[num-1] = temp_df
    for num in clust_to_num[3]:
        temp_df = X_trains[num-1]
        temp_df['working_time'] = ((temp_df['hour'].isin([8,9,10,11,12,13,14,15,16,17,18,19]))&(temp_df['weekday'].isin([0,1,2,3,4]))).astype(int)
        temp_df['lunch_time'] = ((temp_df['hour'].isin([11,12]))&(temp_df['weekday'].isin([0,1,2,3,4]))).astype(int)
        X_trains[num-1] = temp_df
    # drop unnecessary columns
    X_trains = [df.drop(['num', 'datetime', 'hour', 'month', 'day', 'date', 'weekday','solar_flag','nelec_cool_flag'], axis=1).reset_index().drop('index', axis=1) for df in X_trains]
    # standard scaling on numerical features
    num_features = ['temperature', 'windspeed', 'humidity', 'precipitation', 'insolation','min_temperature','THI', 'mean_THI','CDH','mean_CDH', 'date_num']
    for i, (df, mean, std) in enumerate(zip(X_trains, means, stds)):
        df.loc[:,num_features] = (df.loc[:,num_features] - mean) / std
        X_trains[i] = df
    return X_trains
