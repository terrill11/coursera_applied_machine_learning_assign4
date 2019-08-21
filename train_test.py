import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

df = pd.read_csv('train.csv', encoding='windows-1252', low_memory=False)


def state_cols(df):
    data = df.copy()
    data.dropna(subset=['compliance'], inplace=True)
    data = data.loc[:, ['state','compliance']]
    data = pd.get_dummies(data)
 
    state_df = data.groupby(['compliance']).sum().transpose()

    state_df = state_df[0:].reset_index()
    state_df.columns = ['state', '0', '1']
    state_df['0_pc_total'] = state_df['0']/(state_df['0']+state_df['1'])
    state_df['1_pc_total'] = state_df['1']/(state_df['0']+state_df['1'])
    state_df['count'] = state_df['0']+state_df['1']

    final = state_df[((state_df['0_pc_total'] > 0.98) | (state_df['1_pc_total'] > 0.1)) & (state_df['count'] > 25)]

    #return list(final['state'])
    return list(state_df[~state_df['state'].isin(list(final['state']))]['state'])
    
#print(state_cols(df))

def cleanup(train_df):
    extra_cols = ['payment_amount', 'payment_date', 'payment_status',
                  'balance_due', 'collection_status', 'compliance_detail']
    useless_cols = ['violation_street_name', 'mailing_address_str_name', 'country', 'violation_description',
                    'inspector_name', 'violator_name', 'grafitti_status', 'non_us_str_code', 'violation_street_number']

    train_df.drop(extra_cols, axis=1, inplace=True)
    train_df.drop(useless_cols, axis=1, inplace=True)
    train_df.dropna(subset=['compliance'], inplace=True)
    
    # Dropping location columns for the sake of simplicity.
    # Dropping violation_code for now and get baseline results.
    # May add it back into columns if needed.
    train_df.drop(['ticket_issued_date', 'hearing_date', 'city',
                   'zip_code', 'violation_code', 'state'], axis=1, inplace=True)
                    #'state'
    # Dropping remaining location columns.
    # Added benefit of removing columns with high number of null values to simplify further
    train_df.drop(['violation_zip_code', 'mailing_address_str_number'], axis=1, inplace=True)

    # dropping more columns before dummies (remaining columns)
    # ['ticket_id', 'agency_name', 'state', 'disposition', 'fine_amount', 'admin_fee', 'state_fee', 'late_fee', 'discount_amount',\
    #  'clean_up_cost', 'judgment_amount', 'compliance', 'state']

    #return train_df.corr()['compliance'].sort_values()
    #train_df.drop(['fine_amount', 'judgment_amount', 'late_fee', 'state_fee', 'clean_up_cost', 'admin_fee'], axis=1, inplace=True)
    #return train_df.columns

    #getting dummies
    train_df = pd.get_dummies(train_df)
    # unwanted_states = state_cols(df)
    # train_df.drop(unwanted_states, axis=1, inplace=True)

    #return train_df.corr()['compliance'].sort_values()
    #test data
    train_labels = train_df['compliance']
    #ticket_ids = train_df['ticket_id']
    train_df.drop(columns=['compliance', 'ticket_id'], inplace=True)

    imp_feats = ['disposition_Responsible by Default', 'late_fee', 'disposition_Responsible by Admission',\
                 'disposition_Responsible by Determination', 'discount_amount', 'disposition_Responsible (Fine Waived) by Deter']
    train_df.drop(columns=['admin_fee','state_fee','clean_up_cost'], inplace=True)
    train_df.dropna(inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(train_df[imp_feats], train_labels, random_state=0)
    #X_train, X_test, y_train, y_test = train_test_split(train_df, train_labels, random_state=0)

    #return X_train.corr()['compliance'].sort_values()
    #return X_train.columns
    
    return X_train, X_test, y_train, y_test


def train_model():
    X_train, X_test, y_train, y_test = cleanup(df)

    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler()
    # X_train = scaler.fit_transform(X_train)
    # X_test = scaler.fit_transform(X_test)

    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression().fit(X_train, y_train)
    lr_predict = lr.predict(X_test)
    lr_score = roc_auc_score(y_test, lr_predict)

    from sklearn.svm import LinearSVC
    l_svc = LinearSVC().fit(X_train, y_train)
    l_svc_predict = l_svc.predict(X_test)
    l_svc_score = roc_auc_score(y_test, l_svc_predict)

    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier().fit(X_train, y_train)
    dtc_predict = dtc.predict(X_test)
    dtc_score = roc_auc_score(y_test, dtc_predict)

    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=10, random_state=0).fit(X_train, y_train)
    rfc_predict = rfc.predict(X_test)
    rfc_score = roc_auc_score(y_test, rfc_predict)

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier().fit(X_train, y_train)
    knn_predict = knn.predict(X_test)
    knn_score = roc_auc_score(y_test, knn_predict)


    #return l_svc_score
    return lr_score, l_svc_score, dtc_score, rfc_score, knn_score


#print(cleanup(df))
print(train_model())


# importan features?
# ['late_fee', 'disposition_Responsible by Admission', 'discount_amount', 'disposition_Responsible by Default',\
#     'judgment_amount', 'fine_amount', 'agency_name_Buildings, Safety Engineering & Env Department',\
#     'agency_name_Detroit Police Department', 'agency_name_Department of Public Works', 'agency_name_Health Department',\
#     'disposition_Responsible by Determination', 'agency_name_Neighborhood City Halls', 'admin_fee', 'state_fee',\
#     'clean_up_cost', 'disposition_Responsible (Fine Waived) by Deter']
# drop admin_fee, state_fee, clean_up_cost


# things to do:
# - feature engineering
# - min/max scaling with mean/median
# - correlation
# - Univariate Selection - https://towardsdatascience.com/feature-selection-techniques-in-machine-learning-with-python-f24e7da3f36e
