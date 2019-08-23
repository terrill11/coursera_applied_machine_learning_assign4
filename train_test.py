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

    final = state_df[(state_df['1_pc_total'] > 0.15) & (state_df['count'] > 25)]
    # return final
    # return list(final['state'])
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
    
    # more useless columns
    more_useless_cols = ['ticket_issued_date', 'hearing_date', 'city', 'zip_code', 'violation_code', 'admin_fee',
                         'state_fee', 'clean_up_cost', 'mailing_address_str_number', 'violation_code', 'violation_zip_code']
    train_df.drop(more_useless_cols, axis=1, inplace=True)

    #getting dummies
    train_df = pd.get_dummies(train_df)

    #add states in features
    unwanted_states = state_cols(df)
    train_df.drop(unwanted_states, axis=1, inplace=True)

    imp_feats = ['fine_amount', 'late_fee', 'discount_amount', 'judgment_amount', 'state_KS', 'state_NB', 'state_RI',\
                'disposition_Responsible (Fine Waived) by Deter', 'disposition_Responsible by Admission',\
                'disposition_Responsible by Determination']
    train_df.dropna(inplace=True)

    train_labels = train_df['compliance']
    X_train, X_test, y_train, y_test = train_test_split(train_df[imp_feats], train_labels, random_state=0)

    return X_train, X_test, y_train, y_test


def train_model():
    X_train, X_test, y_train, y_test = cleanup(df)

    #from sklearn.metrics import confusion_matrix

    from sklearn.tree import DecisionTreeClassifier
    dtc = DecisionTreeClassifier().fit(X_train, y_train)
    dtc_predict = dtc.predict(X_test)
    dtc_score = roc_auc_score(y_test, dtc_predict)
    #print('DecisionTreeClassifier:\n', confusion_matrix(y_test, dtc_predict))

    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(random_state=0).fit(X_train, y_train)
    rfc_predict = rfc.predict(X_test)
    rfc_score = roc_auc_score(y_test, rfc_predict)
    #print('RandomForestClassifier:\n', confusion_matrix(y_test, rfc_predict))

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier().fit(X_train, y_train)
    knn_predict = knn.predict(X_test)
    knn_score = roc_auc_score(y_test, knn_predict)
    #print('KNeighborsClassfier:\n', confusion_matrix(y_test, knn_predict))

    
    return dtc_score, rfc_score, knn_score


def evaulate_targets(df):
    final = df.groupby(['compliance']).std()

    for i in final.columns:
        print(i)
        print(final[i])
        print(final[i].loc[0]/final[i].loc[1] - 1)
        print('\n')

    pass

#print(evaulate_targets(cleanup(df)))

#print(cleanup(df))
print(train_model())
