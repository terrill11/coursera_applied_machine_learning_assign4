import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


def train_model():
    train_df = pd.read_csv('train.csv', encoding='windows-1252', low_memory=False)

    extra_cols = ['payment_amount', 'payment_date', 'payment_status',\
                  'balance_due', 'collection_status', 'compliance_detail']
    useless_cols = ['violation_street_name', 'mailing_address_str_name', 'country', 'violation_description',\
                    'inspector_name', 'violator_name', 'grafitti_status', 'non_us_str_code']
    train_df.drop(extra_cols, axis=1, inplace=True)
    train_df.drop(useless_cols, axis=1, inplace=True)
    train_df.dropna(subset=['compliance'], inplace=True)

    # Dropping location columns for the sake of simplicity. 
    # Dropping violation_code for now and get baseline results. 
    # May add it back into columns if needed.
    train_df.drop(['ticket_issued_date', 'hearing_date', 'city',\
                   'zip_code', 'violation_code', 'state'], axis=1, inplace=True)

    # Dropping remaining location columns. 
    # Added benefit of removing columns with high number of null values to simplify further
    train_df.drop(['violation_zip_code', 'mailing_address_str_number'], axis=1, inplace=True)

    #getting dummies
    train_df = pd.get_dummies(train_df)

    #test data
    train_labels = train_df['compliance']
    ticket_ids = train_df['ticket_id']
    train_df.drop(columns=['compliance','ticket_id'], inplace=True)

    X_train, X_test, y_train, y_test = train_test_split(train_df, train_labels, random_state=0)

    from sklearn.svm import LinearSVC

    l_svc = LinearSVC().fit(X_train, y_train)
    l_svc_predict = l_svc.predict(X_test)
    l_svc_score = roc_auc_score(y_test, l_svc_predict)
    return l_svc_score

    # from sklearn.linear_model import LogisticRegression

    # # lr = LinearSVC().fit(X_train, y_train)
    # # lr_predict = lr.predict(X_test)
    # # lr_score = roc_auc_score(y_test, lr_predict)
    # grid_lr = GridSearchVC(LogisticRegression(), param_grid=grid_values_lr, scoring='roc_auc')
    # grid_lr.fit(X_train, y_train)

    #return l_svc_score

print(train_model())

# LogisticRegression score: 0.547
# LinearSVC score: 0.603
