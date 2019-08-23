import pandas as pd
import numpy as np


def blight_model():
    
    # Your code here
    train_df = pd.read_csv('train.csv', encoding='windows-1252')
    test_df = pd.read_csv('test.csv')
    
    train_df.dropna(subset=['compliance'], inplace=True)
    
    extra_cols = ['payment_amount', 'payment_date', 'payment_status', 'balance_due', 'collection_status', 'compliance_detail']
    useless_cols = ['violation_street_name', 'mailing_address_str_name', 'country', 'violation_description',\
                    'inspector_name', 'violator_name', 'grafitti_status', 'non_us_str_code', 'violation_street_number']
    
    train_df.drop(extra_cols, axis=1, inplace=True)
    train_df.drop(useless_cols, axis=1, inplace=True)
    train_df.dropna(subset=['compliance'], inplace=True)
    test_df.drop(useless_cols, axis=1, inplace=True)
    
    more_useless_cols = ['ticket_issued_date', 'hearing_date', 'city', 'zip_code', 'violation_code', 'violation_zip_code',\
                        'mailing_address_str_number']
    train_df.drop(more_useless_cols, axis=1, inplace=True)
    test_df.drop(more_useless_cols, axis=1, inplace=True)
    
    train_df = pd.get_dummies(train_df)
    test_df = pd.get_dummies(test_df)

    
    test_ticket_id = test_df['ticket_id']
    y_train = train_df['compliance']
    imp_feats = ['fine_amount', 'late_fee', 'discount_amount', 'judgment_amount', 'state_KS', 'state_NB', 'state_RI',\
                 'disposition_Responsible (Fine Waived) by Deter', 'disposition_Responsible by Admission',\
                 'disposition_Responsible by Determination']
    train_df = train_df[imp_feats]
    train_df, test_df = train_df.align(test_df, join='inner', axis=1)

    train_df.dropna(inplace=True)
    
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(random_state=0).fit(train_df, y_train)
    predict = rfc.predict_proba(test_df)[:,1]

    results = pd.DataFrame()
    results['ticket_id'] = test_ticket_id#.iloc[:,:]
    results['target'] = predict
    results.set_index('ticket_id', inplace=True)
    
    return results

print(blight_model())