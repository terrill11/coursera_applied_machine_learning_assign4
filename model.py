import pandas as pd
import numpy as np

# df = pd.read_csv('train.csv', encoding='windows-1252', low_memory=False)
# print(df[['compliance','compliance_detail']])

def blight_model():

    # Your code here
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.tree import DecisionTreeClassifier

    train_df = pd.read_csv('train.csv', encoding='windows-1252', low_memory=False)
    test_df = pd.read_csv('test.csv')

    #ticket_ids
    train_ticket_id, test_ticket_id = train_df['ticket_id'], test_df['ticket_id']

    #cleaning train sets
    extra_cols = ['payment_amount','payment_date','payment_status','balance_due','collection_status','compliance_detail']
    useless_cols = ['ticket_id', 'violation_street_name', 'mailing_address_str_name', 'country', 'violation_description',\
                    'inspector_name', 'violator_name', 'grafitti_status', 'non_us_str_code']
    train_df.drop(extra_cols, axis=1, inplace=True)
    train_df.drop(useless_cols, axis=1, inplace=True)
    train_df.dropna(subset=['compliance'], inplace=True)

    #cleaning test_sets
    test_df.drop(useless_cols, axis=1, inplace=True)

    return train_df.select_dtypes('object').apply(pd.Series.nunique, axis=0)
    # for i in train_df.columns:
    #     print(i)
    #     print(train_df[i].head())

    #need to clean test as well
    #print(test_df['non_us_str_code'])
    #pass


    # X_train, y_train, X_test = train_df.iloc[:,1:-1], train_df['compliance'], test_df
    # model = DecisionTreeClassifier().fit(X_train, y_train)
    # prediction = model.predict(X_test)
    # return prediction


print(blight_model())

####NOTES

#change categoricals to ints
#'agency_name', 'inspector_name', 'violator_name', disposition

#addresses ints
#'violation_street_number', 'violation_zip_code', 'mailing_address_str_number', 'zip_code'
#check 3rd row zip code

#address str
#city, state, country

#idk
#non_us_str_code, grafitti_status

#datetimes
#ticket_issued_date, hearing_date,

#categorical ints
#violation_code

#continuous ints
#fine_amount, admin_fee, state_fee, late_fee, discount_amount, clean_up_cost, judgement_amount