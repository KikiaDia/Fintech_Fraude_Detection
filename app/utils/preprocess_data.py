import pandas as pd

def process_distro(encoder, scaler, df):
    if not df.empty:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df["Month"] = df['Timestamp'].dt.month
        df["Year"] = df['Timestamp'].dt.year
        df["Day"] = df['Timestamp'].dt.day
        df['Hour'] = df['Timestamp'].dt.hour
        df['Minute'] = df['Timestamp'].dt.minute
        df['Second'] = df['Timestamp'].dt.second
        df['Date'] = df['Timestamp'].dt.date
        cashin_grouped = df[df['Type'] == 'CASHIN'].groupby(['Date','Origine']).agg({'Montant': ['count', 'mean']})
        cashin_grouped.columns = ['CashIn_Count', 'CashIn_AvgAmount']
        cashin_grouped.reset_index(inplace=True)
        df = df.merge(cashin_grouped, how='left', on=['Date', 'Origine'])
        cashout_grouped = df[df['Type'] == 'CASHOUT'].groupby(['Date', 'Destination']).agg({'Montant': ['count', 'mean']})
        cashout_grouped.columns = ['CashOut_Count', 'CashOut_AvgAmount']
        cashout_grouped.reset_index(inplace=True)
        df = df.merge(cashout_grouped, how='left', on=['Date', 'Destination'])
        df[['CashIn_Count', 'CashIn_AvgAmount', 'CashOut_Count', 'CashOut_AvgAmount']] = df[['CashIn_Count', 'CashIn_AvgAmount', 'CashOut_Count', 'CashOut_AvgAmount']].fillna(0)
        df.drop(columns=["Date", "Timestamp"], inplace=True)
        df['Type'] = encoder.transform(df['Type'])
        columns_to_normalize = ['Montant','CashIn_AvgAmount','CashOut_AvgAmount']
        df[columns_to_normalize] = scaler.transform(df[columns_to_normalize])
        return df
    return None


def process_client(encoder, scaler, df):
    if not df.empty:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df['Hour'] = df['Timestamp'].dt.hour
        df['Minute'] = df['Timestamp'].dt.minute
        df['Second'] = df['Timestamp'].dt.second
        df["Month"] = df['Timestamp'].dt.month
        df["Year"] = df['Timestamp'].dt.year
        df["Day"] = df['Timestamp'].dt.day
        df.drop(columns=["Timestamp"], inplace=True)
        df['Type'] = encoder.transform(df['Type'])
        columns_to_normalize = ['Montant']
        df[columns_to_normalize] = scaler.transform(df[columns_to_normalize])
        return df
    return None
