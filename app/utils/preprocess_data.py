import pandas as pd
import numpy as np

def process_dataframe(encoder, scaler, df):
    if not df.empty:
        cashin_df = df[df['Type'] == 'CASHIN']
        cashout_df = df[df['Type'] == 'CASHOUT']
        p2p_df = df[df['Type'] == 'P2P']
        payment_df = df[df['Type'] == 'PAYMENT']
        cashin_to_cashout = set(cashin_df[cashin_df['Origine'].isin(cashout_df['Destination'])]['Origine'])
        p2p_to_cashin_cashout = set(p2p_df[p2p_df['Origine'].isin(cashin_df['Destination']) | p2p_df['Origine'].isin(cashout_df['Origine'])]['Origine'])
        payment_to_cashin_cashout = set(payment_df[payment_df['Origine'].isin(cashin_df['Origine']) | payment_df['Origine'].isin(cashout_df['Destination'])]['Origine'])
        combined_set = cashin_to_cashout.union(p2p_to_cashin_cashout, payment_to_cashin_cashout)
        combined_dataframes = {}
        for value in combined_set:
            combined_dataframes[value] = pd.concat([cashin_df[cashin_df['Origine'] == value],
                                            cashout_df[cashout_df['Destination'] == value],
                                            p2p_df[p2p_df['Origine'] == value],
                                            payment_df[payment_df['Origine'] == value]])
            combined_dataframes[value]['Timestamp'] = pd.to_datetime(combined_dataframes[value]['Timestamp'])
            combined_dataframes[value] = combined_dataframes[value].sort_values(by='Timestamp')
        combined_df = pd.concat(combined_dataframes.values())
        combined_df['Timestamp'] = pd.to_datetime(combined_df['Timestamp'])
        combined_df["Month"] = combined_df['Timestamp'].dt.month
        combined_df["Year"] = combined_df['Timestamp'].dt.year
        combined_df["Day"] = combined_df['Timestamp'].dt.day
        combined_df['Hour'] = combined_df['Timestamp'].dt.hour
        combined_df['Minute'] = combined_df['Timestamp'].dt.minute
        combined_df['Second'] = combined_df['Timestamp'].dt.second
        combined_df['Date'] = combined_df['Timestamp'].dt.date
        cashin_grouped = combined_df[combined_df['Type'] == 'CASHIN'].groupby(['Date','Origine']).agg({'Montant': ['count', 'mean']})
        cashin_grouped.columns = ['CashIn_Count', 'CashIn_AvgAmount']
        cashin_grouped.reset_index(inplace=True)
        combined_df = combined_df.merge(cashin_grouped, how='left', on=['Date', 'Origine'])
        cashout_grouped = combined_df[combined_df['Type'] == 'CASHOUT'].groupby(['Date', 'Destination']).agg({'Montant': ['count', 'mean']})
        cashout_grouped.columns = ['CashOut_Count', 'CashOut_AvgAmount']
        cashout_grouped.reset_index(inplace=True)
        combined_df = combined_df.merge(cashout_grouped, how='left', on=['Date', 'Destination'])
        p2p_grouped = combined_df[combined_df['Type'] == 'P2P'].groupby(['Date', 'Origine']).agg({'Montant': ['count', 'mean']})
        p2p_grouped.columns = ['P2P_Out_Count', 'P2P_Out_AvgAmount']
        p2p_grouped.reset_index(inplace=True)
        combined_df = combined_df.merge(p2p_grouped, how='left', on=['Date', 'Origine'])
        payment_grouped = combined_df[combined_df['Type'] == 'PAYMENT'].groupby(['Date', 'Origine']).agg({'Montant': ['count', 'mean']})
        payment_grouped.columns = ['PAYMENT_Out_Count', 'PAYMENT_Out_AvgAmount']
        payment_grouped.reset_index(inplace=True)
        combined_df = combined_df.merge(payment_grouped, how='left', on=['Date', 'Origine'])
        p2p_grouped = combined_df[combined_df['Type'] == 'P2P'].groupby(['Date', 'Destination']).agg({'Montant': ['count', 'mean']})
        p2p_grouped.columns = ['P2P_In_Count', 'P2P_In_AvgAmount']
        p2p_grouped.reset_index(inplace=True)
        combined_df = combined_df.merge(p2p_grouped, how='left', on=['Date', 'Destination'])
        payment_grouped = combined_df[combined_df['Type'] == 'PAYMENT'].groupby(['Date', 'Destination']).agg({'Montant': ['count', 'mean']})
        payment_grouped.columns = ['PAYMENT_In_Count', 'PAYMENT_In_AvgAmount']
        payment_grouped.reset_index(inplace=True)
        combined_df = combined_df.merge(payment_grouped, how='left', on=['Date', 'Destination'])
        combined_df[['CashIn_Count', 'CashIn_AvgAmount', 'CashOut_Count', 'CashOut_AvgAmount','P2P_Out_Count', 'P2P_Out_AvgAmount','PAYMENT_Out_Count', 'PAYMENT_Out_AvgAmount','P2P_In_Count', 'P2P_In_AvgAmount','PAYMENT_In_Count', 'PAYMENT_In_AvgAmount']] = combined_df[['CashIn_Count', 'CashIn_AvgAmount', 'CashOut_Count', 'CashOut_AvgAmount','P2P_Out_Count', 'P2P_Out_AvgAmount','PAYMENT_Out_Count', 'PAYMENT_Out_AvgAmount','P2P_In_Count', 'P2P_In_AvgAmount','PAYMENT_In_Count', 'PAYMENT_In_AvgAmount']].fillna(0)
        combined_df.drop(columns=["Date", "Timestamp"], inplace=True)
        combined_df['Type'] = encoder.transform(combined_df['Type'])
        columns_to_normalize = ['Montant','CashIn_AvgAmount','CashOut_AvgAmount','P2P_Out_AvgAmount','PAYMENT_Out_AvgAmount','P2P_In_AvgAmount','PAYMENT_In_AvgAmount']
        combined_df[columns_to_normalize] = scaler.transform(combined_df[columns_to_normalize])
        return combined_df
    return None
