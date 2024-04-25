import os
import random
import pandas as pd
import numpy as np
import wfdb
import ast
from torch.utils.data import Dataset
# from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer


def load_raw_data(df, sampling_rate, path):
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data


class PTBXL_ECG_Text(Dataset):
    """
    PTB-XL dataset for ECG-Text SSL pretraining
    """
    def __init__(self, path, sampling_rate=100, train=True):
        self.path = path
        self.sampling_rate = sampling_rate
        self.train = train

        # preprocess and save the data into .npy files
        self.preprocess()

        # load and convert annotation data
        self.X, self.y, self.report = self.load_data()

    def preprocess(self):
        if os.path.exists(self.path+'ecg_train.npy') and os.path.exists(self.path+'report_train.npy') and os.path.exists(self.path+'label_train.npy') and \
            os.path.exists(self.path+'ecg_test.npy') and os.path.exists(self.path+'report_test.npy') and os.path.exists(self.path+'label_test.npy'):
            print("Preprocessed data found, skipping preprocessing...")
        else:
            print("Preprocessed data not found, preprocessing...")
            Y = pd.read_csv(self.path+'ptbxl_database.csv', index_col='ecg_id')
            Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

            # Load raw signal data, in the numpy format
            X = load_raw_data(Y, self.sampling_rate, self.path)

            # Load scp_statements.csv for diagnostic aggregation
            agg_df = pd.read_csv(self.path+'scp_statements.csv', index_col=0)
            agg_df = agg_df[agg_df.diagnostic == 1]

            def aggregate_diagnostic(y_dic):
                tmp = []
                for key in y_dic.keys():
                    if key in agg_df.index:
                        tmp.append(agg_df.loc[key].diagnostic_class)
                return list(set(tmp))

            # Apply diagnostic superclass
            Y['diagnostic_superclass'] = Y.scp_codes.apply(aggregate_diagnostic)

            # Split data into train and test
            test_fold = 10  # as suggested by PTB-XL dataset
            # Train split
            ecg_train = X[np.where(Y.strat_fold != test_fold)]
            diagnostic_train = Y[(Y.strat_fold != test_fold)].diagnostic_superclass
            report_train = Y[(Y.strat_fold != test_fold)].report

            label_train = []
            for item in diagnostic_train:
                try:
                    if item[0] == 'NORM':
                        label_train.append(0)
                    else:
                        label_train.append(1)
                except:
                    pass
            label_train = np.array(label_train)

            # Test split
            ecg_test = X[np.where(Y.strat_fold == test_fold)]
            diagnostic_test = Y[Y.strat_fold == test_fold].diagnostic_superclass
            report_test = Y[Y.strat_fold == test_fold].report

            label_test = []
            for item in diagnostic_test:
                try:
                    if item[0] == 'NORM':
                        label_test.append(0)
                    else:
                        label_test.append(1)
                except:
                    pass
            label_test = np.array(label_test)

            # convert the data into numpy format
            np.save(self.path+'ecg_train.npy', ecg_train)
            np.save(self.path+'label_train.npy', label_train)
            np.save(self.path+'report_train.npy', report_train)
            np.save(self.path+'ecg_test.npy', ecg_test)
            np.save(self.path+'label_test.npy', label_test)
            np.save(self.path+'report_test.npy', report_test)

    def load_data(self):
        # load preprocessed data
        split = "train" if self.train else "test"
        assert os.path.exists(self.path+'ecg_'+split+'.npy') and os.path.exists(self.path+'report_'+split+'.npy') and \
            os.path.exists(self.path+'label_'+split+'.npy'), "Preprocessed data not found, please preprocess the data first"
        X = np.load(self.path+'ecg_'+split+'.npy', allow_pickle=True)
        y = np.load(self.path+'label_'+split+'.npy', allow_pickle=True)
        report = np.load(self.path+'report_'+split+'.npy', allow_pickle=True)

        return X, y, report

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.report[idx]


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    path = '/home/xu0005/Desktop/ECG_data/ptb-xl/1.0.3/'
    dataset = PTBXL_ECG_Text(path, train=True)
    for i in range(10):
        print(dataset[i][1])


    dataloader = DataLoader(
        dataset, 
        batch_size=32, 
        shuffle=True,
        num_workers=2
    )

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        ecg, report = batch[0], batch[1]
        print(ecg.shape)
        print(len(report))

        break