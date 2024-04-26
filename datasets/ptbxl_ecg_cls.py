import os
import random
import pandas as pd
import numpy as np
import wfdb
import pickle
import ast
from tqdm import tqdm
from scipy.signal import resample
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer

def load_dataset(path, sampling_rate):
    # load and convert annotation data
    Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Load raw signal data
    X = load_raw_data_ptbxl(Y, sampling_rate, path)

    return X, Y


def load_raw_data_ptbxl(df, sampling_rate, path):
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path+'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_lr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path+'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_hr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
    elif sampling_rate < 500:
        if os.path.exists(path + f'raw{sampling_rate}.npy'):
            data = np.load(path + f'raw{sampling_rate}.npy', allow_pickle=True)
        else:
            # load the data at 500Hz first
            if os.path.exists(path + 'raw500.npy'):
                data = np.load(path + 'raw500.npy', allow_pickle=True)
            else:
                data = [wfdb.rdsamp(path + f) for f in tqdm(df.filename_hr)]
                data = np.array([signal for signal, meta in data])
                pickle.dump(data, open(path + 'raw500.npy', 'wb'), protocol=4)
        
            # downsample the data to the target sampling rate
            downsampled_data = []
            for signal in tqdm(data, desc=f'Downsampling to {sampling_rate}Hz'):
                # Calculate the downsampling factor
                downsampling_factor = 500 / sampling_rate
            
                # Resample the signal
                downsampled_signal = resample(signal, int(len(signal) / downsampling_factor))
            
                # Append the downsampled signal to the list
                downsampled_data.append(downsampled_signal)
        
            # Convert the list of downsampled signals to a NumPy array
            data = np.array(downsampled_data)
        
            # Optionally, you can save the downsampled data for future use
            np.save(path + f'raw{sampling_rate}.npy', data)

    return data


def compute_label_aggregations(df, folder, ctype):

    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))

    agg_df = pd.read_csv(folder+'scp_statements.csv', index_col=0)

    if ctype in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:

        def aggregate_all_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))

        def aggregate_subdiagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_subclass
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        def aggregate_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_class
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        diag_agg_df = agg_df[agg_df.diagnostic == 1.0]
        if ctype == 'diagnostic':
            df['diagnostic'] = df.scp_codes.apply(aggregate_all_diagnostic)
            df['diagnostic_len'] = df.diagnostic.apply(lambda x: len(x))
        elif ctype == 'subdiagnostic':
            df['subdiagnostic'] = df.scp_codes.apply(aggregate_subdiagnostic)
            df['subdiagnostic_len'] = df.subdiagnostic.apply(lambda x: len(x))
        elif ctype == 'superdiagnostic':
            df['superdiagnostic'] = df.scp_codes.apply(aggregate_diagnostic)
            df['superdiagnostic_len'] = df.superdiagnostic.apply(lambda x: len(x))

    elif ctype == 'form':
        form_agg_df = agg_df[agg_df.form == 1.0]

        def aggregate_form(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in form_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['form'] = df.scp_codes.apply(aggregate_form)
        df['form_len'] = df.form.apply(lambda x: len(x))

    elif ctype == 'rhythm':
        rhythm_agg_df = agg_df[agg_df.rhythm == 1.0]

        def aggregate_rhythm(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in rhythm_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['rhythm'] = df.scp_codes.apply(aggregate_rhythm)
        df['rhythm_len'] = df.rhythm.apply(lambda x: len(x))

    elif ctype == 'all':
        df['all_scp'] = df.scp_codes.apply(lambda x: list(set(x.keys())))

    return df


def select_data(XX,YY, ctype, outputfolder, min_samples=0):
    # convert multilabel to multi-hot
    mlb = MultiLabelBinarizer()

    if ctype == 'diagnostic':
        X = XX[YY.diagnostic_len > 0]
        Y = YY[YY.diagnostic_len > 0]
        mlb.fit(Y.diagnostic.values)
        y = mlb.transform(Y.diagnostic.values)
    elif ctype == 'subdiagnostic':
        counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.subdiagnostic = YY.subdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['subdiagnostic_len'] = YY.subdiagnostic.apply(lambda x: len(x))
        X = XX[YY.subdiagnostic_len > 0]
        Y = YY[YY.subdiagnostic_len > 0]
        mlb.fit(Y.subdiagnostic.values)
        y = mlb.transform(Y.subdiagnostic.values)
    elif ctype == 'superdiagnostic':
        counts = pd.Series(np.concatenate(YY.superdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.superdiagnostic = YY.superdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['superdiagnostic_len'] = YY.superdiagnostic.apply(lambda x: len(x))
        X = XX[YY.superdiagnostic_len > 0]
        Y = YY[YY.superdiagnostic_len > 0]
        mlb.fit(Y.superdiagnostic.values)
        y = mlb.transform(Y.superdiagnostic.values)
    elif ctype == 'form':
        # filter
        counts = pd.Series(np.concatenate(YY.form.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.form = YY.form.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['form_len'] = YY.form.apply(lambda x: len(x))
        # select
        X = XX[YY.form_len > 0]
        Y = YY[YY.form_len > 0]
        mlb.fit(Y.form.values)
        y = mlb.transform(Y.form.values)
    elif ctype == 'rhythm':
        # filter 
        counts = pd.Series(np.concatenate(YY.rhythm.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.rhythm = YY.rhythm.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['rhythm_len'] = YY.rhythm.apply(lambda x: len(x))
        # select
        X = XX[YY.rhythm_len > 0]
        Y = YY[YY.rhythm_len > 0]
        mlb.fit(Y.rhythm.values)
        y = mlb.transform(Y.rhythm.values)
    elif ctype == 'all':
        # filter 
        counts = pd.Series(np.concatenate(YY.all_scp.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.all_scp = YY.all_scp.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['all_scp_len'] = YY.all_scp.apply(lambda x: len(x))
        # select
        X = XX[YY.all_scp_len > 0]
        Y = YY[YY.all_scp_len > 0]
        mlb.fit(Y.all_scp.values)
        y = mlb.transform(Y.all_scp.values)
    else:
        pass

    # save LabelBinarizer
    with open(outputfolder+'mlb.pkl', 'wb') as tokenizer:
        pickle.dump(mlb, tokenizer)

    return X, Y, y, mlb


def preprocess_signals(X_train, X_validation, X_test, outputfolder):
    # Standardize data such that mean 0 and variance 1
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:,np.newaxis].astype(float))
    
    # Save Standardizer data
    with open(outputfolder+'standard_scaler.pkl', 'wb') as ss_file:
        pickle.dump(ss, ss_file)

    return apply_standardizer(X_train, ss), apply_standardizer(X_validation, ss), apply_standardizer(X_test, ss)


def apply_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:,np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp


class PTBXL_CLS(Dataset):
    """
    PTB-XL dataset for ECG classification task
    """
    def __init__(self, path, task="diagnostic", sampling_rate=100, split="train", train_fold=8, val_fold=9, test_fold=10):
        self.datafolder = path
        self.task = task
        self.sampling_rate = sampling_rate
        self.split = split

        # recommended by PTB-XL dataset
        # use folds 1-8 as training set, fold 9 as validation set and fold 10 as test set.
        self.train_fold = train_fold
        self.val_fold = val_fold
        self.test_fold = test_fold

        # preprocess and save the data into .npy files
        self.preprocess()

        # load data
        self.load_data()

    def preprocess(self):
        if not os.path.exists(self.datafolder + self.task):
            os.makedirs(self.datafolder + self.task)

        # if preprocessed data found, then just return
        if os.path.exists(self.datafolder + self.task + '/X_train_' + str(self.sampling_rate) + '.npy') and \
            os.path.exists(self.datafolder + self.task + '/y_train_' + str(self.sampling_rate) + '.npy') and \
            os.path.exists(self.datafolder + self.task + '/X_val_' + str(self.sampling_rate) + '.npy') and \
            os.path.exists(self.datafolder + self.task + '/y_val_' + str(self.sampling_rate) + '.npy') and \
            os.path.exists(self.datafolder + self.task + '/X_test_' + str(self.sampling_rate) + '.npy') and \
            os.path.exists(self.datafolder + self.task + '/y_test_' + str(self.sampling_rate) + '.npy'):
            print("Preprocessed data found, skipping preprocessing...")
            return

        # Load PTB-XL data
        print("Loading PTB-XL data...")
        self.data, self.raw_labels = load_dataset(self.datafolder, self.sampling_rate)

        # Preprocess label data
        print("Preprocessing label data...")
        self.labels = compute_label_aggregations(self.raw_labels, self.datafolder, self.task)

        # Select relevant data and convert to one-hot
        print("Selecting relevant data and converting to one-hot...")
        self.data, self.labels, self.Y, _ = select_data(self.data, self.labels, self.task, self.datafolder + self.task + '/')
        self.input_shape = self.data[0].shape
        
        print("Splitting data...")
        # 10th fold for testing
        self.X_test = self.data[self.labels.strat_fold == self.test_fold]
        self.y_test = self.Y[self.labels.strat_fold == self.test_fold]
        # 9th fold for validation
        self.X_val = self.data[self.labels.strat_fold == self.val_fold]
        self.y_val = self.Y[self.labels.strat_fold == self.val_fold]
        # rest for training
        self.X_train = self.data[self.labels.strat_fold <= self.train_fold]
        self.y_train = self.Y[self.labels.strat_fold <= self.train_fold]

        # Preprocess signal data
        print("Preprocessing signals...")
        self.X_train, self.X_val, self.X_test = preprocess_signals(self.X_train, self.X_val, self.X_test, self.datafolder + self.task + '/')
        self.n_classes = self.y_train.shape[1]

        # Save preprocessed data
        print("Saving preprocessed data...")
        np.save(self.datafolder + self.task + '/X_train_' + str(self.sampling_rate) + '.npy', self.X_train)
        np.save(self.datafolder + self.task + '/y_train_' + str(self.sampling_rate) + '.npy', self.y_train)
        np.save(self.datafolder + self.task + '/X_val_' + str(self.sampling_rate) + '.npy', self.X_val)
        np.save(self.datafolder + self.task + '/y_val_' + str(self.sampling_rate) + '.npy', self.y_val)
        np.save(self.datafolder + self.task + '/X_test_' + str(self.sampling_rate) + '.npy', self.X_test)
        np.save(self.datafolder + self.task + '/y_test_' + str(self.sampling_rate) + '.npy', self.y_test)

    def load_data(self):
        self.X = np.load(self.datafolder + self.task + '/X_' + self.split + '_' + str(self.sampling_rate) + '.npy', allow_pickle=True)
        self.y = np.load(self.datafolder + self.task + '/y_' + self.split + '_' + str(self.sampling_rate) + '.npy', allow_pickle=True)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx].astype(np.float32), self.y[idx].astype(np.float32)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    TASK = "form"     # 'diagnostic', 'subdiagnostic', 'superdiagnostic', 'form', 'rhythm', 'all'
    SAMPLING_RATE = 200

    dataset = PTBXL_CLS(
        path="/home/xu0005/Desktop/ECG_data/ptb-xl/1.0.3/",
        task=TASK,
        sampling_rate=SAMPLING_RATE,
        split="train",
    )

    print(len(dataset))


    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=True,
        num_workers=1
    )

    for batch_idx, batch in enumerate(tqdm(dataloader)):
        ecg, label = batch[0], batch[1]
        print(ecg.shape)
        print(label.shape)

        if batch_idx == 10:
            break