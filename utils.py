import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
from io import StringIO
from Bio import Phylo
from lib.gcforest.gcforest import GCForest
from similarity_network_fusion import micobiome_SNF
import sys

from sklearn.utils import resample

def load():
    '''
    load the training data and adj matrix
    :return:
    '''
    #### construct the network
    sparcc = pd.read_csv('sparcc/sparcc_otu_adj.txt', sep='\t', index_col=0).values
    mic = pd.read_csv('MIC/mic_otu_adj.txt', sep='\t', index_col=0).values
    # spearman = pd.read_csv('MIC/spearman_otu_adj.txt', sep='\t', index_col=0).values
    spieceasi = pd.read_csv('Spieceasi/spieceasi_adj_out.txt', sep=',', index_col=0).values

    identity_mat = np.identity(n=spieceasi.shape[0], dtype=np.int)
    spieceasi = spieceasi + identity_mat

    merged = sparcc + mic + spieceasi
    merged[merged > 1] = 1

    # U = union_Adjac_matrix(left, right)
    X = pd.read_csv('output/ibd_otus.txt', sep='\t', index_col=0, header=0)
    X = shuffle(X)
    y = X['label'].values
    X = X.drop(columns=['label'])
    # print(check_symmetric(spieceasi))
    merged_flat_list = [item for sublist in merged.tolist() for item in sublist]

    sparcc_flat_list = [item for sublist in sparcc.tolist() for item in sublist]

    spieceasi_flat_list = [item for sublist in spieceasi.tolist() for item in sublist]

    print("merged=" + str(merged_flat_list.count(1)), "sparcc=" + str(sparcc_flat_list.count(1)),
          "spieceasi=" + str(spieceasi_flat_list.count(1)))

    # print(X.values.max())
    # print(np.min(X.values[np.nonzero(X.values)]))
    # X[X < 1e-5] = 0

    # fused_networks = micobiome_SNF.build_fused_network().values
    # fused_networks = micobiome_SNF.build_all_fused_network().values

    return X, y, sparcc, merged


def apply_column_filter(row):
    taxa = row[1]
    taxonomy = taxa.rfind(';')
    taxa = taxa[:taxonomy]
    return taxa

def matthews_correlation(y_true, y_pred):
    '''Calculates the Matthews correlation coefficient measure for quality
    of binary classification problems.
    '''
    y_pred_pos = np.round(np.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = np.round(np.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = np.sum(y_pos * y_pred_pos)
    tn = np.sum(y_neg * y_pred_neg)

    fp = np.sum(y_neg * y_pred_pos)
    fn = np.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + sys.float_info.epsilon)




