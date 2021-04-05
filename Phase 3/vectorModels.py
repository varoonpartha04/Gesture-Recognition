from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
import pandas as pd

def convert_dataframe_to_dict(transformed_data, index, semantic_features):
    transformed_data = pd.DataFrame(transformed_data, index = index, columns = semantic_features)
    return transformed_data.to_dict('index')

def sort_non_incr(a):
    for each_dict in a:
        b = {k: v for k, v in sorted(a[each_dict].items(), key=lambda item: item[1], reverse=True)}
        a[each_dict] = b
    return a

# pca code
def pca(k, input_data, semantic_features):
    pca = PCA(n_components = k)
    input_pca = pca.fit_transform(input_data)
    input_pca = convert_dataframe_to_dict(input_pca, input_data.index, semantic_features)
    words_and_scores_pca =  pd.DataFrame(pca.components_,columns=input_data.columns,index = semantic_features)
    pca_output = words_and_scores_pca.T.to_dict()
    sorted_pca = sort_non_incr(pca_output)
    return sorted_pca, input_pca

# svd code 
def svd(k, input_data, semantic_features):
    svd = TruncatedSVD(n_components = k)
    input_svd = svd.fit_transform(input_data)
    input_svd = convert_dataframe_to_dict(input_svd, input_data.index, semantic_features)
    words_and_scores_svd =  pd.DataFrame(svd.components_,columns=input_data.columns,index = semantic_features)
    svd_output = words_and_scores_svd.T.to_dict()
    sorted_svd = sort_non_incr(svd_output)
    return sorted_svd, input_svd

#nmf code
def nmf(k, input_data, semantic_features):
    nmf = NMF(n_components = k)
    input_nmf = nmf.fit_transform(input_data)
    input_nmf = convert_dataframe_to_dict(input_nmf, input_data.index, semantic_features)
    words_and_scores_nmf =  pd.DataFrame(nmf.components_,columns=input_data.columns,index = semantic_features)
    nmf_output = words_and_scores_nmf.T.to_dict()
    sorted_nmf = sort_non_incr(nmf_output)
    return sorted_nmf, input_nmf

#lda code
def lda(k, input_data, semantic_features):
    lda = LatentDirichletAllocation(n_components = k)
    input_lda = lda.fit_transform(input_data)
    input_lda = convert_dataframe_to_dict(input_lda, input_data.index, semantic_features)
    words_and_scores_lda =  pd.DataFrame(lda.components_,columns=input_data.columns,index = semantic_features)
    lda_output = words_and_scores_lda.T.to_dict()
    sorted_lda = sort_non_incr(lda_output)
    return sorted_lda, input_lda