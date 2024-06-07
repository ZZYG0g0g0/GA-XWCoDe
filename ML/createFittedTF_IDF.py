from sklearn.feature_extraction.text import TfidfTransformer


def createFittedTF_IDF(cv, corpus):
    # Generate the word counts for the corpus
    word_count_vector = cv.fit_transform(corpus)
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    fittedTF_IDF = tfidf_transformer.fit(word_count_vector)
    return (fittedTF_IDF)


if __name__ == '__main__':
    fname = "../iTrust/code_feature/CN_MN_VN_CMT_clear.txt"
    tname = "../iTrust/UC_clear.txt"
    output_fname = "../iTrust/QQ"
    createFittedTF_IDF(cv, corpus)
