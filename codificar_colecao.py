import os
import numpy as np
from sklearn.externals import joblib

def main():

    folder = "acmTesteTreino/"

    ### Carregando coleção original ###
    print('Carregando coleção original...')

    arquivo = "teste0.txt"

    file_c = folder + arquivo
    file = open(file_c, 'r', encoding="utf8")
    docs = file.readlines()
    file.close()

    n_termos = 59991

    #Y (classes)
    Y = []
    for doc in docs:
        Y.append(doc.split()[0])
    
    #X (features)
    X = np.zeros((len(docs), n_termos), dtype=np.int_)

    for index,doc in enumerate(docs):    
        for termo in doc.split()[1:]:
            X[index][int(termo.split(':')[0])] = int(termo.split(':')[1])

    ### Carregando modelo em disco ###
    print('Carregando codificador...')

    joblib_file = "encoder(50).pkl"
    model_eforest = joblib.load("modelos/" + joblib_file)

    ### Codificando a coleção ###
    print('Codificando a coleção...')

    X_encode = model_eforest.encode(X)

    ### Salvando codificação ###
    print('Salvando codificação...')

    colecao_codificada_folder = "ColecaoCodificada/"

    file = colecao_codificada_folder + arquivo
    output = open(file, 'w+')

    for d in range(len(docs)):
        output.write(str(Y[d]))
        for f in range(len(X_encode[d])):
            output.write(' ' + str(f) + ':' + str(X_encode[d][f]))
        output.write('\n')
    
    output.close()

# -------------------------------------------------------------------------#
# Executa o metodo main
if __name__ == "__main__": main()
