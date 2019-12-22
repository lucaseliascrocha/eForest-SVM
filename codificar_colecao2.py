import os
import numpy as np
from sklearn.externals import joblib

def main():

    folder = "acmTesteTreino/"

    ### Carregando coleção original ###
    print('Carregando coleção original...')

    arquivo = "treino0.txt"

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

    joblib_file = "encoder(sem restricao).pkl"
    model_eforest = joblib.load("modelos/Supervisionado/" + joblib_file)

    ### Codificando a coleção ###
    print('Codificando a coleção...')

    X_encode = model_eforest.encode(X)

    ### Salvando codificação ###
    print('Salvando codificação...')

    classe = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11']

    colecao_codificada_folder = "ColecaoCodificada/"

    file = colecao_codificada_folder + arquivo.split('.')[0] + '.csv'
    output = open(file, 'w+')


    for i in range(0,300):
        output.write(str(i) + ',')

    output.write('class\n')
    

    for d in range(len(docs)):
        for f in range(len(X_encode[d])):
            output.write(str(X_encode[d][f]) + ',')
        
        output.write(classe[int(Y[d])-1]) # classe nominal
        output.write('\n')
    
    output.close()

# -------------------------------------------------------------------------#
# Executa o metodo main
if __name__ == "__main__": main()
