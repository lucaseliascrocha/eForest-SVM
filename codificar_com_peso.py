import os
import numpy as np
from sklearn import tree
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

    joblib_file = "encoder0.pkl"
    model_eforest = joblib.load("modelos/Supervisionado/" + joblib_file)

    ### Codificando a coleção ###
    print('Codificando a coleção...')

    X_encode = model_eforest.encode(X)

	### Calculando pesos e salvando codificação ###
    print('Calculando pesos e salvando codificação...')

    node_samples = []
	
    for t in range(len(X_encode[0])):
        node_samples.append({})
        for f in range(len(docs)):
            if X_encode[f][t] in node_samples[t].keys():
                node_samples[t][X_encode[f][t]] += 1
            else:
                node_samples[t][X_encode[f][t]] = 1
	
    colecao_codificada_folder = "ColecaoCodificada/"

	### Calculando profundidade dos nós das árvores ###
    trees_depth = []
    leaf_map = []
    aux = 1

    #Alocando map#
    for t in range(300):
        leaf_map.append({})
		
    for t in range(300):

        arvore = model_eforest.estimators_[t]
        n_nodes = arvore.tree_.node_count
        children_left = arvore.tree_.children_left
        children_right = arvore.tree_.children_right
        node_depth = np.zeros(shape=n_nodes)

        stack = [(0, -1)]  # seed is the root node id and its parent depth
        while len(stack) > 0:
            node_id, parent_depth = stack.pop()
            node_depth[node_id] = parent_depth + 1
			# If we have a test node
            if (children_left[node_id] != children_right[node_id]):
                stack.append((children_left[node_id], parent_depth + 1))
                stack.append((children_right[node_id], parent_depth + 1))
            else:
                leaf_map[t][node_id] =  aux
                aux += 1

        trees_depth.append(node_depth)
		

    file_code = colecao_codificada_folder + arquivo
    output_code = open(file_code, 'w+')

    count = 0
    for d in range(len(docs)):
        count += 1
        output_code.write(str(Y[d]))
        for t in range(len(X_encode[d])):
            arvore = model_eforest.estimators_[t]
            altura_max = arvore.tree_.max_depth

			### Calculando peso da folha ###
            alpha = 1
            beta = 0
            altura = altura_max - trees_depth[t][X_encode[d][t]]
            peso = alpha * (altura/(altura_max-1)) + beta * (node_samples[t][X_encode[d][t]]/len(docs))

            ### Escrevendo no arquivo ###
            output_code.write(' ' + str(leaf_map[t][X_encode[d][t]]) + ':' + str(peso))
        output_code.write('\n')
        print(str(count)+'/'+str(len(docs)))
    
    output_code.close()

# -------------------------------------------------------------------------#
# Executa o metodo main
if __name__ == "__main__": main()
