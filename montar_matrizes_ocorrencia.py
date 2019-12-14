import os
import numpy as np

def quant_termos(docs):
    quant = 1
    for doc in docs:
        split = doc.split()

        for termo in split[1:]:
            if int(termo.split(':')[0]) > quant:
                quant = int(termo.split(':')[0])

    return quant


def main():
    folder = "acmTesteTreino/"
    matriz_ocorrencia = "matrizes_de_ocorrencia/"
    classes_folder = "classes/"

    for arquivo in os.listdir(folder):
        print("\n--> " + arquivo)

        file = folder + arquivo
        f = open(file, 'r', encoding="utf8")

        lines = f.readlines()
        f.close()

        # Montando documentos Y (classes)
        print('Montando documento das classes...')
        file = classes_folder + arquivo
        output = open(file, 'w+')

        for doc in lines:
            split = doc.split()
            output.write(split[0] + ' ')
        
        output.close()
        print('Documento das classes (ok)')

        # Montando e salvando as matrizes de ocorrência
        print('Montando matriz de ocorrência...')
        file = matriz_ocorrencia + arquivo
        output = open(file, 'w+')

        # n_termos = quant_termos(lines)
        n_termos = 59991

        X = np.zeros((len(lines), n_termos), dtype=np.int_)

        for index,doc in enumerate(lines):
            split = doc.split()

            for termo in split[1:]:
                X[index][int(termo.split(':')[0])] = int(termo.split(':')[1])

        print('Matriz de ocorrência (ok).')
        print('Salvando matriz de ocorrência...')

        for i in range(0,len(lines)):
            for j in range(0,n_termos):
                output.write(str(X[i][j]) + ' ')
            output.write('\n')

        output.close()
        print('Finalizado.')

# -------------------------------------------------------------------------#
# Executa o metodo main
if __name__ == "__main__": main()