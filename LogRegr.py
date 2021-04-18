import numpy as np
import urllib
from sklearn.naive_bayes import GaussianNB
from sklearn import  model_selection
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.svm import SVC
import csv

#Carrega os csvs
y = pd.read_csv('/Users/tecnicaicp/Documents/PlantCLEF2015/yleaf_trainvgg.csv', header=None)
y = y.to_numpy()
y = np.ravel(y)
print(y.shape)

# handcrafted features
#X = pd.read_csv('/Users/tecnicaicp/Documents/MESTRADO/Projeto_Guilherme_Impl/Matriz de confusão LEAFSCAN/Xlscan.csv', header=None, dtype=float)
#X = X.to_numpy()
#print(X.shape)

#deep features
X = pd.read_csv('/Users/tecnicaicp/Documents/PlantCLEF2015/X_deepleaf_trainvgg.csv', header=None)
X = X.to_numpy()
print(X.shape)

# EXEMPLO USANDO HOLDOUT
# Holdout -> dividindo a base em treinamento (70%) e teste (30%), estratificada
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)   #

# Treina o classificador

clfa = SVC(kernel='linear', probability=True, random_state=42)
#clfa = GaussianNB()
#clfa = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial', max_iter=5000)
clfa = clfa.fit(X_train, y_train)

# testa usando a base de testes
predicted=clfa.predict(X_test)
predp=clfa.predict_proba(X_test)

# calcula a acurÃ¡cia na base de teste
score=clfa.score(X_test, y_test)

# calcula a matriz de confusÃ£o
matrix = confusion_matrix(y_test, predicted)

# apresenta os resultados
print("Accuracy = %.2f " % score)
print("Confusion Matrix:")
print(matrix)


# EXEMPLO USANDO VALIDAÃ‡ÃƒO CRUZADA
clfb = SVC(kernel='linear', probability=True, random_state=42)
#clfb = LogisticRegression(C=1e5, solver='lbfgs', multi_class='multinomial', max_iter=1000)
#clfb = GaussianNB()
folds=10
result = model_selection.cross_val_score(clfb, X, y, cv=folds)
print("\nCross Validation Results %d folds:" % folds)
print("Mean Accuracy: %.2f" % result.mean())
print("Mean Std: %.2f" % result.std())

# matriz de confusÃ£o da validaÃ§Ã£o cruzada
Z = model_selection.cross_val_predict(clfb, X, y, cv=folds)
cm=confusion_matrix(y, Z)
print("Confusion Matrix:")
print(cm)

for i in range(len(predicted)):
    if (predicted[i] != y_test[i]):
        dist=1
        j=0
        while (j<len(X) and dist !=0):
            dist = np.linalg.norm(X[j]-X_test[i])
            j+=1
        print("Label:", y[j-1], class_names[y[j-1]], "  /  Prediction: ", predicted[i], class_names[predicted[i]], predp[i][predicted[i]])
        print("Label:", y[j - 1], "  /  Prediction: ", predicted[i], predp[i][predicted[i]])
        name= "/content/drive/My Drive/Leaves/" + str(class_names[y[j-1]]) + "/" + str(j)+ ".jpg"
        print(name)
        im=cv2.imread(name)
        im = cv2.resize(im,(299,299))
        cv2_imshow(im)
        print("=============================================================================")