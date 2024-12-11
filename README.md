English description below

Sigmoid Function and Logistic Regression Analysis

Introdução

Neste projeto, decidi implementar a regressão logística de forma manual usando a função sigmoide.
Utilizei o conjunto de dados Iris e o objetivo principal foi explorar os fundamentos da regressão logística.

Passos:

Carregamento do Conjunto de Dados Iris

Usei o load_iris da biblioteca sklearn para carregar o conjunto de dados Iris.

Extraí os dados e os rótulos, e selecionei uma classe específica ("setosa") para análise binária.

Pré-processamento dos Dados

Dividi os dados em conjuntos de treinamento e teste utilizando train_test_split, mantendo a estratificação das classes.

Adicionei um termo de viés (bias) aos dados.

Normalizei os dados com StandardScaler para padronizar a escala e melhorar a convergência.

Treinamento com Regressão Logística

Treinei um modelo de regressão logística utilizando a biblioteca sklearn.

Calculei as probabilidades preditivas e as comparei com as classes reais para avaliar o desempenho.

Métricas de Avaliação

Avaliei o modelo usando:

Acurácia

Precisão

Recall
F1-Score

Log-Loss

Implementação Manual da Função Sigmoide

Implementei a função sigmoide para cálculos diretos de probabilidades.

Desenvolvi a função de custo logarítmico para avaliar erros e melhorar o modelo iterativamente.

Busca de Hiperparâmetros (eta):

Realizei uma busca por gradiente (grid search) para encontrar o melhor valor de taxa de aprendizado (eta).

Avaliei a convergência usando várias taxas e selecionei o valor que minimizou a função de custo.

Regressão Logística Multiclasse (One-vs-Rest):

Expandi a análise para problemas multiclasse, treinando modelos binários para cada classe do conjunto Iris.

Combinei as predições dos modelos usando a abordagem One-vs-Rest.

Resultados e Conclusões:

O modelo binário foi altamente eficaz ao diferenciar a classe "setosa".

A abordagem multiclasse demonstrou boa generalização para o restante das classes do conjunto de dados.

O uso manual da função sigmoide e do gradiente logístico ajudou a compreender melhor os detalhes matemáticos por trás da regressão logística.




Softmax Regression e Classificação Multiclasse

Introdução

No segundo arquivo, avancei para a implementação e análise da Regressão Softmax de forma manual, expandindo o escopo para problemas de classificação multiclasse. 
Utilizei novamente o conjunto de dados Iris para explorar os conceitos por trás da regressão Softmax.

Passos

1. Divisão e Pré-processamento dos Dados
   
Carreguei o conjunto de dados Iris utilizando a função load_iris.
Dividi os dados em conjuntos de treinamento e teste com train_test_split.
Encodei as classes de destino (target) usando OneHotEncoder para converter os rótulos em uma representação binária compatível com Softmax.

2. Treinamento Inicial com Regressão Logística (Sklearn)
   
Utilizei o LogisticRegression do Scikit-Learn para treinar um modelo de regressão logística multiclasse, estabelecendo um benchmark inicial.

Avaliei o modelo com métricas como:

Acurácia

Precisão ponderada

Revocação ponderada

F1-score ponderado

Exibi a matriz de confusão normalizada para visualização do desempenho por classe.

3. Implementação Manual da Regressão Softmax
   
Desenvolvi o algoritmo Softmax manualmente para treinar o modelo:

Função Softmax: 

Transformei os logits em probabilidades interpretáveis.

Função de Custo: 

Usei cross entropy para calcular o erro entre predições e rótulos reais.

Gradiente Descendente: 

Atualizei os pesos do modelo iterativamente, minimizando a função de custo.

4. Treinamento e Convergência
   
Treinei o modelo manualmente por 10.000 épocas, ajustando a taxa de aprendizado (eta = 0.01).

Monitorei a convergência da função de custo ao longo das épocas.

5. Avaliação Manual do Modelo
   
Para cada amostra no conjunto de teste:

Calculei as probabilidades preditivas.

Identifiquei a classe prevista e comparei com a classe real.

Reportei o custo individual de cada predição.

Determinei se a predição foi correta ou incorreta.

Calculei a acurácia geral do modelo manualmente, com o modelo alcançando uma acurácia de 96.43%.

Conclusão:

A implementação manual da Regressão Softmax foi eficaz, alcançando uma acurácia de 96.43% no conjunto de teste.

A matriz de confusão ajudou a visualizar quais classes o modelo confundiu com mais frequência.







Sigmoid Function and Logistic Regression Analysis

Introduction

In this project, I decided to implement logistic regression manually using the sigmoid function. 
I used the Iris dataset, and the main objective was to explore the fundamentals of logistic regression.

Steps:

Loading the Iris Dataset

I used load_iris from the sklearn library to load the Iris dataset.

I extracted the data and labels and selected a specific class ("setosa") for binary analysis.

Data Preprocessing

I split the data into training and testing sets using train_test_split, maintaining class stratification.

I added a bias term to the data.

I normalized the data with StandardScaler to standardize the scale and improve convergence.

Training with Logistic Regression

I trained a logistic regression model using the sklearn library.

I calculated the predicted probabilities and compared them with the actual classes to evaluate performance.

Evaluation Metrics

I evaluated the model using:

Accuracy

Precision

Recall

F1-Score

Log-Loss

Manual Implementation of the Sigmoid Function

I implemented the sigmoid function for direct probability calculations.

I developed the log-likelihood cost function to evaluate errors and iteratively improve the model.

Hyperparameter Search (eta):

I performed a gradient search (grid search) to find the best learning rate (eta).

I evaluated convergence using various learning rates and selected the value that minimized the cost function.

Multiclass Logistic Regression (One-vs-Rest):

I expanded the analysis to multiclass problems by training binary models for each class in the Iris dataset.

I combined the predictions from the models using the One-vs-Rest approach.

Results and Conclusions:

The binary model was highly effective in distinguishing the "setosa" class.

The multiclass approach showed good generalization for the other classes in the dataset.

The manual use of the sigmoid function and logistic gradient descent helped deepen the understanding of the mathematical details behind logistic regression.

Softmax Regression and Multiclass Classification:

Introduction

In the second file, I advanced to the manual implementation and analysis of Softmax Regression, expanding the scope to multiclass classification problems. 
I used the Iris dataset again to explore the concepts behind Softmax regression.

Steps:

Data Splitting and Preprocessing

I loaded the Iris dataset using the load_iris function.

I split the data into training and testing sets using train_test_split.

I encoded the target classes using OneHotEncoder to convert the labels into a binary representation compatible with Softmax.

Initial Training with Logistic Regression (Sklearn)

I used LogisticRegression from Scikit-Learn to train a multiclass logistic regression model, establishing an initial benchmark.

I evaluated the model using metrics such as:

Accuracy

Weighted Precision

Weighted Recall

Weighted F1-score

I displayed the normalized confusion matrix for visualizing class performance.

Manual Implementation of Softmax Regression

I developed the Softmax algorithm manually to train the model:

Softmax Function: 

I transformed logits into interpretable probabilities.

Cost Function: 

I used cross-entropy to compute the error between predictions and actual labels.

Gradient Descent: 

I updated the model weights iteratively to minimize the cost function.

Training and Convergence

I trained the model manually for 10,000 epochs, adjusting the learning rate (eta = 0.01).

I monitored the convergence of the cost function over the epochs.

Manual Evaluation of the Model

For each sample in the test set:

I calculated the predicted probabilities.

I identified the predicted class and compared it to the actual class.

I reported the individual cost for each prediction.

I determined whether the prediction was correct or incorrect.

I manually calculated the overall accuracy of the model, which achieved an accuracy of 96.43%.

Conclusion:

The manual implementation of Softmax Regression was effective, achieving an accuracy of 96.43% on the test set.
The confusion matrix helped visualize which classes the model confused most frequently.
