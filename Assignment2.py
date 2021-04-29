import numpy as np
import pandas as pd
import math
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier

df = pd.read_csv('week2.csv') 
X1=df.iloc[:, 0]
X2=df.iloc[:, 1]
X3=X1**2
X4=X2**2
X=np.column_stack((X1,X2, X3, X4)) 
y=df.iloc[:, 2]

# Visualise the downloaded data
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
ax.set_title("Visualisation of Data", fontsize=14)
ax.set_xlabel("X1 (Normalised)", fontsize=12)
ax.set_ylabel("X2 (Normalised)", fontsize=12)
x_min, x_max = X[:, 0].min() - 0.25, X[:, 0].max() + 0.25
y_min, y_max = X[:, 1].min() - 0.25, X[:, 1].max() + 0.25
ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])
colors = ['C0' if marker == -1 else 'C3' for marker in y]
ax.scatter(X[:, 0], X[:, 1], marker='o', facecolors=colors, linewidth=1.0, alpha=0.5)
legend_elements = [Line2D([0], [0], marker='o', color='C0', label='C1 True', 
                          alpha=0.5, linestyle = 'None', markersize=10),
                   Line2D([0], [0], marker='o', color='C3', label='C2 True', 
                          alpha=0.5, linestyle = 'None', markersize=10)]
ax.legend(handles=legend_elements, loc='lower right')
plt.savefig('data_visualisation')

# Train a logistic regression classifier on the data
logreg = LogisticRegression(penalty='none', solver='lbfgs')
logreg.fit(X[:,0:2], y)
print('Logistic Regression Parameter Values Table:')
print('%-20s %-21s %-20s' % ('Linear Model', 'Intercept', 'Slope'))
print('%-20s %-21a %-20a' % ('Logistic Regression', logreg.intercept_, logreg.coef_))

x = np.array([x_min, x_max])
decbond = -(logreg.coef_[0][0] * x + logreg.intercept_)/logreg.coef_[0][1]
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
h = 0.02
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Classify values using trained logistic regression classifer and
# show the decision boundary of the classifier
ax.set_title("Logistic Regression Classifier", fontsize=14)
colors = ['C0' if marker == -1 else 'C3' for marker in logreg.predict(X[:,0:2])]
ax.scatter(X[:, 0], X[:, 1], marker='x', c=colors, linewidth=1.0, alpha=1.0)
ax.plot(x, decbond, c='black', linewidth=1.0)
ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.25)
legend_elements = [Line2D([0], [0], marker='o', color='C0', label='C1 True', 
                          alpha=0.5, linestyle = 'None', markersize=10),
                   Line2D([0], [0], marker='o', color='C3', label='C2 True', 
                          alpha=0.5, linestyle = 'None', markersize=10),
                   Line2D([0], [0], marker='x', color='w', label='C1 Pred', 
                          markeredgecolor='C0', linestyle = 'None', markersize=10),
                   Line2D([0], [0], marker='x', color='w', label='C2 Pred', 
                          markeredgecolor='C3', linestyle = 'None', markersize=10),
                   Line2D([0],[0], color='black', lw=1, label='Decision Boundary')]
ax.legend(handles=legend_elements, loc='lower right')
plt.savefig('lr_classifier_with_2_params')
plt.show()

# Train linear SVM classifer for a range of penalty parameter values 
# and use each trained classifiers to predict the target values in
# the training data
print('SVM Parameter Values Table:')
print('%-8s %-21s %-20s' % ('Penalty', 'Intercept', 'Slope'))
penalties = [0.001, 1, 1000]
svms = []
for penalty in penalties:
    svm = LinearSVC(C=penalty)
    svm.fit(X[:,0:2], y)
    svms.append(svm)
    print('%-8s %-21a %-20a' % (penalty, svm.intercept_, svm.coef_))

# Create two addition features by adding the square of each feature and
# train a logistic regression classifier
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12)); 
ax = ax.flatten()
fig.suptitle('SVM Classifiers', fontsize=16)
fig.tight_layout(pad=4.0)
fig.subplots_adjust(top=0.925)
for itr, penalty in enumerate(penalties):    
    ax[itr].set_title('C = ' + str(penalty), fontsize=14)
    ax[itr].set_xlabel('X1 (Normalised)', fontsize=12)
    ax[itr].set_ylabel('X2 (Normalised)', fontsize=12)
    x_min, x_max = X[:, 0].min() - 0.25, X[:, 0].max() + 0.25
    y_min, y_max = X[:, 1].min() - 0.25, X[:, 1].max() + 0.25
    ax[itr].set_xlim([x_min, x_max])
    ax[itr].set_ylim([y_min, y_max])
    colors = ['C0' if marker == -1 else 'C3' for marker in y]
    ax[itr].scatter(X[:, 0], X[:, 1], marker='o', facecolors=colors, linewidth=1.0, alpha=0.5)
    colors = ['C0' if marker == -1 else 'C3' for marker in svms[itr].predict(X[:,0:2])]
    ax[itr].scatter(X[:, 0], X[:, 1], marker='x', c=colors, linewidth=1.0, alpha=1.0)
    
    x = np.array([x_min, x_max])
    decbond = -(svms[itr].coef_[0][0] * x + svms[itr].intercept_)/svms[itr].coef_[0][1]
    ax[itr].plot(x, decbond, c='black', linewidth=1.0)
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = svms[itr].predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax[itr].contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.25)
    
    legend_elements = [Line2D([0], [0], marker='o', color='C0', label='C1 True', 
                          alpha=0.5, linestyle = 'None', markersize=10),
                       Line2D([0], [0], marker='o', color='C3', label='C2 True', 
                              alpha=0.5, linestyle = 'None', markersize=10),
                       Line2D([0], [0], marker='x', color='w', label='C1 Pred', 
                              markeredgecolor='C0', linestyle = 'None', markersize=10),
                       Line2D([0], [0], marker='x', color='w', label='C2 Pred', 
                              markeredgecolor='C3', linestyle = 'None', markersize=10),
                       Line2D([0],[0], color='black', lw=1, label='Decision Boundary')]
    ax[itr].legend(handles=legend_elements, loc='lower right')
ax[-1].axis('off')
plt.savefig('svm_classifiers')    
plt.show()

# Compare the performance of the classsifier against a reasonable baseline predictor
logreg = LogisticRegression(penalty='none', solver='lbfgs')
logreg.fit(X, y)
print('Logistic Regression Parameter Values Table:')
print('%-20s %-21s %-20s' % ('Linear Model', 'Intercept', 'Slope'))
print('%-20s %-21a %-20a' % ('Logistic Regression', logreg.intercept_, logreg.coef_))

# Plot the classifier decision boundary
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 6))
ax.set_title("Logistic Regression Classifier", fontsize=14)
ax.set_xlabel("X1 (Normalised)", fontsize=12)
ax.set_ylabel("X2 (Normalised)", fontsize=12)
x_min, x_max = X[:, 0].min() - 0.25, X[:, 0].max() + 0.25
y_min, y_max = X[:, 1].min() - 0.25, X[:, 1].max() + 0.25
ax.set_xlim([x_min, x_max])
ax.set_ylim([y_min, y_max])
colors = ['C0' if marker == -1 else 'C3' for marker in y]
ax.scatter(X[:, 0], X[:, 1], marker='o', facecolors=colors, linewidth=1.0, alpha=0.5)
colors = ['C0' if marker == -1 else 'C3' for marker in logreg.predict(X)]
ax.scatter(X[:, 0], X[:, 1], marker='x', c=colors, linewidth=1.0, alpha=1.0)

x = np.arange(x_min, x_max, h)
a = (logreg.coef_[0][2] * x**2 + logreg.coef_[0][0] * x + logreg.intercept_[0])/logreg.coef_[0][3]
b = logreg.coef_[0][1]/(2 * logreg.coef_[0][3])
decbond = -b - np.sqrt(-a + b**2)
ax.plot(x, decbond, c='black', linewidth=1.0)
# x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
# y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
# h = 0.02
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
# Z = Z.reshape(xx.shape)
# ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.25)

legend_elements = [Line2D([0], [0], marker='o', color='C0', label='C1 True', 
                          alpha=0.5, linestyle = 'None', markersize=10),
                   Line2D([0], [0], marker='o', color='C3', label='C2 True', 
                          alpha=0.5, linestyle = 'None', markersize=10),
                   Line2D([0], [0], marker='x', color='w', label='C1 Pred', 
                          markeredgecolor='C0', linestyle = 'None', markersize=10),
                   Line2D([0], [0], marker='x', color='w', label='C2 Pred', 
                          markeredgecolor='C3', linestyle = 'None', markersize=10),
                   Line2D([0],[0], color='black', lw=1, label='Decision Boundary')]
ax.legend(handles=legend_elements, loc='lower right')
plt.savefig('lr_classifier_with_4_params')
plt.show()

dummy = DummyClassifier(strategy='stratified')
dummy.fit(X[:, 2:4], y)
print('Classifier Comparison:')
print('%-20s %-20s' % ('Linear Model', 'Score'))
print('%-20s %-20f' % ('Logistic Regression', logreg.score(X, y)))
print('%-20s %-20f' % ('Dummy', dummy.score(X, y)))