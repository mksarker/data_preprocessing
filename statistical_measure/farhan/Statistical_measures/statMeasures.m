function [Accuracy, DSC, JI, Sensitivity, Specificity, F1Score] = statMeasures(TP, FP, TN, FN)

Accuracy= (TP+TN)/(TP+TN+FP+FN);
DSC= 2*TP./(TP+TP+FN+FP);
JI= TP./(TP+FP+FN);
Sensitivity= TP./(TP+FN);
Specificity= TN./(TN+FP);
F1Score= 2*TP./(2*TP+FP+FN);
