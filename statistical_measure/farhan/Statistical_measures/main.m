clear all;
close all;
clc

GTdir = 'D:\MATLAB\data\GT\';
resultdir= 'D:\MATLAB\data\test_results\';

GTfiles=dir(strcat(GTdir,'*.png'));
resultfiles=dir(strcat(resultdir,'*.jpg'));
L1=length(resultfiles);
L2=length(GTfiles);
count=0;
if L1==L2
    for i=1:1:L1
        resultname=strsplit(resultfiles(i).name,'.');
        GTname=strsplit(GTfiles(i).name,'.');
        if (strcmp(char(resultname(1)), char(GTname(1))))
            count=count+1;
            nm1= strcat(resultdir,resultfiles(i).name);
            nm2= strcat(GTdir,GTfiles(i).name);
            resultimg=im2bw(imread(nm1));
            GTimg=im2bw(imread(nm2));
         
            [TP, FP, TN, FN] = calError(double(GTimg), double(resultimg));
            [Accuracy(i), DSC(i), JI(i), Sensitivity(i), Specificity(i), F1Score(i)] = statMeasures(TP, FP, TN, FN);
        else
            break;
        end        
    end 
    avgAccuracy= mean(Accuracy);
    avgDSC = mean(DSC);
    avgJI= mean(JI);
    avgSensitivity=mean(Sensitivity);
    avgSpecificity=mean(Specificity);
    avgF1Score=mean(F1Score);
    
    stdAccuracy= std(Accuracy);
    stdDSC = std(DSC);
    stdJI= std(JI);
    stdSensitivity=std(Sensitivity);
    stdSpecificity=std(Specificity);
    stdF1Score=std(F1Score);
    
else
    print('Number of files in both directories are different');
end