#!/usr/bin/python 
# -*- coding: UTF-8 -*-

def precision(y_true, y_pred):
    counter = 0.0
    for i in range(len(y_pred)):
        if y_pred[i] == y_true[i]:
            counter += 1
    return counter / len(y_pred)*1.0

def recall(y_true, y_pred):
    counter = 0.0
    total = 0.0
    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            total += 1
            if y_pred[i] == y_true[i]:
                counter += 1
    return counter / total

def F1(p, r):
    return (2.0*p*r)/(p+r)

def precison_recall_f1(y_true, y_pred):
    pre = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    f1 = F1(pre, rec)
    print 'precision:', pre
    print 'recall:', rec
    print 'f1:', f1

