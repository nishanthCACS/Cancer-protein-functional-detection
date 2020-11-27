# -*- coding: utf-8 -*-
"""
Created on %03-Feb-2020 at 9.35 A.m
@author: %A.Nishanth C00294860
"""
#import os
import numpy as np
from itertools import cycle
from scipy import interp
import matplotlib.pyplot as plt
#from copy import deepcopy
from sklearn.metrics import roc_curve, auc,roc_auc_score


class ROC_for_given_3_classes_with_probability:
    """
    This class calculte the ROCcurve and cnfusion matrix for the given probabilities
    """
    
    def __init__(self,OG_test_probs_numpy,TSG_test_probs_numpy,Fusion_test_probs_numpy,minimum_prob_decide= 0.25,plot_on=True,bin_conf_mat=False):
        '''
        plot_on: DEcides the output of ROC plottable or nor
        conf_mat: It decides the o/p of binary confucion matrix of all classes
        '''
        self.OG_test_probs_numpy = OG_test_probs_numpy
        self.TSG_test_probs_numpy=TSG_test_probs_numpy
        self.Fusion_test_probs_numpy=Fusion_test_probs_numpy
        self.plot_on=plot_on
        self.bin_conf_mat=bin_conf_mat
        
        self.minimum_prob_decide = minimum_prob_decide

    def results_final(self):
        
        plot_on= self.plot_on
        
        confusion_matrix_test= self.confusion_matrix()
        AUCs_OG_TSG_Fusion = self.ROC_OG_Vs_TSG_Vs_Fusion(plot_on)
        
        if self.bin_conf_mat:
            AUCs_OG_TSG,conf_OG_TSG = self.ROC_OG_VS_TSG(plot_on,conf_mat=True)
            AUCs_OG_Fusion,conf_OG_Fusion = self.ROC_OG_VS_Fusion(plot_on,conf_mat=True)
            AUCs_TSG_Fusion,conf_TSG_Fusion = self.ROC_TSG_VS_Fusion(plot_on,conf_mat=True)
            conf_binary_all=[conf_OG_TSG,conf_OG_Fusion,conf_TSG_Fusion]
        else:
            AUCs_OG_TSG = self.ROC_OG_VS_TSG(plot_on)
            AUCs_OG_Fusion = self.ROC_OG_VS_Fusion(plot_on)
            AUCs_TSG_Fusion = self.ROC_TSG_VS_Fusion(plot_on)
        
        AUC_matrix = np.empty((4,5))
        
        AUC_matrix[0,:]=[AUCs_OG_TSG_Fusion[0],AUCs_OG_TSG_Fusion[1],AUCs_OG_TSG_Fusion[2],AUCs_OG_TSG_Fusion['micro'],AUCs_OG_TSG_Fusion['macro']]
        #OG_VS_TSG_since_Fusion is not applicaple
        AUC_matrix[1,:]=[AUCs_OG_TSG[0],AUCs_OG_TSG[1],np.NaN,AUCs_OG_TSG['micro'],AUCs_OG_TSG['macro']]
        AUC_matrix[2,:]=[AUCs_OG_Fusion[0],np.NaN,AUCs_OG_Fusion[1],AUCs_OG_Fusion['micro'],AUCs_OG_Fusion['macro']]
        AUC_matrix[3,:]=[np.NaN,AUCs_TSG_Fusion[0],AUCs_TSG_Fusion[1],AUCs_TSG_Fusion['micro'],AUCs_TSG_Fusion['macro']]
        
        AUC_matrix = np.round(AUC_matrix,4)
        if self.bin_conf_mat:
            return confusion_matrix_test,AUC_matrix,conf_binary_all
        else:
            return confusion_matrix_test,AUC_matrix
            
    def confusion_matrix_helper(self,list_1):
        '''
        This function help to create the confusion matrix
        '''
        count_OG=0
        count_TSG=0
        count_Fusion=0
        for i in range(0,len(list_1)):
            if list_1[i][0]>=list_1[i][1] and list_1[i][0]>=list_1[i][2]:
                count_OG=count_OG+1
            elif list_1[i][1]>=list_1[i][0] and list_1[i][1]>=list_1[i][2]:
                count_TSG=count_TSG+1
            elif list_1[i][2]>=list_1[i][0] and list_1[i][2]>=list_1[i][1]:
                count_Fusion=count_Fusion+1
        return count_OG,count_TSG,count_Fusion
    
    def confusion_matrix(self):
        '''
        This function create the confusion matrix
        where the rows are predicted one
        coloumns are the actual one
        '''
        ONGO_count_OG,ONGO_count_TSG,ONGO_count_Fusion=self.confusion_matrix_helper(self.OG_test_probs_numpy)
        Fusion_count_OG,Fusion_count_TSG,Fusion_count_Fusion=self.confusion_matrix_helper(self.Fusion_test_probs_numpy)
        TSG_count_OG,TSG_count_TSG,TSG_count_Fusion=self.confusion_matrix_helper(self.TSG_test_probs_numpy)
        confusion_matrix_fin=np.empty((3,3))
        confusion_matrix_fin[:,0]=[ONGO_count_OG,ONGO_count_TSG,ONGO_count_Fusion]
        confusion_matrix_fin[:,1]=[TSG_count_OG,TSG_count_TSG,TSG_count_Fusion]
        confusion_matrix_fin[:,2]=[Fusion_count_OG,Fusion_count_TSG,Fusion_count_Fusion]  
        OG_acc =round(100*ONGO_count_OG/(ONGO_count_OG+ONGO_count_TSG+ONGO_count_Fusion),2)
        TSG_acc=round(100*TSG_count_TSG/(TSG_count_OG+TSG_count_TSG+TSG_count_Fusion),2)
        Fusion_acc=round(100*Fusion_count_Fusion/(Fusion_count_OG+Fusion_count_TSG+Fusion_count_Fusion),2)
        
        over_all_acc=round(100*((ONGO_count_OG+TSG_count_TSG+Fusion_count_Fusion)/np.sum(confusion_matrix_fin)),2)
        self.Accuracy_gene=[over_all_acc,OG_acc,TSG_acc,Fusion_acc]
        return confusion_matrix_fin
            
    def get_scores_and_assign_n_to_2(self,probabilities,class_given,class_sel_1,class_sel_2):
        '''
        Mainly used ONGO vs TSG
        ploting ROC curve slected classes
        Eg: If ONGO Vs TSG just leaving Fusion
                then ONGO's class_given=0 & TSG's class_given=1
            class_sel_1: index of probability of class ONGO
            class_sel_2: index of probability of class TSG
        '''
        y_score=np.zeros((len(probabilities),2)) 
        y_test=np.zeros((len(probabilities),2),int) 
        correct=0
        for i in range(0,len(probabilities)):
            if sum(probabilities[i])==0:
                y_score[i,:]=[0.5,0.5]
            else:
                prob_1_normal = probabilities[i][class_sel_1]/sum(probabilities[i])#to make sure
                prob_2_normal = probabilities[i][class_sel_2]/sum(probabilities[i])
                if (prob_1_normal+prob_2_normal)==0:
                    y_score[i,:]=[0.5,0.5]
                else:
                    prob_2_normal = probabilities[i][class_sel_2]/sum(probabilities[i])
                if (prob_1_normal+prob_2_normal)<0.00001:
                    y_score[i,:]=[0.5,0.5]
                else:
                    y_score[i,:]=[prob_1_normal/(prob_1_normal+prob_2_normal),prob_2_normal/(prob_1_normal+prob_2_normal)]
                if class_given==0 and (prob_1_normal/(prob_1_normal+prob_2_normal))>=(prob_2_normal/(prob_1_normal+prob_2_normal)):
                    correct=correct+1
                elif class_given==1 and (prob_1_normal/(prob_1_normal+prob_2_normal))<(prob_2_normal/(prob_1_normal+prob_2_normal)):
                    correct=correct+1
            y_test[i][class_given]=1
        return y_score, y_test,correct
    
    def ROC_OG_VS_TSG(self,plot_on,conf_mat=False):
        '''ROC ONGO VS TSG only'''
        ONGO_y_score,ONGO_y_test,OG_correct = self.get_scores_and_assign_n_to_2(self.OG_test_probs_numpy,0,0,1) 
        TSG_y_score,TSG_y_test,TSG_correct = self.get_scores_and_assign_n_to_2(self.TSG_test_probs_numpy,1,0,1)
        y_test = np.concatenate((ONGO_y_test, TSG_y_test), axis=0)
        y_score = np.concatenate((ONGO_y_score, TSG_y_score), axis=0)
      

        print(" ")
        print(" ")
        print("Only ONGO vs TSG over all accuracy: ",100*(OG_correct+TSG_correct)/(len(self.OG_test_probs_numpy)+len(self.TSG_test_probs_numpy)),'%')
        print("Plotting of ROC of ONGO_VS_TSG")
        print("  v            ")
        print("  v            ")
        if conf_mat:
            confusion_matrix_binary=np.empty((2,2))
            confusion_matrix_binary[:,0]=[OG_correct,len(ONGO_y_test)-OG_correct]
            confusion_matrix_binary[:,1]=[len(TSG_y_test)-TSG_correct,TSG_correct]
            return self.ROC_plot(y_test,y_score,2,OG_TSG_only=True,plot_on=plot_on),confusion_matrix_binary    
        else:
            return self.ROC_plot(y_test,y_score,2,OG_TSG_only=True,plot_on=plot_on)      

    def ROC_OG_VS_Fusion(self,plot_on,conf_mat=False):
        '''ROC Fusion VS TSG only'''
        ONGO_y_score,ONGO_y_test,OG_correct = self.get_scores_and_assign_n_to_2(self.OG_test_probs_numpy,0,0,2) 
        Fusion_y_score,Fusion_y_test,Fusion_correct = self.get_scores_and_assign_n_to_2(self.Fusion_test_probs_numpy,1,0,2)
        y_test = np.concatenate((ONGO_y_test, Fusion_y_test), axis=0)
        y_score = np.concatenate((ONGO_y_score, Fusion_y_score), axis=0)
        print("Only ONGO vs Fusion over all accuracy: ",100*(OG_correct+Fusion_correct)/(len(self.OG_test_probs_numpy)+len(self.Fusion_test_probs_numpy)),'%')
        print("Plotting of ROC of ONGO_VS_Fusion is turtned off")
        if conf_mat:
            confusion_matrix_binary=np.empty((2,2))
            confusion_matrix_binary[:,0]=[OG_correct,len(ONGO_y_test)-OG_correct]
            confusion_matrix_binary[:,1]=[len(Fusion_y_test)-Fusion_correct,Fusion_correct]
            return self.ROC_plot(y_test,y_score,2,OG_Fusion_only=True,plot_on=False),confusion_matrix_binary 
        else:
            return self.ROC_plot(y_test,y_score,2,OG_Fusion_only=True,plot_on=False)      

    def ROC_TSG_VS_Fusion(self,plot_on,conf_mat=False):
        '''ROC ONGO VS TSG only'''
        TSG_y_score,TSG_y_test,TSG_correct = self.get_scores_and_assign_n_to_2(self.TSG_test_probs_numpy,0,1,2)
        Fusion_y_score,Fusion_y_test,Fusion_correct = self.get_scores_and_assign_n_to_2(self.Fusion_test_probs_numpy,1,1,2)
        y_test = np.concatenate((TSG_y_test,Fusion_y_test), axis=0)
        y_score = np.concatenate((TSG_y_score,Fusion_y_score), axis=0)
        print("Only TSG vs Fusion over all accuracy: ",100*(TSG_correct+Fusion_correct)/(len(self.Fusion_test_probs_numpy)+len(self.TSG_test_probs_numpy)),'%')
        print("Plotting of ROC of TSG_VS_Fusion is turtned off")
        if conf_mat:
            confusion_matrix_binary=np.empty((2,2))
            confusion_matrix_binary[:,0]=[TSG_correct,len(TSG_y_test)-TSG_correct]
            confusion_matrix_binary[:,1]=[len(Fusion_y_test)-Fusion_correct,Fusion_correct]
            return self.ROC_plot(y_test,y_score,2,TSG_Fusion_only=True,plot_on=False),confusion_matrix_binary     
        else:
            return self.ROC_plot(y_test,y_score,2,TSG_Fusion_only=True,plot_on=False)      
        
    def ROC_OG_Vs_TSG_Vs_Fusion(self,plot_on):
        '''ROC ONGO Vs TSG Vs Fusion only'''       
        y_score=np.zeros((len(self.OG_test_probs_numpy)+len(self.TSG_test_probs_numpy)+len(self.Fusion_test_probs_numpy),3)) 
        y_test=np.zeros((len(self.OG_test_probs_numpy)+len(self.TSG_test_probs_numpy)+len(self.Fusion_test_probs_numpy),3),int) 
        k=0
        for i in range(0,len(self.OG_test_probs_numpy)):
            y_score[k,:]=self.OG_test_probs_numpy[i]
            y_test[k,0]=1
            k=k+1
        for i in range(0,len(self.TSG_test_probs_numpy)):
            y_score[k,:]=self.TSG_test_probs_numpy[i]
            y_test[k,1]=1
            k=k+1
        for i in range(0,len(self.Fusion_test_probs_numpy)):
            y_score[k,:]=self.Fusion_test_probs_numpy[i]
            y_test[k,2]=1
            k=k+1   
        print(" ")
        print(" ")
        print("Plotting of ROC of ONGO_VS_TSG_VS_Fusion")
        print("  v            ")
        print("  v            ")
        return self.ROC_plot(y_test,y_score,3,plot_on=plot_on)        

    def ROC_plot(self,y_test,y_score,n_classes,OG_TSG_only=False,TSG_Fusion_only=False,OG_Fusion_only=False,plot_on=True):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        lw = 2
        
        # Compute macro-average ROC curve and ROC area
        '''the code for macro averaging credit goes to 
        https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
        on 14-Feb-2020 at 8.34pm
        '''
        # First aggregate all false positive rates
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        # %then interpolate all ROC curves at this points
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += interp(all_fpr, fpr[i], tpr[i])
        
        # Finally average it and compute AUC
        mean_tpr /= n_classes
        
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        if plot_on:
            # Plot all ROC curves
            plt.figure()
            plt.plot(fpr["micro"], tpr["micro"],
                     label='micro-average ROC curve (area = {0:0.4f})'
                           ''.format(roc_auc["micro"]),
                     color='deeppink', linestyle=':', linewidth=4)
            
            plt.plot(fpr["macro"], tpr["macro"],
                     label='macro-average ROC curve (area = {0:0.4f})'
                           ''.format(roc_auc["macro"]),
                     color='red', linestyle=':', linewidth=4)
            if n_classes==3:
                colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
                for i, color in zip(range(n_classes), colors):
                    if i==0:
                        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                                 label='ROC curve of ONGO (area = {1:0.4f})'
                                 ''.format(i, roc_auc[i]))
                    elif i==1:
                        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                                 label='ROC curve of TSG (area = {1:0.4f})'
                                 ''.format(i, roc_auc[i]))
                    elif i==2:
                        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                                 label='ROC curve of Fusion (area = {1:0.4f})'
                                 ''.format(i, roc_auc[i]))
            elif n_classes==2:
                colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
                for i, color in zip(range(n_classes), colors):
                    if i==0 and not TSG_Fusion_only:
                        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                                 label='ROC curve of ONGO (area = {1:0.4f})'
                                 ''.format(i, roc_auc[i]))
                    elif (i==1 and OG_TSG_only) or (i==0 and TSG_Fusion_only):
                        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                                 label='ROC curve of TSG (area = {1:0.4f})'
                                 ''.format(i, roc_auc[i]))
                    elif (i==1 and TSG_Fusion_only) or (i==1 and OG_Fusion_only):
                        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                                 label='ROC curve of Fusion (area = {1:0.4f})'
                                 ''.format(i, roc_auc[i]))
    
            plt.plot([0, 1], [0, 1], 'k--', lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            #plt.title('Receiver operating characteristic to ONCO Vs TSG Vs -class')
            plt.legend(loc="lower right")
            plt.show()
        return roc_auc