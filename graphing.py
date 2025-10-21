import numpy
import matplotlib.pyplot as plt
import idx2numpy
import time
import pandas as pd
import seaborn as sns

#ensures that the output stays on one line in the terminal (only if it should)
numpy.set_printoptions(linewidth=numpy.inf)
#Suppresses scientific notion when printing out floating point numbers
numpy.set_printoptions(suppress=True)


class Test:
    def __init__(self):
        pass

#========================================================================================================================
# TEST SPECIFIC PLOTTING
#========================================================================================================================
    """
    Reads csv files of both epoch summary and Calibration Curve,

    Figure 1 plots all 4 different graphs
    Graph 1 = Loss vs Epoch | Graph 2 = Accuracy vs Epoch | Graph 3 = Calibration Curve | Graph 4 = Normalized Gradients

    Figure 2 plots the confusion matrix.
    A confusion matrix every 10 epochs from [1,10,20,30,40,50] for a total of 6 Confusion Matrices.
    """
    # Learning rate: 0.01
    def Test_01(self):
        """Reads csv files of both epoch summary and Calibration Curve,
        Figure 1 plots all 4 different graphs
        Figure 2 plots the confusion matrix."""
        
        epoch_summary = pd.read_csv("Test_1.1_LR0-01/epoch_summary_T1.1.csv")
        CC = pd.read_csv("Test_1.1_LR0-01/CC_V_T1.1.csv")

        fig1 = self.Plotting(epoch_summary,CC,LR=0.01)
        fig2 = self.ConfusionMatrix(file_name_1 = "Test_1.1_LR0-01/CM_T1.1/CM_e", file_name_2 = "_T1.1.npz",LR=0.01)

        plt.show()
        return
    
    # Learning rate: 0.001
    def Test_001(self):
        """Reads csv files of both epoch summary and Calibration Curve,
        Figure 1 plots all 4 different graphs
        Figure 2 plots the confusion matrix."""

        epoch_summary = pd.read_csv("Test_2.1_LR0-001/epoch_summary_T2.1.csv")
        CC = pd.read_csv("Test_2.1_LR0-001/CC_V_T2.1.csv")

        fig1 = self.Plotting(epoch_summary,CC,LR=0.001)
        fig2 = self.ConfusionMatrix(file_name_1 = "Test_2.1_LR0-001/CM_T2.1/CM_e", file_name_2 = "_T2.1.npz",LR=0.001)

        plt.show()
        return
    
    # Learning rate: 0.005
    def Test_005(self):
        """Reads csv files of both epoch summary and Calibration Curve,
        Figure 1 plots all 4 different graphs
        Figure 2 plots the confusion matrix."""

        epoch_summary = pd.read_csv("Test_3.1_LR0-005/epoch_summary_T3.1.csv")
        CC = pd.read_csv("Test_3.1_LR0-005/CC_V_T3.1.csv")

        fig1 = self.Plotting(epoch_summary,CC,LR=0.005)
        fig2 = self.ConfusionMatrix(file_name_1 = "Test_3.1_LR0-005/CM_T3.1/CM_e", file_name_2 = "_T3.1.npz",LR=0.005)

        plt.show()
        return

    # Learning rate: 0.00005
    def Test_00005(self):
        """Reads csv files of both epoch summary and Calibration Curve,
        Figure 1 plots all 4 different graphs
        Figure 2 plots the confusion matrix."""

        epoch_summary = pd.read_csv("Test_4.1_LR0-00005/epoch_summary_T4.1.csv")
        CC = pd.read_csv("Test_4.1_LR0-00005/CC_V_T4.1.csv")

        fig1 = self.Plotting(epoch_summary,CC,LR=0.00005)
        fig2 = self.ConfusionMatrix(file_name_1 = "TTest_4.1_LR0-00005/CM_T4.1/CM_e", file_name_2 = "_T4.1.npz",LR=0.00005)

        plt.show()
        return
    
    # Learning rate: 0.0001
    def Test_0001(self):
        """Reads csv files of both epoch summary and Calibration Curve,
        Figure 1 plots all 4 different graphs
        Figure 2 plots the confusion matrix."""

        epoch_summary = pd.read_csv("Test_5.1_LR0-0001/epoch_summary_T5.1.csv")
        CC = pd.read_csv("Test_5.1_LR0-0001/CC_V_T5.1.csv")

        fig1 = self.Plotting(epoch_summary,CC,LR=0.0001) 
        fig2 = self.ConfusionMatrix(file_name_1 = "Test_5.1_LR0-0001/CM_T5.1/CM_e", file_name_2 = "_T5.1.npz",LR=0.0001)

        plt.show()
    
    def Plotting(self,epoch_summary,CC,LR):
        "Handles Plotting all 4 graphs for figure 1"

        fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(10,5))   # Initializing a (2,2) figure

        fig = plt.suptitle(f"Model's results from a LR of {LR}",size=30)
        fig = plt.subplots_adjust(hspace=0.5)

        epoch = self.Epoch(epoch_summary)   # getting epoch values from specific Test

        #======================================================================================================
        # LOSS
        #======================================================================================================
        ax1 = self.Loss(epoch,epoch_summary,ax1)    # Loss Subplot
        ax1.set_title("Loss vs Epoch")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.legend()
        ax1.grid(True)

        #======================================================================================================
        # ACCURACY
        #======================================================================================================
        ax2 = self.Accuracy(epoch,epoch_summary,ax2)    # Accuracy Subplot
        ax2.set_title("Accuracy vs Epoch")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        ax2.legend()
        ax2.grid(True)
        
        #======================================================================================================
        # CALIBRATION CURVE
        #======================================================================================================
        ax3 = self.Calibration(CC,ax3)  # Calibration Curve Subplot
        ax3.set_title("Calibration Cruve")
        ax3.set_ylabel("Preidicted Probablity per Bin")
        ax3.set_xlabel("True Accuracy per Bin")
        ax3.grid(True)
        
        #======================================================================================================
        # NORMALIZED GRADIENT
        #======================================================================================================
        ax4 = self.NormGrad(epoch,epoch_summary,ax4)    # Normalized Gradient Subplot
        ax4.set_title("Normalized Gradient vs Epoch")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Normalized Gradient")
        ax4.grid(True)

        return fig

    def Epoch(self,epoch_summary):
        """Returns the number of epochs ran by a specific test"""

        epoch = epoch_summary['Epoch']

        return epoch

    def Loss(self,epoch,epoch_summary,ax):
        """Returns the subplot for both training and validation loss"""

        train_loss = epoch_summary['Train Loss']
        val_loss = epoch_summary['Val Loss']

        ax.plot(epoch,train_loss,label="Training",color="Blue")
        ax.plot(epoch,val_loss,label="Validation",color="Green")

        return ax

    def Accuracy(self,epoch,epoch_summary,ax):
        """Returns the training and validation accuracy"""

        train_acc = epoch_summary['Train Acc']
        train_acc = train_acc.astype(float)

        val_acc = epoch_summary['Val Acc']
        val_acc = val_acc.astype(float)   

        ax.plot(epoch,train_acc,label="Training",color="Blue")
        ax.plot(epoch,val_acc,label="Validation",color="Green")

        return ax
    
    def ConfusionMatrix(self,file_name_1,file_name_2,LR):
        """Returns the figure for Confusion Matrix."""

        fig1, axes = plt.subplots(2,3,figsize=(8,8))    # Initializing (2,3) figure
        fig1.suptitle(f"Confusion Matrix every 10 Epochs with a LR of {LR}")
        fig1.subplots_adjust(hspace=0.5)
        
        # Storing each subplot in a dictionary
        axes_dict = {"ax1":axes[0,0],
                     "ax2":axes[0,1],
                     "ax3":axes[0,2],
                     "ax4":axes[1,0],
                     "ax5":axes[1,1],
                     "ax6":axes[1,2],}
        
        epoch = numpy.array([1,10,20,30,40,50]) # Epoch Numbers

        #====================================================================================
        # EPOCH #1
        #====================================================================================
        data = numpy.load(f"{file_name_1}1{file_name_2}")
        correct_matrix = data['confusion_matrx']

        incorrect_matrix = correct_matrix.copy()

        mask= numpy.eye(len(correct_matrix),dtype=bool) # Boolean Identity Matrix

        correct_matrix = numpy.where(mask,correct_matrix,numpy.nan) # Replaces all false values (incorrect predictions) with nan
        
        initial_correct_max = numpy.nanmax(correct_matrix)  # getting max value after replacing incorrect predictions with nan
        correct_matrix = correct_matrix/initial_correct_max # Normalizing all values between 0-1


        numpy.fill_diagonal(incorrect_matrix,numpy.nan) # Replaces all true values (correct predictions) with nan
        initial_incorrect_max = numpy.nanmax(incorrect_matrix)  # max value after replacing correct predictions with nan
        incorrect_matrix = incorrect_matrix/initial_incorrect_max   # Normalizing all values between 0-1

        # Heatmap for correct predictions only
        sns.heatmap(correct_matrix,cbar=False, annot=False, cmap='Greys', vmin=0,vmax=1, fmt='.1f', ax = axes_dict[f'ax1'], annot_kws={'size': 10})

        #Heatmap for incorrect predictions only
        sns.heatmap(incorrect_matrix,cbar=False, annot=False, cmap='Oranges',vmin=0,vmax=1, fmt='.1f', ax = axes_dict[f'ax1'], annot_kws={'size': 10})

        # Universal color bar for both heatmaps
        cbar = fig1.colorbar(axes[0,0].collections[0], ax = axes, orientation='vertical',fraction=0.02,pad=0.03)
        cbar = fig1.colorbar(axes[0,0].collections[1], ax = axes, orientation='vertical',fraction=0.02,pad=0.03)

        axes_dict[f'ax1'].set_title(f"Epoch 1")
        axes_dict[f'ax1'].set_xlabel("Predicted Label")
        axes_dict[f'ax1'].set_ylabel("True Label")
        axes_dict[f'ax1'].set_xticks(numpy.arange(10)+0.5)
        axes_dict[f'ax1'].set_xticklabels(numpy.arange(10))

        #====================================================================================
        # EPOCH #10,20,30,40,50
        #====================================================================================
        for subplot_num in range (2,7):
            data = numpy.load(f"{file_name_1}{epoch[subplot_num-1]}{file_name_2}")
            correct_matrix = data['confusion_matrx']

            
            incorrect_matrix = correct_matrix.copy()

            mask= numpy.eye(len(correct_matrix),dtype=bool) # Boolean Identity Matrix

            correct_matrix = numpy.where(mask,correct_matrix,numpy.nan) # Replaces all false values (incorrect predictions) with nan
            
            correct_matrix = correct_matrix/initial_correct_max # Normalizing all values between 0-1, SCALED BY FIRST EPOCH (used as reference)

            numpy.fill_diagonal(incorrect_matrix,numpy.nan) # Replaces all true values (correct predictions) with nan
            incorrect_matrix = incorrect_matrix/initial_incorrect_max   # Normalizing all values between 0-1, SCALED BY FIRST EPOCH (used as reference)

            # Heatmap for correct predictions only
            sns.heatmap(correct_matrix,cbar=False, annot=False, cmap='Greys', vmin=0,vmax=1, fmt='.1f', ax = axes_dict[f'ax{subplot_num}'], annot_kws={'size': 10})
            # Heatmap for icorrect predictions only
            sns.heatmap(incorrect_matrix,cbar=False, annot=False, cmap='Oranges',vmin=0,vmax=1, fmt='.1f', ax = axes_dict[f'ax{subplot_num}'], annot_kws={'size': 10})

            

            axes_dict[f'ax{subplot_num}'].set_title(f"Epoch {epoch[subplot_num-1]}")
            axes_dict[f'ax{subplot_num}'].set_xlabel("Predicted Label")
            axes_dict[f'ax{subplot_num}'].set_ylabel("True Label")
            axes_dict[f'ax{subplot_num}'].set_xticks(numpy.arange(10)+0.5)
            axes_dict[f'ax{subplot_num}'].set_xticklabels(numpy.arange(10))

        
        return fig1
    
    def Calibration(self,CC,ax):
        "Returns subplot for calibration curve"
        val_pp = pd.Series(numpy.zeros(10))
        val_acc_bin = pd.Series(numpy.zeros(10))

        #====================================================================================
        # PREDICTED PROBABILITY
        #====================================================================================
        #Bins 0 - 8
        for bin in range(1,9):
            val_pp[bin] = (CC[f'PP0.{bin}-0.{bin+1}'].sum())/len(CC)    # Calculating average predicted probability for each bin between 0-8
        #Bin 9
        val_pp[9] = (CC['PP0.9-1.0'].sum())/len(CC) # Calculating average predicted probability for bin 9

        #====================================================================================
        # ACCURACY
        #====================================================================================
        #Bins 0 - 8
        for bin in range(1,9):
            val_acc_bin[bin] = (CC[f'A0.{bin}-0.{bin+1}'].sum())/len(CC)    # Calculating average predicted probability for each bin between 0-8
        #Bin 9
        val_acc_bin[9] = (CC['A0.9-1.0'].sum())/len(CC) # Calculating average predicted probability for bin 9
        
        # Computing diagonal line
        x = numpy.linspace(0,100,11)
        y = x

        ax.plot(val_acc_bin*100,val_pp*100,color="brown")
        ax.plot(x,y,color="Black")

        return ax
    
    def NormGrad(self,epoch,epoch_summary,ax):
        """Returns subplot for normalized gradient"""
        norm_grad = epoch_summary['Grad Norm']

        ax.plot(epoch,norm_grad,color="red")    
        return ax

class betweenTest:
    def __init__(self):
        pass

    #Handles Loss across all 5 tests

    #Handles Acuracy  across all 5 tests

    #CM across all 5 tests

    #Calibration across all 5 tests


#Test1(epoch_summary_T1,CC_V_T1)
#Test2(epoch_summary_T2,CC_V_T2)
# Loss_Compare(epoch_summary_T1_1, epoch_summary_T2_1,epoch_summary_T3_1)
#lr_finder()

plot = Test()
# plot.Test_005()
# plot.Test_001()
# plot.Test_01()
# plot.Test_0001()

plot.Test_0001()
