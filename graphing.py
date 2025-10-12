import numpy
import matplotlib.pyplot as plt
import idx2numpy
import time
import pandas as pd

epoch_summary = pd.read_csv("epoch_summary.csv")

epoch = epoch_summary['Epoch']

train_loss = epoch_summary['Train Loss']
val_loss = epoch_summary['Val Loss']

train_acc = epoch_summary['Train Acc'].str.replace('%','',regex=False)
train_acc = train_acc.astype(float)
val_acc = epoch_summary['Val Acc'].str.replace('%','',regex=False)
val_acc = val_acc.astype(float)

norm_grad = epoch_summary['Grad Norm']


fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,5))
fig.suptitle("Test 1: Fixed Learning Rate of 0.01")

ax1.plot(epoch,train_loss,label='Training',color='blue')
ax1.plot(epoch,val_loss,label='Validation', color ='green')
ax1.set_title("Loss Value Per Epoch")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss Value")
ax1.legend()

ax2.plot(epoch,train_acc,label='Training',color='blue')
ax2.plot(epoch,val_acc,label='Validation', color ='green')
ax2.set_title("Accuracy Per Epoch")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()

ax3.plot(epoch,norm_grad,label='Normalized Gradient',color='brown')
ax3.set_title("Normalized Gradient Per Epoch")
ax3.set_xlabel("Epoch")
ax3.set_ylabel("Normalized Gradient")
ax3.legend()



plt.tight_layout()
plt.show()
