import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc # roc curve tools
from skimage.filters import try_all_threshold, threshold_mean, threshold_minimum, threshold_otsu, threshold_isodata
from skimage.morphology import disk, ball
from skimage.filters.rank import otsu

def make_confusion_matrix(ax, cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          cmap='Blues',
                          title=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.
    Arguments
    ---------
    cf:            confusion matrix to be passed in
    group_names:   List of strings that represent the labels row by row to be shown in each square.
    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'
    count:         If True, show the raw number in the confusion matrix. Default is True.
    normalize:     If True, show the proportions for each category. Default is True.
    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.
    xyticks:       If True, show x and y ticks. Default is True.
    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.
    sum_stats:     If True, display summary statistics below the figure. Default is True.
    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.
    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.
    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        tn, fp, fn, tp = cf.ravel()
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = (tn+tp)*100/(tp+tn+fp+fn) 

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = tp*100/(tp+fp)
            recall    = tp*100/(tp+fn)
            f1_score  = (2*precision*recall)/(precision+recall)
            specificity = tn*100/(tn+fp)
            misclassification = (fp+fn)*100/(fp+fn+tp+tn)
            stats_text = "\n\nAccuracy={:0.2f}%:\nPrecision={:0.2f}%:\nRecall={:0.2f}%:\nF1 Score={:0.2f}%:\nSpecificity={:0.2f}%:\nMisclassification={:0.2f}%:".format(
                accuracy,precision,recall,f1_score, specificity, misclassification)
        else:
            stats_text = "\n\nAccuracy={:0.2f}%:".format(accuracy)
    else:
        stats_text = ""

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories, ax=ax)

    ax.text(0.5, 1.25,stats_text, bbox={'facecolor': 'gray', 'alpha': 0.5})
    ax.set_title(title)
    

    
def plot_roc_curve(ax, mask_truth, recon, label=""):
    ground_truth_labels = mask_truth.ravel() # we want to make them into vectors
    score_value = recon.ravel() # we want to make them into vectors
    fpr, tpr, _ = roc_curve(ground_truth_labels,score_value)
    roc_auc = auc(fpr,tpr)
    roc_formatted = "{:.2f}".format(roc_auc)
    ax.plot(fpr, tpr, label=f'ROC curve {label} (area = {roc_formatted})')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver operating characteristic curve')
    ax.legend(loc="lower right")

def plot_confusion_matrix(ax, mask_truth, mask_recon, label=""):
    y_true = mask_truth.astype(np.uint8).ravel()
    y_pred = mask_recon.astype(np.uint8).ravel()
    cm=confusion_matrix(y_true, y_pred)

    labels = ["True Neg","False Pos","False Neg","True Pos"]
    categories = ["Background", "Particle"]
    make_confusion_matrix(ax,cm, 
                          group_names=labels,
                          categories=categories, 
                          cmap="Blues", percent=False, title=label)
    
def plot_images(axes, images_to_plot, titles):
    for i,f in enumerate(titles):
        ax = axes.ravel()
        ax[i].imshow(images_to_plot[i,:,:], cmap="gray")
        ax[i].set_title(f' {f}')
        
def get_mask(data, t):
    mask = np.zeros_like(data)
    for i in range(data.shape[0]):
        thresh = t(data[i,:,:])
        mask[i,:,:] = data[i,:,:] > thresh
    return mask
            
def plot_results(recon, ground_truth, titles, t_recon, t_gt):
    fig = plt.figure(constrained_layout=True)
    subfigs = fig.subfigures(4, 1,hspace=2, wspace=0.7)
    axsRecon = subfigs[0].subplots(1, 4)
    axsReconMask = subfigs[1].subplots(1, 4)
    axsGTMask = subfigs[2].subplots(1, 4)
    axsDiffMasks = subfigs[3].subplots(1, 4)

    subfigs[0].set_facecolor('0.75')
    subfigs[1].set_facecolor('0.75')
    subfigs[2].set_facecolor('0.75')
    subfigs[3].set_facecolor('0.75')

    subfigs[0].suptitle('Reconstruction', fontsize='x-large')
    subfigs[1].suptitle('Reconstruction Mask (R)', fontsize='x-large')
    subfigs[2].suptitle('Ground Truth Mask (G)', fontsize='x-large')
    subfigs[3].suptitle('G-R', fontsize='x-large')


    plot_images(axsRecon, recon, titles)
    plot_images(axsReconMask, get_mask(recon, t_recon), titles)
    plot_images(axsGTMask, get_mask(ground_truth, t_gt), titles)
    plot_images(axsDiffMasks, get_mask(ground_truth, t_gt)-get_mask(recon, t_recon), titles)
    
def plot_stats(ground_truth, recon, titles, t_recon, t_gt):
    fig = plt.figure(constrained_layout=True, figsize=(12, 25))
    subfigs = fig.subfigures(2, 1, wspace=0.07)
    axsRight = subfigs[1].subplots(1, 1)
    axsLeft = subfigs[0].subplots(2, 2)
    subfigs[0].set_facecolor('0.75')
    subfigs[1].set_facecolor('0.75')

    mask_truth = get_mask(ground_truth, t_recon)
    mask_recon = get_mask(recon, t_gt)

    for i,ax in enumerate(axsLeft.ravel()):
        plot_confusion_matrix(ax, mask_truth[i], mask_recon[i], label=titles[i])

    for i, l in enumerate(titles):
        plot_roc_curve(axsRight, mask_truth[i], recon[i], label=l)

def argand(a, color):
    for x in range(len(a)):
        plt.plot([0,a[x].real],[0,a[x].imag],'ro-',label='python', color = color, alpha=0.2)
    limit=np.max(np.ceil(np.absolute(a))) # set limits for axis
    plt.xlim((-limit,limit))
    plt.ylim((-limit,limit))
    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    
def P2R(radii, angles):
    return radii * np.exp(1j*angles)