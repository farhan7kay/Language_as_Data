# Plotting the Loss and Perplexity
import matplotlib.pyplot as plt


def plot_two(list1,label1,list2,label2,axLabel1 = None, axLabel2 = None,save = False):
    if not axLabel1:
        axLabel1 = ("Epoch","Loss")
    if not axLabel2:
        axLabel2 = ("Epoch","Perplexity")
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(list1, label=label1) #linestyle='dashed',marker="o"
    plt.title(label1)
    plt.xlabel(axLabel1[0])
    plt.ylabel(axLabel1[1])
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(list2, label=label2,)
    plt.title(label2)
    plt.xlabel(axLabel2[0])
    plt.ylabel(axLabel2[1])
    plt.legend()
    if save:
        plt.savefig('temp.png')
    plt.show()