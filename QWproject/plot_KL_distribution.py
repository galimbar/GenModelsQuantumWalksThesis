from matplotlib import pyplot as plt
import numpy as np

def vis_KL_list(KL_score_list, movement_list, pc, save_path):
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 5))
    ax1.hist(KL_score_list)
    ax1.set(xlabel = "asymmetric KL value", ylabel = "Count")
    ax1.set_title("KL values\nmean = " + str(round(np.mean(KL_score_list), 3)) + ", std = " + str(round(np.std(KL_score_list), 3)))
    ax2.scatter(movement_list, KL_score_list)
    ax2.set(xlabel ="movement on PC", ylabel = "KL score")
    ax2.set_title("KL score vs movement")
    plt.suptitle("KL results for PC number " + str(pc))
    plt.tight_layout()
    plt.savefig(save_path + str("KL_results_pc_" + str(pc) + ".png"), dpi=400)
    
