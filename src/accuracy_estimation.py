'''
Created on 1 mar 2017
@author: 	Dario Zanca
@summary: 	Collection of functions to compute visual attention metrics for:
                - saliency maps similarity
                    - AUC Judd (Area Under the ROC Curve, Judd version)
                    - KL Kullback Leiber divergence
                    - NSS Normalized Scanpath Similarity
                - scanpaths similarity
'''

#########################################################################################

# IMPORT EXTERNAL LIBRARIES

import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import os
import cv2
import tqdm
from scipy.stats import entropy
#########################################################################################

##############################  saliency metrics  #######################################

#########################################################################################

''' created: Tilke Judd, Oct 2009
    updated: Zoya Bylinskii, Aug 2014
    python-version by: Dario Zanca, Jan 2017
This measures how well the saliencyMap of an image predicts the ground truth human
fixations on the image. ROC curve created by sweeping through threshold values determined
by range of saliency map values at fixation locations;
true positive (tp) rate correspond to the ratio of saliency map values above threshold
at fixation locations to the total number of fixation locations, false positive (fp) rate
correspond to the ratio of saliency map values above threshold at all other locations to
the total number of posible other locations (non-fixated image pixels) '''


def AUC_Judd(saliencyMap, fixationMap, jitter=True, toPlot=False):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map (binary matrix)
    # jitter=True will add tiny non-zero random constant to all map locations to ensure
    # 		ROC can be calculated robustly (to avoid uniform region)
    # if toPlot=True, displays ROC curve

    # If there are no fixations to predict, return NaN
    if not fixationMap.any():
        print('Error: no fixationMap')
        score = float('nan')
        return score

    # make the saliencyMap the size of the image of fixationMap

    if not np.shape(saliencyMap) == np.shape(fixationMap):
        from skimage.transform import resize
        saliencyMap = resize(saliencyMap, np.shape(fixationMap))

    # jitter saliency maps that come from saliency models that have a lot of zero values.
    # If the saliency map is made with a Gaussian then it does not need to be jittered as
    # the values are varied and there is not a large patch of the same value. In fact
    # jittering breaks the ordering in the small values!
    if jitter:
        # jitter the saliency map slightly to distrupt ties of the same numbers
        saliencyMap = saliencyMap + np.random.random(np.shape(saliencyMap)) / 10 ** 7

    # normalize saliency map
    saliencyMap = (saliencyMap - saliencyMap.min()) \
                  / (saliencyMap.max() - saliencyMap.min())
    #cv2.imshow("das",saliencyMap)
    #cv2.waitKey(0)
    if np.isnan(saliencyMap).all():
        print('NaN saliencyMap')
        score = float('nan')
        return score

    S = saliencyMap.flatten()
    F = fixationMap.flatten()

    Sth = S[F > 0]  # sal map values at fixation locations
    Nfixations = len(Sth)
    Npixels = len(S)

    allthreshes = sorted(Sth, reverse=True)  # sort sal map values, to sweep through values
    tp = np.zeros((Nfixations + 2))
    fp = np.zeros((Nfixations + 2))
    tp[0], tp[-1] = 0, 1
    fp[0], fp[-1] = 0, 1





    for i in range(Nfixations):
        thresh = allthreshes[i]
        aboveth = (S >= thresh).sum()  # total number of sal map values above threshold
        tp[i + 1] = float(i + 1) / Nfixations  # ratio sal map values at fixation locations
        # above threshold
        fp[i + 1] = float(aboveth - i) / (Npixels - Nfixations)  # ratio other sal map values
        # above threshold

    score = np.trapz(tp, x=fp)
    allthreshes = np.insert(allthreshes, 0, 0)
    allthreshes = np.append(allthreshes, 1)

    if toPlot:
        import matplotlib.pyplot as plt
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.matshow(saliencyMap, cmap='gray')
        ax.set_title('SaliencyMap with fixations to be predicted')
        [y, x] = np.nonzero(fixationMap)
        s = np.shape(saliencyMap)
        plt.axis((-.5, s[1] - .5, s[0] - .5, -.5))
        plt.plot(x, y, 'ro')

        ax = fig.add_subplot(1, 2, 2)
        plt.plot(fp, tp, '.b-')
        ax.set_title('Area under ROC curve: ' + str(score))
        plt.axis((0, 1, 0, 1))
        plt.show()

    return score
''' created: Zoya Bylinskii, Aug 2014
    python-version by: Dario Zanca, Jan 2017
This finds the KL-divergence between two different saliency maps when viewed as
distributions: it is a non-symmetric measure of the information lost when saliencyMap
is used to estimate fixationMap. '''


def KLdiv(saliencyMap, fixationMap):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map

    # convert to float
    eps = 1e-12
    map1 = eps + saliencyMap.astype(float)
    map2 = eps + fixationMap.astype(float)

    # make sure maps have the same shape
    if map1.shape != map2.shape:
        from skimage.transform import resize
        map1 = resize(map1, map2.shape)

    # make sure map1 and map2 sum to 1
    map1 = map1 / map1.sum()
    map2 = map2 / map2.sum()

    return (map2 * np.log(map2 / map1)).sum()


######################################################################################

''' created: Zoya Bylinskii, Aug 2014
    python-version by: Dario Zanca, Jan 2017
This finds the normalized scanpath saliency (NSS) between two different saliency maps.
NSS is the average of the response values at human eye positions in a model saliency
map that has been normalized to have zero mean and unit standard deviation. '''


def NSS(saliencyMap, fixationMap):
    # saliencyMap is the saliency map
    # fixationMap is the human fixation map (binary matrix)

    # If there are no fixations to predict, return NaN
    if not fixationMap.any():
        print('Error: no fixationMap')
        score = float('nan')
        return score

    # make sure maps have the same shape
    from skimage.transform import resize
    map1 = resize(saliencyMap, np.shape(fixationMap))
    if not map1.max() == 0:
        map1 = map1.astype(float) / map1.max()

    # normalize saliency map
    if not map1.std(ddof=1) == 0:
        map1 = (map1 - map1.mean()) / map1.std(ddof=1)

    # mean value at fixation locations
    score = map1[fixationMap > 0].mean()

    return score


#########################################################################################

##############################  scanpaths metrics  ######################################

#########################################################################################

''' created: Dario Zanca, July 2017
    Implementation of the Euclidean distance between two scanpath of the same length. '''


def euclidean_distance(human_scanpath, simulated_scanpath):
    if len(human_scanpath) == len(simulated_scanpath):

        dist = np.zeros(len(human_scanpath))
        for i in range(len(human_scanpath)):
            P = human_scanpath[i]
            Q = simulated_scanpath[i]
            dist[i] = np.sqrt((P[0] - Q[0]) ** 2 + (P[1] - Q[1]) ** 2)
        return dist

    else:

        print('Error: The two sequences must have the same length!')
        return False

################################################################################

slices = [500, 3000, 5000]

gt_directory = '/Users/ludo/Documents/Semester Project/TemporalSaliencyPrediction/data/3-sliced_maps/'
fx_directory = '/Users/ludo/Documents/Semester Project/TemporalSaliencyPrediction/data/3-sliced_fixations/'
sm_directory = '/Users/ludo/Documents/Semester Project/TemporalSaliencyPrediction/data/3-sliced_mdsem_preds/'

gt_path = os.path.join(gt_directory)
fx_path = os.path.join(fx_directory)
sm_path = os.path.join(sm_directory)

filenames = [f for f in sorted(os.listdir(sm_path + str(slices[0]))) if f != ".DS_Store"][:10]

AUC_score = 0
KL_score = 0
NSS_score = 0
gt_related =[]
sm_related = []

for filename in tqdm.tqdm(filenames):
    for slice in slices:
        img_gt = cv2.imread(gt_path + str(slice) + '/' + filename,cv2.IMREAD_GRAYSCALE)
        img_fx = cv2.imread(fx_path + str(slice) + '/' + filename,cv2.IMREAD_GRAYSCALE)
        img_sm = cv2.imread(sm_path + str(slice) + '/' + filename,cv2.IMREAD_GRAYSCALE)

        AUC_score += AUC_Judd(img_sm, img_fx)
        KL_score += KLdiv(img_sm, img_gt)
        NSS_score += NSS(img_sm, img_fx)

n = len(filenames) * len(slices)
print("AUC Judd: ", AUC_score / n)
print("KL: ", KL_score / n)
print("NSS: ", NSS_score / n)
