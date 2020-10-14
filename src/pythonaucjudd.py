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
    map1 = saliencyMap.astype(float)
    map2 = fixationMap.astype(float)

    # make sure maps have the same shape
    from skimage.transform import resize
    map1 = resize(map1, np.shape(map2))

    # make sure map1 and map2 sum to 1
    if map1.any():
        map1 = map1 / map1.sum()
    if map2.any():
        map2 = map2 / map2.sum()

    # compute KL-divergence
    eps = 10 ** -12
    score = map2 * np.log(eps + map2 / (map1 + eps))

    return score.sum()


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


gt_directory = '/Users/ludo/Documents/Semester Project/salicon-api-master/3-sliced_fixations/500/'#'/Users/aydemir/Desktop/saliencyautoencoder/saliency-master/data/mit1003/fixations/'
sm_directory = '/Users/ludo/Documents/Semester Project/salicon-api-master/3-sliced_predictions/500/'#'/Users/aydemir/Documents/matlab_saliency/saliency/outputs_default_relu'

gt_path = os.path.join(gt_directory)#, str(i))
sm_path = os.path.join(sm_directory)#, str(i))
print(gt_path)
print(sm_path)
#Now to sort based on their file number. The "key" parameter in sorted is a function based on which the sorting will happen (I use split to exclude the jpg/png from the).

gt_files_sorted = [f for f in sorted(os.listdir(gt_path)) if f != ".DS_Store"]#, key = lambda x: int(x.split(".")[0]) )
sm_files_sorted = [f for f in sorted(os.listdir(gt_path)) if f != ".DS_Store"]#, key = lambda x: int(x.split(".")[0]) )

AUC_score = 0
KL_score = 0
NSS_score = 0
gt_related =[]
sm_related = []
# for sm in sm_files_sorted:
#     if "DS_Store" in sm:
#              continue
#     sm_related.append(sm)

# for gt in gt_files_sorted:

#     if "DS_Store" in gt:
#              continue
#     gt_related.append(gt)

# for sm in sm_files_sorted:
#     for gt in gt_files_sorted:
#         if "DS_Store" in sm:
#             continue
#         else:
#             if "DS_Store" in gt:
#                 continue
#             elif sm.split(".")[0] in gt :
#                 sm_related.append(sm)
#                 gt_related.append(gt)
#                 break



#print(len(sm_related))
#print(len(gt_related))
idx = 0
pack = zip(gt_files_sorted, sm_files_sorted)
for p in pack :
    img_sm = cv2.imread(os.path.join(sm_path, p[1]),cv2.IMREAD_GRAYSCALE)
    img_gt = cv2.imread(os.path.join(gt_path, p[0]),cv2.IMREAD_GRAYSCALE)
    AUC_score += AUC_Judd(img_sm, img_gt)
    KL_score += KLdiv(img_sm, img_gt)
    NSS_score += NSS(img_sm, img_gt)
    idx +=1
    print(idx)
print("AUC Judd: ", AUC_score / idx)
print("KL: ", KL_score / idx)
print("NSS: ", NSS_score / idx)
