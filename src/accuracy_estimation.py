import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import os
import cv2
import tqdm

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


def AUC_Judd(saliencyMap, fixationMap, jitter=True):
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

    # jitter saliency maps that come from saliency models that have a lot of zero values.
    # If the saliency map is made with a Gaussian then it does not need to be jittered as
    # the values are varied and there is not a large patch of the same value. In fact
    # jittering breaks the ordering in the small values!
    if jitter:
        # jitter the saliency map slightly to distrupt ties of the same numbers
        saliencyMap = saliencyMap + np.random.random(np.shape(saliencyMap)) / 10 ** 7

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

    return score

################################################################################

def AUC(gtsAnn, resAnn, stepSize=.01, Nrand=100000):
    """
    Computer AUC score.
    :param gtsAnn : ground-truth annotations
    :param resAnn : predicted saliency map
    :return score: int : score
    """

    salMap = resAnn - np.min(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.max(salMap)

    S = salMap.reshape(-1)
    Sth = np.asarray([ salMap[y-1][x-1] for y,x in gtsAnn ])

    Nfixations = len(gtsAnn)
    Npixels = len(S)

    # sal map values at random locations
    randfix = S[np.random.randint(Npixels, size=Nrand)]

    allthreshes = np.arange(0,np.max(np.concatenate((Sth, randfix), axis=0)),stepSize)
    allthreshes = allthreshes[::-1]
    tp = np.zeros(len(allthreshes)+2)
    fp = np.zeros(len(allthreshes)+2)
    tp[-1]=1.0
    fp[-1]=1.0
    tp[1:-1]=[float(np.sum(Sth >= thresh))/Nfixations for thresh in allthreshes]
    fp[1:-1]=[float(np.sum(randfix >= thresh))/Nrand for thresh in allthreshes]

    auc = np.trapz(tp,fp)
    return auc

################################################################################

def SAUC(gtsAnn, resAnn, shufMap, stepSize=.01):
    """
    Computer SAUC score. A simple implementation
    :param gtsAnn : list of fixation annotations
    :param resAnn : list only contains one element: the result annotation - predicted saliency map
    :return score: int : score
    """

    salMap = resAnn - np.min(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.max(salMap)
    Sth = np.asarray([ salMap[y-1][x-1] for y,x in gtsAnn ])
    Nfixations = len(gtsAnn)

    others = np.copy(shufMap)
    for y,x in gtsAnn:
        others[y-1][x-1] = 0

    ind = np.nonzero(others) # find fixation locations on other images
    nFix = shufMap[ind]
    randfix = salMap[ind]
    Nothers = sum(nFix)

    allthreshes = np.arange(0,np.max(np.concatenate((Sth, randfix), axis=0)),stepSize)
    allthreshes = allthreshes[::-1]
    tp = np.zeros(len(allthreshes)+2)
    fp = np.zeros(len(allthreshes)+2)
    tp[-1]=1.0
    fp[-1]=1.0
    tp[1:-1]=[float(np.sum(Sth >= thresh))/Nfixations for thresh in allthreshes]
    fp[1:-1]=[float(np.sum(nFix[randfix >= thresh]))/Nothers for thresh in allthreshes]

    auc = np.trapz(tp,fp)
    return auc

################################################################################

def CC(gtsAnn, resAnn):
    """
    Computer CC score. A simple implementation
    :param gtsAnn : ground-truth fixation map
    :param resAnn : predicted saliency map
    :return score: int : score
    """

    fixationMap = gtsAnn - np.mean(gtsAnn)
    if np.max(fixationMap) > 0:
        fixationMap = fixationMap / np.std(fixationMap)
    salMap = resAnn - np.mean(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.std(salMap)

    return np.corrcoef(salMap.reshape(-1), fixationMap.reshape(-1))[0][1]

################################################################################

def NSS(gtsAnn, resAnn):
    """
    Computer NSS score.
    :param gtsAnn : ground-truth annotations
    :param resAnn : predicted saliency map
    :return score: int : NSS score
    """

    salMap = resAnn - np.mean(resAnn)
    if np.max(salMap) > 0:
        salMap = salMap / np.std(salMap)
    return np.mean([ salMap[y-1][x-1] for y,x in gtsAnn ])

################################################################################

def KL(saliencyMap, fixationMap):
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

slices = [500, 3000, 5000]

gt_directory = '/Users/ludo/Documents/Semester Project/TemporalSaliencyPrediction/data/3-sliced_maps/'
fx_directory = '/Users/ludo/Documents/Semester Project/TemporalSaliencyPrediction/data/3-sliced_fixations/'
sm_directory = '/Users/ludo/Documents/Semester Project/TemporalSaliencyPrediction/data/3-sliced_mdsem_preds/'

gt_path = os.path.join(gt_directory)
fx_path = os.path.join(fx_directory)
sm_path = os.path.join(sm_directory)

filenames = [f for f in sorted(os.listdir(sm_path + str(slices[0]))) if f != ".DS_Store"]

SAUC_score = []
AUC_Judd_score = []
NSS_score = []
CC_score = []
AUC_score = []
KL_score = []

gt = dict()
fx = dict()
sm = dict()
fx_coord = dict()
shufMap = np.zeros((480,640))

for filename in tqdm.tqdm(filenames):
    for slice in slices:
        id = str(slice) + '/' + filename.split('_')[2][:-4]
        img_gt = cv2.imread(gt_path + str(slice) + '/' + filename,cv2.IMREAD_GRAYSCALE) / 255
        img_fx = cv2.imread(fx_path + str(slice) + '/' + filename,cv2.IMREAD_GRAYSCALE) / 255
        img_sm = cv2.imread(sm_path + str(slice) + '/' + filename,cv2.IMREAD_GRAYSCALE) / 255

        gt[id] = img_gt
        fx[id] = img_fx
        sm[id] = img_sm
        fx_coord[id] = np.transpose(np.nonzero(img_fx))
        shufMap += img_fx

for id in tqdm.tqdm(sm.keys()):
    SAUC_score.append(SAUC(fx_coord[id], sm[id], shufMap))
    AUC_Judd_score.append(AUC_Judd(sm[id], fx[id]))
    NSS_score.append(NSS(fx_coord[id], sm[id]))
    CC_score.append(CC(gt[id], sm[id]))
    AUC_score.append(AUC(fx_coord[id], sm[id]))
    KL_score.append(KL(sm[id], gt[id]))

n = len(filenames) * len(slices)
print("SAUC:        ", np.mean(SAUC_score))
print("AUC_Judd:    ", np.mean(AUC_Judd_score))
print("NSS:         ", np.mean(NSS_score))
print("CC:          ", np.mean(CC_score))
print("AUC:         ", np.mean(AUC_score))
print("KL:          ", np.mean(KL_score))
