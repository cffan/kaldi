import scipy.io
import numpy as np
import tensorflow as tf
import os
from pathlib import Path
import matplotlib.pyplot as plt
from neuralDecoder.utils.handwritingDataUtils import *

def formatSessionGeneral(sessionName, rawDataDir, partitionFolder, tfDataFolder, partitionSuffix='', featType='sp_and_tx', includeIFG=False, cutEnd=False, heldOutBlocks=False, cumulativeMean=False, rollingMean=False, addMeanDriftNoise=False, useRawRedisData=True, noMeanAdaptation=False, addInterWordSil=True):
    
    if useRawRedisData:
        lrr_mat = scipy.io.loadmat(rawDataDir+'/'+sessionName+'_20ms_sentences_raw.mat')
    else:
        lrr_mat = scipy.io.loadmat(rawDataDir+'/'+sessionName+'_20ms_sentences_lrr.mat')

    if cutEnd:
        binsToCut = scipy.io.loadmat(rawDataDir+'/'+sessionName+'_binsToRemoveFromEnd.mat')
    else:
        binsToCut = []
        
    input_features = []
    transcriptions = []
    frame_lens = []
    block_means = []
    block_stds = []
    n_trials = lrr_mat['zeroPaddedSentences'].shape[0]

    for i in range(n_trials):    
        features = lrr_mat['zeroPaddedSentences'][i]

        if cutEnd:
            binsToRemove = binsToCut['binsToRemoveFromEnd'][0, i]-40
            if binsToRemove<0:
                binsToRemove = 0
        else:
            binsToRemove = 0

        sentence_len = (lrr_mat['sentenceDurations'][i, 0] - binsToRemove).astype(np.int32)
        if includeIFG:
            features_tx = features[0:sentence_len, 0:256].astype(np.float32)
            features_sp = features[0:sentence_len, 256:512].astype(np.float32)
        else:
            features_tx = features[0:sentence_len, 0:128].astype(np.float32)
            features_sp = features[0:sentence_len, 256:384].astype(np.float32)

        sentence = lrr_mat['sentences'][i][0][0]

        if featType=='sp_and_tx':
            input_features.append(np.concatenate([features_tx, features_sp], axis=1))
        elif featType=='tx_only':
            input_features.append(features_tx)

        transcriptions.append(sentence)
        frame_lens.append(sentence_len)
        
    blockList = np.unique(lrr_mat['blockNum'])
    blocks = []
    sentBlocks = lrr_mat['blockNum'][lrr_mat['goTrialEpochs'][:,0]]

    for b in range(len(blockList)):
        sentIdx = np.argwhere(sentBlocks==blockList[b])
        sentIdx = sentIdx[:,0].astype(np.int32)
        blocks.append(sentIdx)
        
    if partitionSuffix=='':
        ttp = scipy.io.loadmat(partitionFolder+'/'+sessionName+'.mat')
    else:
        ttp = scipy.io.loadmat(partitionFolder+'/'+sessionName+'_'+partitionSuffix+'.mat')
    
    if heldOutBlocks:
        #hold out block numbers
        trainPartitionIdx = []
        trainBlocks = np.squeeze(ttp['trainBlocks'])
        for x in range(len(trainBlocks)):
            blockIdx = np.argwhere(trainBlocks[x]==blockList)
            trainPartitionIdx.append(blocks[blockIdx[0,0]])

        testPartitionIdx = []
        testBlocks = np.squeeze(ttp['testBlocks'])
        if testBlocks.shape==():
            testBlocks = [testBlocks]

        for x in range(len(testBlocks)):
            blockIdx = np.argwhere(testBlocks[x]==blockList)
            testPartitionIdx.append(blocks[blockIdx[0,0]])

        trainPartitionIdx = np.concatenate(trainPartitionIdx)
        testPartitionIdx = np.concatenate(testPartitionIdx)

    else:
        #random held out trials
        trainPartitionIdx = np.squeeze(ttp['trainTrials'])
        testPartitionIdx = np.squeeze(ttp['testTrials'])

    print(trainPartitionIdx)
    print(testPartitionIdx)

    if heldOutBlocks and not noMeanAdaptation:
        input_features_raw = input_features.copy()

        if cumulativeMean:
            candidateBlocks = np.concatenate([trainBlocks, testBlocks])
        else:
            candidateBlocks = trainBlocks

        #compute block-specific means and mean-sbutract the training blocks
        allMeans = []
        if_train = []
        for b in range(len(candidateBlocks)):
            blockIdx = np.argwhere(candidateBlocks[b]==blockList)
            blockIdx = blockIdx[0,0]

            feats = np.concatenate(input_features[blocks[blockIdx][0]:(blocks[blockIdx][-1]+1)], axis=0)
            feats_mean = np.mean(feats, axis=0, keepdims=True)
            feats_std = np.std(feats, axis=0, keepdims=True)
            allMeans.append(feats_mean)

            if np.any(candidateBlocks[b]==testBlocks):
                continue

            for i in blocks[blockIdx]:
                input_features[i] = (input_features[i] - feats_mean) #/ (feats_std + 1e-8)
                if_train.append(input_features[i])

        all_std = np.std(np.concatenate(if_train, axis=0), axis=0, keepdims=True) + 1e-8

        #mean-subtract the testing blocks with the closest available prior block
        for b in range(len(testBlocks)):
            blockIdx = np.argwhere(testBlocks[b]==blockList)
            blockIdx = blockIdx[0,0]

            validBlocks = candidateBlocks[candidateBlocks<testBlocks[b]] 
            meanBlockIdx = validBlocks[-1]
            meanBlockIdx = np.argwhere(meanBlockIdx==candidateBlocks)
            meanBlockIdx = meanBlockIdx[0,0]

            sentIdx = 0
            for i in blocks[blockIdx]:
                #rolling mean, take from the last 5 sentences
                nSentMin = 10
                nSentMax = 20
                sentIdx += 1
                if sentIdx>nSentMin and rollingMean:
                    nToTake = min(sentIdx-1, nSentMax)
                    feats = np.concatenate(input_features_raw[(i-nToTake):i], axis=0)
                    feats_mean = np.mean(feats, axis=0, keepdims=True)
                    meanToSubtract = feats_mean
                else:
                    meanToSubtract = allMeans[meanBlockIdx]

                input_features[i] = (input_features[i] - meanToSubtract)

        for i in range(len(input_features)):
            input_features[i] = input_features[i] / all_std

        if addMeanDriftNoise:
            for b in range(len(testBlocks)):
                blockIdx = np.argwhere(testBlocks[b]==blockList)
                blockIdx = blockIdx[0,0]

                for i in blocks[blockIdx]:
                    meanDirftNoise = np.random.randn(1,input_features[0].shape[1])*0.6
                    input_features[i] = (input_features[i] + meanDirftNoise)        

    elif not noMeanAdaptation:
        for b in range(len(blocks)):
            feats = np.concatenate(input_features[blocks[b][0]:(blocks[b][-1]+1)], axis=0)
            feats_mean = np.mean(feats, axis=0, keepdims=True)
            feats_std = np.std(feats, axis=0, keepdims=True)
            for i in blocks[b]:
                input_features[i] = (input_features[i] - feats_mean) #/ (feats_std + 1e-8)

        all_std = np.std(np.concatenate(input_features, axis=0), axis=0, keepdims=True) + 1e-8
        for i in range(len(input_features)):
            input_features[i] = input_features[i] / all_std

    plt.figure(figsize=(30, 5))
    plt.imshow(np.concatenate(input_features, 0).T, aspect='auto',
               interpolation='none')
    plt.ylabel('neurons')
    plt.xlabel('time [20ms]')
    plt.tight_layout()

    session_data = {
        'inputFeatures': input_features,
        'transcriptions': transcriptions,
        'frameLens': frame_lens
    }

    suffix = partitionSuffix
    
    if featType=='sp_and_tx':
        suffix+='spikePow'
    elif featType=='tx_only':
        suffix+='txOnly'

    if includeIFG:
        suffix+='_ifg'

    if not cutEnd:
        suffix+='_noCut'

    if heldOutBlocks:
        suffix+='_heldOutBlocks'

    if addMeanDriftNoise:
        suffix += '_mdNoise'

    if useRawRedisData:
        suffix += '_rawRedis'

    if cumulativeMean:
        suffix += '_cMean'

    if rollingMean:
        suffix +='_rMean'

    if noMeanAdaptation:
        suffix += '_noZScore'

    if addInterWordSil:
        suffix += '_iwSil'

    fileName = tfDataFolder+'/'+sessionName+'_'+suffix
    convertToTFRecord(session_data, 
                      fileName,
                      trainTrials=trainPartitionIdx, 
                      testTrials=testPartitionIdx,
                      convertToPhonemes=True, vowelOnly=False, consonantOnly=False, addInterWordSymbol=addInterWordSil)
    
    print(fileName)

    
