#!/usr/bin/python
"""
Generate ROC, PR and Correlation curves.
Format of input: Input format: CSV by --delimiter: len(rocord)==4. Ex: modelId, weight, score, label)
"""
from __future__ import division
from __future__ import print_function
from builtins import str
from builtins import zip

__author__ = 'zf'

import sys, os, time
import bz2, gzip
import uuid
import heapq
import numpy as np
import pickle
import random
import argparse
from collections import namedtuple
from multiprocessing import Process, Pool
from matplotlib import pyplot
import pylab

os.environ["DISPLAY"] = ":0.0"

MERGE_DIR = 'merges'
SORT_DIR = 'sorts'
PROCCESSED_DIR = 'proccessed'
PROCCESSED_FILE = os.path.join(PROCCESSED_DIR, 'data.pickle')
RESULT_DIR = 'results'


PHRASE_GROUP = 0
PHRASE_SORT = 1
PHRASE_PROCESS = 2
PHRASE_CHART = 3


def openFile(file):
    """
  Handles opening different types of files (only normal files and bz2 supported)
  """
    file = file.lower()
    if file.endswith('.bz2'):
        return bz2.BZ2File(file)
    elif file.endswith('.gz'):
        return gzip.open(file)
    return open(file)


def ensure_dir(dirName):
    """
  Create directory if necessary.
  """
    if not os.path.exists(dirName):
        os.makedirs(dirName)


def walktree(input):
    """
    Returns a list of file paths to traverse given input (a file or directory name)
    """
    if os.path.isfile(input):
        return [input]
    else:
        fileNames = []
        for root, dirs, files in os.walk(input):
            fileNames += [os.path.join(root, f) for f in files]
        return fileNames


def batchSort(input, output, key, bufferSize):
    """
    External sort on file using merge sort.
    See http://code.activestate.com/recipes/576755-sorting-big-files-the-python-26-way/
    """
    def merge(key=None, *iterables):
        if key is None:
            keyed_iterables = iterables
        else:
            Keyed = namedtuple("Keyed", ["key", "obj"])
            keyed_iterables = [(Keyed(key(obj), obj) for obj in iterable)
                               for iterable in iterables]
        for element in heapq.merge(*keyed_iterables):
            yield element.obj

    from itertools import islice
    tempdir = os.path.join(args.tempdir, str(uuid.uuid4()))
    os.makedirs(tempdir)
    chunks = []
    try:
        with open(input, 'rb', 64 * 1024) as inputFile:
            inputIter = iter(inputFile)
            while True:
                current_chunk = list(islice(inputIter, bufferSize))
                if not current_chunk:
                    break
                current_chunk.sort(key=key)
                output_chunk = open(
                    os.path.join(tempdir, '%06i' % len(chunks)), 'w+b',
                    64 * 1024)
                chunks.append(output_chunk)
                output_chunk.writelines(current_chunk)
                output_chunk.flush()
                output_chunk.seek(0)
        with open(output, 'wb', 64 * 1024) as output_file:
            output_file.writelines(merge(key, *chunks))
    finally:
        for chunk in chunks:
            try:
                chunk.close()
                os.remove(chunk.name)
            except Exception:
                pass
    print("sorted file %s ready" % (output))


def calculateAUC(rocPoints):
    AUC = 0.0
    lastPoint = (0, 0)
    for point in rocPoints:
        AUC += (point[1] + lastPoint[1]) * (point[0] - lastPoint[0]) / 2.0
        lastPoint = point
    return AUC


def plotCurves(dataByModel):
    """
  Plot ROC, PR and Correlation Curves by model.
  """
    prFigure = pyplot.figure()
    configChart()
    prAx = prFigure.add_subplot(111)
    prAx.set_xlabel('Recall')
    prAx.set_ylabel('Precision')
    prAx.set_title('PR Curve')
    prAx.grid(True)

    rocFigure = pyplot.figure()
    configChart()
    rocAx = rocFigure.add_subplot(111)
    rocAx.set_xlabel('Fallout / FPR')
    rocAx.set_ylabel('Recall')
    rocAx.set_title('ROC Curve')
    rocAx.grid(True)

    corrFigure = pyplot.figure()
    configChart()
    corrAx = corrFigure.add_subplot(111)
    corrAx.set_xlabel('predict score')
    corrAx.set_ylabel('real score')
    corrAx.set_title('Correlation Curve')
    corrAx.grid(True)

    precisionFigure = pyplot.figure()
    configChart()
    precisionAx = precisionFigure.add_subplot(111)
    precisionAx.set_xlabel('score')
    precisionAx.set_ylabel('Precision')
    precisionAx.set_title('Threshold score vs precision')
    precisionAx.grid(True)

    recallFigure = pyplot.figure()
    configChart()
    recallAx = recallFigure.add_subplot(111)
    recallAx.set_xlabel('score')
    recallAx.set_ylabel('Recall')
    recallAx.set_title('Threshold score vs recall')
    recallAx.grid(True)

    falloutFigure = pyplot.figure()
    configChart()
    falloutAx = falloutFigure.add_subplot(111)
    falloutAx.set_xlabel('score')
    falloutAx.set_ylabel('Fallout (False Positive Rate)')
    falloutAx.set_title('Threshold score vs fallout')
    falloutAx.grid(True)

    for (model, data) in list(dataByModel.items()):
        (recalls, precisions) = list(zip(*(data['PR'])))
        prAx.plot(recalls, precisions, marker='o', linestyle='--', label=model)

        (fallouts, recalls) = list(zip(*(data['ROC'])))
        rocAx.plot(fallouts, recalls, marker='o', linestyle='--', label=model)

        (pCtrs, eCtrs) = list(zip(*(data['CORR'])))
        corrAx.plot(pCtrs, eCtrs, label=model)

        (score, recall, precision, fallout) = list(zip(*(data['cutoff'])))

        recallAx.plot(score, recall, label=model + '_recall')
        precisionAx.plot(score, precision, label=model + '_precision')
        falloutAx.plot(score, fallout, label=model + '_fallout')

    # saving figures
    ensure_dir(RESULT_DIR)
    prAx.legend(loc='upper right', shadow=True)
    prFigure.savefig('%s/pr_curve.png' % RESULT_DIR)

    rocAx.legend(loc='lower right', shadow=True)
    rocFigure.savefig('%s/roc_curve.png' % RESULT_DIR)

    corrAx.legend(loc='upper left', shadow=True)
    corrFigure.savefig('%s/corr_curve.png' % RESULT_DIR)

    precisionAx.legend(loc='upper left', shadow=True)
    precisionFigure.savefig('%s/precision.png' % RESULT_DIR)

    recallAx.legend(loc='lower left', shadow=True)
    recallFigure.savefig('%s/recall.png' % RESULT_DIR)

    falloutAx.legend(loc='upper right', shadow=True)
    falloutFigure.savefig('%s/fallout.png' % RESULT_DIR)

    pyplot.close()
    pngs = '{result}/pr_curve.png {result}/roc_curve.png {result}/corr_curve.png {result}/precision.png {result}/recall.png {result}/fallout.png'.format(result=RESULT_DIR)
    print('png: ', pngs)


def groupDataByModel(inputDirs):
    """
    Group data to separated file by model.
    """
    t1 = time.time()
    print("merging files by model to %s" % MERGE_DIR)
    ensure_dir(MERGE_DIR)
    fileByModel = dict()
    randomByModel = dict()
    totalLineMerged = 0
    for inputDir in inputDirs:
        for file in walktree(inputDir):
            for line in openFile(file):
                fields = line.split(args.delimiter)
                if args.ignoreInvalid:
                    if len(fields) != 4 or fields[0] == '' or fields[
                            1] == '' or fields[2] == '' or fields[3] == '':
                        print('Ingonre Invalid line', fields)
                        continue
                model = fields[0]
                if model not in fileByModel:
                    fileByModel[model] = open('%s/%s.txt' % (MERGE_DIR, model), 'w')
                    randomByModel[model] = random.Random()

                if args.sample >= 1.0 or randomByModel[model].random() < args.sample:
                    fileByModel[model].write(line)
                    totalLineMerged += 1
    for file in list(fileByModel.values()):
        file.close()
    t2 = time.time()
    print('Total line proccessed {}'.format(totalLineMerged))
    print("merging files take %ss" % (t2 - t1))

    if args.useMask:
        fileByModel = removeMaskData(fileByModel)
    return fileByModel


def removeMaskData(dataByName):
    toRemoveList = []
    masks = args.useMask.split(',')
    for mask in masks:
        for name in dataByName.keys():
            if mask.endswith('*'):
                if name.startswith(mask[:-1]):
                    print('remove %s because of filter:%s' % (name, mask))
                    toRemoveList.append(name)
            else:
                if name == mask:
                    print('remove %s because of filter:%s' % (name, mask))
                    toRemoveList.append(name)

    for removeName in toRemoveList:
        del dataByName[removeName]
    return dataByName


def loadFileNameByModel(inputDir):
    """
  Load the file names from a directory. Use to restart the process from a given phrase.
  """
    fileNames = walktree(inputDir)
    fileByModel = {}
    for file in fileNames:
        modelName = file.split('/')[-1]
        modelName = modelName.replace('.txt', '')
        fileByModel[modelName] = file
    return fileByModel


def sortDataFileByModel(fileByModel):
    """
  Use external sort to sort data file by score column.
  """
    t1 = time.time()
    print("sorting files....")
    ensure_dir(SORT_DIR)
    processPool = []
    for model in list(fileByModel.keys()):
        mergedFile = '%s/%s.txt' % (MERGE_DIR, model)
        sortedFile = '%s/%s.txt' % (SORT_DIR, model)
        if args.ignoreInvalid:
            key = eval('lambda l: -float(l.split("' + args.delimiter +
                       '")[2] or 0.0)')
        else:
            key = eval('lambda l: -float(l.split("' + args.delimiter +
                       '")[2])')
        process = Process(target=batchSort,
                          args=(mergedFile, sortedFile, key, args.bufferSize))
        process.start()
        processPool.append(process)

    for process in processPool:
        process.join()
    t2 = time.time()
    print("sorting files take %ss" % (t2 - t1))


def processDataByModel(fileByModel):
    """
    Process data by model. Wait all the subprocess finish then plot curves together.
    """
    t1 = time.time()
    print("processing data....")
    pool = Pool(len(fileByModel))
    dataByModel = dict()
    resultList = []
    for model in list(fileByModel.keys()):
        sortedFile = '%s/%s.txt' % (SORT_DIR, model)
        result = pool.apply_async(processData, args=(model, sortedFile))
        resultList.append(result)
    for result in resultList:
        try:
            (model, data) = result.get()
            dataByModel[model] = data
        except Exception as e:
            if not args.ignoreInvalid:
                raise e
    t2 = time.time()

    if args.aucSelect:
        selectLimit = args.selectLimit
        print('Sort model by AUC and select top', selectLimit)
        sortedModelTuple = sorted(list(dataByModel.items()),
                                  key=lambda item: item[1]['AUC'],
                                  reverse=True)
        dataByModel = dict(sortedModelTuple[:selectLimit])

    if args.verbose:
        print(dataByModel)

    ensure_dir(PROCCESSED_DIR)
    pickle.dump(dataByModel, open(PROCCESSED_FILE, 'wb'))
    return dataByModel
    print("processing data take %ss" % (t2 - t1))


def processData(model, input):
    """
  Process data. Bin data into args.shardCount bins. Accumulate data for each bin and populate necessary metrics. 
  """
    print('processData data for %s' % model)
    data = np.loadtxt(input,
                      delimiter=args.delimiter,
                      dtype={
                          'names': ('model', 'weight', 'score', 'label'),
                          'formats': ('S16', 'f4', 'f4', 'i1')
                      })
    dataSize = len(data)
    shardSize = int(dataSize / args.shardCount)

    rocPoints = [(0, 0)]
    prPoints = []
    corrPoints = []
    cutoff = []

    totalConditionPositive = 0.0
    totalConditionNegative = 0.0

    for record in data:
        modelId = record[0]
        weight = record[1]
        score = record[2]
        label = record[3]

        if label == 1:
            totalConditionPositive += weight
        elif label == 0:
            totalConditionNegative += weight
        else:
            assert False, 'label invalid: %d' % label

    truePositive = 0.0
    falsePositive = 0.0
    binTotalScore = 0.0
    binWeight = 0.0
    binPositive = 0.0
    overallTatalScore = 0.0

    partitionSize = 0
    for record in data:
        modelId = record[0]
        weight = record[1]
        score = record[2]
        label = record[3]

        partitionSize += 1
        binWeight += weight
        overallTatalScore += weight * score

        if label == 1:
            truePositive += weight
            binPositive += weight
            binTotalScore += score * weight
        elif label == 0:
            falsePositive += weight

        if partitionSize % shardSize == 0 or partitionSize == dataSize:
            recall = truePositive / totalConditionPositive if totalConditionPositive > 0 else 0.0
            fallout = falsePositive / totalConditionNegative if totalConditionPositive > 0 else 0.0
            precision = truePositive / (truePositive + falsePositive)

            meanPctr = binTotalScore / binWeight
            eCtr = binPositive / binWeight

            rocPoints += [(fallout, recall)]
            prPoints += [(recall, precision)]
            corrPoints += [(eCtr, meanPctr)]
            cutoff += [(score, recall, precision, fallout)]

            binWeight = 0.0
            binTotalScore = 0.0
            binPositive = 0.0

    rocPoints = sorted(rocPoints, key=lambda x: x[0])
    prPoints = sorted(prPoints, key=lambda x: x[0])
    corrPoints = sorted(corrPoints, key=lambda x: x[0])
    cutoff = sorted(cutoff, key=lambda x: x[0])

    AUC = calculateAUC(rocPoints)
    OER = truePositive / overallTatalScore  #Observed Expected Ratio
    F1 = 2 * truePositive / (truePositive + falsePositive + totalConditionPositive)

    print('%s AUC: %f' % (model, AUC))
    print('%s F1: %f' % (model, F1))
    print('%s Observed/Expected Ratio: %f' % (model, OER))
    if args.cutoff:
        print('%s cutoff:' % model, cutoff)

    return model, {
        'ROC': rocPoints,
        'PR': prPoints,
        'CORR': corrPoints,
        'AUC': AUC,
        'OER': OER,
        'F1': F1,
        'cutoff': cutoff
    }


def configChart():
    cf = pylab.gcf()
    defaultSize = cf.get_size_inches()
    plotSizeXRate = args.plotSizeRate
    plotSizeYRate = args.plotSizeRate
    cf.set_size_inches(
        (defaultSize[0] * plotSizeXRate, defaultSize[1] * plotSizeYRate))

def loadProcessedDataByModel():
    return pickle.load(open(PROCCESSED_FILE, 'rb'))

def roc_proccess():
    phrase = args.phrase
    if phrase == PHRASE_GROUP:
        fileByModel = groupDataByModel(args.inputDirs)
        if args.verbose:
            print(fileByModel)
        phrase = PHRASE_GROUP + 1
    else:
        fileByModel = loadFileNameByModel(MERGE_DIR)

    if phrase == PHRASE_SORT:
        sortDataFileByModel(fileByModel)
        phrase = PHRASE_SORT + 1
    else:
        fileByModel = loadFileNameByModel(SORT_DIR)

    if phrase == PHRASE_PROCESS:
        dataByModel = processDataByModel(fileByModel)
        phrase = PHRASE_PROCESS + 1
    else:
        dataByModel = loadProcessedDataByModel()

    plotCurves(dataByModel)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'inputDirs',
        nargs='+',
        help=
        'Input format: CSV by --delimiter: len(rocord)==4. Ex: modelId, weight, score, label'
    )

    parser.add_argument('-p',
                        '--phrase',
                        dest='phrase',
                        default=0,
                        type=int,
                        help='Start phrase. 0 : Group, 1 : Sort,  2 : Process, 3 : Chart')

    parser.add_argument('-d',
                        '--delimiter',
                        dest='delimiter',
                        default='\x01',
                        help='CSV field delimiter. Default is \\x01')

    parser.add_argument(
        '-s',
        '--shard',
        dest='shardCount',
        default=128,
        type=int,
        help=
        'Shard count. Specify how many data point to generate for plotting. default is 128'
    )

    parser.add_argument(
        '--sample',
        dest='sample',
        default=1.0,
        type=float,
        help=
        'Record sample rate. Specify how much percentage of records to keep per model.'
    )

    parser.add_argument('-b',
                        '--buffer',
                        dest='bufferSize',
                        default=32000,
                        type=int,
                        help='bufferSize to use for sorting, default is 32000')

    parser.add_argument('-v',
                        '--verbose',
                        action='store_true',
                        help='Be verbose')

    parser.add_argument('--cutoff',
                        dest='cutoff',
                        action='store_true',
                        help='print cutoff')

    parser.add_argument('-i',
                        '--ignoreInvalid',
                        action='store_true',
                        help='Ignore invalid in thread')

    parser.add_argument('-r',
                        '--rate',
                        type=float,
                        dest='plotSizeRate',
                        default=2,
                        help='Chart size rate. default 2')

    parser.add_argument(
        '--mask',
        dest='useMask',
        default='',
        help=
        'mask certain data. Ex \'metric_nus*,metric_supply*\'. Will remove data collection label start with \'metric_nus and metric_supply\''
    )

    parser.add_argument(
        '--tmp',
        dest='tempdir',
        default='/tmp/',
        help= 'Tmp dir path'
    )

    parser.add_argument('--aucSelect',
                        dest='aucSelect',
                        action='store_true',
                        help='Select top n=selectLimit roc curve by roc AUC')
    parser.add_argument('--selectLimit',
                        dest='selectLimit',
                        default=0,
                        type=int,
                        help='Select top n model')

    global args
    args = parser.parse_args()
    print('Args:', args)
    roc_proccess()


if __name__ == '__main__':
    main()
