#!/usr/bin/python
"""
Generate ROC, PR, Correlation and score to Recall/Pricision/Fallout curves.
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
from itertools import islice
from multiprocessing import Process, Pool
from matplotlib import pyplot
import pylab

os.environ["DISPLAY"] = ":0.0"

MERGE_DIR = 'merges'
SORT_DIR = 'sorts'
OUTPUT_DIR = 'results'
DATA_PICKLE = 'data.pickle'

PHRASE_GROUP = 0
PHRASE_SORT = 1
PHRASE_PROCESS = 2
PHRASE_CHART = 3

DEFAULT_SAMPLE = 1.0
DEFAULT_SHART_COUNT = 100
DEFAULT_BUFFER_SIZE = 32000
DEFAULT_PLOT_SIZE_RATE = 2
DEFAULT_TEMPDIR = '/tmp/'
DEFAULT_DELIMTER = '\x01'

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

def batchSort(input, output, key, buffer_size, tempdir):
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

    tempdir = os.path.join(tempdir, str(uuid.uuid4()))
    os.makedirs(tempdir)
    chunks = []
    try:
        with open(input, 'rb', 64 * 1024) as inputFile:
            inputIter = iter(inputFile)
            while True:
                current_chunk = list(islice(inputIter, buffer_size))
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



class ROC(object):
    def __init__(self,
            args=None, # ArgumentParser args object, override all the below args if set
            input_dirs = '',
            phrase = 0,
            sample = DEFAULT_SAMPLE,
            shard_count = DEFAULT_SHART_COUNT,
            buffer_size = DEFAULT_BUFFER_SIZE,
            plot_size_rate = DEFAULT_PLOT_SIZE_RATE,
            delimiter = DEFAULT_DELIMTER,
            ignore_invalid = False,
            use_mask = '',
            tempdir = DEFAULT_TEMPDIR,
            auc_select = False,
            select_limit = 0,
            verbose = False,
            print_cutoff = False,
            no_plot = False,
            output_dir = OUTPUT_DIR
    ):
        self.args = args

        self.input_dirs = args.input_dirs if args else input_dirs
        self.phrase = args.phrase if args else phrase
        self.shard_count = args.shard_count if args else shard_count
        self.sample = args.sample if args else sample
        self.buffer_size = args.buffer_size if args else buffer_size
        self.plot_size_rate = args.plot_size_rate if args else plot_size_rate
        self.delimiter = args.delimiter if args else delimiter
        self.ignore_invalid = args.ignore_invalid if args else ignore_invalid
        self.use_mask = args.use_mask if args else use_mask
        self.auc_select = args.auc_select if args else auc_select
        self.select_limit = args.select_limit if args else select_limit
        self.verbose = args.verbose if args else verbose
        self.print_cutoff = args.print_cutoff if args else print_cutoff
        self.tempdir = args.tempdir if args else tempdir
        self.no_plot = args.no_plot if args else no_plot
        self.output_dir = args.output_dir if args else output_dir


    @staticmethod
    def calculateAUC(rocPoints):
        AUC = 0.0
        lastPoint = (0, 0)
        for point in rocPoints:
            AUC += (point[1] + lastPoint[1]) * (point[0] - lastPoint[0]) / 2.0
            lastPoint = point
        return AUC

    def plotCurves(self, dataByModel):
        """
        Plot ROC, PR and Correlation Curves by model.
        """
        prFigure = pyplot.figure()
        self.configChart()
        prAx = prFigure.add_subplot(111)
        prAx.set_xlabel('Recall')
        prAx.set_ylabel('Precision')
        prAx.set_title('PR Curve')
        prAx.grid(True)

        rocFigure = pyplot.figure()
        self.configChart()
        rocAx = rocFigure.add_subplot(111)
        rocAx.set_xlabel('Fallout / FPR')
        rocAx.set_ylabel('Recall')
        rocAx.set_title('ROC Curve')
        rocAx.grid(True)

        corrFigure = pyplot.figure()
        self.configChart()
        corrAx = corrFigure.add_subplot(111)
        corrAx.set_xlabel('predict score')
        corrAx.set_ylabel('real score')
        corrAx.set_title('Correlation Curve')
        corrAx.grid(True)

        precisionFigure = pyplot.figure()
        self.configChart()
        precisionAx = precisionFigure.add_subplot(111)
        precisionAx.set_xlabel('score')
        precisionAx.set_ylabel('Precision')
        precisionAx.set_title('Threshold score vs precision')
        precisionAx.grid(True)

        recallFigure = pyplot.figure()
        self.configChart()
        recallAx = recallFigure.add_subplot(111)
        recallAx.set_xlabel('score')
        recallAx.set_ylabel('Recall')
        recallAx.set_title('Threshold score vs recall')
        recallAx.grid(True)

        falloutFigure = pyplot.figure()
        self.configChart()
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
        ensure_dir(self.output_dir)
        prAx.legend(loc='upper right', shadow=True)
        prFigure.savefig('%s/pr_curve.png' % self.output_dir)

        rocAx.legend(loc='lower right', shadow=True)
        rocFigure.savefig('%s/roc_curve.png' % self.output_dir)

        corrAx.legend(loc='upper left', shadow=True)
        corrFigure.savefig('%s/corr_curve.png' % self.output_dir)

        precisionAx.legend(loc='upper left', shadow=True)
        precisionFigure.savefig('%s/precision.png' % self.output_dir)

        recallAx.legend(loc='lower left', shadow=True)
        recallFigure.savefig('%s/recall.png' % self.output_dir)

        falloutAx.legend(loc='upper right', shadow=True)
        falloutFigure.savefig('%s/fallout.png' % self.output_dir)

        pyplot.close()
        pngs = '{result}/pr_curve.png {result}/roc_curve.png {result}/corr_curve.png {result}/precision.png {result}/recall.png {result}/fallout.png'.format(result=self.output_dir)
        print('png: ', pngs)


    def groupDataByModel(self, input_dirs):
        """
        Group data to separated file by model.
        """
        t1 = time.time()
        print("merging files by model to %s" % MERGE_DIR)
        ensure_dir(MERGE_DIR)
        fileByModel = dict()
        randomByModel = dict()
        totalLineMerged = 0
        for inputDir in input_dirs:
            for file in walktree(inputDir):
                for line in openFile(file):
                    fields = line.split(self.delimiter)
                    if self.ignore_invalid:
                        if len(fields) != 4 or fields[0] == '' or fields[
                                1] == '' or fields[2] == '' or fields[3] == '':
                            print('Ingonre Invalid line', fields)
                            continue
                    model = fields[0]
                    if model not in fileByModel:
                        fileByModel[model] = open('%s/%s.txt' % (MERGE_DIR, model), 'w')
                        randomByModel[model] = random.Random()

                    if self.sample >= 1.0 or randomByModel[model].random() < self.sample:
                        fileByModel[model].write(line)
                        totalLineMerged += 1
        for file in list(fileByModel.values()):
            file.close()
        t2 = time.time()
        print('Total line proccessed {}'.format(totalLineMerged))
        print("merging files take %ss" % (t2 - t1))

        if self.use_mask:
            fileByModel = self.removeMaskData(fileByModel)
        return fileByModel


    def removeMaskData(self, dataByName):
        toRemoveList = []
        masks = self.use_mask.split(',')
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


    def loadFileNameByModel(self, inputDir):
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


    def sortDataFileByModel(self, fileByModel):
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
            if self.ignore_invalid:
                key = eval('lambda l: -float(l.split("' + self.delimiter +
                           '")[2] or 0.0)')
            else:
                key = eval('lambda l: -float(l.split("' + self.delimiter +
                           '")[2])')
            process = Process(target=batchSort, args=(mergedFile, sortedFile, key, self.buffer_size, self.tempdir))
            process.start()
            processPool.append(process)

        for process in processPool:
            process.join()
        t2 = time.time()
        print("sorting files take %ss" % (t2 - t1))


    def processDataByModel(self, fileByModel):
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
            result = pool.apply_async(self.computeMetrics, args=(model, sortedFile, self.shard_count, self.delimiter, self.print_cutoff))
            resultList.append(result)
        for result in resultList:
            try:
                (model, data) = result.get()
                dataByModel[model] = data
            except Exception as e:
                if not self.ignore_invalid:
                    raise e
        t2 = time.time()

        if self.auc_select:
            select_limit = self.select_limit
            print('Sort model by AUC and select top', select_limit)
            sortedModelTuple = sorted(list(dataByModel.items()),
                                      key=lambda item: item[1]['AUC'],
                                      reverse=True)
            dataByModel = dict(sortedModelTuple[:select_limit])

        if self.verbose:
            print(dataByModel)

        ensure_dir(self.output_dir)
        pickle.dump(dataByModel, open(os.path.join(self.output_dir, DATA_PICKLE), 'wb'))
        print("processing data take %ss" % (t2 - t1))
        return dataByModel


    @staticmethod
    def computeMetrics(model, input, shard_count, delimiter, print_cutoff=False):
        """
        Process data. Bin data into shard_count bins. Accumulate data for each bin and populate necessary metrics. 
        """
        print('compute metrics for %s' % model)
        data = np.loadtxt(input,
                          delimiter=delimiter,
                          dtype={
                              'names': ('model', 'weight', 'score', 'label'),
                              'formats': ('S16', 'f4', 'f4', 'i1')
                          })
        dataSize = len(data)
        shardSize = int(dataSize / shard_count)

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
                recall = (truePositive / totalConditionPositive) if totalConditionPositive > 0 else 0.0
                fallout = (falsePositive / totalConditionNegative) if totalConditionPositive > 0 else 0.0
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

        AUC = self.calculateAUC(rocPoints)
        OER = truePositive / overallTatalScore  #Observed Expected Ratio
        F1 = 2 * truePositive / (truePositive + falsePositive + totalConditionPositive)

        print('%s AUC: %f' % (model, AUC))
        print('%s F1: %f' % (model, F1))
        print('%s Observed/Expected Ratio: %f' % (model, OER))
        if print_cutoff:
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


    def configChart(self):
        cf = pylab.gcf()
        defaultSize = cf.get_size_inches()
        plotSizeXRate = self.plot_size_rate
        plotSizeYRate = self.plot_size_rate
        cf.set_size_inches(
            (defaultSize[0] * plotSizeXRate, defaultSize[1] * plotSizeYRate))

    def loadProcessedDataByModel(self):
        return pickle.load(open(os.path.join(self.output_dir, DATA_PICKLE), 'rb'))

    def process(self):
        phrase = self.phrase
        if phrase == PHRASE_GROUP:
            fileByModel = self.groupDataByModel(self.input_dirs)
            if self.verbose:
                print(fileByModel)
            phrase = PHRASE_GROUP + 1
        else:
            fileByModel = self.loadFileNameByModel(MERGE_DIR)

        if phrase == PHRASE_SORT:
            self.sortDataFileByModel(fileByModel)
            phrase = PHRASE_SORT + 1
        else:
            fileByModel = self.loadFileNameByModel(SORT_DIR)

        if phrase == PHRASE_PROCESS:
            dataByModel = self.processDataByModel(fileByModel)
            phrase = PHRASE_PROCESS + 1
        else:
            dataByModel = self.loadProcessedDataByModel()

        if not self.no_plot:
            self.plotCurves(dataByModel)
        return dataByModel


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'input_dirs',
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
                        default=DEFAULT_DELIMTER,
                        help='CSV field delimiter. Default is \\x01')

    parser.add_argument(
        '-s',
        '--shard',
        dest='shard_count',
        default=DEFAULT_SHART_COUNT,
        type=int,
        help=
        'Shard count. Specify how many data point to generate for plotting. default is 100'
    )

    parser.add_argument(
        '--sample',
        dest='sample',
        default=DEFAULT_SAMPLE,
        type=float,
        help=
        'Record sample rate. Specify how much percentage of records to keep per model.'
    )

    parser.add_argument('-b',
                        '--buffer',
                        dest='buffer_size',
                        default=DEFAULT_BUFFER_SIZE,
                        type=int,
                        help='buffer_size to use for sorting, default is 32000')


    parser.add_argument('-i',
                        '--ignore_invalid',
                        action='store_true',
                        help='Ignore invalid in thread')

    parser.add_argument('-r',
                        '--rate',
                        type=float,
                        dest='plot_size_rate',
                        default=DEFAULT_PLOT_SIZE_RATE,
                        help='Chart size rate. default 2')

    parser.add_argument(
        '--mask',
        dest='use_mask',
        default='',
        help=
        'mask certain data. Ex \'metric_nus*,metric_supply*\'. Will remove data collection label start with \'metric_nus and metric_supply\''
    )

    parser.add_argument(
        '--tmp',
        dest='tempdir',
        default=DEFAULT_TEMPDIR,
        help= 'Tmp dir path'
    )

    parser.add_argument('--auc_select',
                        dest='auc_select',
                        action='store_true',
                        help='Select top n=select_limit roc curve by roc AUC')

    parser.add_argument('--select_limit',
                        dest='select_limit',
                        default=0,
                        type=int,
                        help='Select top n model')

    parser.add_argument('-v',
                        '--verbose',
                        action='store_true',
                        help='Be verbose')

    parser.add_argument('--print-cutoff',
                        dest='print_cutoff',
                        action='store_true',
                        help='print cutoff')

    parser.add_argument('--no-plot',
                        dest='no_plot',
                        action='store_true',
                        help='do not plot')

    parser.add_argument('--output-dir',
                        dest='output_dir',
                        default=OUTPUT_DIR,
                        help='output data file')

    args = parser.parse_args()
    print('Args:', args)

    roc = ROC(args)
    roc.process()


if __name__ == '__main__':
    main()
