# Roc charting tools
Ploting ROC charts is not easy, which usually requires a MapReduce like data proccessing code.

This tools is highly optimized and run on single machine super fast.
In the brenchmarking, it processed 1.1G compressed GZ data within 300 seconds on a `3.1 GHz Intel Core i7 Macbook pro`.

## Install

`pip install roc-tools`

## Requirements

For ploting support
`pip install matplotlib`

For python2/3 compatiable
`pip install future`

On linux
`sudo apt-get install python-tk`

## Usage

`roc inputfolders` or `roc inputfiles...`

```
> roc -h
usage: roc [-h] [-p PHRASE] [-d DELIMITER] [-s SHARDCOUNT] [-b BUFFERSIZE]
           [-v] [--cutoff] [-i] [-r PLOTSIZERATE] [--mask USEMASK]
           [--aucSelect] [--selectLimit SELECTLIMIT]
           inputDirs

positional arguments:
  inputDirs              Input format: CSV by --delimiter: len(rocord)==4. Ex:
                        modelId, weight, score, label

optional arguments:
  -h, --help            show this help message and exit
  -p PHRASE, --phrase PHRASE
                        Start phrase. 0 : Group, 1 : Sort, 2 : Process
  -d DELIMITER, --delimiter DELIMITER
                        CSV field delimiter. Default is \x01
  -s SHARDCOUNT, --shard SHARDCOUNT
                        Shard count. Specify how many data point to generate
                        for plotting. default is 64
  -b BUFFERSIZE, --buffer BUFFERSIZE
                        bufferSize to use for sorting, default is 32000
  -v, --verbose         Be verbose
  --cutoff              print cutoff
  -i, --ignoreInvalid   Ignore invalid in thread
  -r PLOTSIZERATE, --rate PLOTSIZERATE
                        Chart size rate. default 1.5
  --mask USEMASK        mask certain data. Ex 'metric_nus*,metric_supply*'.
                        Will remove data collection label start with
                        'metric_nus and metric_supply'
  --aucSelect           Select top n=selectLimit roc curve by roc AUC
  --selectLimit SELECTLIMIT
                        Select top n model
```

## Inputfile format
A CSV with four columns, defalut delimiter is `\x01`


`Input format: CSV by --delimiter: len(rocord)==4. Ex: modelId, weight, score, label`


## Exampe

We have a 1G cvs data.
```
> ls -lh data/000001_0.gz
-rw-r--r--  1 zf  staff   1.1G Mar 14 10:46 data/000001_0.gz
```

Using the default CSV delimiter, `\x01`
```
> gzcat data/000001_0.gz | head
channels_use_3_average_1\x011.0\x010.5278635621070862\x010
channels_use_3_average_1\x011.0\x010.28971177339553833\x010
channels_use_3_average_1\x011.0\x010.31590744853019714\x010
```

Let generate the ROC with 10% of data.
```
> roc --sample 0.1 data
```

```
Args: Namespace(aucSelect=False, bufferSize=32000, cutoff=False, delimiter='\x01', ignoreInvalid=False, inputDir='data', phrase=0, plotSizeRate=1.5, sample=0.1, selectLimit=0, shardCount=64, useMask='', verbose=False)
merging files by model to merges
Total line proccessed 12011651
merging files take 287.450480938s
sorting files....
sorted file sorts/channels_use_3_average_1.txt ready
sorting files take 53.800538063s
processing data....
processData data for channels_use_3_average_1
channels_use_3_average_1 AUC: 0.774804
channels_use_3_average_1 F1: 0.000157
channels_use_3_average_1 Observed/Expected Ratio: 0.000232
png:  results/pr_curve.png results/roc_curve.png results/corr_curve.png results/precision.png results/recall.png
```

Let take a look at the result image.

![](images/roc_curve.png)
![](images/pr_curve.png)
![](images/precision.png)
![](images/recall.png)

## Resume from failure from phrase checkpoint

The roc command breakdown the process to 4 phrases.
0. Merge data, group and merge records by model
1. Sort data, sort records by model
2. Proccess, computing roc data
3. Plot, ploting the charts

If your command failed at certain phrase, you can restart it and change the arguments.

For example, if you sort filed, because of limit of file descriptor (See FAQ #1)
You can resume the job with the below command and add argument to change the buffer.
```
roc <your-data> -p 1 --buffer 64000
```

Also you can use the phrase argument to re-chart with different scale ratio.

```
roc <your-data> -p 3 -r 2.0
```

## FAQ

1. ```IOError: [Errno 24] Too many open files: '/tmp/```

If you run it on a very large data, you may run into this issue. It's becaused the external merge sort create too many tmp file during the soring phrase.

Two ways to solve it.

First, turn up the file descriptor limit in your system, the default is usually 256.
You can change it with `> ulimit -n 10000`

Sencod way to solve it.
Make the merge sort buffer bigger, which will lead to less parallism thus less tmp files created.
for example `--buffer 128000`
 



