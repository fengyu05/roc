# roc
Roc charting tools

## Install

`pip install roc-tools`

requirements

`pip install matplotlib`
`pip install future`

On linux
`sudo apt-get install python-tk`

## Usage
`roc -h`

`roc inputfolder` or `roc inputfiles`

## Inputfile format
A CSV with four columns, defalut delimiter is `\x01`


`Input format: CSV by --delimiter: len(rocord)==4. Ex: modelId, weight, score, label`

