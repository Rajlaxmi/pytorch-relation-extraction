**2019.03.09 update**:

- Update to Python3.X
- Updated to Pytorch 0.4+ (Remove Variable, etc.)
- Use mask for Piece Pooling
- Large version dataset NYT is recommended compared to FilterNYT


2019.03.05:

Fix the bug of `mask piece wise`.

- Updated to pytorch 0.4+, version 0.3 is not compatible


2018.11.3:

** Mask based ** `use_pcnn=True` currently has some problems, are being modified, it is recommended:

- Direct use of `use_pcnn=False` test, performance is not too bad
- Use the mask to modify the previous version: https://github.com/ShomyLiu/pytorch-relation-extraction/tree/7e3ef1720d43690fc0da0d81e54bdc0fc0cf822a


2018.10.14 update:

The fully supervised relationship extracts the code address of PCNN (Zeng 2014): [PCNN] (https://github.com/ShomyLiu/pytorch-pcnn)


2018.9.10 update:
- Refer to OpenNRE using mask to quickly calculate piece wise pooling.
    - Modify NYT 53 class data processing (complete)
    - Modify NYT 27 class data processing (unfinished)
    
Data processing has been modified

Use Pytorch to reproduce PCNN+MIL (Zeng 2015) and PCNN+ATT (Lin 2016), and compare the performance of the two models on two large versions of the dataset (27 class relationships / 53 class relationships).



Related blogs:

- [Relationship drawing paper notes] (http://shomy.top/2018/02/28/relation-extraction/)

- [Reproduction result description] (http://shomy.top/2018/07/05/pytorch-relation-extraction/)



In the organization of the code, structural design, the main reference [Chen Yun Pytorch practical guide] (https://zhuanlan.zhihu.com/p/29024978) (personal recommendation). Therefore, some implementation details will not be described again. You can refer to Chen Yun's practical guide.



## Implementation Overview


surroundings:

- Python 2.X
- Pytorch 0.3.1
- fire

Brief introduction to the main directory:

```
├── checkpoints # save preloaded model
├── config.py # parameter
├── dataset #数据目录
│ ├── FilterNYT # SMALL Data
│ ├── NYT # LARGE Data
│ ├── filternyt.py
│ ├── __init__.py
│ ├── nyt.py
├── main_mil.py # PCNN+ONE main file
├── main_att.py # PCNN+ATT main file
├── models #模型目录
│ ├── BasicModule.py
│ ├── __init__.py
│ ├── PCNN_ATT.py
│ ├── PCNN_ONE.py
├── plot.ipynb
├── README.md
├── utils.py #tool function
```



This code is basically written in accordance with Chen Yun's guide imitation. Separate data models, parameter/configure individual files, and use the fire library to manage command line parameters, making it easier to modify parameters.

Because of the training of PCNN+ONE and PCNN+ATT, the test methods are not the same, so for the sake of simplicity, the main files are written: `main_mil.py` and `main_att.py`.

The training method is the same. If you use PCNN+ONE to train big data sets, you can directly modify the parameters later. By default, the parameters of `config.py` are used:

```

Python main_mil.py train --data="NYT" --batch_size=128

```

Note: You need to process the data in the next section in advance (mainly to generate data in npy format, which is convenient to be imported directly by the model).



## Data preprocessing

In order to save space, two pieces of raw data of LARGE and SMALL are uploaded, so data preprocessing is required to generate npy format data.

First download two original data, address:

[Baidu network disk] (https://pan.baidu.com/s/1Mu46NOtrrJhqN68s9WfLKg) [谷云云盘](https://drive.google.com/drive/folders/1kqHG0KszGhkyLA4AZSLZ2XZm9sxD8b58?usp=sharing)

A simple description of the data format:
- First line: Two entity IDs: ent1id ent2id
- The second line: the number of sentences in the bag tag and bag. Since a few bags have multiple labels (no more than 4), the sentence label is represented by 4 integers, and -1 is empty, such as: 2 4 - 1 -1 3 means that the label of the bag is 2 and 4, and then contains 3 sentences.
- The next few lines indicate the sentences in the bag


Put the two zips into the `dataset` directory and extract them. This will form two directories, one NYT, one FilterNYT, where the LARGE dataset is in the NYT directory, and the SMALL data is in FilterNYT. The raw data here is from Zeng 2015. And in the open source code of Lin2016.



For LARGE data:



- Switch to the NYT directory,

- Compile and execute extract.cpp from the extract_cpp directory: `g++ extract.cpp -o extract`, then execute: `./extract`, get `bag_train.txt, bag_test.txt, vector.txt` (in the NYT directory), The cpp is the code for Lin2016 preprocessing

- Switch back to the home directory: Perform data preprocessing: `python dataset/nyt.py` This will generate a series of npy files in the NYT directory.



For SMALL data

- Directly execute `python dataset/filternyt.py` to generate npy files in the FilterNYT directory.



The generated NPY files are directly imported using Pytorch's Dataset. See the `*Data` class of `nyt.py` and `filternyt.py` for specific code.

After the data is pre-processed, you can train/test according to the above commands.



## tuning optimization

In the process of recurring, I spent a lot of effort, stepped on a lot of pits, and simply remembered:

- The optimization function uses `Adadelta` instead of `Adam`, which can be used with `SGD`, but not as good as `Adadelta`.

- In the theano code of Zeng 2015, there are some errors in the place of select instance and predict (there is no instance with the highest probability)

- BatchSize is relatively large and better (128)



A description of the results can be viewed on the blog.



## Reference

- [PCNN+ONE Zeng 2015] (https://github.com/smilelhh/ds_pcnns)
- [PCNN+ATT Lin 2016] (https://github.com/thunlp/OpenNRE)
- [RE-DS-Word-Attention-Models] (https://github.com/SharmisthaJat/RE-DS-Word-Attention-Models)
- [GloRE](https://github.com/ppuliu/GloRE)
