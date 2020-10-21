# French-to-English-Translator
Implemented the Transformer deep learning architecture from the famous 2017 research paper "Attention Is All You Need" using Pytorch and used it make a French to English translator.

## Transformer Architecture
The Transformer architecture can be best described by these the figures from the research paper:

![](https://github.com/Rami97rgb/French-to-English-Translator/blob/master/images/transformer1.png)
![](https://github.com/Rami97rgb/French-to-English-Translator/blob/master/images/transformer2.png)
![](https://github.com/Rami97rgb/French-to-English-Translator/blob/master/images/transformer3.png)

## Dataset
The IWSLT dataset has been used for training, it is is a machine translation dataset that is focused on the automatic transcription and translation of TED and TEDx talks, i.e. public speeches covering many different topics. This dataset is relatively small (the corpus has 130K sentences).

## Testing
After training the model for 10 epochs, which only took one hour on a Google Colab GPU, it achieved relatively good (better than traditional Seq2Seq LSTM models). For example, it translated this sentence: 
