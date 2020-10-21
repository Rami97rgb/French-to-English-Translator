# French-to-English-Translator
Implemented the Transformer deep learning architecture from the famous 2017 research paper "Attention Is All You Need" using Pytorch and used it make a French to English translator.

## Transformer Architecture
The Transformer architecture can be best described by these figures from the research paper:

![](https://github.com/Rami97rgb/French-to-English-Translator/blob/master/images/transformer1.png)
![](https://github.com/Rami97rgb/French-to-English-Translator/blob/master/images/transformer2.png)
![](https://github.com/Rami97rgb/French-to-English-Translator/blob/master/images/transformer3.png)

## Dataset
The IWSLT dataset has been used for training, it is is a machine translation dataset that is focused on the automatic transcription and translation of TED and TEDx talks, i.e. public speeches covering many different topics. This dataset is relatively small (the corpus has 130K sentences).

## Testing
After training the model for 10 epochs, which only took one hour on a Google Colab GPU, it achieved relatively good results (better than traditional Seq2Seq LSTM models). For example, it translated this sentence:

![](https://github.com/Rami97rgb/French-to-English-Translator/blob/master/images/transformer4.png)

Which should be translated into "a black and white cat is cute" and our model translates it into:

![](https://github.com/Rami97rgb/French-to-English-Translator/blob/master/images/transformer5.png)

This seems fairly good considering that the model has only been trained for a couple of epochs at this point and considering the size of the dataset. Note that most of the other examples have been perfectly translated.

## Resources
Inspiration for this project: https://www.youtube.com/watch?v=M6adRGJe5cQ

Research paper: https://arxiv.org/abs/1706.03762

Paper explanation: http://peterbloem.nl/blog/transformers
