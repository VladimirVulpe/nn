## Investigation
* [RNNoise](https://people.xiph.org/~jm/demo/rnnoise) and [RNNoise: Using Deep Learning for Noise Suppression](https://hacks.mozilla.org/2017/09/rnnoise-deep-learning-noise-suppression/)
* [Acoustic Noise Cancellation by Machine Learning](https://towardsdatascience.com/acoustic-noise-cancellation-by-machine-learning-4144af497661) 
* [Recurrent Neural Active Noise Cancellation](https://towardsdatascience.com/deep-active-noise-cancellation-e364ce4562d4) 

## Noise cancelation products
[krisp.ai](https://krisp.ai/)

## Data
* [Telecommunications & Signal Processing Laboratory](http://www-mmsp.ece.mcgill.ca/Documents/Data/)
* [NTT multilingual speech database for telephonometry](https://www.ntt-at.com/product/multilingual/)

## Fragestellungen
* [Audio/Digital Signal Processing/Recurrent NN - Need help understanding and reproducing this paper in Python](https://www.reddit.com/r/MachineLearning/comments/c9qb1l/d_audiodigital_signal_processingrecurrent_nn_need/)
* Noise Pre-Processing: [Mel-Skala vs Bark-Skala](https://www.isip.piconepress.com/courses/msstate/ece_8463/lectures/current/lecture_04/lecture_04_05.html) insb. [im Bezug auf die Performance](https://www.ijitee.org/wp-content/uploads/papers/v8i11/K19990981119.pdf)
* reicht pytorch Audio für die initiale Verarbeitung, insb. bei der Aufteilung in Bark-Skala oder [Mel-Skala](https://pytorch.org/audio/transforms.html#melscale). Alternativ [librosa](https://librosa.github.io/librosa/) verwenden. 
* werden die Daten mittels numpy von pytorch Audio nach keras transportiert? 
* woher bekommt man Training und Testdaten? Kann man sich sich auch selbst generieren? 

## Process

