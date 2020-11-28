# [What ist deep learning](https://machinelearningmastery.com/what-is-deep-learning/)
Deep Learning is a subfield of machine learning concerned with algorithms inspired by the structure and function of the brain called artificial neural networks.

One reason that deep learning has taken off like crazy is because it is fantastic at supervised learning - Andrew Ng

Andrew often mentions that we should and will see more benefits coming from the unsupervised side of the tracks as the field matures to deal with the **abundance of unlabeled data available**.

When you hear the term deep learning, just think of a large deep neural net. **Deep** refers to the **number of layers** typically and so this kind of the popular term that’s been adopted in the press. I think of them as deep neural networks generally.

Important property of Neural Networks: **Results get better** with **more data** + **bigger models** + **more computation** (better algorithms, new insights and improved techniques always help too). 

In addition to scalability, another often cited benefit of deep learning models is their ability to perform automatic feature extraction from raw data, also called feature learning.

Deep learning is really good at learning f, particularly in situations where the data is complex. In fact, artificial neural networks are known as **universal function approximators** because they’re able to learn any function, no matter how wiggly, with just a single **hidden layer**.

Deep neural networks are harder to interpret because the features are learned and aren’t explained anywhere in English. It’s all in the machine’s imagination.

Definition at [towardsdatascience - Understanding Neural Networks. From neuron to RNN, CNN, and Deep Learning](https://towardsdatascience.com/understanding-neural-networks-from-neuron-to-rnn-cnn-and-deep-learning-cd88e90e0a90).

## Backpropagation
[Definition on YouTube](https://www.youtube.com/watch?v=Ilg3gGewQ5U).  

## Rectified linear unit (ReLU)
`f(x)=max(0,x)` ist die ReLU und wird als [Aktivierungsfunktion, anstelle der Sigmoidfunktion, verwendet](https://de.wikipedia.org/wiki/Rectifier_(neuronale_Netzwerke)). 

## Recurrent Neural Network (RNN)

RNNs have a sense of built-in memory and are well-suited for language problems. They’re also important in reinforcement learning since they enable the agent to keep track of where things are and what happened historically even when those elements aren’t all visible at once. [Christopher Olah wrote an excellent walkthrough of RNNs and LSTMs in the context of language problems](http://colah.github.io/posts/2015-08-Understanding-LSTMs/).

### Long Short Term Memory networks (LSTM)

[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## Convulational Neural Network (CNN)
CNNs are designed specifically for taking images as input, and are effective for computer vision tasks. They are also instrumental in deep reinforcement learning. CNNs are specifically inspired by the way animal visual ortices work, and they’re the focus of the deep learning course we’ve been referencing throughout this article, Stanford’s CS231n.



## RNN vs CNN

CNN is a feed forward neural network that is generally used for Image recognition and object classification. While RNN works on the principle of saving the output of a layer and feeding this back to the input in order to predict the output of the layer.

CNN considers only the current input while RNN considers the current input and also the previously received inputs. It can memorize previous inputs due to its internal memory.

CNN has 4 layers namely: Convolution layer, ReLU layer, Pooling and Fully Connected Layer. Every layer has its own functionality and performs feature extractions and finds out hidden patterns.

There are 4 types of RNN namely: One to One, One to Many, Many to One and Many to Many.

RNN can handle sequential data while CNN cannot.

... [images + architecture](https://qr.ae/pNvBGs)

## Deep reinforcement learning
This is one of the most exciting areas of deep learning research, at the heart of recent achievements like OpenAI defeating professional Dota 2 players and DeepMind’s AlphaGo surpassing humans in the game of Go. We’ll dive deeper in Part 5 , but essentially the goal is to apply all of the techniques in this post to the problem of teaching an agent to maximize reward. This can be applied in any context that can be gamified — from actual games like Counter Strike or Pacman, to self-driving cars, to trading stocks, to (ultimately) real life and the real world.


## Examples
* [CNN + RNN Architecture at Tesla](https://www.youtube.com/watch?v=oBklltKXtDE)


