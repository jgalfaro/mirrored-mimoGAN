Pilot Contamination Attack Detection in Massive MIMO Using Generative Adversarial Networks
===

### Fatemeh Banaeizadeh, Carleton University, School of Computer Science, Canada.

### Michel Barbeau, Carleton University, School of Computer Science, Canada.

### Joaquin Garcia-Alfaro, Institut Polytechnique de Paris, Telecom SudParis, France.

### Evangelos Kranakis, Carleton University, School of Computer Science, Canada.

### Tao Wan, CableLabs, Coal Creek Circle Louisville, Colorado, USA.

## Abstract

Supplementary Material to Ref. [1].
Reliable and high throughput communication in Massive Multiple-Input
Multiple-Output (MIMO) systems strongly depends on accurate channel
estimation at the Base Station (BS). However, the channel estimation
process in MIMO systems is vulnerable to the pilot contamination
attack, which not only degrades efficiency of channel estimation, but
also increases the probability of information leakage. In this paper,
we propose a defence mechanism against the pilot contamination attack
using a deep-learning model, namely Generative Adversarial Networks
(GAN), to detect suspicious uplink connections at the BS. Training of
the models is performed via normal data, which consists of received
signals from legitimate users and a real channel matrix. We report in
this companion website some simulation results that show the
feasibility of our approach.

## Keywords

Massive MIMO, Pilot contamination attack, Generative Adversarial Network,
Network security.


## Architecture of our proposed GAN

Figure 1 shows the architecture of our proposed GAN. The discriminator
contains four layers. The three first layers consist of convolutional
(*Con2D*) operations, followed by *BatchNormalization*, *LeakyRelu*,
and *DropOut* operations. The last layer is composed of *Flatten* and
*Dense* operations, followed by a *MSE* linear activation
function, which returns a binary output (e.g., *true* or *false*).


<img src="./figs/GANfigure.png" width="65%"  />

#### Figure 1. Architecture of our proposed GAN.

The generator is designed as an estimator, with encoding and decoding
blocks, to estimate the channel from the noisy received signals (Y)
at the BS. I.e., it acts as an autoencoder estimating real channel
matrices from the noisy signals. The generator is composed of four
encoding blocks and four decoding blocks. Each encoding block consists
of layers with convolutional (*Con2D*) operations, followed by
*BatchNormalization* and *Relu* operations. The output of the last
encoding block is sent to the first decoding block. Each decoding
block is composed of transpose convolutional (*Con2DTranspose*)
operations, followed by *BatchNormalization* and *Relu* operations.
The last layer of the last decoder uses a *tanh* activation function,
to scale the output values in the range -1 and 1.

## Matlab Simulations

We simulate a single-cell network in Matlab with the characteristics
indicated in the table below. Each cell is equipped with a 64-antenna
BS and 8 users. Orthogonal pilot sequences with length ?? = 8 are
generated for each user by hadamard matrix. Training data is only
composed of normal data. Normal data consist of a database of the
received signal (Y) with size of 4000 as an input for the generator,
and a database of the real channels (H) with size of 4000 as an input
for the discriminator. Each received signal and each channel matrix is
a three dimensional matrix with size (64,8,2). The third dimension
shows values of real and imaginary parts of the received signal and
channel matrices. The Matlab code associated to this part is available
<a
href="https://github.com/jgalfaro/mirrored-mimoGAN/tree/main/matlab">here</a>.


#### Table 1. Simulation parameters in Matlab

<img src="./figs/table1.png" width="45%"  />


## Python Simulations

The training process is conducted using Python code. After training,
the trained networks and their weighs are saved to be evaluated in a
testing phase, with both normal and abnormal datasets (each of size
1000). The Python code associated to this part is available <a
href="https://github.com/jgalfaro/mirrored-mimoGAN/tree/main/python">here</a>.

## References

If using this code for research purposes, please cite:

[1] F. Banaeizadeh, M. Barbeau, J. Garcia-Alfaro, E. Kranakis, T. Wan. "Pilot Contamination Attack Detection in 5G Massive MIMOSystems Using Generative Adversarial Networks", IEEE International Mediterranean Conference on Communications and Networking (MeditCom), 2021. 

```
@inproceedings{barbeau2021ccece2021,
  title={{Pilot Contamination Attack Detection in 5G Massive MIMOSystems Using Generative Adversarial Networks}},
  author={Banaeizadeh, Fatemeh and Barbeau, Michel and Garcia-Alfaro, Joaquin and Kranakis, Evangelos and Wan, Tao},
  booktitle={IEEE International Mediterranean Conference on Communications and Networking (MeditCom)},
  pages={},
  year={2021},
  month={ },
  publisher={IEEE},
  doi = {},
  url = {},
}
```
