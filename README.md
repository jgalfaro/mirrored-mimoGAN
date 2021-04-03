Supplementary Material for Pilot Contamination Attack Detection in Massive
MIMO Using Generative Adversarial Networks
===

### Fatemeh Banaeizadeh, Carleton University, School of Computer Science, Canada.

### Michel Barbeau, Carleton University, School of Computer Science, Canada.

### Joaquin Garcia-Alfaro, Institut Polytechnique de Paris, Telecom SudParis, France.

### Evangelos Kranakis, Carleton University, School of Computer Science, Canada.

### Tao Wan, CableLabs, Coal Creek Circle Louisville, Colorado, USA.

## Abstract

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

## Matlab Simulations

We simulate a single-cell network in Matlab with the characteristics
indicated in the table below. Each cell is equipped with a 64-antenna
BS and 8 users. Orthogonal pilot sequences with length τ = 8 are
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

## Python Simulations

The training process is conducted using Python code. After training,
the trained networks and their weighs are saved to be evaluated in a
testing phase, with both normal and abnormal datasets (each of size
1000). The Python code associated to this part is available <a
href="https://github.com/jgalfaro/mirrored-mimoGAN/tree/main/python">here</a>.
