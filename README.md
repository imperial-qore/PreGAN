[![License](https://img.shields.io/badge/License-BSD%203--Clause-red.svg)](https://github.com/imperial-qore/PreGAN/blob/master/LICENSE)
![Python 3.7, 3.8](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fimperial-qore%2FPreGAN&count_bg=%23FFC401&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
[![Actions Status](https://github.com/imperial-qore/SimpleFogSim/workflows/DeFog-Benchmarks/badge.svg)](https://github.com/imperial-qore/PreGAN/actions)
<br>
![Docker pulls yolo](https://img.shields.io/docker/pulls/shreshthtuli/yolo?label=docker%20pulls%3A%20yolo)
![Docker pulls pocketsphinx](https://img.shields.io/docker/pulls/shreshthtuli/pocketsphinx?label=docker%20pulls%3A%20pocketsphinx)
![Docker pulls aeneas](https://img.shields.io/docker/pulls/shreshthtuli/aeneas?label=docker%20pulls%3A%20aeneas)

# PreGAN

GAN based Preemptive Migration Prediction Network for Proactive Fault Tolerance in Fog Environments. This work uses GANs and co-simulations to learn a few-shot anomaly classifier and preemptive migration (load balancing) based fault-tolerance engine for reliable fog computing.

Anomaly classification into: CPU overutilization (CPU), Abnormal disk utilization (ADU), Memory leak (MEL), Abnormal memory allocation (AMA), Network overload (NOL). Ground truth labels obtained by [ADE tool](https://www.openmainframeproject.org/projects/anomaly-detection-engine-for-linux-logs-ade).


## Model

1. ADPE (Anomaly Detection and Prototype Encoder) :
	1. Graph Attention
	2. GRU 
	3. Late Fusion
	4. Self-Attention with scheduling decision (next state embedding prediction)
	5. Anomaly detection, diagnosis and classification prototypes
	6. Embedding = 0 vector for non-anomalous hosts and prototype vector otherwise

2. Generator :
	1. Self-Attention with ADPE embedding
	2. Delta output (change is schedule prediction)
	3. New schedule = Original Schedule + Delta

3. Discriminator : 
	1. Comparison between new and original schedules
	2. Output which is better

## Motivation

Explot the trade-off between no migration (contention) and migration (transfer overheads).

1. Graph Attention :
	- multi-modal feature extraction

2. GRU :
	- temporal trend extraction

3. Self-Attention :
	- localized contextual trend

4. Protype prediction :
	- model stability and training with limited data (FSL)

4. Delta prediction by generator:
	- correct faults based on protype embeddings for each host

5. Adversarial taining :
	- running two co-simulations each time is expensive
	- discriminator adversarial loss makes generator's delta correction lead to more optimum decision

## Pipeline

1. Pretrain time-series based anomaly detection, diagnosis and classification model (ADPE).
	- Trained using unsupervised dataset
	- Outputs binary anomally class for each host (trained using cross-entropy loss)
	- Outputs class prototype vector (trained using triplet loss)

2. GAN training phase (ADPE is fine-tuned)
	- Freeze Generator and get discriminator output
	- Ground truth = co-simulate old and new and get best
	- Backprop discriminator using cross-entropy loss
	- Freeze Discriminator, ground truth = new better
	- Backrpop generator and ADPE using cross-entropy loss

3. Testing phase (ADPE and GAN frozen)
	- Run inference and obtain Discriminator output
	- Use best output as scheduling decision

## Implementation Details

1. Normalization of data for stability
2. Percentile used for ground truth labels.
3. Loss multiplier for class imbalance.
4. Class representation prototypes updated using exponential decay.

* Train Encoder - 200 intervals
* Traing GAN - 400 intervals
* Test - 100 intervals

Algorithms for the three stages.

## Figures and Comparisons

- @Model diagram (Figure) :
	- ADPE
	- Generator
	- Discriminator

- @Visualization of attention score (Plot) :
	- Truncated (say 5 dimension time-series data) with colormap of attention scores for each dimension

- @Visualization of predictions with classes (Plot) :
	- Truncated (say 5 dimension time-series data) with class highlighting

- @Visualization of class prototypes (Plot) :
	- tSNE plot of embeddings on test dataset for different class protypes.

- @Loss and accuracy curves with intervals/epochs for ADPE and GAN (Plot):
	- Detection and classification accuracy, visualization of predicitons/ground-truth, confusion matrix
	- GAN loss (gen and disc), which is better

- @Visualization of migration decisions (Plot) :
	- Migration from hosts.
	- Accuracy (correct prediction / total anomalies), better schedule (migrations / migrations+no-migrations)

- @Experimental setup (Image) :
	- RPi Cluster

- @Dataset Statistics (Table) :
	- Dimension
	- Size (training and testing)
	- Anomaly rate

- @Comparison (Table) :
	- detection - Acc, P, R, F1
	- diagnosis - NDCG@5, HitRate@100%, HitRate@150%
	- classification error
	- %found better schedule = (No of better decisions as per co-simulation / Total anomalies detected after experiment by ADE tool). This means out of say 200 intervals, 40 have true anomalies, 38 were detected, 30 were those where a better schedule was found. So 30 / 40. (Most important metric)

- @QoS (Plots) :
	- Response time, energy, sla, preemptive migrations, anomalies (each class). Overheads.

- @QoS senstivity using simulator (Plot) :
	- QoS with workloads. (energy, sla, Improvement ratio, f1)

- Sensitivity Analysis (Plots) : 
	- detection/diagnosis/classification with window size, prototype dimension, decay factor.

## License

BSD-3-Clause. 
Copyright (c) 2021, Shreshth Tuli.
All rights reserved.

See License file for more details.
