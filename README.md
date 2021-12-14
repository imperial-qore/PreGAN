<h1 align="center">PreGAN</h1>
<div align="center">
  <a href="https://github.com/imperial-qore/PreGAN/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/License-BSD%203--Clause-red.svg" alt="License">
  </a>
   <a>
    <img src="https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg" alt="Python 3.7, 3.8">
  </a>
   <a>
    <img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fimperial-qore%2FPreGAN&count_bg=%23FFC401&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false" alt="Hits">
  </a>
   <a href="https://github.com/imperial-qore/PreGAN/actions">
    <img src="https://github.com/imperial-qore/COSCO/workflows/DeFog-Benchmarks/badge.svg" alt="Actions Status">
  </a>
 <br>
   <a>
    <img src="https://img.shields.io/docker/pulls/shreshthtuli/yolo?label=docker%20pulls%3A%20yolo" alt="Docker pulls yolo">
  </a>
   <a>
    <img src="https://img.shields.io/docker/pulls/shreshthtuli/pocketsphinx?label=docker%20pulls%3A%20pocketsphinx" alt="Docker pulls pocketsphinx">
  </a>
   <a>
    <img src="https://img.shields.io/docker/pulls/shreshthtuli/aeneas?label=docker%20pulls%3A%20aeneas" alt="Docker pulls aeneas">
  </a>
</div>

Building a fault-tolerant edge system that can quickly react to node overloads or failures is challenging due to the unreliability of edge devices and the strict service deadlines of modern applications. Moreover, unnecessary task migrations can stress the system network, giving rise to the need for a smart and parsimonious failure recovery scheme. Prior approaches often fail to adapt to highly volatile workloads or accurately detect and diagnose faults for optimal remediation. There is thus a need for a robust and proactive fault-tolerance mechanism to meet service level objectives. In this work, we propose PreGAN, a composite AI model using a Generative Adversarial Network (GAN) to predict preemptive migration decisions for proactive fault-tolerance in containerized edge deployments. PreGAN uses co-simulations in tandem with a GAN to learn a few-shot anomaly classifier and proactively predict migration decisions for reliable computing. Extensive experiments on a Raspberry-Pi based edge environment show that PreGAN can outperform state-of-the-art baseline methods in fault-detection, diagnosis and classification, thus achieving high quality of service. PreGAN accomplishes this by 5.1% more accurate fault detection, higher diagnosis scores and 23.8% lower overheads compared to the best method among the considered baselines.

## Quick Test
Clone repo.
```console
git clone https://github.com/imperial-qore/PreGAN.git
cd PreGAN/
```
Install dependencies.
```console
sudo apt -y update
python3 -m pip --upgrade pip
python3 -m pip install matplotlib scikit-learn
python3 -m pip install -r requirements.txt
python3 -m pip install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
export PATH=$PATH:~/.local/bin
```
Change line 117 in `main.py` to use one of the implemented fault-tolerance techniques: `PreGANRecovery`, `PCFTRecovery`, `DFTMRecovery`, `ECLBRecovery` or `CMODLBRecovery` and run the code using the following command.
```console
python3 main.py
````

## External Links
| Items | Contents | 
| --- | --- |
| **Pre-print** | https://arxiv.org/abs/2112.02292 |
| **Contact**| Shreshth Tuli ([@shreshthtuli](https://github.com/shreshthtuli))  |
| **Funding**| Imperial President's scholarship |

## Cite this work
Our work is accepted in IEEE Conference on Computer Communications (INFOCOM) 2022. Cite our work using the bibtex entry below.
```bibtex
@inproceedings{tuli2021pregan,
  title={{PreGAN: Preemptive Migration Prediction Network for Proactive Fault-Tolerant Edge Computing}},
  author={Tuli, Shreshth and Casale, Giuliano and Jennings, Nicholas R},
  booktitle={IEEE Conference on Computer Communications (INFOCOM)},
  year={2022},
  organization={IEEE}
}

```

## License

BSD-3-Clause. 
Copyright (c) 2021, Shreshth Tuli.
All rights reserved.

See License file for more details.
