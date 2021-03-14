# Interpreting Multivariate Shapley Interactions in DNNs

Official code implementation for the paper "Interpreting Multivariate Shapley Interactions in DNNs" (AAAI 2021) [paper](https://arxiv.org/abs/2010.05045). 

## Prerequisites

The code was tested with python 3.6, Tensorflow 1.14.0, keras 2.3.1, CUDA 10.1 and Ubuntu 16.04.

## Installation

1. Clone the repository:

   ```
   git clone https://github.com/Yichen-Xie/Multivariate_Shapley_Interactions.git
   ```

2. Install Tensorflow and keras:

   ```
   pip install tensorflow-gpu==1.14.0 keras==2.3.1
   ```

3. Install other necessary packages:

   ```
   pip install numpy matplotlib scipy six
   ```

## Data & Model Preparation

Download the pre-processed SST-2 and CoLA dataset as well as the pre-trained BERT and ELMo models [here](https://drive.google.com/drive/folders/1s2uxXOHhGsJvPYIbIMCe09d0Ygs1Qy2-?usp=sharing).

Make sure to put the files in the following structure:

```
$ROOT$
|	|--GLUE_data
|	|	|--SST-2
|	|	|--CoLA
|	|--models
|	|	|--uncased_L-12_H-768_A-12
|	|	|--Elmo
|	|--elmo
|	|	|--elmo_data.py
|	|	|--tf_module
```

## Demonstration

## Toy Tasks

## Experiments on NLP Tasks

## Citation

If you found our paper or code useful for your research, please cite the following paper:

```
@inproceedings{zhang2020Interpreting ,
      title={Interpreting Multivariate Shapley Interactions in DNNs}, 
      author={Zhang, Hao and Xie, Yichen and Zheng, Longjie and Zhang, Die and Zhang, Quanshi},
      year={2021},
      booktitle = {The AAAI Conference on Artificial Intelligence (AAAI)}
}
```

