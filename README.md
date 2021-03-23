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

Download the pre-processed SST-2 and CoLA dataset as well as the pre-trained BERT, ELMo and CNN models [here](https://drive.google.com/drive/folders/1s2uxXOHhGsJvPYIbIMCe09d0Ygs1Qy2-?usp=sharing).

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

The demonstration quantifies the significance of interaction among words on a certain sentence in either SST-2 or CoLA dataset.  It could also show the partitions of coalition with maximal or minimal interaction.

```
python compute_interaction_bert.py --task_name $cola or sst-2$ --sentence_idx $id of sentence in the val set$  --seg_start $beginning position of the selected coalition in the sentence$ --seg_end $end position of the selected coalition in the sentence$ 

e.g.
python compute_interaction_bert.py --task_name sst-2 --sentence_idx 171  --seg_start 2 --seg_end 5
```

## Toy Tasks

We evaluate the accuracy of estimated partition on three toy datasets:  Add-Multiple Dataset, AND-OR Dataset, and Exponential Dataset. 

- [x] Add-Multiple Dataset
- [x] AND-OR Dataset
- [x] Exponential Dataset

### Add-Multiple Dataset

In the Add-Multiple dataset,  each sample consists of addition operations and multiplication operations *e.g.* $y=f(x)=x_1+x_2×x_3+x_4×x_5+x_6+x_7$.

#### Dataset Generation

We generate this toy dataset automatically. Please run the following command:

```
cd toy_dataset/generators
python generator.py --task multiple_add --size $number of samples in the dataset$ --min_len $minimal length of each sample$ --max_len $maximal length of each sample$

e.g.
cd toy_dataset/generators
python generator.py --task multiple_add --size 1000 --min_len 8 --max_len 12 
```

#### Evaluation

Run our method to estimate the partition and evaluate the accuracy:

```
python compute_interaction_toy.py --data_path toy_dataset/datasets/toy_dataset_multiple_add.json
```

### AND-OR Dataset

In the AND-OR dataset,  each sample only contains AND operations and OR operations *e.g.* $y=f(x)=x_1|x_2\&x_3|x_4\&x_5|x_6|x_7$.

#### Dataset Generation

We generate this toy dataset automatically. Please run the following command:

```
cd toy_dataset/generators
python generator.py --task and_or --size $number of samples in the dataset$ --min_len $minimal length of each sample$ --max_len $maximal length of each sample$

e.g.
cd toy_dataset/generators
python generator.py --task and_or --size 1000 --min_len 8 --max_len 12 
```

#### Evaluation

Run our method to estimate the partition and evaluate the accuracy:

```
python compute_interaction_toy.py --data_path toy_dataset/datasets/toy_dataset_and_or.json
```

### Exponential Dataset

In the Exponential  dataset,  each sample contains exponential operations and addition operations *e.g.* $y=f(x)=x_1^{x_2}+x_3^{x_4}+x_5+x_6$.

#### Dataset Generation

We generate this toy dataset automatically. Please run the following command:

```
cd toy_dataset/generators
python exp_generator.py --size $number of samples in the dataset$ --min_len $minimal length of each sample$ --max_len $maximal length of each sample$

e.g.
cd toy_dataset/generators
python exp_generator.py --size 1000 --min_len 8 --max_len 12 
```

#### Evaluation

Run our method to estimate the partition and evaluate the accuracy:

```
python compute_interaction_toy.py --data_path toy_dataset/datasets/toy_dataset_exp.json
```

## Experiments on NLP Tasks

### Evaluation of the accuracy of $T([A])$

We compared the extracted significance of interactions $T([A])$ estimated by our proposed method with the accurate significance of interactions derived from Shapley value.

We conduct this experiment with multiple different models:

- [x] BERT model
- [x] ELMo model
- [ ] LSTM model
- [ ] CNN model
- [ ] Transformer model

```
cd accuracy_evaluation

# BERT model
python compute_interaction_diff_bert.py

# ELMo model
python compute_interaction_diff_elmo.py
```

Please refer to `accuracy_evaluation/draw_figure.py` to visualize the results like our paper.

### Stability of $T([A])$

We also measured the stability of $T([A])$, when we computed $T([A])$ multiple times with different sampled sets of $g$ and $S$.

- [ ] BERT model
- [x] ELMo model
- [ ] LSTM model
- [x] CNN model
- [ ] Transformer model

```
cd instability

# ELMo model
cd elmo_interaction
python interaction_elmo.py  
# estimate T([A]) multiple times and save the result
python compute_interaction_instability_elmo.py
# calculate the instability with the saved T([A])

# CNN model
cd cnn_interaction
python interaction_cnn.py  
# estimate T([A]) multiple times and save the result
python compute_interaction_instability_cnn.py
# calculate the instability with the saved T([A])
```

You can refer to our code `instability/instability_figure.py` to draw a figure to visualize the stability of estimation.

### Interactions *w.r.t.* the intermediate-layer feature

We could also compute the significance of interactions among a set of input words *w.r.t.* the computation of an intermediate layer feature.

We carry out this experiment on two datasets: SST-2 and CoLA.

```
cd intermediate_layers
python compute_interaction_bert_intermed.py --task_name $cola or sst-2$
```

You can refer to our code `intermediate_layers/draw_figure.py` to visualize the significance of interaction in the  intermediate layers .

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

