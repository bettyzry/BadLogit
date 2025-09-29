

## Specification of dependencies.
The specification of dependencies is shown in `requirements.txt`.


## BadFreq's training code and evaluation code.

Use `demo_finetunning_attack.py` for training and evaluation. 
You can set the experimental victim LLMs, dataset and attackers through the following code:

```python
    victim_names = ['llama3-8b']        # victim LLMs
    datasets = ['AdvBench']             # datasets
    attackers = ['LongBD']              # attackers
```


## Adaptive defender's defection code and evaluation code.

Use `BadLogits.py` for training and evaluation. 
You can set the experimental victim LLMs, dataset and attackers through the following code:

```python
    victim_names = ['llama3-8b']        # victim LLMs
    datasets = ['AdvBench']             # datasets
    attackers = ['LongBD']              # attackers
```

## Backdoor Attack Training (BadChain) code and evaluation code.

Use `BadChain.py` for training and evaluation. 
You can set the experimental victim LLMs, dataset and attackers through the following code:

```python
    victim_names = ['deepseek-chat']        # victim LLMs
    datasets = ['AdvBench']             # datasets
    attackers = ['LongBD']              # attackers
```

The results are recorded in the path "result.csv"

## Results tables and figures.
The experimental results are located in the path "./plot_resource".

Use functions in `plot.py` to plot the following figures.

```python
plot_logits()
```
<img src="plot_resource/logits.jpg" alt="Case True" width="400">  


```python
plot_rate()
```
![results](plot_resource/rate.jpg)

```python
plot_param()
```
![results](plot_resource/param.jpg)

```python
plot_lf_ss()
```
![results](plot_resource/lf-ss.jpg)


```python
plot_badchain()
```
![results](plot_resource/badchain.jpg)


```python
plot_frequency()
```
![results](plot_resource/freq.jpg)

```python
plot_ablation()
```
![results](plot_resource/ablation.jpg)

```python
plot_logit_point()
```
![results](plot_resource/logit-SST-2.jpg)