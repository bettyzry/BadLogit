

## Specification of dependencies.
The specification of dependencies is shown in `requirements.txt`.


## Backdoor Attack Training code and evaluation code.

Use `demo_finetunning_attack.py` for training and evaluation. 
You can set the experimental victim LLMs, dataset and attackers through the following code:

```python
    victim_names = ['llama3-8b']        # victim LLMs
    datasets = ['SST-2']             # datasets
    attackers = ['LongBD']              # attackers
```



## Backdoor Defense code and evaluation code.

Use `BadLogits.py` for training and evaluation. 
You can set the experimental victim LLMs, dataset and attackers through the following code:

```python
    victim_names = ['llama3-8b']        # victim LLMs
    datasets = ['SST-2']             # datasets
    attackers = ['LongBD']              # attackers
```



The results are recorded in the path "result.csv"

## Results tables and figures.
The experimental results are located in the path "./plot_resource".

Use functions in `plot.py` to plot the following figures.

```python
plot_redar()
```

![results](plot_resource/defense_redar.jpg)


```python
plot_rate()
```
![results](plot_resource/rate.jpg)

```python
plot_logits()
```
<img src="plot_resource/logits.png" alt="Case True" width="400">  


```python
plot_param()
```
![results](plot_resource/param.jpg)
