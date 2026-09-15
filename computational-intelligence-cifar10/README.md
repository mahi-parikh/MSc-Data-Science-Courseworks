# Computational Intelligence — CIFAR-10 Image Classification

MSc Data Science group coursework. This project trains a custom AlexNet-inspired
convolutional neural network (CNN) on CIFAR-10, then compares gradient-based and
nature-inspired metaheuristic algorithms for optimizing the network's final layer.
Since training the whole neural network(NN) repeatedly for all the optimization
algorithms was computationally expensive, the full NN for the 
complete CIFAR 10 dataset was only trained once using Gradient Descent, and then 
all the layers of the trained NN were freezed except the last. Following this, the 
optimal weights of the last layer were randomised and multiple optimization algorithms
were used to optimize only the weights of this last layer.

## Project overview

The project compares a gradient-based baseline with several non-gradient and evolutionary optimisation approaches. The CNN contains five convolutional layers, three max-pooling layers and three fully connected layers, with the final layer subsequently used as the optimisation target for the metaheuristic experiments.

- **Custom CNN**: 5 convolutional layers (with batch normalization) + 3 fully
  connected layers, inspired by AlexNet, trained on CIFAR-10.
- **Optimization methods compared**, all applied to the final layer after freezing
  the rest of the network:
  - Gradient Descent (SGD) — baseline
  - Particle Swarm Optimization (PSO)
  - Simulated Annealing (SA)
  - Clonal Selection Algorithm (CSA)
  - NSGA-II — a multi-objective variant balancing validation loss against a
    weight-magnitude regularizer
    
## My contribution

This was a **group project**. My individual contribution was:

- Design and implementation of the custom AlexNet-inspired CNN.
- Implementation and experimentation with the Clonal Selection Algorithm.

The repository also contains the other group experiments because they form part of the complete comparative study. They are not presented as solely my work.

## Key results

Gradient descent outperformed all the metaheuristics on validation accuracy, suggesting the CNN's loss landscape (after feature extraction) was smooth enough to favor local, gradient-based search over global exploration. Among the metaheuristics (each evaluated over 50 fitness evaluations), Clonal Selection reached 64.12% accuracy and PSO 63.52%, both ahead of Simulated Annealing at 59.89%. The NSGA-II experiment, run as a bi-objective problem over 60 generations, produced a Pareto front trading off validation loss against weight magnitude — the best-loss solution (≈1.52e-06 validation loss) achieved only 21.16% full test accuracy, since it was evaluated on a single batch and optimized only the final layer, while the best weight-penalty solution kept weights smaller (≈0.0031 loss, ≈1357.9 summed squared weights) at the cost of higher loss.

## Files

- `report/CompIntelligence_CW.pdf` — full write-up (IEEE conference
  format)
- `notebooks/neural_network_submission.ipynb` — CNN architecture, training loop,
  and all four optimization algorithms

## Reproducing

```bash
pip install -r requirements.txt
```

CIFAR-10 downloads automatically via `torchvision.datasets` on first run — no
manual dataset setup needed.

## Note

This was **group coursework**, submitted as part of an MSc Data Science module.
Personal identifying details for group members have been removed from the
report before publishing here. The code and write-up reflect a shared team
submission, not solo work.
