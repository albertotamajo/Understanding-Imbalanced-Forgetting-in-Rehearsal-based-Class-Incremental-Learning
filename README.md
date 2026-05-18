# Understanding Imbalanced Forgetting in Rehearsal-Based Class-Incremental Learning

> Alberto Tamajo, Srinandan Dasmahapatra and Rahman Attar

Official repository for the paper [Understanding Imbalanced Forgetting in Rehearsal-Based Class Incremental Learning](https://arxiv.org/abs/2605.14785) (under review).

## Abstract
Neural networks suffer from *catastrophic forgetting* in class-incremental learning (CIL) settings. Rehearsal—replaying a subset of past samples—is a well-established mitigation strategy. However, recent results suggest that, despite balanced rehearsal allocation, some classes are forgotten substantially more than others. Despite its relevance, this *imbalanced forgetting* phenomenon remains underexplored. This work shows that imbalanced forgetting arises systematically and severely in rehearsal-based CIL and investigates it extensively. Specifically, we construct, from a principled analysis, three last-layer coefficients that capture different gradient-level sources of interference affecting each past class *during* an incremental step. We then demonstrate that, together, they reliably predict how past classes will rank in terms of forgetting *at the end* of that step. While predictive performance alone does not establish causality, these results support the interpretation of the coefficients as a plausible mechanistic account linking last-layer gradient-level interactions during training to class-level forgetting outcomes. Notably, one coefficient—capturing self-induced interference—emerges as the strongest predictor, with controlled experiments providing evidence consistent with this coefficient being influenced by the new-class interference coefficient. Overall, our findings provide valuable insights and suggest promising directions for mitigating imbalanced forgetting by reducing class-wise disparities in the identified sources of interference.

## Repository Structure

```text
.
├── Analysis/          # Scripts for analyzing data extracted from completed experiments
├── Experiments/       # Scripts for running the experiments presented in the paper
├── Extractions/       # Scripts for extracting data from completed experiments
├── benchmarks/        # CIFAR-100 and TinyImageNet benchmarks adapted to class-incremental learning
├── convnets/          # ResNet backbone implementations
├── metrics/           # Metrics tracked during continual learning
├── models/            # Building blocks for continual learning architectures
├── training/          # Continual learning training loops
├── LICENSE            # Repository license
├── README.md          # Readme file
└── environment.yml    # Conda environment file for reproducing the project environment
```

The `benchmarks`, `metrics`, `models`, and `training` modules build on the corresponding modules from the [Avalanche](https://avalanche.continualai.org/) framework. We thank the authors for their contributions to the continual learning community.

## Reproducibility
This repository provides two different reproducibility workflows.
### Full Reproducibility
To fully reproduce the results presented in the paper, follow the instructions below in the given order:
1. We provide an `environment.yml` file containing a list of the necessary dependencies. Use it to reproduce the same environment in your machine.
2. Run all the scripts inside the `Experiments` folder.
3. Run all the scripts inside the `Extractions` folder.
	- These scripts produce `.pkl` files containing the extracted data. Move these files into the `Analysis` folder.
4. Use the `normal_exps_analysis` and `controlled_exps_analysis` scripts inside the `Analysis` folder to analyze the data extracted from the standard experiments and the controlled experiments, respectively.
### Lightweight Reproducibility
Alternatively, since full reproduction is time-consuming, you can skip Steps 1–3 by downloading from [here](https://1drv.ms/u/c/b2e831b9f4e18bc4/IQCz1LW5eOnKTKJ_9nycnNH_AWIRYC9Ro_Ac7cQPoPJlO2w?e=esjKdS) a zipped folder containing the `.pkl` files generated during our empirical investigation. As before, move these files into the `Analysis` folder and proceed to step 4.


## Citation
If you find our work useful in your research, please consider citing:

	@article{tamajo2026understanding,
  		title={Understanding Imbalanced Forgetting in Rehearsal-Based Class-Incremental Learning},
  		author={Tamajo, Alberto and Dasmahapatra, Srinandan and Attar, Rahman},
  		journal={arXiv preprint arXiv:2605.14785},
  		year={2026}
  	}

	
## License
Our code is released under MIT License (see LICENSE file for details).

