# Understanding Imbalanced Forgetting in Rehearsal-Based Class-Incremental Learning

> Alberto Tamajo, Srinandan Dasmahapatra and Rahman Attar

Official repository for the paper [Understanding Imbalanced Forgetting in Rehearsal-Based Class Incremental Learning](https://arxiv.org/abs/2605.14785) (under review).

## Abstract
Neural networks suffer from *catastrophic forgetting* in class-incremental learning (CIL) settings. Rehearsal—replaying a subset of past samples—is a well-established mitigation strategy. However, recent results suggest that, despite balanced rehearsal allocation, some classes are forgotten substantially more than others. Despite its relevance, this *imbalanced forgetting* phenomenon remains underexplored. This work shows that imbalanced forgetting arises systematically and severely in rehearsal-based CIL and investigates it extensively. Specifically, we construct, from a principled analysis, three last-layer coefficients that capture different gradient-level sources of interference affecting each past class *during* an incremental step. We then demonstrate that, together, they reliably predict how past classes will rank in terms of forgetting *at the end* of that step. While predictive performance alone does not establish causality, these results support the interpretation of the coefficients as a plausible mechanistic account linking last-layer gradient-level interactions during training to class-level forgetting outcomes. Notably, one coefficient—capturing self-induced interference—emerges as the strongest predictor, with controlled experiments providing evidence consistent with this coefficient being influenced by the new-class interference coefficient. Overall, our findings provide valuable insights and suggest promising directions for mitigating imbalanced forgetting by reducing class-wise disparities in the identified sources of interference.


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

