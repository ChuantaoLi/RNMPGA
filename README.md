# README

Multi-population evolutionary models have been widely adopted for feature selection owing to their effective search ability. However, existing studies remain prone to local optima and premature convergence when addressing high-dimensional feature selection problems. To mitigate these limitations, this study proposes the Ring-Neighborhood Multi-Population Genetic Algorithm (RNMPGA) for high-dimensional feature selection. RNMPGA decomposes the feature selection problem into multiple subproblems, each with a distinct search preference and assigned to a dedicated subpopulation. RNMPGA links subpopulations in a ring structure and each subpopulation connects only to its two immediate neighbors. Within each subpopulation, uniform crossover and guided mutation drive local evolution. Between subpopulations, the elite interaction mechanism shares high-quality information among neighboring subpopulations. Experiments on 23 high-dimensional datasets show that RNMPGA achieves the best average ranks for feature selection among the six comparison methods, with average ranks of 1.17, 1.48, and 2.17 for FN, HV, and IGD, respectively.



The run commands are as follows:

- Install dependencies: `pip install -r requirements.txt`
- Comparative experiments: `python main.py --config config.yaml`
- Ablation analysis: `python exp_ablation.py --repeats 10 --datasets auto`
- Parameter sensitivity analysis: `python exp_sensitivity.py --repeats 10 --datasets auto`



![rnmpga-workflow](https://lctmemos.oss-cn-beijing.aliyuncs.com/img/rnmpga-workflow.png)
