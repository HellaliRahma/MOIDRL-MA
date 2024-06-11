# A Multi-Objective Multi-Agent Interactive Deep Reinforcement Learning Approach for Feature Selection (MOIDRL-MA)
 
Feature selection is paramount in data preprocessing for optimizing feature subsets in machine learning tasks. Intriguing recent developments in this area are the Interactive Reinforced Feature Selection  methods, showing promise. However, they often prioritize singular metrics like accuracy, overlooking conflicting metrics or objectives. Moreover, their reliance on a single trainer can hamper efficiency by providing repetitive advice. To tackle these issues, this paper presents a Multi-objective Multi-agent Interactive Deep Reinforcement Learning method. We conceptualize feature selection as a multi-objective problem and introduce a novel reward assignment method that balances the Area Under the Curve (AUC) measure and the number of selected features. Furthermore, we propose a diverse trainer strategy that harnesses multiple trainers to prevent repetitive guidance. This strategy leverages diverse external trainers   for accelerated feature exploration and fosters self-exploration by the agent. Our approach yields Pareto Front solutions, offering flexibility for decision-makers in selecting the final optimal feature set. Empirical results demonstrate superior performance compared to state-of-the-art feature selection methods. Additionally, through the Pareto Front solutions, our method effectively manages the trade-offs between AUC and feature count.

## Table of Contents

- [Requirements](#requirements)
- [Usage](#usage)

## Requirements

- Python 3.x
- Tensorflow 2.11.0

## Usage
(1) python MainMLL.py

(2) Execute the "Pareto_Front_MLL.ipynb" Jupyter notebook to calculate the Pareto front solutions.
