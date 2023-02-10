<!--
 * @Author: Zhou Hao
 * @Date: 2022-04-07 18:04:04
 * @LastEditors: Zhou Hao
 * @LastEditTime: 2022-04-07 18:15:56
 * @Description: file content
 * @E-mail: 2294776770@qq.com
-->

# WRND: An adaptive robust weighted oversampling framework for imbalanced classification with natural neighborhood density


* **Abstract**：Data imbalance and label noise are ubiquitous challenges in data mining and machine learning, severely impairing classification performance. The synthetic minority oversampling technique (SMOTE) and its variants have been proposed, but they are easily constrained by hyperparameter optimization such as k-nearest neighbor, deteriorate performance duo to noise, rarely take data distribution information into account, and cause high complexity. Furthermore, SMOTE-based methods perform random linear interpolation between each minority class sample and its randomly selected k-nearest neighbors, regardless of sample difierences and distribution information. To address the above problems, an adaptive, robust, and general weighted oversampling framework based on relative neighborhood density (WRND) is proposed. It can combine with most SMOTE-based sampling algorithms easily and improve their performance. Firstly, it adaptively distinguishes and filters noisy and outlier samples by introducing the natural neighbor, which inherently avoids extra noise and overlapping samples introduced by the synthesis of noisy samples. Then, the relative neighborhood density of each sample can be obtained, which reects the intra-class and inter-class distribution information within the natural neighborhood. To alleviate the blindness of SMOTE-based methods, the amount and locations of synthetic samples are assigned informedly based on distribution information and reasonable generalization of natural neighborhoods of original samples. Extensive experiments on 24 benchmark datasets and six classic classifiers with eight pairs of representative sampling algorithms and two state-of-the-art frameworks, the experimental results significantly demonstrate the efiectiveness of the WRND framework. Code and framework are available at https://github.com/dream-lm/WRND framework.

* **Keyword**: Imbalanced classification, Label noise,Oversampling framework, Relative neighborhood density.

# Folders and Filers

* **results for each dataset**: This folder corresponds to the experimental results in Chapter 4 of the manuscript.
* **visual experiments results**: This folder corresponds to the visualization of the experimental results in the manuscript.
* **all_smote_v5.py**: SMOTE-WRND， Borderline_SMOTE-WRND.
* **apis.py**: Some functions for synthesizing artificial datasets.
* **fourclass10_change.csv**: A dataset.
* **main.py**: Code entry. Call the oversampling algorithms and visualize.
* **requirements.txt**: Environment required for code.

# Requirements

### Minimal installation requirements (>=Python 3.7):

* Anaconda 3.
* Linux operating system or Windows operating system.
* Sklearn, numpy, pandas, imbalanced_learn.

### Installation requirements (Python 3):

* pip install -r requirements.txt

# Usage

* pip install -r requirements.txt.
* python main.py

# Doesn't work?

* Please contact Hao Zhou at 2294776770@qq.com
