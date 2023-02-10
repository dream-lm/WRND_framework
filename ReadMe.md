<!--
 * @Author: Zhou Hao
 * @Date: 2022-04-07 18:04:04
 * @LastEditors: Zhou Hao
 * @LastEditTime: 2022-04-07 18:15:56
 * @Description: file content
 * @E-mail: 2294776770@qq.com
-->

# WRND: An adaptive robust weighted oversampling framework for imbalanced classification with natural neighborhood density


* **Abstract**：Data imbalance and label noise are ubiquitous challenges in data mining and machine learning. The synthetic minority oversampling technique (SMOTE) and its variants have been proposed, but they are easily constrained by k-nearest neighbor hyperparameters, deteriorate performance duo to noise, rarely take data distribution information into account, and cause high complexity or additional parameter optimization. Furthermore, SMOTE-based methods select nearest neighbor samples and synthesize new samples by interpolating between neighbors arbitrarily, which do not consider the distribution of information and neighborhood oversampling. To address the above problems, an adaptive, robust, and general weighted oversampling framework based on natural neighborhood density (WRND) is proposed. it can combine with most SMOTE-based sampling algorithms easily and improve their performance. Firstly, according to the concepts of natural neighbors, it adaptively searches the natural neighborhood, and distinguishes and filters noisy and outlier samples. Then, the natural neighborhood density of each sample can be obtained, which reflects the intra-class and inter-class distribution information within the natural neighborhood. To alleviate the blindness of SMOTE-based methods, the amount and locations of synthetic samples are assigned informedly based on distribution information and reasonable generalization of natural neighborhoods of original samples. Extensive experiments on 24 benchmark datasets and six classic classifiers with eight pairs of representative sampling algorithms and two state-of-the-art frameworks, the experimental results significantly demonstrate the effectiveness of the WRND framework. Code and framework are available at https://github.com/dream-lm/WRND_framework.
* **Keyword**: Imbalanced classification, Label noise,Oversampling framework, Natural neighborhood density.

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
