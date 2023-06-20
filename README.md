# Bachelor's Thesis Dependencies

This repository contains all publishable dependencies from Dieterich Lab Heidelberg for a Bachelors thesis in Computational Linguistics (B.A. - 2023) at Heidelberg University with the title: 
#### "Interpretability in NLP - A comparative analysis of two post-hoc interpretability methods on the Faitfulness metric"  

#### 🔍 Contents
* [Thesis presentation at Dieterich Lab](BA_thesis_present_Dieterichlab.pdf)
* Notebooks
  * [Faithfulness - Sufficiency](ferret-Suff.ipynb)  
  * [Faithfulness - Comprehensiveness](ferret-Comp.ipynb)  
  * [Training Data Statistics](DataStatistics.ipynb)  
* Additions
  * [Python code for training and testing BERT model](BertSeqCardio.py)
  * [Test Results as plain text](BertSeqCA.txt)
  * [Bash script for running GPU intensive job on Dieterich Lab SLURM cluster](BertSeqCA.sh)

To run the _Faithfulness_ notebooks, please beforehand contact [the author](#Maintenance) to obtain the necessary confidential data. You may then proceed as follows:

#### 🧰 Prerequisites
1. Create and activate a Python virtual environment: `python3 -m venv xai`
2. Install all required packages in [requirements.txt](requirements.txt): `pip install -r requirements.txt`
3. Create the according IPython kernel for Jupyter: `python3 -m ipykernel install --user --name=xai`

#### ⚙️ Run Faithfulness Notebooks
1. Open Jupyter Notebook: `jupyter notebook` and select the created kernel _xai_
2. Both Comprehensiveness & Sufficiency notebooks can be executed

#### ✏️ Maintenance
Author: Raziye Sari - sari@cl.uni-heidelberg.de  
Last updated: June 20th 2023  
