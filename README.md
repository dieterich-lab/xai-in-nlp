# Thesis Repository

This repository contains all publishable dependencies for a Bachelors thesis with the title: "Interpretability in NLP - A comparative analysis of two post-hoc interpretability methods on the *Faitfulness* metric."  
Author: Raziye Sari  
Last updated: May 19th 2023  

#### Prerequisites
1. Create and activate Python venv: `python3 -m venv xai`
2. Install requirements.txt: `pip install -r requirements.txt`
3. Create according IPython kernel: `python3 -m ipykernel install --user --name=xai`

#### Run Faithfulness Notebooks
1. Open Jupyter Notebook: `jupyter notebook` and select _xai_ kernel
2. Both Comprehensiveness & Sufficiency notebooks can be executed

#### Where to find what
* [Thesis presentation](BA_thesis_present_Dieterichlab.pdf)
* Notebooks
  * [Sufficiency Notebook](ferret-Suff.ipynb)  
  * [Comprehensiveness Notebook](ferret-Comp.ipynb)  
  * [Training Data Statistics](DataStatistics.ipynb)  
* Code
  * [Training and testing BERT](BertSeqCardio.py)
  * [Test Results](BertSeqCA.txt)
* Additional Stuff
  * [Bash script](BertSeqCA.sh)
