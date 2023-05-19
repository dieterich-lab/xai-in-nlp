# Thesis Dependencies

This repository contains all publishable dependencies from Dieterich Lab Heidelberg for a Bachelors thesis at Heidelberg University with the title: 
#### "Interpretability in NLP - A comparative analysis of two post-hoc interpretability methods on the *Faitfulness* metric"  

#### 🔍 Where to find what
* [Thesis presentation](BA_thesis_present_Dieterichlab.pdf)
* Notebooks
  * [Faithfulnes - Sufficiency](ferret-Suff.ipynb)  
  * [Faithfulness - Comprehensiveness](ferret-Comp.ipynb)  
  * [Training Data Statistics](DataStatistics.ipynb)  
* Code
  * [Training and testing BERT](BertSeqCardio.py)
  * [Test Results](BertSeqCA.txt)
  * [Bash script for running code on SLURM](BertSeqCA.sh)

To run the _Faithfulness_ notebooks, please proceed as follows:

#### 🧰 Prerequisites
1. Create and activate Python venv: `python3 -m venv xai`
2. Install requirements.txt: `pip install -r requirements.txt`
3. Create according IPython kernel: `python3 -m ipykernel install --user --name=xai`

#### ⚙️ Run Faithfulness Notebooks
1. Open Jupyter Notebook: `jupyter notebook` and select _xai_ kernel
2. Both Comprehensiveness & Sufficiency notebooks can be executed

#### ✏️ Mantainenance
Author: Raziye Sari - sari@cl.uni-heidelberg.de  
Last updated: May 19th 2023  
