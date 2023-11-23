# iEnhancer-ELM
📋 iEnhancer-ELM: Learning Explainable Contextual Information to Improve Enhancer Identification using Enhancer Language Models

## Abstract
&nbsp;&nbsp;&nbsp;&nbsp; Enhancers are important cis-regulatory elements, enhancing the transcription of target genes. De novo design of high-activity enhancers is one of long-standing goals in generated biology for both clinical purpose and artificial life, because of their vital roles on regulation of cell development, differentiation, and apoptosis. But designing the enhancers with specific properties remains challenging, primarily due to the unclear understanding of enhancer regulatory codes. Here, we propose an AI-driven enhancer design method, named Enhancer-GAN, to generate high-activity enhancer sequences. Enhancer-GAN is firstly pre-trained on a large enhancer dataset that contains both low-activity and high-activity enhancers, and then is optimized to generate high-activity enhancers with feedback-loop mechanism. Domain constraint and curriculum learning were introduced into Enhancer-GAN to alleviate the noise from feedback loop and accelerate the training convergence. Experimental results on benchmark datasets demonstrate that the activity of generated enhancers is significantly higher than ones in benchmark dataset. Besides, we find 10 new motifs from generated high-activity enhancers. These results demonstrate Enhancer-GAN is promising to generate and optimize bio-sequences with desired properties.


<div align=center><img src="Figure/framework.png" width="700" /></div>


## Code
### Enviromnent
```
# Clone this repository
git clone https://github.com/chen-bioinfo/iEnhancer-ELM.git
cd iEnhancer-ELM

# download the pre-trained BERT-based DNA models from the link (https://drive.google.com/drive/folders/1qzvCzYbx0UIZV3HY4pEEeIm3d_mqZRcb?usp=sharing);
# With these pre-trained models and the code file of iEnhancer-ELM/code/DNA_bert_finetuning_average_L2.ipynb, 
# we can reproduce the training process.

cd iEnhancer-ELM/code
# download the fine-trained classification models form the link (https://drive.google.com/drive/folders/1EdOYQ2BLcAUtS_dupWdmJ-v6bkne4xAM?usp=sharing);
# With these fine-trained modles and the code file of iEnhancer-ELM/code/DNA_Bert_finetuning_L2_ensemble.ipynb, 
# we can reproduce the best performance in independent dataset. And our motif analysis is based on these fine-trained models. 

# the key elements of 'iEnhancer-ELM' operating environment are listed below:
# python=3.6.9; Torch=1.9.0+cull 
# Numpy=1.19.5; Transformers=3.0.16
# GPU=NVIDIA A100 80GB PCIe

# the more details about code will been shown in the folder of 'code'.
```

## Reference 
[1] Ji Y, Zhou Z, Liu H, et al. DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome[J]. Bioinformatics, 2021, 37(15): 2112-2120.<br>
[2] Bailey T L. STREME: accurate and versatile sequence motif discovery[J]. Bioinformatics, 2021, 37(18): 2834-2840.<br>
[3] Castro-Mondragon J A, Riudavets-Puig R, Rauluseviciute I, et al. JASPAR 2022: the 9th release of the open-access database of transcription factor binding profiles[J]. Nucleic acids research, 2022, 50(D1): D165-D173.<br>
[4] Gupta S, Stamatoyannopoulos J A, Bailey T L, et al. Quantifying similarity between motifs[J]. Genome biology, 2007, 8(2): 1-9.<br>
[5] Basith S, Hasan M M, Lee G, et al. Integrative machine learning framework for the identification of cell-specific enhancers from the human genome[J]. Briefings in Bioinformatics, 2021, 22(6): bbab252.
