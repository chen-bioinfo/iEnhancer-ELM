## Training process
*  DNA_bert_finetuning_average_L2.ipynb can reproduce the our training process;<br>
*  DNA_bert_finetuning_average_L2_K_Flod.ipynb show the process of 5-fold cross validation;<br>
*  DNA_Bert_finetuning_L2_ensemble.ipynb exhibit the best performance of our four classification model and present the integration result of these models.<br>

## Motif  analysis
The precess of moitf analysis mainly refer to [1].

*  motif_Attention_output.ipynb shows the process of calculating weight of each nucleotide in its original sequence by the attention mechanism in iEhancer-ELM. <br>
*  motif_analysis,ipynb shows the process of using attention weight of nucleotide to retrieve the potential patterns, and then filter by hypergeometric test. Finally, merge the patterns into the final motifs based on sequence similarity.
*  The Folder of atten saves the result of motif analysis.
*  The folder of analysis shows the t-SNE analysis redult ofsequence embedding.

## Reference
[1] Ji Y, Zhou Z, Liu H, et al. DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome[J]. Bioinformatics, 2021, 37(15): 2112-2120.
