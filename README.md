# Survival-HCC
Survival Prediction in Liver Cancer with deep learning based multi-omics integration

I have made an auto-encoder based model for multi-omics data analysis. It is based on a bayesian latent factor model, with inference done using artificial neural networks.

Autoencoders are hourglass-shaped neural networks that are trained to reconstruct the input data after passing it through a bottleneck layer. Thereby, autoencoders learn an efficient lower dimension representation of high-dimensional data, a “latent factor” representation of the data. The Multi-modal autoencoders take data from different modalities and learn latent factor representations of the data.

The latent factors that the Variational Autoencoders result are done clusturing with K-means clusturing and the featured from the autoencoder are reduced to optimal number of clusters. I used the scikit learn package for this.

I have further calculated the Harrell's c-index.


I have trained the autoencoder with the miRNA, methylation and rna data of 360 samples. I am currently working in finding the latent factors which are most probable factor for patient survival.

## We can run the file with 
python main.py 


P.S. necessary python packages should be installed first with pip
