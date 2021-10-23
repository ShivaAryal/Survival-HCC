# Survival-HCC
Survival Prediction in Liver Cancer with deep learning based multi-omics integration

I have made an auto-encoder based model for multi-omics data analysis. It is based on a bayesian latent factor model, with inference done using artificial neural networks.

Autoencoders are hourglass-shaped neural networks that are trained to reconstruct the input data after passing it through a bottleneck layer. Thereby, autoencoders learn an efficient lower dimension representation of high-dimensional data, a “latent factor” representation of the data. The Multi-modal autoencoders take data from different modalities and learn latent factor representations of the data.

The latent factors that the Variational Autoencoders result are done clusturing with K-means clusturing and the featured from the autoencoder are reduced to optimal number of clusters. I used the scikit learn package for this.

I have further calculated the Harrell's c-index.


The project isn't completed as I am getting stuck in formatting the datasets found in the internet. Much of my time was spent in understanding the research paper and the new things implemented in this paper. But, I am quite sure, If I get some more time, I can get this basic workflow done.

