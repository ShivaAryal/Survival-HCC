import os
import urllib.request
import pandas as pd

from autoEncoderModel import MainModel
import autoEncoderModel

# from .autoEncoderModel import MainModel

os.makedirs('data', exist_ok=True)

if not os.path.isfile(os.path.join('data', 'meth.csv')):
    urllib.request.urlretrieve(
        url='https://firebasestorage.googleapis.com/v0/b/reactfirebase-142f5.appspot.com/o/new-meth.csv?alt=media&token=27c623b6-b3d5-404a-becc-edb7c159caeb',
        filename=os.path.join('data', 'meth.csv')
    )

if not os.path.isfile(os.path.join('data', 'mir.csv')):
    urllib.request.urlretrieve(
        url='https://firebasestorage.googleapis.com/v0/b/reactfirebase-142f5.appspot.com/o/mir.csv?alt=media&token=09dac88a-d3b4-4fa7-8f14-51adea70e93c',
        filename=os.path.join('data', 'mir.csv')
    )

if not os.path.isfile(os.path.join('data', 'rna.csv')):
    urllib.request.urlretrieve(
        url='https://firebasestorage.googleapis.com/v0/b/reactfirebase-142f5.appspot.com/o/new-rna.csv?alt=media&token=2f487489-04cb-4687-bd0d-e6733086e3f9',
        filename=os.path.join('data', 'rna.csv')
    )

if not os.path.isfile(os.path.join('data', 'survival.csv')):
    urllib.request.urlretrieve(
        url='https://firebasestorage.googleapis.com/v0/b/reactfirebase-142f5.appspot.com/o/survival.csv?alt=media&token=b5519a48-f4b8-48b7-8385-7227d1632ac8',
        filename=os.path.join('data', 'survival.csv')
    )


meth = pd.read_csv(os.path.join('data', 'meth.csv'), index_col=0)
mir = pd.read_csv(os.path.join('data', 'mir.csv'), index_col=0)
rna = pd.read_csv(os.path.join('data', 'rna.csv'), index_col=0)
survival = pd.read_csv(os.path.join('data', 'survival.csv'), index_col=0)

mainModel = MainModel(n_hidden=[900], n_latent=70, epochs=100)
z = mainModel.fit_transform({'meth': meth, 'mir': mir, 'rna': rna})

print(mainModel.hist.plot())

# selecting latent factors which are most relevant for patient survivial
z_clinical = mainModel.select_clinical_factors(survival)
print(z_clinical.head())
