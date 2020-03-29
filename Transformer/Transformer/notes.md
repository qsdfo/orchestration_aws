Transformer non conditionné marchait bien (25 sur lsdb) avec 6 layers and 0.5 de dropout ?!


Transformer non conditionné, 2q: 12 layers, 0.1 dropout, no input dropuot
mega overfitting: 128it [00:29,  4.46it/s]
======= Epoch 312 =======
---Train---
Loss: 10.8966     
---Val---
Loss: 341.364     

Model Transformer_12_False_LsdbDataset(lsdb,32) saved
256it [03:17,  1.31it/s]
128it [00:30,  4.23it/s]
======= Epoch 313 =======
---Train---
Loss: 10.8635     
---Val---
Loss: 339.805     
Best val possible  97



6 couches 0.2 dropout
Model Transformer_6_False_LsdbDataset(lsdb,32) saved
256it [03:59,  1.07it/s]
128it [00:28,  4.54it/s]
======= Epoch 214 =======
---Train---
Loss: 18.0752     
---Val---
Loss: 234.973     

Model Transformer_6_False_LsdbDataset(lsdb,32) saved
256it [04:00,  1.05it/s]
128it [00:28,  4.59it/s]
======= Epoch 215 =======
---Train---
Loss: 18.0853     
---Val---
Loss: 236.412

======= Epoch 221 =======
---Train---
Loss: 17.7961     
---Val---
Loss: 238.111     
Best val 91 (entraînement commencé avec 0.1 dropout)     