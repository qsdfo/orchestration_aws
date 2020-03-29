# Data

## Augmentations
12 pitches up and down as data augmentation improved qualitative results.
More specifically, pitches in piano score are more respected.

- Learning on last frame only, basic transfo. Quali

        ======= Epoch 29 =======
        ---Train---
        Loss: 1.61797
        ---Val---
        Loss: 23.0771

- Minimalist data loader (only p(t) and o(t-1:t)). Quali: lack of continuity... 
probably because can't relate past piano (not given to the model) to past orchestrat, 
and at the same time does not manage to learn voice leading 

        ======= Epoch 84 =======
        ---Train---
        Loss: 2.59689
        ---Val---
        Loss: 36.2784

## Representation
- midi seems to suck. perhaps would need orchestra to be midi too ?
- voice = best ?
- or old school pr ?

# Training
- Overfitting gives better results, even for unseen scores. Use temperature
- last frame learning seems to be best (fits better generation)
- input droppout is abolutely crucial to avoid overfitting

# Model

## Conditioning
## Stacking
Bon scores, mais générations vraiment mauvaise, même sans le conditionnement pas incroyables

### NADE
Seems like the best sampling strategy is:
- T = 1.2
- note order: lowest entropy first

#### Number of passes
When generating, perform several passes.
If not, crowded orchestration.

#### Loss
- Train on masked events only

- Train only on last frame after epoch 5

- Train on everything, even masked events

#### Scheduled masking

- Geometric masking


#### Results

- Large Transformer-NADE (128 per head dim), scheduled masking. 
Quali = good with
    - temperature = 1.2
    - sample low entropy notes first


    ======= Epoch 42 =======
    ---Train---
    Loss: 4.38996
    ---Val---
    Loss: 15.9466

- Transformer-NADE, ordering sampled first, loss on all token (even masked ones). 
Quali = shit. Some notes are super biased (always played), did not manage to change that by playing with sampling.
Looks like it does not look at the piano part.


    ======= Epoch 46 =======
    ---Train---
    Loss: 2.67731
    ---Val---
    Loss: 16.0994
        
- Transformer-NADE, trained only on o(t), no normalisation of the loss in Xent, order sampled from uniform distribution
Quali = pas mal mais pas ouf non plus.
    
        
    ======= Epoch 42 =======
    ---Train---
    Loss: 830.992
    ---Val---
    Loss: 3912.59