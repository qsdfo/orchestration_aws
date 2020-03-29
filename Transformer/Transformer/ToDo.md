# Runnings
## Bach
- Reducter, soft label, small lr

## Franck
- bach, input dropout (fixed)

--------------------------------------------------------------------------------------------
# ToDo
- Retry input dropout on everything now that I stopped doing shit

## Reductor
- input dropout
    - on embedding
    - on tokens
    Both a re already implemented, just need to try (testing on Bach now)
- longer sequences, finer quantization
    - for orchestration, perhaps use bagging embedding for piano (more compact)
- write datamanager using reduction model from both a purely orchestral and mixed dbs
- try with short context for orchestra but longer context for piano
       - sparse transformer?
       
## Transformer
- Transformer-VAE on Bach
    - Attendre training et générer
        - lowest or highest entropy first ?
- FiLM on orchestral Transformer


--------------------------------------------------------------------------------------------
# Reduction
## Hparams
- Number of layers
- Temporal order

## Conditioning
- One encoder output shared across layers
- Different encoder output for each layer

## Relative attention
- Block ?

## Dropout
- input dropout ?

--------------------------------------------------------------------------------------------
# Arrangement
- retry with mixup, better conditioning, larger model, double conditioning, block attentions

## General
- audio rendering ? (for multi-modal)
- toy examples for orchestration
- input all informations (piano, orchestra) in one huge transformer

--------------------------------------------------------------------------------------------
# Ideas
## Model
- Add general information about piano score (through kind of large convent over the whole score)
   - Padding issues
   - How to include in transformer (Feature-wise conditioning ?)
- Spectral conditioning

## Pretraining with GANs ?
- Pretrain on only orchestral part
- Use GAN for learning distrib of orchestral vectors
- Generator is a transformer(-nade) without encoder
- Feed O(t-1), O(t) to both Gen and Disc. Need also O(t-1) in Disc 
(if not just learn to generate vectors O(t), not to relate them with O(t)) 
= Conditional GAN with O(t-1) being the conditioning information