# Data
- Truncate instead of integrate over time segments + rough quantization = 
less events plus miss information, but at least clean mappings

# Runs
## Transformer / Arrangement
Small architecture, small context
--conditioning --num_layers=4 --batch_size=64 --sequence_size=3 --subdivision=16 --dataset_type=arrangement_midiPiano --action=train --num_batches=1024 --lr=1e-4
Final loss  ~2.4
Results moyens, mais pas aberrants

## Transformer / Arrangement_midiPiano
Resultats pourris. Arrive à chopper le thème sur quelques notes, puis "décroche", ne suit même pas la mélodie ni l'harmonie...
Repère quand même les sections où il y aplus de notes dans la partie piano et orchestre avec plus d'instruments.