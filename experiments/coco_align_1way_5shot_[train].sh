# COCO 1-way 5-shot
python train.py with gpu_id=2 mode='train' dataset='COCO' label_sets=0 model.align=True task.n_ways=1 task.n_shots=5
python train.py with gpu_id=2 mode='train' dataset='COCO' label_sets=1 model.align=True task.n_ways=1 task.n_shots=5
python train.py with gpu_id=2 mode='train' dataset='COCO' label_sets=2 model.align=True task.n_ways=1 task.n_shots=5
python train.py with gpu_id=2 mode='train' dataset='COCO' label_sets=3 model.align=True task.n_ways=1 task.n_shots=5