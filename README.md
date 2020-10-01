# Uncertainty-based stepsize adaption and FGSM
Paper: Repurposing Pretrained Models for Robust Out-of-domain Few-Shot Learning

- Running environment
	= python 3.6
	= pytorch 1.4
	= nVidia v100 GPU
		
- Our code forked from https://github.com/dragen1860/MAML-Pytorch

- We use 4 datasets in the paper: miniImageNet, CUB-200-2011, VGG Flower, and Traffic Sign
	= miniImageNet
		: we use the split. https://github.com/twitter/meta-learning-lstm/tree/master/data/miniImagenet
	
	= CUB-200-2011, VGG Flower, and Traffic Sign
		* We did not any preprocess for the datasets.
		* We just change the folder name to use the datasets in the code.
		* We use all datasets for meta-tesing. Therefore we don't need to split the dataset.
		* You can download the dataset.
			+ miniImageNet: You can download miniImageNet on the web
			+ CUB-200-2011: https://drive.google.com/file/d/17WeYdiExqUWTpRF_3qfLnPa0bJKRXEal/view?usp=sharing
			+ VGG Flower: https://drive.google.com/file/d/1Osbl0oB_R7xpow852ot5rDIe0FQRW9f7/view?usp=sharing
			+ Traffic Sign: https://drive.google.com/file/d/17WiJWgYfdFdXrOZ5ACVbS2rarj8rrCsu/view?usp=sharing
			+ If you want to download some files from google drive in your linux machine, I strongly recommend the tool: https://github.com/tanaikech/goodls
	
- Before you run
	= Extract dataset in your machine
	= Change the data path in miniimagenet_train.py
		
- Pretrained models
	= save/model_5way_1shot: Our baseline model for 5-way 1-shot classification. The checkpoint chose based on miniImageNet validation accuracy.
	= save/model_5way_5shot: Our baseline model for 5-way 5-shot classification. The checkpoint chose based on miniImageNet validation accuracy.
	= save/model_5way_5shot_last_checkpoint: Last checkpoint of 5-way 5-shot model training. The checkpoint overfitted miniImageNet training set.
	= save/model_10way_1shot: Our baseline model for 10-way 1-shot classification. The checkpoint chose based on miniImageNet validation accuracy.
	
- How to run (Meta-training)
	= Actually our proposed method is working on meta-testing time.
	= If you want to reproduce the test performance, no need to train MAML. (We provided pretrained models)		
	= We set the training iteration 150,000, but you don't need to the iteration. You can change the iteration to 60,000 same with MAML.
	= command: python miniimagenet_train.py (default is 5-way 1-shot classification)
	
- How to run (Meta-tesing) *Our proposed method works meta-testing time
	= command (SGD) --optim=sgd
		* 5way-1shot flower dataset: python miniimagenet_train.py --mode=test --modelfile=save/model_5way_1shot.pth --update_lr=0.01 --optim=sgd --k_spt=1 --n_way=5 --domain=flower --ad_train_org --enaug
		* 5way-5shot flower dataset (Last checkpoint): python miniimagenet_train.py --mode=test --modelfile=save/model_5way_5shot_last_checkpoint.pth --update_lr=0.01 --optim=sgd --k_spt=5 --n_way=5 --domain=flower --ad_train_org --enaug
		* 5way-5shot flower dataset: python miniimagenet_train.py --mode=test --modelfile=save/model_5way_5shot.pth --update_lr=0.01 --optim=sgd --k_spt=5 --n_way=5 --domain=flower --ad_train_org --enaug
		* 10way-1shot flower dataset: python miniimagenet_train.py --mode=test --modelfile=save/model_10way_1shot.pth --update_lr=0.01 --optim=sgd --k_spt=1 --n_way=10 --domain=flower --ad_train_org --enaug
	= command (Adam) -optim=adam --adaptive
		* 5way-1shot flower dataset: python miniimagenet_train.py --mode=test --modelfile=save/model_5way_1shot.pth --update_lr=0.01 --optim=adam --k_spt=1 --n_way=5 --domain=flower --ad_train_org --enaug --adaptive
		* 5way-5shot flower dataset (Last checkpoint): python miniimagenet_train.py --mode=test --modelfile=save/model_5way_5shot_last_checkpoint.pth --update_lr=0.01 --optim=adam --k_spt=5 --n_way=5 --domain=flower --ad_train_org --enaug --adaptive
		* 5way-5shot flower dataset: python miniimagenet_train.py --mode=test --modelfile=save/model_5way_5shot.pth --update_lr=0.01 --optim=adam --k_spt=5 --n_way=5 --domain=flower --ad_train_org --enaug --adaptive
		* 10way-1shot flower dataset: python miniimagenet_train.py --mode=test --modelfile=save/model_10way_1shot.pth --update_lr=0.01 --optim=adam --k_spt=1 --n_way=10 --domain=flower --ad_train_org --enaug --adaptive
	= Run by script
		* If you use slurm, you can use the script to generate jog list.
		* python make_script_all.py
		* ./run_all.sh
