dryrun_train_clam_camelyon16:
	python train_clam.py --config configs_simea/clam_camelyon16.yaml --k_start 1 --k_end 1 --max_epochs 2


train_clam_camelyon16:
	python train_clam.py --config configs_simea/clam_camelyon16.yaml --max_epochs 100