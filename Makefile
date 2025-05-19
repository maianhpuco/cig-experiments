dryrun_train_clam_camelyon16:
	@echo "Activating conda environment for simea .."
	conda activate clam_env && \
	python train_clam.py --config configs_simea/clam_camelyon16.yaml --k_start 1 --k_end 1 --max_epochs 2

train_clam_camelyon16_1fold:
	@echo "Activating conda environment for simea .."
	conda activate clam_env && \
	python train_clam.py --config configs_simea/clam_camelyon16.yaml --max_epochs 200 --k_start 1 --k_end 1

train_clam_camelyon16_4fold:
	@echo "Activating conda environment for simea .."
	conda activate clam_env && \
	python train_clam.py --config configs_simea/clam_camelyon16.yaml --max_epochs 200 --k_start 2 --k_end 5

test_ig_clam:
	python ig_clam.py --config configs_simea/clam_camelyon16.yaml  




 # Makefile for running different IG variants

CONFIG_CAM=configs_simea/clam_camelyon16.yaml

ig_clam_integrated_gradient:
	python ig_clam.py --config $(CONFIG_CAM) --ig_name integrated_gradient

ig_clam_expected_gradient:
	python ig_clam.py --config $(CONFIG_CAM) --ig_name expected_gradient

ig_clam_integrated_decision_gradient:
	python ig_clam.py --config $(CONFIG_CAM) --ig_name integrated_decision_gradient

ig_clam_contrastive_gradient:
	python ig_clam.py --config $(CONFIG_CAM) --ig_name contrastive_gradient

ig_clam_vanilla_gradient:
	python ig_clam.py --config $(CONFIG_CAM) --ig_name vanilla_gradient

ig_clam_square_integrated_gradient:
	python ig_clam.py --config $(CONFIG_CAM) --ig_name square_integrated_gradient

ig_clam_optim_square_integrated_gradient:
	python ig_clam.py --config $(CONFIG_CAM) --ig_name optim_square_integrated_gradient
