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

train_clam_tcga_1fold:
	@echo "Activating conda environment for simea .."
	conda activate clam_env && \
	python train_tcga.py --config configs_simea/clam_tcga.yaml --max_epochs 200 --k_start 1 --k_end 1

train_clam_tcga_4fold:
	@echo "Activating conda environment for simea .."
	conda activate clam_env && \
	python train_tcga.py --config configs_simea/clam_tcga.yaml --max_epochs 200 --k_start 2 --k_end 5


test_ig_clam:
	python ig_clam.py --config configs_simea/clam_camelyon16.yaml  




 # Makefile for running different IG variants

ig_clam_integrated_gradient:
	python ig_clam.py --config configs_simea/clam_camelyon16.yaml --ig_name integrated_gradient

ig_clam_expected_gradient:
	python ig_clam.py --config configs_simea/clam_camelyon16.yaml --ig_name expected_gradient

ig_clam_integrated_decision_gradient:
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	python ig_clam.py --config configs_simea/clam_camelyon16.yaml --ig_name integrated_decision_gradient 

ig_clam_contrastive_gradient:
	python ig_clam.py --config configs_simea/clam_camelyon16.yaml --ig_name contrastive_gradient

ig_clam_vanilla_gradient:
	python ig_clam.py --config configs_simea/clam_camelyon16.yaml --ig_name vanilla_gradient

ig_clam_square_integrated_gradient:
	python ig_clam.py --config configs_simea/clam_camelyon16.yaml --ig_name square_integrated_gradient

ig_clam_optim_square_integrated_gradient:
	python ig_clam.py --config configs_simea/clam_camelyon16.yaml --ig_name optim_square_integrated_gradient

dr_idg:
	python ig_clam_test.py --config configs_simea/clam_camelyon16.yaml --ig_name integrated_decision_gradient


dr_cig:
	python ig_clam_test.py -sig-config configs_simea/clam_camelyon16.yaml --ig_name contrastive_gradient


dr_sig:
	python ig_clam_test.py --config configs_simea/clam_camelyon16.yaml --ig_name square_integrated_gradient


dr_osig:
	python ig_clam_test.py --config configs_simea/clam_camelyon16.yaml --ig_name optim_square_integrated_gradient

# ----- Grouped Methods -----
group_basic: #done, done , done 
	make ig_clam_integrated_gradient
	make ig_clam_vanilla_gradient
	make ig_clam_expected_gradient

group_adv: #error, done  
	make ig_clam_integrated_decision_gradient
	make ig_clam_contrastive_gradient 

group_square: #running runing  
	make ig_clam_square_integrated_gradient
	make ig_clam_optim_square_integrated_gradient
# # error make ig_clam_integrated_decision_gradient, ig_clam_contrastive_gradientig_clam_square_integrated_gradient Run all methods
# all_ig_methods: group_basic group_advanced group_square
