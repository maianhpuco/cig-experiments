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
	python train_clam.py --config configs_simea/clam_camelyon16.yaml --max_epochs 200 --k_start 2 --k_end 5
	
train_clam_camelyon16_23fold:
	@echo "Activating conda environment for simea .."
	python train_clam.py --config configs_simea/clam_camelyon16.yaml --max_epochs 200 --k_start 2 --k_end 3
 
train_clam_camelyon16_45fold:
	@echo "Activating conda environment for simea .."
	python train_clam.py --config configs_simea/clam_camelyon16.yaml --max_epochs 200 --k_start 4 --k_end 5


train_clam_tcga_1fold:
	@echo "Activating conda environment for simea .."
	python train_tcga.py --config configs_simea/clam_tcga_renal.yaml --max_epochs 200 --k_start 1 --k_end 1

train_clam_tcga_23fold:
	@echo "Activating conda environment for simea .."
	python train_clam_tcga.py --config configs_simea/clam_tcga_renal.yaml --max_epochs 200 --k_start 2 --k_end 3

train_clam_tcga_56fold:
	@echo "Activating conda environment for simea .."
	python train_clam_tcga.py --config configs_simea/clam_tcga_renal.yaml --max_epochs 200 --k_start 4 --k_end 5



train_clam_cam_23fold:
	@echo "Activating conda environment for simea .."
	python train_clam.py --config configs_simea/clam_camelyon16.yaml --max_epochs 200 --k_start 2 --k_end 4
	
train_clam_cam_56fold:
	@echo "Activating conda environment for simea .."
	python train_clam.py --config configs_simea/clam_camelyon16.yaml --max_epochs 200 --k_start 4 --k_end 6
   
# test_ig_clam_camelyon16:
# 	python ig_clam.py --config configs_simea/clam_camelyon16.yaml  

test_ig_clam_tcga:
	python ig_clam_tcga_test.py --config configs_simea/clam_tcga_renal.yaml --ig_name integrated_gradient 


# Makefile for running different IG variants ===== on TCGA-Renal

# Makefile for running different IG variants ===== on camelyon16 

ig_clam_integrated_gradient:
	python ig_clam.py --config configs_simea/clam_camelyon16.yaml --ig_name integrated_gradient

ig_clam_expected_gradient:
	python ig_clam.py --config configs_simea/clam_camelyon16.yaml --ig_name expected_gradient

ig_clam_integrated_decision_gradient:
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	python ig_clam.py --config configs_simea/clam_camelyon16.yaml --ig_name integrated_decision_gradient --device cpu

ig_clam_contrastive_gradient:
	python ig_clam.py --config configs_simea/clam_camelyon16.yaml --ig_name contrastive_gradient

ig_clam_vanilla_gradient:
	python ig_clam.py --config configs_simea/clam_camelyon16.yaml --ig_name vanilla_gradient

ig_clam_square_integrated_gradient:
	python ig_clam.py --config configs_simea/clam_camelyon16.yaml --ig_name square_integrated_gradient

ig_clam_optim_square_integrated_gradient:
	python ig_clam.py --config configs_simea/clam_camelyon16.yaml --ig_name optim_square_integrated_gradient

dr_idg:
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	python ig_clam_test.py --config configs_simea/clam_camelyon16.yaml --ig_name integrated_decision_gradient --device cpu

# ----- Grouped Methods -----
group_basic: #done, done , done 
	make ig_clam_integrated_gradient
	make ig_clam_vanilla_gradient
	make ig_clam_expected_gradient

group_adv: #error_oomr, done  
	make ig_clam_integrated_decision_gradient
	make ig_clam_contrastive_gradient 

group_square: #done  runing  
	make ig_clam_square_integrated_gradient
	make ig_clam_optim_square_integrated_gradient



# # error make ig_clam_integrated_decision_gradient, ig_clam_contrastive_gradientig_clam_square_integrated_gradient Run all methods
# all_ig_methods: group_basic group_advanced group_square


# ==== TCGA-RENAL Methods ====
ig_clam_tcga_renal_integrated_gradient:
	python ig_clam.py --config configs_simea/clam_tcga_renal.yaml --ig_name integrated_gradient

ig_clam_tcga_renal_expected_gradient:
	python ig_clam.py --config configs_simea/clam_tcga_renal.yaml --ig_name expected_gradient

ig_clam_tcga_renal_integrated_decision_gradient:
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	python ig_clam.py --config configs_simea/clam_tcga_renal.yaml --ig_name integrated_decision_gradient --device cpu

ig_clam_tcga_renal_contrastive_gradient:
	python ig_clam.py --config configs_simea/clam_tcga_renal.yaml --ig_name contrastive_gradient

ig_clam_tcga_renal_vanilla_gradient:
	python ig_clam.py --config configs_simea/clam_tcga_renal.yaml --ig_name vanilla_gradient

ig_clam_tcga_renal_square_integrated_gradient:
	python ig_clam.py --config configs_simea/clam_tcga_renal.yaml --ig_name square_integrated_gradient

ig_clam_tcga_renal_optim_square_integrated_gradient:
	python ig_clam.py --config configs_simea/clam_tcga_renal.yaml --ig_name optim_square_integrated_gradient --device cpu

# Grouped commands for batch runs
# group_tcga_basic:
# 	make ig_clam_tcga_renal_integrated_gradient
# 	make ig_clam_tcga_renal_vanilla_gradient
# 	make ig_clam_tcga_renal_expected_gradient

# group_tcga_adv:
# 	make ig_clam_tcga_renal_integrated_decision_gradient
# 	make ig_clam_tcga_renal_contrastive_gradient

# group_tcga_square:
# 	make ig_clam_tcga_renal_square_integrated_gradient
# 	make ig_clam_tcga_renal_optim_square_integrated_gradient

group_tcga_basic:
	make ig_clam_tcga_renal_integrated_gradient
	make ig_clam_tcga_renal_vanilla_gradient
	make ig_clam_tcga_renal_expected_gradient
	make ig_clam_tcga_renal_integrated_decision_gradient

group_tcga_adv:
	make ig_clam_tcga_renal_contrastive_gradient
	make ig_clam_tcga_renal_square_integrated_gradient
	make ig_clam_tcga_renal_optim_square_integrated_gradient 
	

group_cam_basic:
	make ig_clam_integrated_gradient
	make ig_clam_vanilla_gradient
	make ig_clam_expected_gradient
	make ig_clam_integrated_decision_gradient
	
group_cam_adv:
	make ig_clam_contrastive_gradient # 
	make ig_clam_square_integrated_gradient # 
	make ig_clam_optim_square_integrated_gradient # 

#===========PLOT IG ============== 
# CAMELYON 16 
# === CAMELYON16 IG Plotting Commands ===

plot_camelyon16_integrated_gradient:
	python ig_clam_plot.py --config configs_simea/clam_camelyon16.yaml --ig_name integrated_gradient --start_fold 1 --end_fold 1  

plot_camelyon16_vanilla_gradient:
	python ig_clam_plot.py --config configs_simea/clam_camelyon16.yaml --ig_name vanilla_gradient --start_fold 1 --end_fold 1  

plot_camelyon16_expected_gradient:
	python ig_clam_plot.py --config configs_simea/clam_camelyon16.yaml --ig_name expected_gradient --start_fold 1 --end_fold 1  

plot_camelyon16_integrated_decision_gradient:
	python ig_clam_plot.py --config configs_simea/clam_camelyon16.yaml --ig_name integrated_decision_gradient --start_fold 1 --end_fold 1  

plot_camelyon16_contrastive_gradient:
	python ig_clam_plot.py --config configs_simea/clam_camelyon16.yaml --ig_name contrastive_gradient --start_fold 1 --end_fold 1  

plot_camelyon16_square_integrated_gradient:
	python ig_clam_plot.py --config configs_simea/clam_camelyon16.yaml --ig_name square_integrated_gradient --start_fold 1 --end_fold 1  

plot_camelyon16_optim_square_integrated_gradient:
	python ig_clam_plot.py --config configs_simea/clam_camelyon16.yaml --ig_name optim_square_integrated_gradient --start_fold 1 --end_fold 1  

# === Grouped Targets ===

group_plot_cam_basic:
	make plot_camelyon16_integrated_gradient
	make plot_camelyon16_vanilla_gradient
	make plot_camelyon16_expected_gradient
	make plot_camelyon16_integrated_decision_gradient

group_plot_cam_adv:
	make plot_camelyon16_contrastive_gradient
	make plot_camelyon16_square_integrated_gradient
	make plot_camelyon16_optim_square_integrated_gradient

# === TCGA-RENAL IG Plotting Commands ===

plot_tcga_renal_integrated_gradient:
	python ig_clam_plot.py --config configs_simea/clam_tcga_renal.yaml --ig_name integrated_gradient --start_fold 1 --end_fold 1  

plot_tcga_renal_vanilla_gradient:
	python ig_clam_plot.py --config configs_simea/clam_tcga_renal.yaml --ig_name vanilla_gradient --start_fold 1 --end_fold 1  

plot_tcga_renal_expected_gradient:
	python ig_clam_plot.py --config configs_simea/clam_tcga_renal.yaml --ig_name expected_gradient --start_fold 1 --end_fold 1  

plot_tcga_renal_integrated_decision_gradient:
	python ig_clam_plot.py --config configs_simea/clam_tcga_renal.yaml --ig_name integrated_decision_gradient --start_fold 1 --end_fold 1  

plot_tcga_renal_contrastive_gradient:
	python ig_clam_plot.py --config configs_simea/clam_tcga_renal.yaml --ig_name contrastive_gradient --start_fold 1 --end_fold 1  

plot_tcga_renal_square_integrated_gradient:
	python ig_clam_plot.py --config configs_simea/clam_tcga_renal.yaml --ig_name square_integrated_gradient --start_fold 1 --end_fold 1  

plot_tcga_renal_optim_square_integrated_gradient:
	python ig_clam_plot.py --config configs_simea/clam_tcga_renal.yaml --ig_name optim_square_integrated_gradient --start_fold 1 --end_fold 1  

# === Grouped Targets ===

group_plot_tcga_basic:
	make plot_tcga_renal_integrated_gradient
	make plot_tcga_renal_vanilla_gradient
	make plot_tcga_renal_expected_gradient
	make plot_tcga_renal_integrated_decision_gradient

group_plot_tcga_adv:
	make plot_tcga_renal_contrastive_gradient
	make plot_tcga_renal_square_integrated_gradient
	make plot_tcga_renal_optim_square_integrated_gradient




# python check_score.py --config  configs_simea/clam_camelyon16.yaml


#=========== MAKE FILE FOR SANITY CHECK ========
	 

dr_cig:
	python ig_clam_test.py -sig-config configs_simea/clam_camelyon16.yaml --ig_name contrastive_gradient


dr_sig:
	python ig_clam_test.py --config configs_simea/clam_camelyon16.yaml --ig_name square_integrated_gradient


dr_osig:
	python ig_clam_test.py --config configs_simea/clam_camelyon16.yaml --ig_name optim_square_integrated_gradient



plot_cam_integrated_gradient: 
	python ig_clam_plot.py \
	--config configs_simea/clam_camelyon16.yaml \
	--ig_name integrated_gradient \
	--start_fold 1 \
	--end_fold 1 

plot_cam_integrated_gradient: 
	python ig_clam_plot.py \
	--config configs_simea/clam_camelyon16.yaml \
	--ig_name integrated_gradient \
	--start_fold 1 \
	--end_fold 1 

plot_tcga_ig:
	python ig_clam_plot_tcga.py \
	--config configs_simea/clam_tcga_renal.yaml \
	--ig_name integrated_gradient \
	--start_fold 1 \
	--end_fold 1  


	
check_cig:
	python ig_clam_cig_check.py \
	--config configs_simea/clam_camelyon16.yaml \
	--ig_name optim_square_integrated_gradient




check_score: 
	python check_score.py --config  configs_simea/clam_camelyon16.yaml --ig_name contrastive_gradient 
