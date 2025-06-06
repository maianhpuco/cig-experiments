
# === Checkpoint Paths ===
CKPT_CLAM_CAMELYON16 = /home/mvu9/processing_datasets/processing_camelyon16/clam_results/fold_1/s_1_checkpoint.pt
CKPT_CLAM_TCGA_RENAL = /home/mvu9/processing_datasets/processing_tcga_256/clam_tcga_renal_result/result_final_ep200/s_1_checkpoint.pt
CKPT_CLAM_TCGA_LUNG  = /home/mvu9/processing_datasets/processing_tcga_256/clam_tcga_lung_result/result_final_ep200/s_1_checkpoint.pt

CKPT_MLP_CAMELYON16 = /home/mvu9/processing_datasets/processing_camelyon16/mlp_results/fold_1/best_model.pth
CKPT_MLP_TCGA_RENAL = /home/mvu9/processing_datasets/processing_tcga_256/mlp_tcga_renal_results/fold_1/best_model.pth
CKPT_MLP_TCGA_LUNG = /home/mvu9/processing_datasets/processing_tcga_256/mlp_tcga_lung_results/fold_1/best_model.pth 

# ============= TRAIN CLAM =============
dryrun_train_clam_camelyon16:
	python train_clam.py --config configs_simea/clam_camelyon16.yaml --k_start 1 --k_end 1 --max_epochs 1
train_clam_camelyon16_1fold:
	python train_clam.py --config configs_simea/clam_camelyon16.yaml --max_epochs 200 --k_start 1 --k_end 1
train_clam_camelyon16_23fold:
	python train_clam.py --config configs_simea/clam_camelyon16.yaml --max_epochs 200 --k_start 2 --k_end 3
train_clam_camelyon16_45fold:
	python train_clam.py --config configs_simea/clam_camelyon16.yaml --max_epochs 200 --k_start 4 --k_end 5
train_clam_camelyon16_4fold:
	python train_clam.py --config configs_simea/clam_camelyon16.yaml --max_epochs 200 --k_start 2 --k_end 5
train_clam_camelyon16_5fold:
	python train_clam.py --config configs_simea/clam_camelyon16.yaml --max_epochs 200 --k_start 1 --k_end 5


train_clam_tcga_renal_1fold:
	python train_clam_tcga_renal.py --config configs_simea/clam_tcga_renal.yaml --max_epochs 200 --k_start 1 --k_end 1
train_clam_tcga_renal_5fold:
	python train_clam_tcga_renal.py --config configs_simea/clam_tcga_renal.yaml --max_epochs 200 --k_start 1 --k_end 5
train_clam_tcga_lung_1fold:
	python train_clam_tcga_lung.py --config configs_simea/clam_tcga_lung.yaml --max_epochs 200 --k_start 1 --k_end 1
train_clam_tcga_lung_5fold:
	python train_clam_tcga_lung.py --config configs_simea/clam_tcga_lung.yaml --max_epochs 200 --k_start 1 --k_end 5
	

# ============= DSMIL =============
train_dsmil_camelyon16_1fold:
	python train_dsmil.py --config configs_simea/dsmil_camelyon16.yaml --num_epochs 200 --k_start 1 --k_end 1
train_dsmil_camelyon16_4fold:
	python train_dsmil.py --config configs_simea/dsmil_camelyon16.yaml --num_epochs 200 --k_start 2 --k_end 5
# train_dsmil_camelyon16_5fold:
# 	python train_dsmil.py --config configs_simea/dsmil_camelyon16.yaml --num_epochs 200 --k_start 1 --k_end 5


train_dsmil_tcga_renal_1fold:
	python train_dsmil_tcga_renal.py --config configs_simea/dsmil_tcga_renal.yaml --num_epochs 200 --k_start 1 --k_end 1
train_dsmil_tcga_renal_5fold:
	python train_dsmil_tcga_renal.py --config configs_simea/dsmil_tcga_renal.yaml --num_epochs 200 --k_start 1 --k_end 5


train_dsmil_tcga_lung_1fold:
	@echo "Activating conda environment for simea .."
	conda activate clam_env && \
	python train_dsmil_tcga_lung.py --config configs_simea/dsmil_tcga_lung.yaml --num_epochs 200 --k_start 1 --k_end 1
train_dsmil_tcga_lung_5fold:
	@echo "Activating conda environment for simea .."
	conda activate clam_env && \
	python train_dsmil_tcga_lung.py --config configs_simea/dsmil_tcga_lung.yaml --num_epochs 200 --k_start 1 --k_end 5



# ============= DTFD =============
train_dtfd_camelyon16_1fold:
	conda activate dtfd && \
	python train_dtfd.py --config configs_simea/dtfd_camelyon16.yaml --num_epochs 200 --k_start 1 --k_end 1
train_dtfd_camelyon16_4fold:
	conda activate dtfd && \
	python train_dtfd.py --config configs_simea/dtfd_camelyon16.yaml --num_epochs 200 --k_start 2 --k_end 5
train_dtfd_camelyon16_5fold:
	@echo "Activating conda environment for simea .."
	conda activate dtfd && \
	python train_dtfd.py --config configs_simea/dtfd_camelyon16.yaml --num_epochs 200 --k_start 1 --k_end 5


train_dtfd_tcga_renal_1fold:
	@echo "Activating conda environment for simea .."
	conda activate dtfd && \
	python train_dtfd_tcga_renal.py --config configs_simea/dtfd_tcga_renal.yaml --EPOCH 200 --k_start 1 --k_end 1
train_dtfd_tcga_renal_5fold:
	@echo "Activating conda environment for simea .."
	conda activate dtfd && \
	python train_dtfd_tcga_renal.py --config configs_simea/dtfd_tcga_renal.yaml --EPOCH 200 --k_start 1 --k_end 5


train_dtfd_tcga_lung_1fold:
	@echo "Activating conda environment for simea .."
	conda activate dtfd && \
	python train_dtfd_tcga_lung.py --config configs_simea/dtfd_tcga_lung.yaml --EPOCH 200 --k_start 1 --k_end 1
train_dtfd_tcga_lung_5fold:
	@echo "Activating conda environment for simea .."
	conda activate dtfd && \
	python train_dtfd_tcga_lung.py --config configs_simea/dtfd_tcga_lung.yaml --EPOCH 200 --k_start 1 --k_end 5

# ============= TRAIN MLP =============
dryrun_train_mlp_camelyon16:
	python train_mlp.py --config configs_simea/mlp_camelyon16.yaml --k_start 1 --k_end 1 --max_epochs 3
# Full run on GPU 3 for camelyon16
train_mlp_camelyon16_1fold:
	CUDA_VISIBLE_DEVICES=7 python train_mlp.py --config configs_simea/mlp_camelyon16.yaml --k_start 1 --k_end 1 --max_epochs 200


# Full run on GPU 3 for TCGA-Renal
train_mlp_tcga_renal_1fold:
	CUDA_VISIBLE_DEVICES=4 python train_mlp.py --config configs_simea/mlp_tcga_renal.yaml --k_start 1 --k_end 1 --max_epochs 200

# Full run on GPU 3 for TCGA-Lung
train_mlp_tcga_lung_1fold:
	CUDA_VISIBLE_DEVICES=5 python train_mlp.py --config configs_simea/mlp_tcga_lung.yaml --k_start 1 --k_end 1 --max_epochs 200

#======================================= TEST ================================== 
# === MLP Test Targets ===
test_mlp_camelyon16_fold_1:
	CUDA_VISIBLE_DEVICES=7 python test_mlp.py \
		--config configs_simea/mlp_camelyon16.yaml --fold 1 \
		--ckpt_path $(CKPT_MLP_CAMELYON16)

test_mlp_tcga_renal_fold_1:
	CUDA_VISIBLE_DEVICES=7 python test_mlp.py \
		--config configs_simea/mlp_tcga_renal.yaml --fold 1 \
		--ckpt_path $(CKPT_MLP_TCGA_RENAL)

test_mlp_tcga_lung_fold_1:
	CUDA_VISIBLE_DEVICES=1 python test_mlp.py \
		--config configs_simea/mlp_tcga_lung.yaml --fold 1 \
		--ckpt_path $(CKPT_MLP_TCGA_LUNG)

test_mlp_fold_1: test_mlp_camelyon16_fold_1 test_mlp_tcga_renal_fold_1 test_mlp_tcga_lung_fold_1
 

# ============= TEST CLAM ============= 
test_clam_camelyon16_fold_1:
	CUDA_VISIBLE_DEVICES=1 python test_clam.py \
	--config configs_simea/clam_camelyon16.yaml --fold 1 \
	--ckpt_path $(CKPT_CLAM_CAMELYON16)

test_clam_tcga_renal_fold_1:
	CUDA_VISIBLE_DEVICES=1 python test_clam.py \
	--config configs_simea/clam_tcga_renal.yaml --fold 1 \
	--ckpt_path $(CKPT_CLAM_TCGA_RENAL)

test_clam_tcga_lung_fold_1:
	CUDA_VISIBLE_DEVICES=1 python test_clam.py \
	--config configs_simea/clam_tcga_lung.yaml --fold 1 \
	--ckpt_path $(CKPT_CLAM_TCGA_LUNG)

test_clam_fold_1: test_clam_camelyon16_fold_1 test_clam_tcga_renal_fold_1 test_clam_tcga_lung_fold_1
 
# ============= TEST DSMIL ============= 
test_dsmil_camelyon16_fold_1:
	python test_dsmil.py \
	--config configs_simea/dsmil_camelyon16.yaml --fold 1 \
	--ckpt_path /home/mvu9/processing_datasets/processing_camelyon16/dsmil_results/fold_1/fold_1_1.pth 
# --ckpt_path /home/mvu9/processing_datasets/processing_camelyon16/dsmil_results/20250528/fold_1_1.pth
test_dsmil_tcga_renal_fold_1:
	python test_dsmil.py \
	--config configs_simea/dsmil_tcga_renal.yaml --fold 1 \
	--ckpt_path /home/mvu9/processing_datasets/processing_tcga_256/dsmil_results/20250528/fold_1_1.pth
test_dsmil_tcga_lung_fold_1:
	python test_dsmil.py \
	--config configs_simea/dsmil_tcga_lung.yaml --fold 1 \
	--ckpt_path /home/mvu9/processing_datasets/processing_tcga_256/dsmil_tcga_lung_results/20250528/fold_1_1.pth 
# --ckpt_path /home/mvu9/processing_datasets/processing_tcga_256/dsmil_tcga_lung_results/20250528/fold_1_1.pth 	
test_dsmil_fold_1: test_dsmil_camelyon16_fold_1 test_dsmil_tcga_renal_fold_1 test_dsmil_tcga_lung_fold_1

# ============= TEST DTFD =============
test_dtfd_camelyon16_fold_1:
	python test_dtfd.py \
	--config configs_simea/dtfd_camelyon16.yaml --fold 1 \
	--ckpt_path /home/mvu9/processing_datasets/processing_camelyon16/dtfd_results/best_model.pth
test_dtfd_tcga_renal_fold_1:
	python test_dtfd.py \
	--config configs_simea/dtfd_tcga_renal.yaml --fold 1 \
	--ckpt_path /home/mvu9/processing_datasets/processing_tcga_256/dtfd_tcga_renal_results/best_model.pth
test_dtfd_tcga_lung_fold_1:
	python test_dtfd.py  \
	--config configs_simea/dtfd_tcga_lung.yaml --fold 1 \
	--ckpt_path /home/mvu9/processing_datasets/processing_tcga_256/dtfd_tcga_lung_results_1fold/best_model.pth
test_dtfd_fold_1: test_dtfd_camelyon16_fold_1  test_dtfd_tcga_renal_fold_1 test_dtfd_tcga_lung_fold_1

# ==================================PREDICTION ======================================
# ============= PREDICT CLAM =============
# baseline_clam_camelyon16_fp: 
# 	python sampling_baseline_fp.py --config configs_simea/clam_camelyon16.yaml --start_fold 1 --end_fold 1 
baseline_clam_camelyon16_trainvaltest:
	python sampling_baseline_trainvaltest.py --config configs_simea/clam_camelyon16.yaml --start_fold 1 --end_fold 1 

baseline_clam_camelyon16:
	python sampling_baseline.py --config configs_simea/clam_camelyon16.yaml --start_fold 1 --end_fold 1
baseline_clam_tcga_renal:
	python sampling_baseline.py --config configs_simea/clam_tcga_renal.yaml --start_fold 1 --end_fold 1
baseline_clam_tcga_lung:
	python sampling_baseline.py --config configs_simea/clam_tcga_lung.yaml --start_fold 1 --end_fold 1 

all_baseline_clam_tcga: baseline_clam_tcga_renal baseline_clam_tcga_lung 

# ============= PREDICT DSMIL =============
baseline_dsmil_camelyon16:
	python sampling_baseline.py --config configs_simea/dsmil_camelyon16.yaml --start_fold 1 --end_fold 1
baseline_dsmil_tcga_renal:
	python sampling_baseline.py --config configs_simea/dsmil_tcga_renal.yaml --start_fold 1 --end_fold 1
baseline_dsmil_tcga_lung:
	python sampling_baseline.py --config configs_simea/dsmil_tcga_lung.yaml --start_fold 1 --end_fold 1 
all_baseline_dsmil: baseline_dsmil_camelyon16 baseline_dsmil_tcga_renal baseline_dsmil_tcga_lung 
# ============= PREDICT DTFD=============
baseline_dtfd_camelyon16:
	python sampling_baseline.py --config configs_simea/dtfd_camelyon16.yaml --start_fold 1 --end_fold 1
baseline_dtfd_tcga_renal:
	python sampling_baseline.py --config configs_simea/dtfd_tcga_renal.yaml --start_fold 1 --end_fold 1
baseline_dtfd_tcga_lung:
	python sampling_baseline.py --config configs_simea/dtfd_tcga_lung.yaml --start_fold 1 --end_fold 1 
all_baseline_dtfd: baseline_dtfd_camelyon16 baseline_dsmil_tcga_renal baseline_dsmil_tcga_lung 

#=================BASELINE MLP============ 
baseline_mlp_camelyon16:
	python sampling_baseline.py --config configs_simea/mlp_camelyon16.yaml --start_fold 1 --end_fold 1
baseline_mlp_tcga_renal:
	python sampling_baseline.py --config configs_simea/mlp_tcga_renal.yaml --start_fold 1 --end_fold 1
baseline_mlp_tcga_lung:
	python sampling_baseline.py --config configs_simea/mlp_tcga_lung.yaml --start_fold 1 --end_fold 1 






# ========= predict ========= 
predict_clam_camelyon16:
	python predict_clam.py --config configs_simea/clam_camelyon16.yaml --fold_start 1 --fold_end 1

# ========= metric ========= 
metric_clam_camelyon16:
	python metric_clam.py --config configs_simea/clam_camelyon16.yaml --fold_start 1 --fold_end 1 

# test_ig_clam_camelyon16:
# 	python ig_clam.py --config configs_simea/clam_camelyon16.yaml  

test_ig_clam_tcga:
	python ig_clam_tcga_test.py --config configs_simea/clam_tcga_renal.yaml --ig_name integrated_gradient 

pr_metric:
	python print_metrics.py --config configs_simea/clam_camelyon16.yaml --fold_start 1 --fold_end 1
 
#========== IG CLAM CAMELYON16 Methods ============== 
clam_ig_camelyon16:
	python ig_clam.py --config configs_simea/clam_camelyon16.yaml --ig_name ig --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_CLAM_CAMELYON16) 
clam_eg_camelyon16:
	python ig_clam.py --config configs_simea/clam_camelyon16.yaml --ig_name eg --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_CLAM_CAMELYON16)  
clam_idg_camelyon16: 
	python ig_clam.py --config configs_simea/clam_camelyon16.yaml --ig_name idg --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_CLAM_CAMELYON16)  
clam_cig_camelyon16:
	CUDA_VISIBLE_DEVICES=0 python ig_clam.py --config configs_simea/clam_camelyon16.yaml --ig_name cig --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_CLAM_CAMELYON16)  
clam_g_camelyon16:
	python ig_clam.py --config configs_simea/clam_camelyon16.yaml --ig_name g --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_CLAM_CAMELYON16) 

group_basic_camelyon16: clam_ig_camelyon16 clam_g_camelyon16 clam_eg_camelyon16 clam_cig_camelyon16
group_adv_camelyon16: clam_cig_camelyon16 clam_idg_camelyon16

#========== IG CLAM TCGA-RENAL Methods ============== 
clam_ig_tcga_renal:
	CUDA_VISIBLE_DEVICES=3 python ig_clam.py --config configs_simea/clam_tcga_renal.yaml --ig_name ig --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_CLAM_TCGA_RENAL)
clam_eg_tcga_renal:
	CUDA_VISIBLE_DEVICES=3 python ig_clam.py --config configs_simea/clam_tcga_renal.yaml --ig_name eg --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_CLAM_TCGA_RENAL)
clam_idg_tcga_renal:
	CUDA_VISIBLE_DEVICES=3 python ig_clam.py --config configs_simea/clam_tcga_renal.yaml --ig_name idg --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_CLAM_TCGA_RENAL)
clam_cig_tcga_renal:
	CUDA_VISIBLE_DEVICES=1 python ig_clam.py --config configs_simea/clam_tcga_renal.yaml --ig_name cig --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_CLAM_TCGA_RENAL)
clam_g_tcga_renal:
	CUDA_VISIBLE_DEVICES=3 python ig_clam.py --config configs_simea/clam_tcga_renal.yaml --ig_name g --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_CLAM_TCGA_RENAL)

group_basic_tcga_renal: clam_ig_tcga_renal clam_g_tcga_renal clam_eg_tcga_renal
group_adv_tcga_renal: clam_cig_tcga_renal clam_idg_tcga_renal

# ========== IG CLAM TCGA-RENAL Methods ==============  
clam_ig_tcga_lung:
	CUDA_VISIBLE_DEVICES=2 python ig_clam.py --config configs_simea/clam_tcga_lung.yaml --ig_name ig --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_CLAM_TCGA_LUNG)
clam_eg_tcga_lung:
	CUDA_VISIBLE_DEVICES=2  python ig_clam.py --config configs_simea/clam_tcga_lung.yaml --ig_name eg --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_CLAM_TCGA_LUNG)
clam_idg_tcga_lung:
	CUDA_VISIBLE_DEVICES=2  python ig_clam.py --config configs_simea/clam_tcga_lung.yaml --ig_name idg --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_CLAM_TCGA_LUNG)
clam_cig_tcga_lung:
	CUDA_VISIBLE_DEVICES=2 python ig_clam.py --config configs_simea/clam_tcga_lung.yaml --ig_name cig --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_CLAM_TCGA_LUNG)
clam_g_tcga_lung:
	CUDA_VISIBLE_DEVICES=2  python ig_clam.py --config configs_simea/clam_tcga_lung.yaml --ig_name g --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_CLAM_TCGA_LUNG)
group_basic_tcga_lung: clam_ig_tcga_lung clam_g_tcga_lung clam_eg_tcga_lung
group_adv_tcga_lung: clam_cig_tcga_lung clam_idg_tcga_lung

# ----- Grouped Methods -----
group_basic: clam_ig, clam_g, clam_eg 
group_adv: clam_cig, clam_idg 

clam_cig: clam_cig_tcga_lung clam_cig_tcga_renal clam_cig_camelyon16

group_square: #done  runing  
	make ig_clam_square_integrated_gradient
	make ig_clam_optim_square_integrated_gradient



# # error make ig_clam_integrated_decision_gradient, ig_clam_contrastive_gradientig_clam_square_integrated_gradient Run all methods
# all_ig_methods: group_basic group_advanced group_square


# ==== TCGA-RENAL Methods ====
# ig_clam_tcga_renal_integrated_gradient:
# 	python ig_clam.py --config configs_simea/clam_tcga_renal.yaml --ig_name integrated_gradient

# ig_clam_tcga_renal_expected_gradient:
# 	python ig_clam.py --config configs_simea/clam_tcga_renal.yaml --ig_name expected_gradient

# ig_clam_tcga_renal_integrated_decision_gradient:
# 	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
# 	python ig_clam.py --config configs_simea/clam_tcga_renal.yaml --ig_name integrated_decision_gradient --device cpu

# ig_clam_tcga_renal_contrastive_gradient:
# 	python ig_clam.py --config configs_simea/clam_tcga_renal.yaml --ig_name contrastive_gradient

# ig_clam_tcga_renal_vanilla_gradient:
# 	python ig_clam.py --config configs_simea/clam_tcga_renal.yaml --ig_name vanilla_gradient

# ig_clam_tcga_renal_square_integrated_gradient:
# 	python ig_clam.py --config configs_simea/clam_tcga_renal.yaml --ig_name square_integrated_gradient

# ig_clam_tcga_renal_optim_square_integrated_gradient:
# 	python ig_clam.py --config configs_simea/clam_tcga_renal.yaml --ig_name optim_square_integrated_gradient --device cpu

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
	python scripts/plotting/ig_clam_plot_cam.py --config configs_simea/clam_camelyon16.yaml --ig_name integrated_gradient --start_fold 1 --end_fold 1  
plot_camelyon16_vanilla_gradient:
	python scripts/plotting/ig_clam_plot_cam.py --config configs_simea/clam_camelyon16.yaml --ig_name vanilla_gradient --start_fold 1 --end_fold 1  
plot_camelyon16_expected_gradient:
	python scripts/plotting/ig_clam_plot_cam.py --config configs_simea/clam_camelyon16.yaml --ig_name expected_gradient --start_fold 1 --end_fold 1  
plot_camelyon16_integrated_decision_gradient:
	python scripts/plotting/ig_clam_plot_cam.py --config configs_simea/clam_camelyon16.yaml --ig_name integrated_decision_gradient --start_fold 1 --end_fold 1  
plot_camelyon16_contrastive_gradient:
	python scripts/plotting/ig_clam_plot_cam.py --config configs_simea/clam_camelyon16.yaml --ig_name contrastive_gradient --start_fold 1 --end_fold 1  
plot_camelyon16_square_integrated_gradient:
	python scripts/plotting/ig_clam_plot_cam.py --config configs_simea/clam_camelyon16.yaml --ig_name square_integrated_gradient --start_fold 1 --end_fold 1  
plot_camelyon16_optim_square_integrated_gradient:
	python scripts/plotting/ig_clam_plot_cam.py --config configs_simea/clam_camelyon16.yaml --ig_name optim_square_integrated_gradient --start_fold 1 --end_fold 1  

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
	python ig_clam_plot_tcga.py --config configs_simea/clam_tcga_renal.yaml --ig_name integrated_gradient --start_fold 1 --end_fold 1  
plot_tcga_renal_vanilla_gradient:
	python ig_clam_plot_tcga.py --config configs_simea/clam_tcga_renal.yaml --ig_name vanilla_gradient --start_fold 1 --end_fold 1  
plot_tcga_renal_expected_gradient:
	python ig_clam_plot_tcga.py --config configs_simea/clam_tcga_renal.yaml --ig_name expected_gradient --start_fold 1 --end_fold 1  
plot_tcga_renal_integrated_decision_gradient:
	python ig_clam_plot_tcga.py --config configs_simea/clam_tcga_renal.yaml --ig_name integrated_decision_gradient --start_fold 1 --end_fold 1  
plot_tcga_renal_contrastive_gradient:
	python ig_clam_plot_tcga.py --config configs_simea/clam_tcga_renal.yaml --ig_name contrastive_gradient --start_fold 1 --end_fold 1  
plot_tcga_renal_square_integrated_gradient:
	python ig_clam_plot_tcga.py --config configs_simea/clam_tcga_renal.yaml --ig_name square_integrated_gradient --start_fold 1 --end_fold 1  
plot_tcga_renal_optim_square_integrated_gradient:
	python ig_clam_plot_tcga.py --config configs_simea/clam_tcga_renal.yaml --ig_name optim_square_integrated_gradient --start_fold 1 --end_fold 1  

# === Grouped Targets ===

group_plot_tcga_basic: #running in tmux == plot_tcga 
	make plot_tcga_renal_integrated_gradient
	make plot_tcga_renal_vanilla_gradient
	make plot_tcga_renal_expected_gradient
	make plot_tcga_renal_integrated_decision_gradient
	make plot_tcga_renal_contrastive_gradient


group_plot_tcga_adv:
# make plot_tcga_renal_contrastive_gradient
# make plot_tcga_renal_square_integrated_gradient
# make plot_tcga_renal_optim_square_integrated_gradient

# python check_score.py --config  configs_simea/clam_camelyon16.yaml


#=========== MAKE FILE FOR SANITY CHECK ========
	 

dr_cig:
	python ig_clam_test.py -sig-config configs_simea/clam_camelyon16.yaml --ig_name contrastive_gradient

dr_sig:
	python ig_clam_test.py --config configs_simea/clam_camelyon16.yaml --ig_name square_integrated_gradient

dr_osig:
	python ig_clam_test.py --config configs_simea/clam_camelyon16.yaml --ig_name optim_square_integrated_gradient

dr_idg:
	PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
	python ig_clam_test.py --config configs_simea/clam_camelyon16.yaml --ig_name integrated_decision_gradient --device cpu
 
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

check_plot_tcga:
	python ig_clam_plot_tcga.py \
	--config configs_simea/clam_tcga_renal.yaml \
	--ig_name integrated_gradient \
	--start_fold 1 \
	--end_fold 1  
	
check_cig:
	python ig_clam_cig_check.py \
	--config configs_simea/clam_camelyon16.yaml \
	--ig_name optim_square_integrated_gradient


check_acc: 
	python scripts/metrics/compute_acc.py --config configs_simea/clam_camelyon16.yaml  --start_fold 1 --end_fold 1
	

check_score: 
	python check_score.py --config  configs_simea/clam_camelyon16.yaml --ig_name contrastive_gradient 

check_pic:
	python check_pic.py --config configs_simea/clam_camelyon16.yaml 

pic_clam:
	python metric_pic_clam.py --config configs_simea/clam_camelyon16.yaml 
pic_clam_topk:
	python metric_pic_clam_topk.py --config configs_simea/clam_camelyon16.yaml 

rise_clam:
	CUDA_VISIBLE_DEVICES=2 python metric_rise_clam.py --config configs_simea/clam_camelyon16.yaml 
   
mean_std_camelyon16:
	python compute_mean_std_folds.py \
	--config configs_simea/clam_camelyon16.yaml \
	--start_fold 1 \
	--end_fold 5

# ==============MAUI================
# SANITY CHECK FOR MAUI 
maui_check_clam_ig:
	python check_ig_metrics_maui.py --config configs_maui/clam_camelyon16.yaml 

# pic_clam_id:
# 	CUDA_VISIBLE_DEVICES=2  python metric_pic_clam.py --config configs_simea/clam_camelyon16.yaml  --ig_name ig 


pic_clam_ig_camelyon16:
	CUDA_VISIBLE_DEVICES=2 python metric_pic_clam.py --config configs_simea/clam_camelyon16.yaml --ig_name ig

pic_clam_eg_camelyon16:
	CUDA_VISIBLE_DEVICES=2 python metric_pic_clam.py --config configs_simea/clam_camelyon16.yaml --ig_name eg

pic_clam_idg_camelyon16:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_clam.py --config configs_simea/clam_camelyon16.yaml --ig_name idg

pic_clam_cig_camelyon16:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_clam.py --config configs_simea/clam_camelyon16.yaml --ig_name cig

pic_clam_g_camelyon16:
	CUDA_VISIBLE_DEVICES=2 python metric_pic_clam.py --config configs_simea/clam_camelyon16.yaml --ig_name g

group_basic_pic_camelyon16: pic_clam_ig_camelyon16 pic_clam_g_camelyon16 pic_clam_eg_camelyon16 pic_clam_cig_camelyon16
group_adv_pic_camelyon16: pic_clam_cig_camelyon16 pic_clam_idg_camelyon16


# group_adv_rise_camelyon16: rise_clam_cig_camelyon16 rise_clam_idg_camelyon16

#====== Metric TOP-K PIC CAMELYON16 =============== 
# === PIC Top-K Targets for Camelyon16 ===
pictopk_clam_ig_camelyon16:
	CUDA_VISIBLE_DEVICES=4 python metric_pic_clam_topk.py \
		--config configs_simea/clam_camelyon16.yaml \
		--ckpt_path $(CKPT_CLAM_CAMELYON16) \
		--ig_name ig

pictopk_clam_eg_camelyon16:
	CUDA_VISIBLE_DEVICES=4 python metric_pic_clam_topk.py \
		--config configs_simea/clam_camelyon16.yaml \
		--ckpt_path $(CKPT_CLAM_CAMELYON16) \
		--ig_name eg

pictopk_clam_idg_camelyon16:
	CUDA_VISIBLE_DEVICES=5 python metric_pic_clam_topk.py \
		--config configs_simea/clam_camelyon16.yaml \
		--ckpt_path $(CKPT_CLAM_CAMELYON16) \
		--ig_name idg

pictopk_clam_cig_camelyon16:
	CUDA_VISIBLE_DEVICES=5 python metric_pic_clam_topk.py \
		--config configs_simea/clam_camelyon16.yaml \
		--ckpt_path $(CKPT_CLAM_CAMELYON16) \
		--ig_name cig

pictopk_clam_g_camelyon16:
	CUDA_VISIBLE_DEVICES=4 python metric_pic_clam_topk.py \
		--config configs_simea/clam_camelyon16.yaml \
		--ckpt_path $(CKPT_CLAM_CAMELYON16) \
		--ig_name g

pictopk_clam_random_camelyon16:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_clam_topk.py \
		--config configs_simea/clam_camelyon16.yaml \
		--ckpt_path $(CKPT_CLAM_CAMELYON16) \
		--ig_name random

group_basic_pictopk_camelyon16: pictopk_clam_ig_camelyon16 pictopk_clam_g_camelyon16 pictopk_clam_eg_camelyon16 pictopk_clam_cig_camelyon16
group_adv_pictopk_camelyon16: pictopk_clam_cig_camelyon16 pictopk_clam_idg_camelyon16


# === PIC Top-K Commands for TCGA Renal ===
pictopk_clam_ig_tcga_renal:
	CUDA_VISIBLE_DEVICES=7 python metric_pic_clam_topk.py \
		--config configs_simea/clam_tcga_renal.yaml \
		--ckpt_path=$(CKPT_CLAM_TCGA_RENAL) \
		--ig_name ig

pictopk_clam_eg_tcga_renal:
	CUDA_VISIBLE_DEVICES=7 python metric_pic_clam_topk.py \
		--config configs_simea/clam_tcga_renal.yaml \
		--ckpt_path=$(CKPT_CLAM_TCGA_RENAL) \
		--ig_name eg

pictopk_clam_idg_tcga_renal:
	CUDA_VISIBLE_DEVICES=7 python metric_pic_clam_topk.py \
		--config configs_simea/clam_tcga_renal.yaml \
		--ckpt_path=$(CKPT_CLAM_TCGA_RENAL) \
		--ig_name idg

pictopk_clam_cig_tcga_renal:
	CUDA_VISIBLE_DEVICES=6 python metric_pic_clam_topk.py \
		--config configs_simea/clam_tcga_renal.yaml \
		--ckpt_path=$(CKPT_CLAM_TCGA_RENAL) \
		--ig_name cig

pictopk_clam_g_tcga_renal:
	CUDA_VISIBLE_DEVICES=7 python metric_pic_clam_topk.py \
		--config configs_simea/clam_tcga_renal.yaml \
		--ckpt_path=$(CKPT_CLAM_TCGA_RENAL) \
		--ig_name g
pictopk_clam_random_tcga_renal:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_clam_topk.py \
		--config configs_simea/clam_tcga_renal.yaml \
		--ckpt_path=$(CKPT_CLAM_TCGA_RENAL) \
		--ig_name random

# === Group Rules ===
group_basic_pictopk_tcga_renal: pictopk_clam_ig_tcga_renal pictopk_clam_g_tcga_renal pictopk_clam_eg_tcga_renal pictopk_clam_cig_tcga_renal pictopk_clam_idg_tcga_renal
pictopk_clam_random_tcga: pictopk_clam_random_camelyon16 pictopk_clam_random_tcga_renal  

# group_adv_pictopk_tcga_renal: pictopk_clam_cig_tcga_renal pictopk_clam_idg_tcga_renal


# === PIC Top-K Commands for TCGA Lung ===
pictopk_clam_ig_tcga_lung:
	CUDA_VISIBLE_DEVICES=7 python metric_pic_clam_topk.py \
		--config configs_simea/clam_tcga_lung.yaml \
		--ckpt_path=$(CKPT_CLAM_TCGA_LUNG) \
		--ig_name ig

pictopk_clam_eg_tcga_lung:
	CUDA_VISIBLE_DEVICES=7 python metric_pic_clam_topk.py \
		--config configs_simea/clam_tcga_lung.yaml \
		--ckpt_path=$(CKPT_CLAM_TCGA_LUNG) \
		--ig_name eg

pictopk_clam_idg_tcga_lung:
	CUDA_VISIBLE_DEVICES=7 python metric_pic_clam_topk.py \
		--config configs_simea/clam_tcga_lung.yaml \
		--ckpt_path=$(CKPT_CLAM_TCGA_LUNG) \
		--ig_name idg

pictopk_clam_cig_tcga_lung:
	CUDA_VISIBLE_DEVICES=7 python metric_pic_clam_topk.py \
		--config configs_simea/clam_tcga_lung.yaml \
		--ckpt_path=$(CKPT_CLAM_TCGA_LUNG) \
		--ig_name cig

pictopk_clam_g_tcga_lung:
	CUDA_VISIBLE_DEVICES=7 python metric_pic_clam_topk.py \
		--config configs_simea/clam_tcga_lung.yaml \
		--ckpt_path=$(CKPT_CLAM_TCGA_LUNG) \
		--ig_name g

pictopk_clam_random_tcga_lung:
	CUDA_VISIBLE_DEVICES=1 python metric_pic_clam_topk_fast.py \
		--config configs_simea/clam_tcga_lung.yaml \
		--ckpt_path=$(CKPT_CLAM_TCGA_LUNG) \
		--ig_name random

# === Group Rules ===
group_basic_pictopk_tcga_lung: pictopk_clam_ig_tcga_lung pictopk_clam_g_tcga_lung pictopk_clam_eg_tcga_lung pictopk_clam_cig_tcga_lung pictopk_clam_idg_tcga_lung pictopk_clam_random_tcga_lung 
group_missing_pictopk_tcga_lung:  pictopk_clam_eg_tcga_lung pictopk_clam_idg_tcga_lung 
# ====================================================== Metric TOP-K PIC MLP ===============
# === PIC Top-K Targets for Camelyon16 (MLP) ===
pictopk_mlp_ig_camelyon16:
	CUDA_VISIBLE_DEVICES=1 python metric_pic_mlp_topk.py \
		--config configs_simea/mlp_camelyon16.yaml \
		--ckpt_path $(CKPT_MLP_CAMELYON16) \
		--ig_name ig

pictopk_mlp_eg_camelyon16:
	CUDA_VISIBLE_DEVICES=1 python metric_pic_mlp_topk.py \
		--config configs_simea/mlp_camelyon16.yaml \
		--ckpt_path $(CKPT_MLP_CAMELYON16) \
		--ig_name eg

pictopk_mlp_idg_camelyon16:
	CUDA_VISIBLE_DEVICES=0 python metric_pic_mlp_topk.py \
		--config configs_simea/mlp_camelyon16.yaml \
		--ckpt_path $(CKPT_MLP_CAMELYON16) \
		--ig_name idg

pictopk_mlp_cig_camelyon16:
	CUDA_VISIBLE_DEVICES=0 python metric_pic_mlp_topk.py \
		--config configs_simea/mlp_camelyon16.yaml \
		--ckpt_path $(CKPT_MLP_CAMELYON16) \
		--ig_name cig

pictopk_mlp_g_camelyon16:
	CUDA_VISIBLE_DEVICES=1 python metric_pic_mlp_topk.py \
		--config configs_simea/mlp_camelyon16.yaml \
		--ckpt_path $(CKPT_MLP_CAMELYON16) \
		--ig_name g

pictopk_mlp_random_camelyon16:
	CUDA_VISIBLE_DEVICES=2 python metric_pic_mlp_topk.py \
		--config configs_simea/mlp_camelyon16.yaml \
		--ckpt_path $(CKPT_MLP_CAMELYON16) \
		--ig_name random

group_basic_pictopk_mlp_camelyon16: pictopk_mlp_ig_camelyon16 pictopk_mlp_g_camelyon16 pictopk_mlp_eg_camelyon16 pictopk_mlp_cig_camelyon16
group_adv_pictopk_mlp_camelyon16: pictopk_mlp_cig_camelyon16 pictopk_mlp_idg_camelyon16


# === PIC Top-K Targets for TCGA Renal (MLP) ===
pictopk_mlp_ig_tcga_renal:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_mlp_topk.py \
		--config configs_simea/mlp_tcga_renal.yaml \
		--ckpt_path=$(CKPT_MLP_TCGA_RENAL) \
		--ig_name ig

pictopk_mlp_eg_tcga_renal:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_mlp_topk.py \
		--config configs_simea/mlp_tcga_renal.yaml \
		--ckpt_path=$(CKPT_MLP_TCGA_RENAL) \
		--ig_name eg

pictopk_mlp_idg_tcga_renal:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_mlp_topk.py \
		--config configs_simea/mlp_tcga_renal.yaml \
		--ckpt_path=$(CKPT_MLP_TCGA_RENAL) \
		--ig_name idg

pictopk_mlp_cig_tcga_renal:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_mlp_topk.py \
		--config configs_simea/mlp_tcga_renal.yaml \
		--ckpt_path=$(CKPT_MLP_TCGA_RENAL) \
		--ig_name cig

pictopk_mlp_g_tcga_renal:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_mlp_topk.py \
		--config configs_simea/mlp_tcga_renal.yaml \
		--ckpt_path=$(CKPT_MLP_TCGA_RENAL) \
		--ig_name g

pictopk_mlp_random_tcga_renal:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_mlp_topk.py \
		--config configs_simea/mlp_tcga_renal.yaml \
		--ckpt_path=$(CKPT_MLP_TCGA_RENAL) \
		--ig_name random

group_basic_pictopk_mlp_tcga_renal: pictopk_mlp_ig_tcga_renal pictopk_mlp_g_tcga_renal pictopk_mlp_eg_tcga_renal 
group_random_pictopk_mlp_tcga: pictopk_mlp_random_camelyon16 pictopk_mlp_random_tcga_renal
group_adv_pictopk_mlp_tcga_renal: pictopk_mlp_cig_tcga_renal pictopk_mlp_idg_tcga_renal 

# === PIC Top-K Targets for TCGA Lung (MLP) ===
pictopk_mlp_ig_tcga_lung:
	CUDA_VISIBLE_DEVICES=2 python metric_pic_mlp_topk.py \
		--config configs_simea/mlp_tcga_lung.yaml \
		--ckpt_path=$(CKPT_MLP_TCGA_LUNG) \
		--ig_name ig

pictopk_mlp_eg_tcga_lung:
	CUDA_VISIBLE_DEVICES=2 python metric_pic_mlp_topk.py \
		--config configs_simea/mlp_tcga_lung.yaml \
		--ckpt_path=$(CKPT_MLP_TCGA_LUNG) \
		--ig_name eg

pictopk_mlp_idg_tcga_lung:
	CUDA_VISIBLE_DEVICES=4 python metric_pic_mlp_topk.py \
		--config configs_simea/mlp_tcga_lung.yaml \
		--ckpt_path=$(CKPT_MLP_TCGA_LUNG) \
		--ig_name idg

pictopk_mlp_cig_tcga_lung:
	CUDA_VISIBLE_DEVICES=4 python metric_pic_mlp_topk.py \
		--config configs_simea/mlp_tcga_lung.yaml \
		--ckpt_path=$(CKPT_MLP_TCGA_LUNG) \
		--ig_name cig

pictopk_mlp_g_tcga_lung:
	CUDA_VISIBLE_DEVICES=2 python metric_pic_mlp_topk.py \
		--config configs_simea/mlp_tcga_lung.yaml \
		--ckpt_path=$(CKPT_MLP_TCGA_LUNG) \
		--ig_name g

pictopk_mlp_random_tcga_lung:
	CUDA_VISIBLE_DEVICES=2 python metric_pic_mlp_topk.py \
		--config configs_simea/mlp_tcga_lung.yaml \
		--ckpt_path=$(CKPT_MLP_TCGA_LUNG) \
		--ig_name random

group_basic_pictopk_mlp_tcga_lung: pictopk_mlp_ig_tcga_lung pictopk_mlp_g_tcga_lung pictopk_mlp_eg_tcga_lung
group_adv_pictopk_mlp_tcga_lung: pictopk_mlp_cig_tcga_lung pictopk_mlp_idg_tcga_lung  

# ====================================================== MLP ig method ===============
#========== IG MLP CAMELYON16 Methods (CUDA 1) ============== 
mlp_ig_camelyon16:
	CUDA_VISIBLE_DEVICES=1 python ig_mlp.py --config configs_simea/mlp_camelyon16.yaml --ig_name ig --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_MLP_CAMELYON16)
mlp_eg_camelyon16:
	CUDA_VISIBLE_DEVICES=1 python ig_mlp.py --config configs_simea/mlp_camelyon16.yaml --ig_name eg --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_MLP_CAMELYON16)
mlp_idg_camelyon16:
	CUDA_VISIBLE_DEVICES=1 python ig_mlp.py --config configs_simea/mlp_camelyon16.yaml --ig_name idg --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_MLP_CAMELYON16)
mlp_cig_camelyon16:
	CUDA_VISIBLE_DEVICES=1 python ig_mlp.py --config configs_simea/mlp_camelyon16.yaml --ig_name cig --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_MLP_CAMELYON16)
mlp_g_camelyon16:
	CUDA_VISIBLE_DEVICES=1 python ig_mlp.py --config configs_simea/mlp_camelyon16.yaml --ig_name g --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_MLP_CAMELYON16)

group_basic_mlp_camelyon16: mlp_ig_camelyon16 mlp_g_camelyon16 mlp_eg_camelyon16 mlp_cig_camelyon16 mlp_idg_camelyon16
group_adv_mlp_camelyon16: mlp_cig_camelyon16 mlp_idg_camelyon16

#========== IG MLP TCGA-RENAL Methods (CUDA 2) ============== 
mlp_ig_tcga_renal:
	CUDA_VISIBLE_DEVICES=2 python ig_mlp.py --config configs_simea/mlp_tcga_renal.yaml --ig_name ig --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_MLP_TCGA_RENAL)
mlp_eg_tcga_renal:
	CUDA_VISIBLE_DEVICES=2 python ig_mlp.py --config configs_simea/mlp_tcga_renal.yaml --ig_name eg --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_MLP_TCGA_RENAL)
mlp_idg_tcga_renal:
	CUDA_VISIBLE_DEVICES=2 python ig_mlp.py --config configs_simea/mlp_tcga_renal.yaml --ig_name idg --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_MLP_TCGA_RENAL)
mlp_cig_tcga_renal:
	CUDA_VISIBLE_DEVICES=2 python ig_mlp.py --config configs_simea/mlp_tcga_renal.yaml --ig_name cig --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_MLP_TCGA_RENAL)
mlp_g_tcga_renal:
	CUDA_VISIBLE_DEVICES=2 python ig_mlp.py --config configs_simea/mlp_tcga_renal.yaml --ig_name g --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_MLP_TCGA_RENAL)

group_basic_mlp_tcga_renal: mlp_ig_tcga_renal mlp_g_tcga_renal mlp_eg_tcga_renal mlp_cig_tcga_renal mlp_idg_tcga_renal
group_adv_mlp_tcga_renal: mlp_cig_tcga_renal mlp_idg_tcga_renal

#========== IG MLP TCGA-LUNG Methods (CUDA 3) ============== 
mlp_ig_tcga_lung:
	CUDA_VISIBLE_DEVICES=3 python ig_mlp.py --config configs_simea/mlp_tcga_lung.yaml --ig_name ig --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_MLP_TCGA_LUNG)
mlp_eg_tcga_lung:
	CUDA_VISIBLE_DEVICES=3 python ig_mlp.py --config configs_simea/mlp_tcga_lung.yaml --ig_name eg --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_MLP_TCGA_LUNG)
mlp_idg_tcga_lung:
	CUDA_VISIBLE_DEVICES=3 python ig_mlp.py --config configs_simea/mlp_tcga_lung.yaml --ig_name idg --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_MLP_TCGA_LUNG)
mlp_cig_tcga_lung:
	CUDA_VISIBLE_DEVICES=3 python ig_mlp.py --config configs_simea/mlp_tcga_lung.yaml --ig_name cig --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_MLP_TCGA_LUNG)
mlp_g_tcga_lung:
	CUDA_VISIBLE_DEVICES=3 python ig_mlp.py --config configs_simea/mlp_tcga_lung.yaml --ig_name g --fold_start 1 --fold_end 1 \
	--ckpt_path $(CKPT_MLP_TCGA_LUNG)

group_basic_mlp_tcga_lung: mlp_ig_tcga_lung mlp_g_tcga_lung mlp_eg_tcga_lung mlp_cig_tcga_lung mlp_idg_tcga_lung
group_adv_mlp_tcga_lung: mlp_cig_tcga_lung mlp_idg_tcga_lung





# === TOP-K NPC Targets for Camelyon16 (MLP) ===
topknpc_mlp_ig_camelyon16:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_mlp_topknpc.py \
		--config configs_simea/mlp_camelyon16.yaml \
		--ckpt_path $(CKPT_MLP_CAMELYON16) \
		--ig_name ig

topknpc_mlp_eg_camelyon16:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_mlp_topknpc.py \
		--config configs_simea/mlp_camelyon16.yaml \
		--ckpt_path $(CKPT_MLP_CAMELYON16) \
		--ig_name eg

topknpc_mlp_idg_camelyon16:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_mlp_topknpc.py \
		--config configs_simea/mlp_camelyon16.yaml \
		--ckpt_path $(CKPT_MLP_CAMELYON16) \
		--ig_name idg

topknpc_mlp_cig_camelyon16:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_mlp_topknpc.py \
		--config configs_simea/mlp_camelyon16.yaml \
		--ckpt_path $(CKPT_MLP_CAMELYON16) \
		--ig_name cig

topknpc_mlp_g_camelyon16:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_mlp_topknpc.py \
		--config configs_simea/mlp_camelyon16.yaml \
		--ckpt_path $(CKPT_MLP_CAMELYON16) \
		--ig_name g

topknpc_mlp_random_camelyon16:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_mlp_topknpc.py \
		--config configs_simea/mlp_camelyon16.yaml \
		--ckpt_path $(CKPT_MLP_CAMELYON16) \
		--ig_name random

group_basic_topknpc_mlp_camelyon16: topknpc_mlp_ig_camelyon16 topknpc_mlp_g_camelyon16 topknpc_mlp_eg_camelyon16
group_adv_topknpc_mlp_camelyon16: topknpc_mlp_cig_camelyon16 topknpc_mlp_idg_camelyon16
group_all_topknpc_mlp_camelyon16: \
	topknpc_mlp_ig_camelyon16 topknpc_mlp_g_camelyon16 topknpc_mlp_eg_camelyon16 \
	topknpc_mlp_cig_camelyon16 topknpc_mlp_idg_camelyon16 


# === TOP-K NPC Targets for TCGA Renal (MLP) ===
topknpc_mlp_ig_tcga_renal:
	CUDA_VISIBLE_DEVICES=2 python metric_pic_mlp_topknpc.py \
		--config configs_simea/mlp_tcga_renal.yaml \
		--ckpt_path=$(CKPT_MLP_TCGA_RENAL) \
		--ig_name ig

topknpc_mlp_eg_tcga_renal:
	CUDA_VISIBLE_DEVICES=2 python metric_pic_mlp_topknpc.py \
		--config configs_simea/mlp_tcga_renal.yaml \
		--ckpt_path=$(CKPT_MLP_TCGA_RENAL) \
		--ig_name eg

topknpc_mlp_idg_tcga_renal:
	CUDA_VISIBLE_DEVICES=2 python metric_pic_mlp_topknpc.py \
		--config configs_simea/mlp_tcga_renal.yaml \
		--ckpt_path=$(CKPT_MLP_TCGA_RENAL) \
		--ig_name idg

topknpc_mlp_cig_tcga_renal:
	CUDA_VISIBLE_DEVICES=2 python metric_pic_mlp_topknpc.py \
		--config configs_simea/mlp_tcga_renal.yaml \
		--ckpt_path=$(CKPT_MLP_TCGA_RENAL) \
		--ig_name cig

topknpc_mlp_g_tcga_renal:
	CUDA_VISIBLE_DEVICES=2 python metric_pic_mlp_topknpc.py \
		--config configs_simea/mlp_tcga_renal.yaml \
		--ckpt_path=$(CKPT_MLP_TCGA_RENAL) \
		--ig_name g

topknpc_mlp_random_tcga_renal:
	CUDA_VISIBLE_DEVICES=2 python metric_pic_mlp_topknpc.py \
		--config configs_simea/mlp_tcga_renal.yaml \
		--ckpt_path=$(CKPT_MLP_TCGA_RENAL) \
		--ig_name random

group_basic_topknpc_mlp_tcga_renal: topknpc_mlp_ig_tcga_renal topknpc_mlp_g_tcga_renal topknpc_mlp_eg_tcga_renal 
group_adv_topknpc_mlp_tcga_renal: topknpc_mlp_cig_tcga_renal topknpc_mlp_idg_tcga_renal 
group_all_topknpc_mlp_tcga_renal: \
	topknpc_mlp_ig_tcga_renal topknpc_mlp_g_tcga_renal topknpc_mlp_eg_tcga_renal \
	topknpc_mlp_cig_tcga_renal topknpc_mlp_idg_tcga_renal 


# === TOP-K NPC Targets for TCGA Lung (MLP) ===
topknpc_mlp_ig_tcga_lung:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_mlp_topknpc.py \
		--config configs_simea/mlp_tcga_lung.yaml \
		--ckpt_path=$(CKPT_MLP_TCGA_LUNG) \
		--ig_name ig

topknpc_mlp_eg_tcga_lung:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_mlp_topknpc.py \
		--config configs_simea/mlp_tcga_lung.yaml \
		--ckpt_path=$(CKPT_MLP_TCGA_LUNG) \
		--ig_name eg

topknpc_mlp_idg_tcga_lung:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_mlp_topknpc.py \
		--config configs_simea/mlp_tcga_lung.yaml \
		--ckpt_path=$(CKPT_MLP_TCGA_LUNG) \
		--ig_name idg

topknpc_mlp_cig_tcga_lung:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_mlp_topknpc.py \
		--config configs_simea/mlp_tcga_lung.yaml \
		--ckpt_path=$(CKPT_MLP_TCGA_LUNG) \
		--ig_name cig

topknpc_mlp_g_tcga_lung:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_mlp_topknpc.py \
		--config configs_simea/mlp_tcga_lung.yaml \
		--ckpt_path=$(CKPT_MLP_TCGA_LUNG) \
		--ig_name g

topknpc_mlp_random_tcga_lung:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_mlp_topknpc.py \
		--config configs_simea/mlp_tcga_lung.yaml \
		--ckpt_path=$(CKPT_MLP_TCGA_LUNG) \
		--ig_name random

group_basic_topknpc_mlp_tcga_lung: topknpc_mlp_ig_tcga_lung topknpc_mlp_g_tcga_lung topknpc_mlp_eg_tcga_lung
group_adv_topknpc_mlp_tcga_lung: topknpc_mlp_cig_tcga_lung topknpc_mlp_idg_tcga_lung
group_all_topknpc_mlp_tcga_lung: \
	topknpc_mlp_ig_tcga_lung topknpc_mlp_g_tcga_lung topknpc_mlp_eg_tcga_lung \
	topknpc_mlp_cig_tcga_lung topknpc_mlp_idg_tcga_lung 



# === TOP-K NPC Targets for Camelyon16 (CLAM) ===
topknpc_clam_ig_camelyon16:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_clam_topknpc.py \
		--config configs_simea/clam_camelyon16.yaml \
		--ckpt_path $(CKPT_CLAM_CAMELYON16) \
		--ig_name ig

topknpc_clam_eg_camelyon16:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_clam_topknpc.py \
		--config configs_simea/clam_camelyon16.yaml \
		--ckpt_path $(CKPT_CLAM_CAMELYON16) \
		--ig_name eg

topknpc_clam_idg_camelyon16:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_clam_topknpc.py \
		--config configs_simea/clam_camelyon16.yaml \
		--ckpt_path $(CKPT_CLAM_CAMELYON16) \
		--ig_name idg

topknpc_clam_cig_camelyon16:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_clam_topknpc.py \
		--config configs_simea/clam_camelyon16.yaml \
		--ckpt_path $(CKPT_CLAM_CAMELYON16) \
		--ig_name cig

topknpc_clam_g_camelyon16:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_clam_topknpc.py \
		--config configs_simea/clam_camelyon16.yaml \
		--ckpt_path $(CKPT_CLAM_CAMELYON16) \
		--ig_name g

topknpc_clam_random_camelyon16:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_clam_topknpc.py \
		--config configs_simea/clam_camelyon16.yaml \
		--ckpt_path $(CKPT_CLAM_CAMELYON16) \
		--ig_name random

group_basic_topknpc_clam_camelyon16: topknpc_clam_ig_camelyon16 topknpc_clam_g_camelyon16 topknpc_clam_eg_camelyon16
group_adv_topknpc_clam_camelyon16: topknpc_clam_cig_camelyon16 topknpc_clam_idg_camelyon16
group_all_topknpc_clam_camelyon16: \
	topknpc_clam_ig_camelyon16 topknpc_clam_g_camelyon16 topknpc_clam_eg_camelyon16 \
	topknpc_clam_cig_camelyon16 topknpc_clam_idg_camelyon16 topknpc_clam_random_camelyon16


# === TOP-K NPC Targets for TCGA Renal (CLAM) ===
topknpc_clam_ig_tcga_renal:
	CUDA_VISIBLE_DEVICES=2 python metric_pic_clam_topknpc.py \
		--config configs_simea/clam_tcga_renal.yaml \
		--ckpt_path=$(CKPT_CLAM_TCGA_RENAL) \
		--ig_name ig

topknpc_clam_eg_tcga_renal:
	CUDA_VISIBLE_DEVICES=2 python metric_pic_clam_topknpc.py \
		--config configs_simea/clam_tcga_renal.yaml \
		--ckpt_path=$(CKPT_CLAM_TCGA_RENAL) \
		--ig_name eg

topknpc_clam_idg_tcga_renal:
	CUDA_VISIBLE_DEVICES=2 python metric_pic_clam_topknpc.py \
		--config configs_simea/clam_tcga_renal.yaml \
		--ckpt_path=$(CKPT_CLAM_TCGA_RENAL) \
		--ig_name idg

topknpc_clam_cig_tcga_renal:
	CUDA_VISIBLE_DEVICES=2 python metric_pic_clam_topknpc.py \
		--config configs_simea/clam_tcga_renal.yaml \
		--ckpt_path=$(CKPT_CLAM_TCGA_RENAL) \
		--ig_name cig

topknpc_clam_g_tcga_renal:
	CUDA_VISIBLE_DEVICES=2 python metric_pic_clam_topknpc.py \
		--config configs_simea/clam_tcga_renal.yaml \
		--ckpt_path=$(CKPT_CLAM_TCGA_RENAL) \
		--ig_name g

topknpc_clam_random_tcga_renal:
	CUDA_VISIBLE_DEVICES=2 python metric_pic_clam_topknpc.py \
		--config configs_simea/clam_tcga_renal.yaml \
		--ckpt_path=$(CKPT_CLAM_TCGA_RENAL) \
		--ig_name random

group_basic_topknpc_clam_tcga_renal: topknpc_clam_ig_tcga_renal topknpc_clam_g_tcga_renal topknpc_clam_eg_tcga_renal
group_adv_topknpc_clam_tcga_renal: topknpc_clam_cig_tcga_renal topknpc_clam_idg_tcga_renal
group_all_topknpc_clam_tcga_renal: \
	topknpc_clam_ig_tcga_renal topknpc_clam_g_tcga_renal topknpc_clam_eg_tcga_renal \
	topknpc_clam_cig_tcga_renal topknpc_clam_idg_tcga_renal topknpc_clam_random_tcga_renal


# === TOP-K NPC Targets for TCGA Lung (CLAM) ===
topknpc_clam_ig_tcga_lung:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_clam_topknpc.py \
		--config configs_simea/clam_tcga_lung.yaml \
		--ckpt_path=$(CKPT_CLAM_TCGA_LUNG) \
		--ig_name ig

topknpc_clam_eg_tcga_lung:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_clam_topknpc.py \
		--config configs_simea/clam_tcga_lung.yaml \
		--ckpt_path=$(CKPT_CLAM_TCGA_LUNG) \
		--ig_name eg

topknpc_clam_idg_tcga_lung:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_clam_topknpc.py \
		--config configs_simea/clam_tcga_lung.yaml \
		--ckpt_path=$(CKPT_CLAM_TCGA_LUNG) \
		--ig_name idg

topknpc_clam_cig_tcga_lung:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_clam_topknpc.py \
		--config configs_simea/clam_tcga_lung.yaml \
		--ckpt_path=$(CKPT_CLAM_TCGA_LUNG) \
		--ig_name cig

topknpc_clam_g_tcga_lung:
	CUDA_VISIBLE_DEVICES=3 python metric_pic_clam_topknpc.py \
		--config configs_simea/clam_tcga_lung.yaml \
		--ckpt_path=$(CKPT_CLAM_TCGA_LUNG) \
		--ig_name g

topknpc_clam_random_tcga_lung:
	CUDA_VISIBLE_DEVICES=4 python metric_pic_clam_topknpc.py \
		--config configs_simea/clam_tcga_lung.yaml \
		--ckpt_path=$(CKPT_CLAM_TCGA_LUNG) \
		--ig_name random

group_basic_topknpc_clam_tcga_lung: topknpc_clam_ig_tcga_lung topknpc_clam_g_tcga_lung topknpc_clam_eg_tcga_lung
group_adv_topknpc_clam_tcga_lung: topknpc_clam_cig_tcga_lung topknpc_clam_idg_tcga_lung
group_all_topknpc_clam_tcga_lung: \
	topknpc_clam_ig_tcga_lung topknpc_clam_g_tcga_lung topknpc_clam_eg_tcga_lung \
	topknpc_clam_cig_tcga_lung topknpc_clam_idg_tcga_lung topknpc_clam_random_tcga_lung



#=====================================================
pre_plot_camelyon16:
	CUDA_VISIBLE_DEVICES=3 python pre_plot.py --config configs_simea/clam_camelyon16.yaml
pre_plot_tcga_renal:
	CUDA_VISIBLE_DEVICES=3 python pre_plot.py --config configs_simea/clam_tcga_renal.yaml  
pre_plot_tcga_lung:
	CUDA_VISIBLE_DEVICES=3 python pre_plot.py --config configs_simea/clam_tcga_lung.yaml 
#=====================================================



#=====================================================
gt_plot_camelyon16:
	CUDA_VISIBLE_DEVICES=3 python plot_downscale_slide_gt.py --config configs_simea/clam_camelyon16.yaml
gt_plot_tcga_renal:
	CUDA_VISIBLE_DEVICES=4 python plot_downscale_slide_gt.py --config configs_simea/clam_tcga_renal.yaml  
gt_plot_tcga_lung:
	CUDA_VISIBLE_DEVICES=5 python plot_downscale_slide_gt.py --config configs_simea/clam_tcga_lung.yaml 
#===================================================== 
 
plot_clam_camelyon16_ig:
	CUDA_VISIBLE_DEVICES=3 python plot.py --config configs_simea/clam_camelyon16.yaml --ig_name ig 

### --------------------- CAMELYON16 ---------------------
plot_camelyon:
	make plot_camelyon_clam
	make plot_camelyon_mlp

plot_camelyon_clam:
	CUDA_VISIBLE_DEVICES=5 python plot.py --config configs_simea/clam_camelyon16.yaml --ig_name g
	CUDA_VISIBLE_DEVICES=5 python plot.py --config configs_simea/clam_camelyon16.yaml --ig_name ig
	CUDA_VISIBLE_DEVICES=5 python plot.py --config configs_simea/clam_camelyon16.yaml --ig_name cig
	CUDA_VISIBLE_DEVICES=5 python plot.py --config configs_simea/clam_camelyon16.yaml --ig_name idg
	CUDA_VISIBLE_DEVICES=5 python plot.py --config configs_simea/clam_camelyon16.yaml --ig_name eg

plot_camelyon_mlp:
	CUDA_VISIBLE_DEVICES=5 python plot.py --config configs_simea/mlp_camelyon16.yaml --ig_name g
	CUDA_VISIBLE_DEVICES=5 python plot.py --config configs_simea/mlp_camelyon16.yaml --ig_name ig
	CUDA_VISIBLE_DEVICES=5 python plot.py --config configs_simea/mlp_camelyon16.yaml --ig_name cig
	CUDA_VISIBLE_DEVICES=5 python plot.py --config configs_simea/mlp_camelyon16.yaml --ig_name idg
	CUDA_VISIBLE_DEVICES=5 python plot.py --config configs_simea/mlp_camelyon16.yaml --ig_name eg


### --------------------- TCGA RENAL ---------------------
plot_tcga_renal:
	make plot_tcga_renal_clam
	make plot_tcga_renal_mlp

plot_tcga_renal_clam:
	CUDA_VISIBLE_DEVICES=6 python plot.py --config configs_simea/clam_tcga_renal.yaml --ig_name g
	CUDA_VISIBLE_DEVICES=6 python plot.py --config configs_simea/clam_tcga_renal.yaml --ig_name ig
	CUDA_VISIBLE_DEVICES=6 python plot.py --config configs_simea/clam_tcga_renal.yaml --ig_name cig
	CUDA_VISIBLE_DEVICES=6 python plot.py --config configs_simea/clam_tcga_renal.yaml --ig_name idg
	CUDA_VISIBLE_DEVICES=6 python plot.py --config configs_simea/clam_tcga_renal.yaml --ig_name eg

plot_tcga_renal_mlp:
	CUDA_VISIBLE_DEVICES=6 python plot.py --config configs_simea/mlp_tcga_renal.yaml --ig_name g
	CUDA_VISIBLE_DEVICES=6 python plot.py --config configs_simea/mlp_tcga_renal.yaml --ig_name ig
	CUDA_VISIBLE_DEVICES=6 python plot.py --config configs_simea/mlp_tcga_renal.yaml --ig_name cig
	CUDA_VISIBLE_DEVICES=6 python plot.py --config configs_simea/mlp_tcga_renal.yaml --ig_name idg
	CUDA_VISIBLE_DEVICES=6 python plot.py --config configs_simea/mlp_tcga_renal.yaml --ig_name eg


### --------------------- TCGA LUNG ---------------------
plot_tcga_lung:
	make plot_tcga_lung_clam
	make plot_tcga_lung_mlp

plot_tcga_lung_clam:
	CUDA_VISIBLE_DEVICES=7 python plot.py --config configs_simea/clam_tcga_lung.yaml --ig_name g
	CUDA_VISIBLE_DEVICES=7 python plot.py --config configs_simea/clam_tcga_lung.yaml --ig_name ig
	CUDA_VISIBLE_DEVICES=7 python plot.py --config configs_simea/clam_tcga_lung.yaml --ig_name cig
	CUDA_VISIBLE_DEVICES=7 python plot.py --config configs_simea/clam_tcga_lung.yaml --ig_name idg
	CUDA_VISIBLE_DEVICES=7 python plot.py --config configs_simea/clam_tcga_lung.yaml --ig_name eg

plot_tcga_lung_mlp:
	CUDA_VISIBLE_DEVICES=7 python plot.py --config configs_simea/mlp_tcga_lung.yaml --ig_name g
	CUDA_VISIBLE_DEVICES=7 python plot.py --config configs_simea/mlp_tcga_lung.yaml --ig_name ig
	CUDA_VISIBLE_DEVICES=7 python plot.py --config configs_simea/mlp_tcga_lung.yaml --ig_name cig
	CUDA_VISIBLE_DEVICES=7 python plot.py --config configs_simea/mlp_tcga_lung.yaml --ig_name idg
	CUDA_VISIBLE_DEVICES=7 python plot.py --config configs_simea/mlp_tcga_lung.yaml --ig_name eg

proc_anno:
	python metric2_processing_anno.py --config configs_simea/clam_camelyon16.yaml

# proc_anno >> compute_iou_dice_camelyon 16 
test_dice_iou:
	python metric2_compute_iou_dice.py --config configs_simea/clam_camelyon16.yaml --ig_name g
dice_iou_clam_camelyon16:
	python metric2_compute_iou_dice.py --config configs_simea/clam_camelyon16.yaml --ig_name g
	python metric2_compute_iou_dice.py --config configs_simea/clam_camelyon16.yaml --ig_name ig
	python metric2_compute_iou_dice.py --config configs_simea/clam_camelyon16.yaml --ig_name cig
	python metric2_compute_iou_dice.py --config configs_simea/clam_camelyon16.yaml --ig_name idg
	python metric2_compute_iou_dice.py --config configs_simea/clam_camelyon16.yaml --ig_name eg

dice_iou_mlp_camelyon16:
	python metric2_compute_iou_dice.py --config configs_simea/mlp_camelyon16.yaml --ig_name g
	python metric2_compute_iou_dice.py --config configs_simea/mlp_camelyon16.yaml --ig_name ig
	python metric2_compute_iou_dice.py --config configs_simea/mlp_camelyon16.yaml --ig_name cig
	python metric2_compute_iou_dice.py --config configs_simea/mlp_camelyon16.yaml --ig_name idg
	python metric2_compute_iou_dice.py --config configs_simea/mlp_camelyon16.yaml --ig_name eg

plot_alpha_compute: 
	CUDA_VISIBLE_DEVICES=5 python plot_alpha_compute.py --config configs_simea/clam_camelyon16.yaml --ig_name ig --ckpt_path $(CKPT_CLAM_CAMELYON16)
	CUDA_VISIBLE_DEVICES=6 python plot_alpha_compute.py --config configs_simea/clam_camelyon16.yaml --ig_name g --ckpt_path $(CKPT_CLAM_CAMELYON16) 
	CUDA_VISIBLE_DEVICES=7 python plot_alpha_compute.py --config configs_simea/clam_camelyon16.yaml --ig_name cig --ckpt_path $(CKPT_CLAM_CAMELYON16) 


plot_alpha_plot_ig:
	CUDA_VISIBLE_DEVICES=5 python plot_alpha_plot.py --config configs_simea/clam_camelyon16.yaml --ig_name ig 
plot_alpha_plot_g:
	CUDA_VISIBLE_DEVICES=6 python plot_alpha_plot.py --config configs_simea/clam_camelyon16.yaml --ig_name g 
plot_alpha_plot_cig:
	CUDA_VISIBLE_DEVICES=7 python plot_alpha_plot.py --config configs_simea/clam_camelyon16.yaml --ig_name cig 



rs_topk:
	python combine_result_topk.py


result:
	python combine_result_topknpc.py

result_iou:
	python combine_iou.py


plot_gt_heatmap:
	CUDA_VISIBLE_DEVICES=5 python plot_gt_heatmap.py --config configs_simea/clam_camelyon16.yaml 