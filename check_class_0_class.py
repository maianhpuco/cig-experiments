from scipy.spatial.distance import cosine
from scipy.stats import pearsonr

# Collect all slide IDs that are present in both class folders
class_0_folder = os.path.join(args.paths['attribution_scores_folder'], method, f'fold_{fold_id}', 'class_0')
class_1_folder = os.path.join(args.paths['attribution_scores_folder'], method, f'fold_{fold_id}', 'class_1')

shared_slide_ids = set(
    f.replace('.npy', '') for f in os.listdir(class_0_folder)
) & set(
    f.replace('.npy', '') for f in os.listdir(class_1_folder)
)

print(f"\n🔍 Comparing scores between class_0 and class_1 ({len(shared_slide_ids)} shared slides):")

for slide_id in sorted(shared_slide_ids):
    score_0 = np.load(os.path.join(class_0_folder, f"{slide_id}.npy"))
    score_1 = np.load(os.path.join(class_1_folder, f"{slide_id}.npy"))

    # Ensure equal length
    if len(score_0) != len(score_1):
        print(f"⚠️ Mismatched shapes for {slide_id}: {score_0.shape} vs {score_1.shape}")
        continue

    diff_mean = np.mean(score_0 - score_1)
    cos_sim = 1 - cosine(score_0, score_1)
    pear_corr, _ = pearsonr(score_0, score_1)

    print(f"Slide: {slide_id}")
    print(f" - Mean diff      : {diff_mean:.4f}")
    print(f" - Cosine sim     : {cos_sim:.4f}")
    print(f" - Pearson corr   : {pear_corr:.4f}")
