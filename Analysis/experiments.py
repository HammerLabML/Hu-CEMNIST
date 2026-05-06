#!/usr/bin/env python
# coding: utf-8
import pickle
import numpy as np
import pandas as pd
import dice_ml
from skimage.metrics import structural_similarity as ssim

from utils import load_original_images, compute_percentage_change, compute_img_quality, \
    load_human_generated_counterfactuals, compute_area_of_change, compare_area_of_change
from dnn import train_dnn
from cf_proto import ProtoExplainer
from cf_nf import NFexplainer


# Load training data and train a MLP for digit classification
dnn, (X_train, y_train), (X_test, y_test) = train_dnn()


# Load human generated counterfactuals
cf_data = load_human_generated_counterfactuals()


orig_data = load_original_images()
def _get_orig_img(digit_label: int):
    for d in orig_data:
        if d["label"] == digit_label:
            return d["img"]


def _get_all_orig_digits(label: int):
    test_idx = y_test == label
    return X_test[test_idx, :]


def compute_metrics(x_orig: np.ndarray, x_cf: np.ndarray, target_label: int) -> dict:
    return {"amount_of_change": np.sum(np.abs(x_orig - x_cf)),
            "percentage_of_change": compute_percentage_change(x_orig, x_cf),
            "area_of_change": compute_area_of_change(x_orig, x_cf),
            "img_quality": compute_img_quality(x_cf, _get_all_orig_digits(target_label))}


# ### Human-generated Counterfactuals
human_cfs_eval = []
for orig_digit, target_digit, img in cf_data:
    orig_img = _get_orig_img(orig_digit)

    human_cfs_eval.append({"orig_digit": orig_digit, "target_digit": target_digit,
                           "metrics": compute_metrics(orig_img, img, target_digit)})
    
with open("human_generated_cfs-eval.pickle", "wb") as f_out:
    pickle.dump(human_cfs_eval, f_out)


def compare_with_human_generated_cfs(x_orig, x_cf, y_cf):
    agreement_scores = []
    similarity_scores = []

    for orig_digit, target_digit, img in cf_data:
        orig_img = _get_orig_img(orig_digit)

        if target_digit == y_cf:
            agreement_scores.append(compare_area_of_change(compute_area_of_change(orig_img, img),
                                                           compute_area_of_change(x_orig, x_cf)))
            similarity_scores.append(ssim(img.reshape((28, 28)), x_cf.reshape((28, 28)),
                                          data_range=img.max() - img.min()))
            
    return np.max(agreement_scores), np.max(similarity_scores)



# ### Algorithm-generated Counterfactuals

# #### Counterfactuals guided by prototypes
expl = ProtoExplainer(dnn, X_train, y_train)

protoguided_cfs_eval = []
no_cf_found = 0
for orig_digit, target_digit, _ in cf_data:
    orig_img = _get_orig_img(orig_digit)

    try:
        [x_cf] = expl.compute_counterfactual(orig_img, y_target=target_digit)
        x_cf = x_cf.flatten()

        metrics = compute_metrics(orig_img, x_cf, target_digit)
        agreement_score, similarity_score = compare_with_human_generated_cfs(orig_img, x_cf, target_digit)
        metrics["human_agreement_score"] = agreement_score
        metrics["human_similarity_score"] = similarity_score

        protoguided_cfs_eval.append({"orig_digit": orig_digit, "target_digit": target_digit,
                                     "metrics": metrics, "xcf": x_cf})
    except:
        no_cf_found += 1
    
print(f"No counterfactuals found in {no_cf_found}/{len(cf_data)} -- i.e., {no_cf_found / len(cf_data)}%")
with open("protoguided_cfs-eval.pickle", "wb") as f_out:
    pickle.dump(protoguided_cfs_eval, f_out)


# ### DiCE

# Wrap data as a pandas dataframe
cols = list(range(X_test.shape[1]))
test_data = pd.DataFrame(data=np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1),
                         columns=cols + ["y"])
data = dice_ml.Data(dataframe=test_data, continuous_features=cols, outcome_name="y")

# Compute counterfactuals
m = dice_ml.Model(model=dnn, backend="sklearn", func=None)
exp = dice_ml.Dice(data, m, method="genetic")

dice_cfs_eval = []
no_cf_found = 0
for orig_digit, target_digit, _ in cf_data:
    orig_img = _get_orig_img(orig_digit)

    try:
        x_orig = pd.DataFrame(orig_img.reshape(1, -1), columns=cols)
        cf_result = exp.generate_counterfactuals(x_orig, total_CFs=1, desired_class=target_digit, verbose=False)
        x_cf = cf_result.cf_examples_list[0].final_cfs_df[cols].to_numpy().flatten()

        metrics = compute_metrics(orig_img, x_cf, target_digit)
        agreement_score, similarity_score = compare_with_human_generated_cfs(orig_img, x_cf, target_digit)
        metrics["human_agreement_score"] = agreement_score
        metrics["human_similarity_score"] = similarity_score

        dice_cfs_eval.append({"orig_digit": orig_digit, "target_digit": target_digit,
                              "metrics": metrics, "xcf": x_cf})
    except:
        no_cf_found += 1

print(f"No counterfactuals found in {no_cf_found}/{len(cf_data)} -- i.e., {no_cf_found / len(cf_data)}%")
with open("dice_cfs-eval.pickle", "wb") as f_out:
    pickle.dump(dice_cfs_eval, f_out)


# #### Normalizing Flows
expl = NFexplainer(X_train, y_train, X_test, y_test)

nf_cfs_eval = []
no_cf_found = 0
for orig_digit, target_digit, _ in cf_data:
    orig_img = _get_orig_img(orig_digit)

    try:
        x_cf = expl.compute_counterfactual(orig_img, np.array(orig_digit), np.array(target_digit))
        x_cf = x_cf.flatten()

        metrics = compute_metrics(orig_img, x_cf, target_digit)
        agreement_score, similarity_score = compare_with_human_generated_cfs(orig_img, x_cf, target_digit)
        metrics["human_agreement_score"] = agreement_score
        metrics["human_similarity_score"] = similarity_score

        nf_cfs_eval.append({"orig_digit": orig_digit, "target_digit": target_digit,
                            "metrics": metrics, "xcf": x_cf})
    except Exception as ex:
        print(ex)
        no_cf_found += 1

print(f"No counterfactuals found in {no_cf_found}/{len(cf_data)} -- i.e., {no_cf_found / len(cf_data)}%")
with open("nf_cfs-eval.pickle", "wb") as f_out:
    pickle.dump(nf_cfs_eval, f_out)
