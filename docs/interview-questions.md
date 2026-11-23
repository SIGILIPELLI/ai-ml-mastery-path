---
title: "AI & Machine Learning Interview Questions and Answers (180 Questions)"
description: "180 AI & Machine Learning interview questions with answers and explanations, from beginner to advanced, organised by level. Free, from the AI & Machine Learning Mastery Path."
---

# AI & Machine Learning Interview Questions and Answers

**180 questions** with the correct answer and a short explanation, organised from beginner (Level 1) to advanced (Level 4). Every question comes from a lesson in the free AI & Machine Learning Mastery Path, so you can open the lesson to go deeper.

!!! tip "Practise, don't just read"
    Each lesson also ends with an interactive quiz, and the [final quiz](quiz.md) tests all four levels. Try to answer each question below before reading the answer.

## Level 1 · Entry

### [01 · Setup &amp; the Scientific Python Stack](level-1/01-setup-scientific-python.md)

**Q1.** Why work inside a virtual environment?

**Answer:** It isolates each project's packages so one project's versions can't break another's

pip installs into .venv instead of the system Python.

**Q2.** Which library is used for classical machine learning in this stack?

**Answer:** scikit-learn

NumPy is for arrays, pandas for tables, matplotlib for plots and PyTorch for neural networks.

**Q3.** What is the recommended habit about notebooks and scripts?

**Answer:** Explore in a notebook, then consolidate what worked into a script

Scripts are version-control friendly and look like production code.

**Q4.** How do you know the setup is done?

**Answer:** All five library versions print when check_stack.py runs

Exact versions don't matter; anything from the last few years works.

### [02 · NumPy &amp; Pandas Essentials](level-1/02-numpy-pandas.md)

**Q5.** What is the single most-checked property in ML code?

**Answer:** shape

A dataset of 150 examples with 4 features is (150, 4); shape mismatches are the most common bug.

**Q6.** What is vectorization?

**Answer:** Applying operations element-wise with the loop running in optimized C instead of Python

The lesson's benchmark was roughly 50–100x faster than a Python loop.

**Q7.** What does M.mean(axis=0) compute for a 2-row, 3-column array?

**Answer:** The mean of each column

axis=0 goes down the columns, giving [2.5, 3.5, 4.5] for the example.

**Q8.** What is the practical rule when you write a for loop over data points?

**Answer:** Look for the array operation instead

Training does millions of multiply-adds, so vectorization matters.

### [03 · Data Preparation](level-1/03-data-preparation.md)

**Q9.** Why set aside a test set before doing anything else?

**Answer:** To measure performance on data the model hasn't seen, and not touch it until the end

A model can simply memorize training data, so evaluating on it is meaningless.

**Q10.** What does stratify=y do in train_test_split?

**Answer:** Keeps each class in the same proportion in both splits

It's used for classification.

**Q11.** Why impute with the median rather than the mean?

**Answer:** The median is robust to outliers

SimpleImputer(strategy="median") learns per-column medians from the data it's fit on.

**Q12.** What is data leakage?

**Answer:** A silent bug that makes bad models look good

It's the most important concept in practical ML data preparation.

### [04 · Regression](level-1/04-regression.md)

**Q13.** What does regression predict?

**Answer:** A number, such as a house price

Classification predicts categories.

**Q14.** What does least squares training minimize?

**Answer:** The squared difference between predictions and actual values on the training set

It finds the weights w and intercept b.

**Q15.** Why is linear regression the baseline of choice?

**Answer:** You can read and reason about each weight

For example each extra unit of MedInc adds about 0.449 (≈$44,900) to the prediction.

**Q16.** Which two failure modes haunt every model?

**Answer:** Underfitting and overfitting

The module introduces both after fitting a line.

### [05 · Classification](level-1/05-classification.md)

**Q17.** What does logistic regression do despite its name?

**Answer:** It is a classifier: a weighted sum squashed through a sigmoid into a probability, thresholded at 0.5

It outputs probabilities between 0 and 1.

**Q18.** Why does predict_proba matter in practice?

**Answer:** '97% likely benign' supports different decisions than '51% likely benign', though predict returns 1 for both

Probabilities carry confidence that labels hide.

**Q19.** Why is scaling done inside a pipeline for logistic regression?

**Answer:** It, like most non-tree models, cares about feature scale

Decision trees need no scaling and are directly inspectable.

**Q20.** What does stratify=y preserve in the breast cancer split?

**Answer:** The class ratio (about 63/37) in both splits

The dataset has 357 benign and 212 malignant tumors.

### [06 · Clustering &amp; Unsupervised Learning](level-1/06-clustering-unsupervised.md)

**Q21.** What distinguishes unsupervised learning?

**Answer:** It finds structure with no labels at all

fit_predict in the k-means example consumes only X.

**Q22.** Why scale features before k-means on real data?

**Answer:** k-means is pure distance math, so scale dominates distances

Use StandardScaler.

**Q23.** What does the elbow method look for?

**Answer:** The bend in inertia versus k where improvements level off

In the example, inertia drops hugely until k=4, then only marginally.

**Q24.** What is true of cluster IDs?

**Answer:** They are arbitrary: only the grouping is meaningful

Cluster 2 this run may be cluster 0 next run.

### [07 · Model Evaluation &amp; Cross-Validation](level-1/07-model-evaluation.md)

**Q25.** What does the five-seed experiment show?

**Answer:** Accuracy wobbles by 3+ points depending on the split, so a single number is partly luck

Same model, same data.

**Q26.** How does k-fold cross-validation work?

**Answer:** Train k times, holding out a different fold each time, and report mean ± std

Every row is used for testing exactly once.

**Q27.** Why is the scaler inside a Pipeline important for CV?

**Answer:** It's re-fit inside each fold on that fold's training portion, avoiding leakage

Scaling X once up front leaks test fold statistics.

**Q28.** What is the cost of cross-validation?

**Answer:** k model fits instead of 1

cv=5 or cv=10 are standard.

### [08 · Feature Engineering &amp; Pipelines](level-1/08-feature-engineering-pipelines.md)

**Q29.** What does a scikit-learn Pipeline guarantee?

**Answer:** Preprocessing is learned only from training data, with the whole pipeline refit per CV fold

You literally cannot forget to re-fit the scaler per fold.

**Q30.** What does pipe.predict/pipe.score do with the scaler?

**Answer:** Only transform (no fitting)

pipe.fit runs fit_transform through each step, then fits the final model.

**Q31.** What is feature engineering?

**Answer:** Deriving columns that expose the signal more directly

Models can only use the information you give them, in the form you give it.

**Q32.** What is the standard real-world skeleton the module builds toward?

**Answer:** ColumnTransformer inside a Pipeline

It handles different column types in one object.

### [09 · Neural Network Fundamentals with PyTorch](level-1/09-neural-networks-pytorch.md)

**Q33.** What two additions does a torch.Tensor have over a NumPy ndarray?

**Answer:** It can live on a GPU and it can track gradients

Shapes, vectorization and reductions transfer directly, using dim instead of axis.

**Q34.** What does loss.backward() compute for loss = (w - 5)^2 at w = 3?

**Answer:** w.grad = -4

d(loss)/dw = 2(3−5) = −4.

**Q35.** In which direction does gradient descent nudge w?

**Answer:** A small step against the gradient: w ← w − lr · grad

A negative gradient means loss decreases if w increases, so w increases.

**Q36.** What is a neural net, under the hype?

**Answer:** Layers of linear regression with non-linear squashes between them, trained by gradient descent

Tensors, autograd, an MLP and the training loop are the four core ideas.

### [10 · Capstone — End-to-End Tabular ML Project](level-1/10-capstone-project.md)

**Q37.** Why does all data logic live in prepare.py, imported by both training and prediction?

**Answer:** So the exact same code path that prepared training data prepares future data

Divergence between those paths is where production ML bugs come from.

**Q38.** What data quirk does the exploration step reveal about the target?

**Answer:** MedHouseVal is capped at 5.0 ($500k)

It also shows outliers such as 141 rooms per household.

**Q39.** Which feature correlates most with house value?

**Answer:** MedInc (about 0.688)

Income dominates.

**Q40.** What does train.py save for reuse?

**Answer:** The trained pipeline as model.joblib

predict.py loads it to predict for new districts.

## Level 2 · Intermediate

### [01 · Ensemble Methods Deep Dive](level-2/01-ensemble-methods.md)

**Q41.** What is bagging?

**Answer:** Training many copies of a model on bootstrap samples (drawn with replacement) and averaging predictions

It mostly tames variance, not bias.

**Q42.** What extra randomness does a random forest add?

**Answer:** Only a random subset of features is considered at each split

This decorrelates the trees.

**Q43.** What does max_features="sqrt" mean?

**Answer:** Each split considers about the square root of the number of features

It's the default for classification.

**Q44.** Why does a single decision tree overfit easily?

**Answer:** It memorizes noise along with signal

Ensembles combine many models so their errors average out.

### [02 · Gradient Boosting &amp; XGBoost](level-2/02-gradient-boosting-xgboost.md)

**Q45.** How does boosting differ from a random forest?

**Answer:** Boosting builds trees sequentially, each correcting the ensemble's errors; forests build independent trees and average

Boosting is usually the strongest technique for tabular data.

**Q46.** What is a weak learner here?

**Answer:** A shallow tree (e.g. max_depth=3) adding a small learning_rate-scaled correction

The ensemble solves the problem collectively over many small steps.

**Q47.** What does early stopping do in XGBoost?

**Answer:** Stops adding trees once validation loss stalls

E.g. 143 of 2000 trees used.

**Q48.** What do subsample and colsample_bytree do?

**Answer:** Add bagging-style row and column subsampling per tree to fight overfitting

XGBoost is boosting plus a bit of bagging.

### [03 · Time Series Forecasting Basics](level-2/03-time-series.md)

**Q49.** Why is a random train_test_split invalid for time series?

**Answer:** It would let the model train on future days and be tested on earlier ones

Split by time instead, or use TimeSeriesSplit.

**Q50.** Why write shift(1).rolling(7).mean() instead of rolling(7).mean()?

**Answer:** So the feature for day t only uses data available before day t

An un-shifted window silently includes today's own target value, the most common time-series ML bug.

**Q51.** What do lag features capture?

**Answer:** Earlier values of the series, such as the value 1, 2 or 7 days ago

They are created with shift(lag).

**Q52.** What does TimeSeriesSplit do?

**Answer:** Multiple folds that each still respect time order

Training windows grow with each fold.

### [04 · NLP Basics — Text Features &amp; Classification](level-2/04-nlp-basics.md)

**Q53.** What is a bag of words?

**Answer:** Each document as a vector of word counts, with word order thrown away

CountVectorizer builds the vocabulary and counts.

**Q54.** What problem does TF-IDF address?

**Answer:** Raw counts over-value words that are frequent everywhere; it rescales by how distinctive a word is across the corpus

It down-weights common words.

**Q55.** Why call tfidf.transform (not fit_transform) on the test set?

**Answer:** To reuse the training vocabulary and IDF

Fitting on test data would leak.

**Q56.** What does the lesson do with the classifier's coefficients?

**Answer:** Inspect the highest and lowest to see the most 'space' and most 'baseball' words

It reveals what the model actually learned.

### [05 · CNNs for Image Classification](level-2/05-cnns-images.md)

**Q57.** Why are CNNs better than plain MLPs for images?

**Answer:** They exploit spatial structure instead of treating pixels as a flat list

Nearby pixels are related.

**Q58.** What shape is a FashionMNIST image tensor?

**Answer:** (1, 28, 28): channels, height, width

One grayscale channel; color photos would have 3.

**Q59.** What does nn.MaxPool2d(2) do?

**Answer:** Halves height and width

28×28 becomes 14×14, then 7×7.

**Q60.** What does the final layer output?

**Answer:** Raw logits for 10 classes

CrossEntropyLoss takes logits.

### [06 · Transfer Learning Basics](level-2/06-transfer-learning.md)

**Q61.** What is transfer learning?

**Answer:** Reusing a network pretrained on a huge dataset and adapting it to a smaller task

It needs far less data and training time.

**Q62.** How many parameters train when only the new 3-class head is trainable?

**Answer:** 1,539 (512 × 3 + 3)

The other ~11.2M parameters stay frozen.

**Q63.** Why must you use the same normalization statistics as pretraining?

**Answer:** Differently scaled images shift activations away from the ranges the filters were tuned for, destroying the benefit

Use weights.transforms().

**Q64.** What does requires_grad = False do?

**Answer:** Freezes a weight so it isn't updated

It freezes the backbone.

### [07 · Model Interpretability](level-2/07-model-interpretability.md)

**Q65.** What does permutation importance measure?

**Answer:** The drop in performance when a feature's values are shuffled, breaking its relationship to the target

It is model-agnostic and computed on held-out data.

**Q66.** Why can feature_importances_ be misleading?

**Answer:** It counts how often a feature is used for splitting, which can be biased toward high-cardinality features

Permutation importance is computed on X_test instead.

**Q67.** What does partial dependence show that importance does not?

**Answer:** The shape of the relationship between a feature and the prediction

For MedInc, value rises steeply then flattens: diminishing returns.

**Q68.** Why does interpretability matter in credit, healthcare and hiring?

**Answer:** A model that predicts well but can't explain why is a liability in regulated domains

It's also a debugging aid everywhere else.

### [08 · Imbalanced Data &amp; Advanced Evaluation](level-2/08-imbalanced-data.md)

**Q69.** Why apply resampling only to the training set?

**Answer:** The test set must stay a faithful sample of real-world class proportions for an honest estimate

Never resample X_test.

**Q70.** What does SMOTE do?

**Answer:** Synthesizes new minority-class rows

RandomOverSampler duplicates; RandomUnderSampler drops majority rows.

**Q71.** What trade-off do the resampling results show versus the baseline?

**Answer:** Recall rises sharply (0.18 → ~0.8) while precision falls

Baseline precision 0.833 with recall 0.179.

**Q72.** Why choose a decision threshold deliberately?

**Answer:** scikit-learn's default of 0.5 may not suit the cost of errors

ROC and precision-recall curves evaluate across all thresholds.

### [09 · Recommender System Fundamentals](level-2/09-recommender-systems.md)

**Q73.** What is the idea of user-based collaborative filtering?

**Answer:** Find users with similar rating patterns and recommend what they liked that you haven't seen

The lesson uses cosine similarity.

**Q74.** How sparse are real recommender datasets typically?

**Answer:** More than 95% sparse

Most users have rated only a handful of items.

**Q75.** What does matrix factorization learn?

**Answer:** A small vector of latent factors per user and item whose dot product approximates observed ratings

Neighborhood methods scale poorly and struggle with sparsity.

**Q76.** What fallback does predict_rating use when no neighbor rated the item?

**Answer:** The item's global average rating

It avoids returning nothing.

### [10 · Project — End-to-End Modeling Workflow](level-2/10-project-modeling-workflow.md)

**Q77.** Why does the churn project use average_precision for the search score?

**Answer:** It suits an imbalanced (~80/20) classification problem

RandomizedSearchCV scores with it.

**Q78.** What does scale_pos_weight do in XGBoost here?

**Answer:** Counteracts the 80/20 class imbalance

Candidates include 1, 3 and 4.

**Q79.** What does ColumnTransformer allow?

**Answer:** Scaling numeric columns and one-hot encoding the categorical column in one preprocessing step

has_addon is passed through unchanged with remainder="passthrough".

**Q80.** Which factors drive churn in the simulated data?

**Answer:** Tenure, contract type and support tickets

Month-to-month contracts and tickets raise churn; tenure lowers it.

## Level 3 · Advanced

### [01 · Transformers &amp; Attention from Scratch](level-3/01-transformers-attention.md)

**Q81.** What is self-attention?

**Answer:** Every position in a sequence looks at every other position and decides how much to attend to it

It is the core idea of the transformer.

**Q82.** In the from-scratch code, what do Q, K and V represent?

**Answer:** What I'm looking for, what I offer, and what I contribute if attended to

Scores are Q @ K.T scaled by sqrt(d_model), then softmaxed.

**Q83.** What does each row of the attention weights satisfy?

**Answer:** It sums to 1, a distribution over all tokens

The output is a weighted average of all value vectors.

**Q84.** Why use multi-head attention?

**Answer:** Parallel smaller attentions let the model attend to different kinds of relationships at once

d_model must be divisible by n_heads.

### [02 · Modern NLP with Hugging Face](level-3/02-huggingface-nlp.md)

**Q85.** Why do modern tokenizers rarely hit words they can't represent?

**Answer:** Rare words are split into known subword pieces instead of an out-of-vocabulary token

For example 'unfamiliarly' becomes un, ##familiar, ##ly.

**Q86.** Why fine-tune a pretrained transformer rather than train from zero?

**Answer:** Nobody retrains a transformer from scratch for every task; Hugging Face provides pretrained models and a standard workflow

The module covers tokenizers, loading a model and fine-tuning for classification.

**Q87.** What does torch.no_grad() do during inference?

**Answer:** Skips gradient tracking

It's used while computing logits.

**Q88.** What does the lesson use for the fine-tuning dataset size?

**Answer:** A shuffled subset: 2000 training and 500 test IMDB reviews

Small subsets keep the example fast.

### [03 · Advanced PyTorch Patterns](level-3/03-advanced-pytorch.md)

**Q89.** Which two methods does a custom Dataset require?

**Answer:** __len__ and __getitem__

DataLoader handles batching, shuffling and parallel loading.

**Q90.** What does num_workers &gt; 0 do?

**Answer:** Loads data in parallel background worker processes so the GPU isn't waiting on I/O

It keeps the GPU fed.

**Q91.** What does CosineAnnealingLR do?

**Answer:** Smoothly decays the learning rate along a cosine curve toward 0 over T_max epochs

No manual step-function tuning is needed.

**Q92.** When is scheduler.step() called in the example?

**Answer:** Once per epoch, after all batches

The learning rate falls from 1e-2 toward 0.

### [04 · Experiment Tracking (MLflow &amp; W&amp;B)](level-3/04-experiment-tracking.md)

**Q93.** Why track experiments with MLflow?

**Answer:** You can't compare 40 runs by memory; logging makes them reproducible and comparable

It logs parameters, metrics and artifacts.

**Q94.** What does each start_run() create?

**Answer:** A new, independently addressable record of params, metrics and the model, timestamped

Runs live inside a named experiment.

**Q95.** How do you view the recorded runs?

**Answer:** mlflow ui

The lesson prints this hint.

**Q96.** What does client.search_runs(..., order_by=["metrics.accuracy DESC"]) return?

**Answer:** Past runs sorted by accuracy, best first

It lets you query runs programmatically.

### [05 · MLOps Foundations](level-3/05-mlops-foundations.md)

**Q97.** Why isn't git alone enough for ML projects?

**Answer:** Multi-gigabyte datasets and model checkpoints aren't cheap to diff

DVC tracks large files and stores lightweight pointers in git.

**Q98.** What does data/train.csv.dvc contain?

**Answer:** A small pointer: a content hash and size, tracked by git

The actual file lives in the DVC remote, fetched with dvc pull.

**Q99.** What does dvc repro run?

**Answer:** Only the stages whose dependencies changed since the last run

Stages declare deps, outs, params and metrics in dvc.yaml.

**Q100.** Why pin versions in requirements.txt and use a Dockerfile?

**Answer:** 'Works on my machine' becomes 'works in this exact, reproducible container'

The same image runs on a laptop, CI and production.

### [06 · Hyperparameter Optimization at Scale](level-3/06-hyperparameter-optimization.md)

**Q101.** What is random search's blind spot?

**Answer:** Trial 50 learns nothing from trials 1–49, since samples are independent of past results

Bayesian optimization searches more intelligently.

**Q102.** Why sample learning_rate with log=True?

**Answer:** The effect of 0.01 vs 0.02 is much larger than 0.20 vs 0.21

A log scale matches how learning rate matters.

**Q103.** What does pruning do in Optuna?

**Answer:** Abandons a trial once it's clearly worse than the best so far, without finishing training

It monitors intermediate performance for iteratively trained models.

**Q104.** What does direction="maximize" mean in the study?

**Answer:** Optuna seeks the highest objective value, here CV ROC AUC

The sampler is TPE with a seed for reproducibility.

### [07 · Distributed &amp; Accelerated Training](level-3/07-distributed-training.md)

**Q105.** What do GPUs speed up in neural network training?

**Answer:** Large matrix multiplications, parallelized across thousands of cores

Every nn.Linear and attention Q @ K.T is a matmul.

**Q106.** Why call torch.cuda.synchronize() when timing?

**Answer:** GPU ops are asynchronous, so you wait for completion before reading the clock

Otherwise timing would be wrong.

**Q107.** Why does production code prefer DistributedDataParallel over DataParallel?

**Answer:** One process per GPU with no single bottleneck device, unlike DataParallel's gradient aggregation on one GPU

It's launched with torchrun --nproc_per_node=4.

**Q108.** What is data parallelism?

**Answer:** Replicate the model and split the batch across devices

Each process handles a different data shard.

### [08 · Model Compression &amp; Efficiency](level-3/08-model-compression.md)

**Q109.** What are the three standard compression techniques?

**Answer:** Quantization, pruning and knowledge distillation

Fewer bits per number, fewer weights, and a smaller model taught to mimic a bigger one.

**Q110.** What does quantize_dynamic do to Linear weights?

**Answer:** Converts 32-bit floats to 8-bit integers, cutting weight memory roughly 4x with usually minor accuracy loss

The int8 and fp32 outputs differ slightly but not zero.

**Q111.** What does prune.l1_unstructured(layer, 'weight', amount=0.4) do?

**Answer:** Zeroes the smallest 40% of weights by magnitude

Small-magnitude weights contribute least to the layer's output.

**Q112.** Why call prune.remove(layer, 'weight')?

**Answer:** To bake the mask in permanently and drop the pruning wrapper

It finalizes the pruned weights.

### [09 · Advanced Computer Vision](level-3/09-advanced-computer-vision.md)

**Q113.** What does an object detection model output?

**Answer:** A variable-length list of (box, label, score) triples

Unlike classification, it's one triple per detected object.

**Q114.** What does semantic segmentation produce?

**Answer:** A class label per pixel

argmax(1) collapses the 21-channel score map to a class index per pixel.

**Q115.** Why filter predictions with scores &gt; 0.7?

**Answer:** To keep only confident detections

It drops low-confidence boxes.

**Q116.** What does IoU measure for a detected box?

**Answer:** Overlap between two boxes: intersection area over union area

The worked example computes it from box coordinates.

### [10 · Project — Reproduce a Research Paper](level-3/10-project-paper-reproduction.md)

**Q117.** Which paper's claim does the capstone reproduce?

**Answer:** Dropout: A Simple Way to Prevent Neural Networks from Overfitting (Srivastava et al., 2014)

Its central claim is testable in minutes.

**Q118.** What is the claim being tested?

**Answer:** A network with dropout generalizes better (smaller train/test gap) than an identical one without it

Same data, architecture and optimizer.

**Q119.** Why must the comparison keep everything else identical?

**Answer:** So any difference is attributable to dropout

The experiment uses the same seed, lr and epochs with dropout_p of 0.0 vs 0.5.

**Q120.** Why call model.eval() when measuring accuracy?

**Answer:** To turn dropout off during evaluation

Dropout should only be active during training.

## Level 4 · Master

### [01 · Production ML System Design](level-4/01-production-ml-design.md)

**Q121.** What is training/serving skew?

**Answer:** The model is trained on one feature distribution but served another because of mismatched feature code

It is the single most common source of production ML failures.

**Q122.** In the lesson's example, what bug causes the skew?

**Answer:** The serving path substitutes a 90-day cached value for avg_purchase_30d

It happens with no error or crash.

**Q123.** What is a feature store for?

**Answer:** Computing features consistently for both training and online lookup

It reduces the two-codebases problem.

**Q124.** What closes the loop in the production architecture?

**Answer:** Online monitoring and feedback into new training data

Predictions feed back into monitoring.

### [02 · Model Serving at Scale](level-4/02-model-serving.md)

**Q125.** What does pydantic do in the FastAPI serving example?

**Answer:** Validates the request shape and types, returning a clear 422 for malformed requests before they reach the model

PredictRequest and PredictResponse define the schema.

**Q126.** Why export a model to ONNX?

**Answer:** Portable, optimized inference that runs from Python, C++ or mobile without PyTorch installed

ONNX Runtime applies framework-agnostic optimizations like operator fusion.

**Q127.** What does dynamic_axes batch_size allow?

**Answer:** The exported model to accept varying batch sizes

It marks the batch dimension as variable.

**Q128.** What is the point of batching requests?

**Answer:** Higher throughput by running several predictions together within a max size and wait time

BatchPredictor uses max_batch_size and max_wait_ms.

### [03 · Feature Stores &amp; Data Engineering for ML](level-4/03-feature-stores-data-eng.md)

**Q129.** What does a feature store eliminate?

**Answer:** Duplicated feature logic: one definition computed once and served from batch (training) and low-latency (online) paths

It directly addresses training/serving skew.

**Q130.** How do the offline and online stores differ?

**Answer:** Offline keeps full point-in-time history for training; online keeps only the current value per entity for live predictions

Each holds what its use case needs.

**Q131.** What does the naive feature lookup do wrong for training?

**Answer:** It grabs the latest value, ignoring when the label happened, leaking the future

For a label on Jan 20 it returned 55.3 from Feb instead of 42.1.

**Q132.** What is point-in-time correctness?

**Answer:** Using only feature values that existed before the label's timestamp

The lesson compares it to the rolling-window leakage bug.

### [04 · CI/CD for Machine Learning](level-4/04-ci-cd-ml.md)

**Q133.** Why isn't code-only CI enough for ML?

**Answer:** ML fails in ways like silently dropped features, valid-but-worse models and shifting metric behavior

You also test data and model behavior.

**Q134.** What does test_model_beats_naive_baseline check?

**Answer:** The model beats predicting the majority class by a margin

It asserts accuracy exceeds baseline + 0.05.

**Q135.** What does the invariance test show?

**Answer:** Changing an irrelevant feature like customer ID shouldn't change predictions

It compares predictions before and after perturbing the ID.

**Q136.** What does the no-regression test guard against?

**Answer:** Deploying a retrained model that is worse than the production model

It allows at most a 0.01 drop in accuracy.

### [05 · Monitoring, Drift Detection &amp; Retraining](level-4/05-monitoring-drift.md)

**Q137.** Why can a model that passed every CI gate still degrade?

**Answer:** The world keeps changing after deployment while the model stays frozen

You need monitoring and drift detection.

**Q138.** Why log features, predictions and model version at prediction time?

**Answer:** Without the actual production inputs, drift detection is impossible after the fact

You compare production inputs to the training distribution.

**Q139.** What does PSI measure?

**Answer:** How much a feature's production distribution has shifted from the training distribution

PSI &lt; 0.1 means no significant drift; 0.1–0.25 moderate; above that is serious.

**Q140.** What does 'close the loop' mean here?

**Answer:** Automated retraining triggers when drift is detected

Detect statistically, not by gut feel.

### [06 · A/B Testing &amp; Online Evaluation](level-4/06-ab-testing-online-eval.md)

**Q141.** Why assign variants by hashing (user_id, experiment_name)?

**Answer:** It's stateless, stable per user, and gives independent splits across experiments

The experiment name acts as a salt.

**Q142.** Why compute required sample size before launching?

**Answer:** To know how long the test must run and whether the effect is even detectable at your traffic

The example needs 14,057 users per variant for a 1pp lift at 80% power.

**Q143.** What is the risk of early peeking?

**Answer:** Fooling yourself with significance you don't actually have

The lesson stresses reading results without overclaiming.

**Q144.** What do alpha=0.05 and power=0.8 mean in sizing?

**Answer:** 5% false-positive tolerance and 80% chance to detect the true effect

They set the z values used in the sample-size formula.

### [07 · Responsible AI — Fairness, Privacy, Security](level-4/07-responsible-ai.md)

**Q145.** What is demographic parity?

**Answer:** Equal selection rates across groups

Equal opportunity is equal true positive rates.

**Q146.** Why does a fairness report compute both demographic parity and equal opportunity?

**Answer:** They're different, sometimes mutually exclusive, definitions: a model can satisfy one and violate the other

The lesson doesn't pick one number to optimize.

**Q147.** How can an accurate model still fail?

**Answer:** It can systematically fail a subgroup, leak information about individuals, or be tricked by a crafted input

These are things you compute and test, not just ethics theory.

**Q148.** What does the Laplace mechanism in differential privacy do?

**Answer:** Adds calibrated noise scaled by sensitivity and epsilon

It is the lesson's basic mechanics of differential privacy.

### [08 · Cost Optimization &amp; Efficient Inference](level-4/08-efficient-inference.md)

**Q149.** What variable does this module target compared with Module 02?

**Answer:** Total infrastructure cost per served prediction, rather than per-request latency

It uses quantization, distillation and caching.

**Q150.** What did int8 dynamic quantization achieve in the example?

**Answer:** About 4x smaller (42.3MB → 10.9MB) and faster (1.842ms → 0.612ms)

The accuracy cost must be validated, not assumed negligible.

**Q151.** What does the distillation loss combine?

**Answer:** Matching the teacher's soft output distribution (KL divergence) and getting the true label right

alpha weights the distill term; temperature softens the distributions.

**Q152.** Why multiply the distillation term by temperature**2?

**Answer:** To keep gradient magnitudes comparable after softening

It's part of the standard formulation used in the lesson's code.

### [09 · Building an ML Platform](level-4/09-ml-platform-practices.md)

**Q153.** What is an ML platform?

**Answer:** Shared infrastructure that lets many teams train, serve and monitor many models without reinventing the basics

It standardizes a model registry, self-serve training/serving and multi-tenant isolation.

**Q154.** What does a model registry provide?

**Answer:** A single source of truth for what's deployed where

Instead of a spreadsheet or a teammate's memory.

**Q155.** What happens to the old production version when a new one is promoted?

**Answer:** It's moved to ARCHIVED; only one PRODUCTION version at a time

The promote method enforces this.

**Q156.** What stage does a newly registered model start in?

**Answer:** STAGING

register() creates it as STAGING.

### [10 · Capstone — Operate a Production ML Service](level-4/10-capstone-production-service.md)

**Q157.** What service does the capstone build?

**Answer:** A fraud-scoring API

It's trained, tested, served, drift-monitored, CI/CD-gated and A/B tested.

**Q158.** What does the /score endpoint log for each prediction?

**Answer:** Timestamp, model version, features and fraud probability, for later drift detection

It appends JSON lines to prediction_log.jsonl.

**Q159.** When is a transaction flagged?

**Answer:** When the fraud probability exceeds 0.5

flagged = proba &gt; 0.5.

**Q160.** Why does train.py use a stratified split and fixed seeds?

**Answer:** Reproducible training that preserves class balance

It follows the reproducible pattern from earlier modules.

## Mixed questions across all levels

**Q161.** What shape does `np.array([[1, 2, 3], [4, 5, 6]])` have?

**Answer:** (2, 3)

2 rows and 3 columns; shape is the most-checked property in ML code.

**Q162.** For `M = np.array([[1., 2., 3.], [4., 5., 6.]])`, what does `M.mean(axis=0)` return?

**Answer:** [2.5 3.5 4.5]

axis=0 goes down the columns, giving one mean per column.

**Q163.** Which line is leakage-free?

**Answer:** `pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression())]); cross_val_score(pipe, X, y, cv=5)`

Inside a Pipeline the scaler is re-fit on each fold's training portion only.

**Q164.** You evaluate a 1-nearest-neighbor model on its own training set and get 100%. What does it tell you?

**Answer:** Nothing: it memorized the data; evaluate on held-out data

A model can simply memorize training data.

**Q165.** Which call keeps the class ratio the same in train and test?

**Answer:** `train_test_split(X, y, stratify=y)`

stratify=y preserves class proportions.

**Q166.** A fraud model has 3% positives. It scores 97% accuracy by predicting 'no fraud' always. What should you use instead?

**Answer:** Precision/recall and PR curves, with a deliberately chosen threshold

Accuracy lies on imbalanced data.

**Q167.** Which technique may you apply to the test set when handling imbalance?

**Answer:** None: resampling is only for the training set

The test set must keep real-world class proportions.

**Q168.** k-means with k=4 gives inertia 942.3 and with k=5 gives 868.7, after 3673.8 at k=3. What does the elbow suggest?

**Answer:** k=4, because improvements level off after it

Big drops until k=4, then marginal gains.

**Q169.** You fit a `StandardScaler` before `KMeans` because...

**Answer:** k-means is pure distance math, so feature scale would dominate

Unscaled large-range features dominate distances.

**Q170.** Which metric tells you how much of the target's variance a regression explains?

**Answer:** R²

R² compares your model with predicting the mean.

**Q171.** A random forest differs from plain bagging because it...

**Answer:** considers only a random subset of features at each split

That extra randomness decorrelates the trees.

**Q172.** Gradient boosting differs from a random forest because...

**Answer:** trees are built sequentially, each correcting the errors so far

Boosting fits residuals step by step.

**Q173.** For time-series features you write `df['sales'].shift(1).rolling(7).mean()`. Why shift(1)?

**Answer:** So the feature for day t only uses data before day t

An unshifted rolling window includes today's own target value, leaking the label.

**Q174.** You split a daily-sales dataset for forecasting. Which split is valid?

**Answer:** Train on earlier dates, test on later dates (TimeSeriesSplit)

A random split lets the model train on the future.

**Q175.** In PyTorch, `w = torch.tensor(3.0, requires_grad=True); loss = (w - 5) ** 2; loss.backward()` leaves `w.grad` equal to...

**Answer:** -4

d/dw (w−5)² = 2(w−5) = −4 at w = 3.

**Q176.** In a training loop, what is the correct order inside each step?

**Answer:** optimizer.zero_grad(); loss.backward(); optimizer.step()

Clear gradients, compute new ones, then update the weights.

**Q177.** Why freeze the backbone with `requires_grad = False` in transfer learning?

**Answer:** So only the new head trains and the pretrained features are kept

Only the new final layer's parameters are updated.

**Q178.** A teammate's serving code computes `avg_purchase_30d` differently from training. What is this called?

**Answer:** Training/serving skew

Duplicated feature logic in two code paths is a top production failure.

**Q179.** Which tool compares the production distribution of a feature with its training distribution?

**Answer:** Population Stability Index (PSI)

PSI above about 0.25 signals significant drift.

**Q180.** You want to deploy a PyTorch model to run without PyTorch installed. What do you export to?

**Answer:** ONNX

ONNX Runtime serves the model framework-independently.

## Practical coding questions

### Train/test split by hand

Without any libraries, split the list `data = list(range(10))` into a training list (first 80%) and a test list (last 20%) in order, as you would for time-ordered data, and print both lists.

Starter code:

```
data = list(range(10))
# split in order: first 80% train, last 20% test
# print(train) then print(test)
```

Expected output:

```text
[0, 1, 2, 3, 4, 5, 6, 7]
[8, 9]
```

??? success "Show a solution"

    ```
    data = list(range(10))
    cut = int(len(data) * 0.8)
    train, test = data[:cut], data[cut:]
    print(train)
    print(test)
    ```

    For time-ordered data you split by position, never randomly, so the model never trains on the future.

### Mean absolute error

Write a function `mae(y_true, y_pred)` in plain Python that returns the mean absolute error, and print it for `y_true = [3, -0.5, 2, 7]` and `y_pred = [2.5, 0.0, 2, 8]`.

Starter code:

```
def mae(y_true, y_pred):
    # your code here
    pass

print(mae([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]))
```

Expected output:

```text
0.5
```

??? success "Show a solution"

    ```
    def mae(y_true, y_pred):
        return sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_true)

    print(mae([3, -0.5, 2, 7], [2.5, 0.0, 2, 8]))
    ```

    The absolute errors are 0.5, 0.5, 0 and 1, which sum to 2 and average to 0.5.

---

More free practice: the [AI & Machine Learning Mastery Path](index.md) curriculum, the [Interview Prep Pack](https://learnmasterypath.com/interview-pack/) and the [free resume ATS check](https://learnmasterypath.com/resume/).
