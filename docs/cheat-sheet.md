---
title: "AI & Machine Learning Cheat Sheet: Quick Reference, Beginner to Advanced"
description: "A free AI & Machine Learning cheat sheet: 40 quick-reference tables from every level of the AI & Machine Learning Mastery Path, each linked to its full lesson."
---

# AI & Machine Learning Cheat Sheet

Quick-reference tables from **40 lessons** of the free AI & Machine Learning Mastery Path, in course order from beginner to advanced. Each section links to the full lesson with explanations and runnable examples.

## Level 1 · Entry

### [01 · Setup &amp; the Scientific Python Stack](level-1/01-setup-scientific-python.md)

| Command | Purpose |
|---------|---------|
| `python3 -m venv .venv` | Create a virtual environment in `.venv/`. |
| `source .venv/bin/activate` | Activate it (prompt shows `(.venv)`). |
| `pip install numpy pandas scikit-learn matplotlib torch` | Install the full course stack. |
| `pip freeze > requirements.txt` | Snapshot exact versions for reproducibility. |
| `jupyter notebook` | Launch the notebook interface. |
| `model.fit(X_train, y_train)` | Train a scikit-learn model. |
| `model.predict(X_test)` | Predict labels for new data. |

### [02 · NumPy &amp; Pandas Essentials](level-1/02-numpy-pandas.md)

| Operation | NumPy | pandas |
|-----------|-------|--------|
| Shape of the data | `a.shape` | `df.shape` |
| First rows | `a[:5]` | `df.head()` |
| Select column | `M[:, 2]` | `df["col"]` |
| Filter rows | `M[M[:, 0] > 3]` | `df[df["col"] > 3]` |
| Per-feature mean | `M.mean(axis=0)` | `df.mean(numeric_only=True)` |
| Elementwise math | `x * 2 + 1` | `df["col"] * 2 + 1` |
| Per-group summary | — | `df.groupby("k")["v"].mean()` |
| Random numbers | `np.random.default_rng(42)` | — |
| To the other type | — | `df.to_numpy()` |

### [03 · Data Preparation](level-1/03-data-preparation.md)

| Task | Tool | Key detail |
|------|------|-----------|
| Hold out a test set | `train_test_split(X, y, test_size=0.2, random_state=42)` | Do it *first*; `stratify=y` for classification. |
| Fill missing values | `SimpleImputer(strategy="median")` | Fit on train only. |
| Drop missing rows | `df.dropna()` | Only if few rows affected. |
| Standardize features | `StandardScaler()` | Needed for k-NN, linear models, neural nets; not trees. |
| Scale to [0, 1] | `MinMaxScaler()` | Alternative to standardization. |
| Encode unordered categories | `OneHotEncoder(handle_unknown="ignore")` | One 0/1 column per category. |
| Encode ordered categories | `OrdinalEncoder(categories=[...])` | Only when order is real. |
| Learn vs. apply | `.fit()` / `.transform()` | `fit_transform` on train, `transform` on test. |

### [04 · Regression](level-1/04-regression.md)

| Task | Code |
|------|------|
| Fit linear regression | `LinearRegression().fit(X_train, y_train)` |
| Inspect the model | `model.coef_`, `model.intercept_` |
| Predict | `model.predict(X_test)` |
| Metrics | `mean_absolute_error`, `mean_squared_error`, `r2_score` |
| R² shortcut | `model.score(X_test, y_test)` |
| Baseline | `DummyRegressor(strategy="mean")` |
| Curved fits | `PolynomialFeatures(degree=d)` before the model |
| Regularized linear | `Ridge(alpha=...)`, `Lasso(alpha=...)` |
| Diagnose overfitting | Compare train vs. test scores |

### [05 · Classification](level-1/05-classification.md)

| Task | Code / rule |
|------|-------------|
| Probabilistic linear classifier | `LogisticRegression(max_iter=1000)` — scale features first |
| Rule-based, no scaling needed | `DecisionTreeClassifier(max_depth=3)` |
| Distance-based | `KNeighborsClassifier(n_neighbors=5)` — must scale |
| Get probabilities | `model.predict_proba(X)` |
| All-in-one report | `classification_report(y_test, y_pred)` |
| Error breakdown | `confusion_matrix(y_test, y_pred)` |
| "Model says yes — trust it?" | Precision |
| "Did we find them all?" | Recall |
| Imbalanced data | Never trust bare accuracy; use `stratify=`, `class_weight="balanced"` |
| Sanity baseline | `DummyClassifier(strategy="most_frequent")` |

### [06 · Clustering &amp; Unsupervised Learning](level-1/06-clustering-unsupervised.md)

| Task | Code | Notes |
|------|------|-------|
| Cluster into k groups | `KMeans(n_clusters=k, n_init="auto").fit_predict(X)` | Scale features first; IDs are arbitrary. |
| Cluster quality (per k) | `km.inertia_` (elbow), `silhouette_score(X, labels)` (peak) | Heuristics, not oracles. |
| Shape-flexible clustering + noise | `DBSCAN(eps=..., min_samples=...)` | Label `-1` = noise; tune `eps`. |
| Reduce dimensions | `PCA(n_components=2).fit_transform(X_scaled)` | Always standardize first. |
| Variance kept | `pca.explained_variance_ratio_` | `n_components=0.95` targets 95%. |
| Synthetic data | `make_blobs`, `make_moons` | Perfect for building intuition. |

### [07 · Model Evaluation &amp; Cross-Validation](level-1/07-model-evaluation.md)

| Task | Code |
|------|------|
| 5-fold CV score | `cross_val_score(model, X_train, y_train, cv=5)` |
| Report | `f"{scores.mean():.3f} +/- {scores.std():.3f}"` |
| Multiple metrics | `cross_validate(model, X, y, scoring=["accuracy","f1"])` |
| Grid search | `GridSearchCV(pipe, param_grid, cv=5, n_jobs=-1)` |
| Pipeline params in grids | `"stepname__param": [values]` |
| Best result | `search.best_params_`, `search.best_score_` |
| Cheap big-grid search | `RandomizedSearchCV(..., n_iter=50)` |
| Non-default metric | `scoring="f1"` / `"r2"` / `"neg_mean_absolute_error"` |
| Diagnose fit | Compare train score vs. CV score (the gap) |

### [08 · Feature Engineering &amp; Pipelines](level-1/08-feature-engineering-pipelines.md)

| Task | Code |
|------|------|
| Chain steps | `Pipeline([("scale", StandardScaler()), ("clf", model)])` |
| Quick anonymous version | `make_pipeline(StandardScaler(), model)` |
| Per-column-type prep | `ColumnTransformer([("num", ..., num_cols), ("cat", ..., cat_cols)])` |
| Untouched columns | `ColumnTransformer(..., remainder="passthrough")` |
| Select columns by dtype | `X.select_dtypes(include="number").columns` |
| Names after transform | `preprocess.get_feature_names_out()` |
| Tune through a pipeline | `"stepname__param"` (nested: `"prep__num__impute__strategy"`) |
| Inspect a fitted step | `pipe.named_steps["scale"].mean_` |
| Why pipelines | Leakage-proof CV + a single shippable object |

### [09 · Neural Network Fundamentals with PyTorch](level-1/09-neural-networks-pytorch.md)

| Task | Code |
|------|------|
| Tensor from data | `torch.tensor(arr, dtype=torch.float32)` |
| Labels for CrossEntropy | dtype `torch.long` |
| Track gradients | `requires_grad=True`; `loss.backward()`; `.grad` |
| Define an MLP | `nn.Sequential(nn.Linear(i,h), nn.ReLU(), nn.Linear(h,o))` |
| Classification loss | `nn.CrossEntropyLoss()` (takes raw logits) |
| Regression loss | `nn.MSELoss()` |
| Optimizer | `torch.optim.Adam(model.parameters(), lr=0.01)` |
| The loop | forward → `loss` → `zero_grad()` → `backward()` → `step()` |
| Inference mode | `model.eval()` + `with torch.no_grad():` |
| Predicted class | `logits.argmax(dim=1)` |
| Reproducibility | `torch.manual_seed(42)` |
| Mini-batches | `DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)` |

### [10 · Capstone — End-to-End Tabular ML Project](level-1/10-capstone-project.md)

| Project stage | Tool / habit |
|---------------|--------------|
| EDA | `df.describe()`, `df.corr()["target"]`, histograms |
| Shared data logic | one `prepare.py` imported by train *and* predict |
| Baselines first | `DummyRegressor`, then `LinearRegression` |
| Compare fairly | `cross_val_score(pipe, X_train, y_train, cv=5)` |
| Tune the winner | `GridSearchCV` with `model__` params |
| Final claim | test set, once, in real units |
| Save / load | `joblib.dump(pipe, "model.joblib")` / `joblib.load` |

## Level 2 · Intermediate

### [01 · Ensemble Methods Deep Dive](level-2/01-ensemble-methods.md)

| Technique | Reduces | Key knob |
|---|---|---|
| `BaggingClassifier` | Variance (overfitting) | `n_estimators`, `max_samples` |
| `RandomForestClassifier` | Variance, correlation between trees | `max_features`, `n_estimators` |
| `VotingClassifier(voting="soft")` | Variance across model *types* | which base models you pick |
| `StackingClassifier` | Both, adaptively | `final_estimator`, `cv` |

### [02 · Gradient Boosting &amp; XGBoost](level-2/02-gradient-boosting-xgboost.md)

| Knob | Effect |
|---|---|
| `n_estimators` | More trees = more correction rounds; too many overfits without early stopping |
| `learning_rate` | Shrinks each tree's contribution; smaller = needs more trees, generalizes better |
| `max_depth` | Boosted trees are shallow (3-6) by design — depth here isn't "how complex a single learner is" |
| `subsample` / `colsample_bytree` | Row/column bagging per tree, fights overfitting |
| `early_stopping_rounds` | Stop once validation metric stops improving |

### [03 · Time Series Forecasting Basics](level-2/03-time-series.md)

| Task | Code |
|---|---|
| Lag feature | `df["sales"].shift(k)` |
| Rolling stat (leak-safe) | `df["sales"].shift(1).rolling(w).mean()` |
| Time-respecting split | slice by date, or `TimeSeriesSplit` |
| Naive baseline | `shift(1)` |
| Seasonal naive baseline | `shift(period)` |
| Error metric | `mean_absolute_error`, `mean_absolute_percentage_error` |

### [04 · NLP Basics — Text Features &amp; Classification](level-2/04-nlp-basics.md)

| Task | Code |
|---|---|
| Word counts | `CountVectorizer(stop_words="english")` |
| TF-IDF weighting | `TfidfVectorizer(stop_words="english")` |
| Fit vocabulary + transform | `.fit_transform(X_train)` (train only) |
| Reuse vocabulary | `.transform(X_test)` (never `fit_transform` on test) |
| Include phrases | `ngram_range=(1, 2)` |
| Inspect learned words | `feature_names_out()` + `clf.coef_` |

### [05 · CNNs for Image Classification](level-2/05-cnns-images.md)

| Layer | Purpose |
|---|---|
| `nn.Conv2d(in_c, out_c, kernel_size, padding)` | Learn local spatial filters |
| `nn.MaxPool2d(k)` | Downsample, add translation tolerance |
| `.flatten(1)` | Turn feature maps into a vector before `nn.Linear` |
| `padding=1` with `kernel_size=3` | Keeps spatial size unchanged after the conv |

### [06 · Transfer Learning Basics](level-2/06-transfer-learning.md)

| Stage | What trains | Typical `lr` |
|---|---|---|
| Feature extraction | Only the new head | `1e-3` |
| Fine-tuning | Head + last few blocks | `1e-5`–`1e-4` |
| Full fine-tuning | Everything | `1e-5` or lower, only with enough data |

### [07 · Model Interpretability](level-2/07-model-interpretability.md)

| Question | Tool |
|---|---|
| Which features matter overall? | `permutation_importance` |
| What shape is the relationship? | `PartialDependenceDisplay` |
| Why did the model predict *this* for *this row*? | `shap.TreeExplainer` |
| Fast but potentially biased importance | `model.feature_importances_` |

### [08 · Imbalanced Data &amp; Advanced Evaluation](level-2/08-imbalanced-data.md)

| Technique | Effect |
|---|---|
| `class_weight="balanced"` | Reweights loss, no data duplication |
| `RandomOverSampler` | Duplicates minority rows |
| `SMOTE` | Synthesizes new minority rows by interpolation |
| `RandomUnderSampler` | Drops majority rows (loses data) |
| ROC AUC | Threshold-independent, but optimistic on imbalance |
| Average precision (PR AUC) | Better summary for rare-class problems |
| Custom threshold | `(proba >= t)` instead of the default 0.5 |

### [09 · Recommender System Fundamentals](level-2/09-recommender-systems.md)

| Approach | Idea | Scales to |
|---|---|---|
| User-based CF | Similar users → similar tastes | Small user bases |
| Item-based CF | Similar items → co-rated together | Slightly better scaling |
| Matrix factorization (`NMF`) | Learn compact latent factors | Large sparse matrices |
| `precision@k` / `recall@k` | Overlap between top-k and true likes | Any ranked recommender |

### [10 · Project — End-to-End Modeling Workflow](level-2/10-project-modeling-workflow.md)

| Step | Module this reuses |
|---|---|
| Realistic imbalanced target | 08 · Imbalanced Data |
| `ColumnTransformer` for mixed feature types | Level 1 · Feature Engineering |
| Boosted trees | 02 · Gradient Boosting & XGBoost |
| `RandomizedSearchCV` scored on `average_precision` | 08 · Imbalanced Data |
| SHAP explanation of the final model | 07 · Model Interpretability |

## Level 3 · Advanced

### [01 · Transformers &amp; Attention from Scratch](level-3/01-transformers-attention.md)

| Component | Role |
|---|---|
| Query / Key / Value | Learned projections deciding what to look for / offer / contribute |
| `scores = QK^T / sqrt(d_head)` | Scaled similarity between every pair of positions |
| `softmax(scores)` | Turns scores into a probability distribution per row |
| Multi-head | Several attention "views" run in parallel, concatenated |
| Positional encoding | Injects order, since attention alone is order-agnostic |
| Residual + LayerNorm | Stabilizes training in deep stacks |

### [02 · Modern NLP with Hugging Face](level-3/02-huggingface-nlp.md)

| Task | Code |
|---|---|
| Load a tokenizer | `AutoTokenizer.from_pretrained(name)` |
| Load a model for classification | `AutoModelForSequenceClassification.from_pretrained(name, num_labels=k)` |
| Tokenize a batch | `tokenizer(texts, padding=True, truncation=True, return_tensors="pt")` |
| Fine-tune | `Trainer(model, args, train_dataset, eval_dataset).train()` |
| Reuse an existing head | Omit `num_labels` when the pretrained head already matches your task |

### [03 · Advanced PyTorch Patterns](level-3/03-advanced-pytorch.md)

| Pattern | Code |
|---|---|
| Custom data source | Subclass `Dataset`, implement `__len__`/`__getitem__` |
| Parallel data loading | `DataLoader(..., num_workers=N, pin_memory=True)` |
| LR schedule | `torch.optim.lr_scheduler.CosineAnnealingLR/StepLR/OneCycleLR` |
| Mixed precision | `torch.amp.autocast()` + `GradScaler` |
| Prevent exploding gradients | `torch.nn.utils.clip_grad_norm_` |

### [04 · Experiment Tracking (MLflow &amp; W&amp;B)](level-3/04-experiment-tracking.md)

| Task | Code |
|---|---|
| Group runs | `mlflow.set_experiment(name)` |
| Start a tracked run | `with mlflow.start_run():` |
| Log hyperparameters | `mlflow.log_params({...})` |
| Log a metric | `mlflow.log_metric(name, value)` |
| Save the model artifact | `mlflow.sklearn.log_model(model, "model")` |
| Browse runs | `mlflow ui` (local web dashboard) |
| Query programmatically | `MlflowClient().search_runs(...)` |

### [05 · MLOps Foundations](level-3/05-mlops-foundations.md)

| Concern | Tool / pattern |
|---|---|
| Version large data/model files | `dvc add`, `dvc push`/`dvc pull` |
| Reproducible dependencies | Pinned `requirements.txt` |
| Reproducible runtime | `Dockerfile` |
| Pipeline with explicit stage dependencies | `dvc.yaml` + `dvc repro` |
| Skip re-running unchanged stages | Content hashing of declared deps |

### [06 · Hyperparameter Optimization at Scale](level-3/06-hyperparameter-optimization.md)

| Concept | Tool |
|---|---|
| Search space definition | `trial.suggest_int/float/categorical` |
| Search strategy | `TPESampler` (Bayesian) vs. `RandomSampler` |
| Early-kill bad trials | `optuna.pruners.MedianPruner` |
| Which params mattered | `optuna.importance.get_param_importances` |
| Log-scale search | `suggest_float(..., log=True)` for rate-like params |

### [07 · Distributed &amp; Accelerated Training](level-3/07-distributed-training.md)

| Concept | Tool |
|---|---|
| Quick multi-GPU, single process | `nn.DataParallel` (simple, has bottlenecks) |
| Production multi-GPU/multi-node | `DistributedDataParallel` + `torchrun` |
| Gradient sync across replicas | Automatic all-reduce inside `backward()` |
| Scaling learning rate with batch size | Linear scaling rule |

### [08 · Model Compression &amp; Efficiency](level-3/08-model-compression.md)

| Technique | Reduces | Typical cost |
|---|---|---|
| Dynamic quantization | Memory (~4x), inference latency | Small accuracy drop |
| Unstructured pruning | Number of nonzero weights | Sharp accuracy drop past ~60-70% |
| Knowledge distillation | Model size (student << teacher) | Needs distillation training pass |

### [09 · Advanced Computer Vision](level-3/09-advanced-computer-vision.md)

| Task | Output shape | Model family |
|---|---|---|
| Classification | 1 label per image | ResNet, ViT |
| Detection | N boxes + labels + scores per image | Faster R-CNN, YOLO |
| Segmentation | 1 label per pixel | DeepLabV3, U-Net |
| Box quality metric | IoU (0 to 1) | — |

### [10 · Project — Reproduce a Research Paper](level-3/10-project-paper-reproduction.md)

| Step | What to nail down |
|---|---|
| Pick a falsifiable claim | "X causes measurable effect Y," not "the model is good" |
| Match the essential mechanism | Here: dropout applied at training time, disabled at eval |
| Control everything else | Same architecture, optimizer, data, seed |
| Compare a clear metric | Generalization gap, not just raw accuracy |
| Sweep the key hyperparameter | Confirms the effect isn't a one-off artifact of one `p` |

## Level 4 · Master

### [01 · Production ML System Design](level-4/01-production-ml-design.md)

| Failure mode | Cause | Fix |
|---|---|---|
| Training/serving skew | Feature logic duplicated in two codebases | Shared feature functions / feature store |
| Silent data drift | Input distribution shifts after deploy | Monitoring (Module 05) |
| Stale features at inference | Online store lags the source data | Freshness SLAs, staleness checks |
| "Works in the notebook" | No parity test between offline/online paths | Parity tests in CI |

### [02 · Model Serving at Scale](level-4/02-model-serving.md)

| Concern | Tool |
|---|---|
| Request/response validation | FastAPI + `pydantic` models |
| Framework-agnostic, optimized inference | ONNX export + `onnxruntime` |
| High throughput under load | Request batching (with a max-wait bound) |
| Concurrent request handling | `uvicorn --workers N`, `async def` handlers |

### [03 · Feature Stores &amp; Data Engineering for ML](level-4/03-feature-stores-data-eng.md)

| Concept | Purpose |
|---|---|
| Offline store | Full history, point-in-time queries, training data assembly |
| Online store | Current snapshot only, low-latency lookup at inference |
| Point-in-time join | Only use feature values known *before* the label's timestamp |
| Single write path | Eliminates the duplicate-logic root cause of training/serving skew |

### [04 · CI/CD for Machine Learning](level-4/04-ci-cd-ml.md)

| Concern | Test type | Where it runs |
|---|---|---|
| Schema/null/range violations | Data validation tests | Before training |
| Model beats trivial baseline | Model behavior tests | After training |
| Model doesn't regress vs prod | Comparison gate | Before deploy |
| Gradual rollout | Canary deploy (5% → 25% → 100%) | After gate passes |
| Fresh data drift | Scheduled nightly retrain | Cron trigger |

### [05 · Monitoring, Drift Detection &amp; Retraining](level-4/05-monitoring-drift.md)

| Signal | Metric | What it catches |
|---|---|---|
| Feature distribution shift | PSI per feature | Input data changed shape |
| Prediction distribution shift | KS test on output scores | Model behaving differently on new inputs |
| Label drift (when labels arrive late) | Accuracy/AUC on delayed ground truth | Model actually getting worse, not just inputs shifting |
| Operational health | Latency, error rate, request volume | Infra problems, not model problems |

### [06 · A/B Testing &amp; Online Evaluation](level-4/06-ab-testing-online-eval.md)

| Concern | Tool |
|---|---|
| Stable, stateless user assignment | Hash `(user_id, experiment)` into buckets |
| How long to run the experiment | `required_sample_size` before launch |
| Reading results correctly | Fixed analysis at a pre-committed sample size |
| Avoiding peeking bias | Sequential testing correction, or just don't peek |

### [07 · Responsible AI — Fairness, Privacy, Security](level-4/07-responsible-ai.md)

| Concern | Metric / technique |
|---|---|
| Equal approval rates across groups | Demographic parity difference |
| Equal accuracy for those who deserve approval | Equal opportunity difference |
| Bound what a query reveals about one individual | Differential privacy (Laplace mechanism) |
| Robustness to crafted malicious inputs | Adversarial evaluation (FGSM, PGD) |
| Gate deployment on fairness, not just accuracy | Fairness check in the CI/CD pipeline |

### [08 · Cost Optimization &amp; Efficient Inference](level-4/08-efficient-inference.md)

| Lever | What it trades | Typical win |
|---|---|---|
| Quantization (fp32→int8) | Small accuracy loss | ~4x smaller, 2-3x faster |
| Distillation | Training complexity upfront | 5-10x smaller model, most of the accuracy |
| Response caching | Staleness (bounded by TTL) | Near-zero cost on cache hits |
| Batching (Module 02) | Slightly higher per-request latency | Much higher throughput per instance |

### [09 · Building an ML Platform](level-4/09-ml-platform-practices.md)

| Platform concern | Building block |
|---|---|
| "What's deployed where, and who owns it" | Model registry with staged promotion |
| Teams self-serve training without reinventing infra | Declarative config + shared job-submission pipeline |
| Fair resource sharing across teams | Per-tenant quotas enforced before job submission |
| Stable caller-facing names across version churn | A router that resolves name → current production endpoint |

### [10 · Capstone — Operate a Production ML Service](level-4/10-capstone-production-service.md)

| Stage | Module | Artifact in this capstone |
|---|---|---|
| Design the system, avoid train/serve skew | 01 | Shared feature computation in `train.py`/`serve.py` |
| Serve with a validated API | 02 | `serve.py` FastAPI endpoint |
| Feature consistency at scale | 03 | (would move to a feature store beyond this size) |
| Test data + model, gate promotion | 04 | `tests/test_service.py`, `fraud-service.yml` |
| Detect drift, trigger retraining | 05 | `nightly_drift_check` |
| Validate the change with real traffic | 06 | `run_fraud_model_experiment` |
| Check fairness before shipping | 07 | `test_fairness_gap_within_bound` |
| Control serving cost | 08 | (quantize `model.pkl` if latency/cost demands it) |
| Multi-team infrastructure | 09 | `register_model.py` against a shared registry |
| **All of the above, one decision** | — | `full_promotion_decision` |

---

Go deeper with the full [AI & Machine Learning Mastery Path](index.md) curriculum or test yourself with the [interview questions](interview-questions.md).
