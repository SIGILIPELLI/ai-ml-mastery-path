---
description: "Building an ML Platform — Every module so far solved one team's problem for one model. An ML platform is the shared infrastructure that lets many teams…"
---

# 09 · Building an ML Platform

Every module so far solved one team's problem for one model. An ML
platform is the shared infrastructure that lets *many* teams train, serve,
and monitor *many* models without each one reinventing Modules 01-08. This
module covers what a platform actually standardizes: a model registry,
a self-serve training/serving abstraction, and multi-tenant isolation.

## A model registry: the single source of truth for "what's deployed where"

```python
# registry.py — a minimal model registry backed by a database table
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class Stage(Enum):
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"

@dataclass
class ModelVersion:
    name: str
    version: int
    stage: Stage
    artifact_uri: str          # e.g. s3://models/fraud-detector/v14/model.pkl
    metrics: dict
    registered_at: datetime
    registered_by: str

class ModelRegistry:
    def __init__(self):
        self._versions: dict[str, list[ModelVersion]] = {}

    def register(self, name: str, artifact_uri: str, metrics: dict, registered_by: str) -> ModelVersion:
        existing = self._versions.setdefault(name, [])
        version = len(existing) + 1
        mv = ModelVersion(name, version, Stage.STAGING, artifact_uri, metrics,
                           datetime.utcnow(), registered_by)
        existing.append(mv)
        return mv

    def promote(self, name: str, version: int, stage: Stage):
        for mv in self._versions[name]:
            if mv.stage == stage and stage == Stage.PRODUCTION:
                mv.stage = Stage.ARCHIVED   # only one PRODUCTION version at a time
            if mv.version == version:
                mv.stage = stage

    def get_production_model(self, name: str) -> ModelVersion | None:
        return next((mv for mv in self._versions.get(name, [])
                     if mv.stage == Stage.PRODUCTION), None)

registry = ModelRegistry()
v1 = registry.register("fraud-detector", "s3://models/fraud-detector/v1/", {"auc": 0.91}, "alice")
registry.promote("fraud-detector", version=1, stage=Stage.PRODUCTION)
print(registry.get_production_model("fraud-detector"))
```

This is the same promotion concept as Module 04's CI/CD gate, generalized
across every model any team on the platform trains — instead of each team
tracking "which model is live" in a spreadsheet or a teammate's memory, the
registry is the one place that answers it programmatically, for any
downstream tool (serving, monitoring, rollback) to query.

## A self-serve training abstraction

```yaml
# train_config.yaml — what a team submits; the platform handles the rest
model_name: fraud-detector
framework: sklearn
entry_point: train.py
compute:
  instance_type: cpu.large
  max_runtime_minutes: 60
data:
  source: s3://data-lake/fraud/training/
tracking:
  experiment_name: fraud-detector-experiments
```

```python
# platform/submit_job.py — what the platform does with that config
def submit_training_job(config: dict, submitted_by: str) -> str:
    job_id = f"job-{config['model_name']}-{int(time.time())}"

    provision_compute(config["compute"]["instance_type"])          # Module 01/02 territory
    container = build_training_container(config["entry_point"])    # Module 05's Dockerfile pattern
    run_with_tracking(container, config["tracking"]["experiment_name"])  # Module 04 experiment tracking

    audit_log.write({"job_id": job_id, "submitted_by": submitted_by, "config": config})
    return job_id
```

A team writes `train.py` and a small YAML file; the platform handles
provisioning, containerization, experiment tracking, and audit logging
uniformly — the same infrastructure code serves every model, so
improvements (a faster base image, a new tracking integration) benefit
every team at once instead of requiring N teams to each adopt it.

## Multi-tenant isolation: resource quotas

```python
class TenantQuotaManager:
    def __init__(self):
        self.quotas = {}       # tenant -> {"cpu_hours": limit, "storage_gb": limit}
        self.usage = {}        # tenant -> current usage

    def set_quota(self, tenant: str, cpu_hours: float, storage_gb: float):
        self.quotas[tenant] = {"cpu_hours": cpu_hours, "storage_gb": storage_gb}
        self.usage.setdefault(tenant, {"cpu_hours": 0.0, "storage_gb": 0.0})

    def can_submit_job(self, tenant: str, estimated_cpu_hours: float) -> tuple[bool, str]:
        quota = self.quotas.get(tenant)
        if quota is None:
            return False, f"no quota configured for tenant {tenant!r}"
        used = self.usage[tenant]["cpu_hours"]
        if used + estimated_cpu_hours > quota["cpu_hours"]:
            return False, (
                f"would exceed cpu_hours quota: {used:.1f} + {estimated_cpu_hours:.1f} "
                f"> {quota['cpu_hours']:.1f}"
            )
        return True, "within quota"

    def record_usage(self, tenant: str, cpu_hours: float):
        self.usage[tenant]["cpu_hours"] += cpu_hours

qm = TenantQuotaManager()
qm.set_quota("team-fraud", cpu_hours=500, storage_gb=1000)
qm.record_usage("team-fraud", cpu_hours=470)
print(qm.can_submit_job("team-fraud", estimated_cpu_hours=50))
# (False, 'would exceed cpu_hours quota: 470.0 + 50.0 > 500.0')
```

## Worked example: routing a prediction request through a multi-model platform

```python
class PlatformRouter:
    """Given a model name, look up the current production version from the
    registry and route the request to its serving endpoint -- the layer
    that lets 'call fraud-detector' stay stable while versions rotate underneath."""

    def __init__(self, registry: ModelRegistry, endpoint_map: dict[str, str]):
        self.registry = registry
        self.endpoint_map = endpoint_map   # artifact_uri -> live endpoint URL

    def route(self, model_name: str, features: dict) -> dict:
        prod = self.registry.get_production_model(model_name)
        if prod is None:
            raise ValueError(f"no production version registered for {model_name!r}")

        endpoint = self.endpoint_map.get(prod.artifact_uri)
        if endpoint is None:
            raise RuntimeError(f"registry says v{prod.version} is production but no live endpoint found")

        return {"routed_to": endpoint, "model_version": prod.version, "request": features}

router = PlatformRouter(registry, {"s3://models/fraud-detector/v1/": "http://serving-v1.internal:8080"})
print(router.route("fraud-detector", {"amount": 250.0, "merchant_category": "electronics"}))
```

Callers depend only on the model *name* ("fraud-detector"), never a
specific version or endpoint URL — a promotion in the registry
(`registry.promote(...)`) instantly changes what every caller routes to,
without any caller-side code change, which is the platform doing exactly
what Module 04's canary rollout needs at the infrastructure layer.

## Cheat sheet

| Platform concern | Building block |
|---|---|
| "What's deployed where, and who owns it" | Model registry with staged promotion |
| Teams self-serve training without reinventing infra | Declarative config + shared job-submission pipeline |
| Fair resource sharing across teams | Per-tenant quotas enforced before job submission |
| Stable caller-facing names across version churn | A router that resolves name → current production endpoint |

## How It Actually Works

**A registry's value comes from being the single place that decouples
"which version exists" from "which version is live," which every other
platform piece can then depend on instead of re-deriving.** Without a
registry, "what's in production" lives implicitly in whatever config was
last deployed to a server — invisible to a monitoring job, a rollback
script, or a new teammate. `get_production_model` gives every consumer
(the router, a monitoring dashboard, an audit tool) one authoritative
query instead of each reimplementing "figure out what's live" by
inspecting running infrastructure. The `promote` method's invariant — only
one `PRODUCTION` version per model name at a time, enforced by
auto-archiving the previous one — is what makes `get_production_model`
safe to treat as returning a single unambiguous answer rather than a list
callers have to disambiguate themselves.

**The self-serve training abstraction works by pushing the *variable* part
(what the model does) into a small user-owned file, while keeping the
*invariant* part (how a job runs) as shared platform code.** `train.py`
and `train_config.yaml` are the only things a team writes; provisioning,
containerization (Module 05's Dockerfile pattern generalized to build any
`entry_point`), and tracking integration all live in
`platform/submit_job.py`, owned by the platform team. This is a deliberate
inversion from Module 05's single-team Dockerfile: there, one team's
Dockerfile encodes one model's environment; here, the *pattern* is
extracted into shared infrastructure so a fix or upgrade (e.g. a new base
image with a security patch) is applied once, centrally, and takes effect
for every team's next job submission — the platform is precisely the
generalization of per-team plumbing into a shared, versioned service.

**Quota enforcement has to happen *before* job submission, not after,
because compute is far more expensive to claw back than to withhold.**
`can_submit_job` checks projected usage against the quota *before* any
compute is provisioned — rejecting a job that would exceed quota costs
nothing, whereas discovering an over-quota situation after a job has
already consumed 400 CPU-hours means that cost is sunk regardless of what
happens next. This mirrors the same "gate before the expensive step"
pattern as Module 04's CI/CD pipeline (data validation runs before
training, not after) and Module 06's sample-size calculation (computed
before running the experiment, not after) — a recurring principle across
this whole level: whenever an expensive or hard-to-reverse step is
involved, the check that could prevent wasting it belongs strictly before
it, not as a post-hoc audit.

## Exercise

Extend `PlatformRouter.route` to fall back to the most recent `STAGING`
version (with a logged warning) if no `PRODUCTION` version is registered,
instead of raising. Then extend `TenantQuotaManager` with a second
resource dimension, `gpu_hours`, with a stricter default quota than
`cpu_hours`, and explain — referencing the "gate before the expensive
step" principle above — why a platform would deliberately set tighter
default quotas on GPU time than CPU time even for teams who haven't
requested either yet.
