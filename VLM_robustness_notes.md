## Project: VLM Robustness to Background and Context

This file collects key observations from `vlm_robustness_eval.ipynb` and ideas for extensions beyond the original proposal.

---

## 1. What CLIP is actually doing

- **Similarity model, not fixed classifier**
  - CLIP has an image encoder \(f_\theta\) and a text encoder \(g_\phi\).
  - Images and prompts are mapped into a shared embedding space; the score is (scaled) cosine similarity.
  - Inference is implemented by encoding a set of prompts and picking the one with the highest similarity to the image.
  - Prompts effectively *are* the classifier weights; changing prompt wording changes the decision boundary.

- **Prompt dependence**
  - Simple object prompts: `\"a photo of a car\"`, `\"a photo of a horse\"`, etc.
  - Relational prompts: `\"a photo of a person riding a horse\"`, `\"a photo of a person standing next to a horse\"`, etc.
  - The model does not have an explicit symbolic notion of objects/relations; everything is implicit in the learned embedding geometry.

---

## 2. Object-level robustness findings (COCO)

- **Setup**
  - Categories: initial focus on `boat`, `airplane`, `horse`, later extended to more vehicles/animals.
  - Prompts: `TEXT_PROMPTS = [f\"a photo of a {c}\" for c in CONFIG[\"categories\"]]`.
  - Perturbations (applied via COCO bboxes + masks):
    - `blur`: Gaussian blur of the background.
    - `mask`: replace background with a constant color.
    - `crop`: crop tightly around the object bbox with padding.

- **Key observation**
  - With many categories and **all** val2017 images, accuracies sometimes **increase** under perturbations (especially `crop`).
  - Interpretation:
    - Original COCO images are cluttered; multiple objects + strong scene priors.
    - Cropping/blur/mask often *remove* misleading context and co-occurring objects.
    - For simple prompts like `\"a photo of a car\"`, this can move the distribution closer to CLIP's training regime (isolated object photos), so accuracy rises.
  - Conclusion:
    - Object-level results primarily probe **shortcut use vs. object isolation**, not high-level reasoning.

---

## 3. Relational reasoning experiment: person riding a horse

- **Dataset construction**
  - Source: COCO val2017 instances + captions.
  - We restrict to images containing both `person` and `horse`.
  - Use caption patterns to define **clean relational labels**:
    - Positive (label = 1): captions mention *riding* a horse (e.g., `\"riding a horse\"`, `\"on horseback\"`) and not clearly standing.
    - Negative (label = 0): captions mention *standing next to/beside/near* a horse.
    - Ambiguous cases (no such phrases) are **dropped**.
  - Foreground region: union bbox over all person+horse instances, used to build masks and crops.

- **Prompt set**
  - `REL_PROMPTS_RIDING` includes:
    - Positive:
      - `\"a photo of a person riding a horse\"` (index 0, treated as the \"true\" relation).
    - Strong foils:
      - `\"a photo of a person standing next to a horse\"`
      - `\"a photo of a person standing beside a horse\"`
      - `\"a photo of a person standing near a horse\"`
      - `\"a photo of a horse with no person riding it\"`

- **Relational accuracy metric**
  - For each image:
    - Compute similarities to all prompts.
    - Let `top_idx = argmax(similarities)`.
    - If `label == 1` (riding): correct iff `top_idx == 0`.
    - If `label == 0` (standing/not riding): correct iff `top_idx != 0`.

---

## 4. Relational robustness under perturbations

- **Perturbations reused**
  - `blur`, `mask`, `crop` are applied using the person+horse union bbox as foreground.

- **Observations (example run)**
  - Baseline relational accuracy around ~0.8 on the cleaned riding-vs-standing dataset.
  - Under perturbations:
    - `blur`: slight decrease.
    - `mask`: larger decrease (background replaced by constant).
    - `crop`: often close to baseline.
  - Importantly, changes are **modest**, not catastrophic.

- **Stability metrics**
  - For each perturbation type, we compute:
    - \(P(\\text{correct after} \\mid \\text{correct before})\): robustness of already-correct decisions.
    - \(P(\\text{correct after} \\mid \\text{wrong before})\): how often perturbations fix previously wrong decisions.
  - These reveal whether perturbations mostly:
    - Preserve correct relational understanding, or
    - Shake the model into/away from the correct answer.

- **Interpretation**
  - Riding vs standing is largely determined by **local person–horse configuration**, which survives our current perturbations.
  - Background blur/mask remove far context, but not the crucial contact/pose region, so relational accuracy remains relatively stable.
  - This suggests CLIP has some genuine sensitivity to local relational cues in these cases, though it is not a dedicated relational reasoning model.

---

## 5. How CLIP was trained (and why relations are implicit)

- **Training regime**
  - Trained on hundreds of millions of (image, text) pairs via contrastive learning.
  - Text is noisy web data (alt-text, captions, etc.), not structured relational labels.
  - Objective: make each image close to its own text and far from others in the batch.

- **Relational supervision is weak**
  - Many captions naturally contain relational phrases (\"a man riding a horse\", \"a dog under a table\"), so CLIP **does see relational language**.
  - But there is no explicit supervision of the form:
    - \"relation = riding(person, horse)\" or \"left_of(cube, sphere)\".
  - Relational behavior is **emergent**, not guaranteed; probing it requires careful dataset construction (as we did with COCO captions) and perturbations.

---

## 6. Possible extensions

### 6.1. More relational templates on COCO

- Build additional relational datasets based on COCO annotations + captions, for example:
  - `train_on_tracks`: train on/at tracks vs train elsewhere.
  - `bus_in_city_street`: bus in a city street vs bus in parking lot/depot.
  - `person_holding_object`: person holding an object vs object nearby on a table.
- For each relation:
  - Use object category co-occurrence + caption patterns to define clean positives/negatives.
  - Design **multiple hard foils** for the text prompts (swapping the relation, location, or count).
  - Evaluate baseline vs `blur`/`mask`/`crop` and compute stability metrics as in the riding experiment.

### 6.2. Stronger, relation-targeted perturbations

- Current perturbations mostly alter **far background**:
  - Consider perturbations that target the *relation-critical* region, e.g.:
    - Remove most of the person while keeping the horse (or vice versa).
    - Randomly shift/crop so that relative spatial cues are degraded.
  - If relational accuracy stays unrealistically high even when the relation is visually destroyed, that would be strong evidence of shortcut reliance (e.g., relying on caption priors alone).

### 6.3. CLEVR-style controlled reasoning

- Use CLEVR or CLEVR-like synthetic data as a complementary benchmark:
  - Advantages:
    - Fully known scene graph and ground truth programs.
    - Clean supervision of attributes and relations (left-of, behind, closest, etc.).
    - No real-world background bias.
  - Possible experiments:
    - Treat CLEVR captions/questions as prompts and evaluate CLIP (or another VLM) on:
      - Counting (\"how many red spheres left of the cube?\").
      - Spatial comparisons (\"is the sphere closer to the cube than to the cylinder?\").
    - Add synthetic perturbations that corrupt spatial relations while preserving object identity, then measure reasoning robustness.

### 6.4. ImageNet(-R/-A) and ObjectNet

- To align with the proposal:
  - **ImageNet-R / ImageNet-A**:
    - Evaluate CLIP with object prompts on OOD variants to study generalization beyond clean ImageNet.
    - Compare effects of artificial background perturbations vs natural distribution shift.
  - **ObjectNet**:
    - Object-centric, reduced background bias.
    - Serves as a control: if robustness issues mostly vanish on ObjectNet but not on COCO/ImageNet, that supports the shortcut-learning hypothesis.

---

## 7. Summary of main conceptual points

- CLIP is a **prompt-driven similarity model**, not a fixed classifier; prompts are crucial at inference.
- Background perturbations can **improve** object-level accuracy by removing misleading context, so rising accuracy under blur/mask/crop does *not* necessarily contradict the shortcut-learning story.
- To study \"reasoning\" robustness, we need tasks where correctness depends on:
  - **Relations, counts, and configurations**, not just object identity.
  - Our riding-vs-standing COCO subset is a first step in that direction.
- Controlled datasets like **CLEVR** and distribution-shift benchmarks like **ImageNet-R / ObjectNet** are natural next steps to separate:
  - Low-level object robustness,
  - Background shortcut reliance,
  - Genuine compositional and relational reasoning.

