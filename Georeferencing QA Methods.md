# Georeferencing Quality Assessment Methods for UAV Orthomosaics over Web Basemaps

## Overview

This report compares seven candidate methods for assessing the georeferencing quality of UAV orthomosaics by analyzing alignment between orthomosaics and web basemap tiles (streets and satellite imagery) in 1024×1024 browser screenshots at zoom level 17.
The focus is on binary QA (good vs. misaligned), with primarily translational errors but possible rotation and scale errors, under constraints of a small labeled dataset, Windows deployment, and CPU‑first inference.

## High-Level Comparison Table

**Legend:** ✅ well suited, ⚠️ usable with caveats / adaptation, ❌ generally not suited for this setup.

| # | Method | Task fit (alignment scoring, translation/rot/scale, cross-domain) | Data requirements | Robustness (texture, lighting, tiles, large shifts) | Computational requirements | Implementation complexity | Interpretability | Published evidence |
|---|--------|---------------------------------------------------------------------|-------------------|------------------------------------------------------|----------------------------|---------------------------|------------------|--------------------|
| 1 | Phase Correlation + FFT | ✅ Natively estimates global translation; extensions exist for rotation/scale via log-polar / Fourier–Mellin; works best when common structure dominates, but raw intensity differences between ortho RGB and styled OSM tiles can hurt, so edge/gradient/phase-congruency preprocessing is recommended.[^1][^2][^3][^4] | ✅ No labels required; fully unsupervised; suitable immediately with current data. | ⚠️ Robust to global illumination but degrades on featureless regions and when only a small fraction of the image overlaps or tiles are missing; enhanced variants using structural representations and masking improve multisensor robustness and featureless cases.[^2][^5] | ✅ Very fast on CPU due to FFT; widely used for large remote-sensing images; rotation/scale‑invariant variants add modest overhead.[^2][^6][^7] | ⚠️ Classical signal processing; libraries exist but need some engineering (windowing, masking, PSR computation, log‑polar if you want rotation/scale). | ✅ Provides explicit shift vector and a peak/PSR score that is easy to threshold; extensions can also estimate rotation/scale.[^1][^2][^7] | ✅ Many works on multisensor and multimodal remote sensing registration (optical–SAR, optical–LiDAR, multispectral) using enhanced phase correlation or phase congruency.[^2][^3][^8][^6][^9] |
| 2 | SuperPoint + LightGlue + RANSAC | ✅ Designed for pairwise feature matching; RANSAC homography / fundamental matrix gives a direct inlier ratio and geometric consistency score; handles translation, moderate rotation and scale; has been validated on UAV and aerial imagery.[^10][^11][^12][^13] | ✅ Works out of the box with pretrained weights; only a small labeled set is needed to calibrate a decision threshold on inlier statistics or derived "good_probability"; no pixel‑level labels required.[^14][^15] | ⚠️ Very strong in textured urban scenes; with default training it can struggle in large low‑texture fields or forests, but retraining or domain‑specific variants (e.g., ForestGlue) improve low‑texture robustness.[^11][^12] | ⚠️ Inference is heavier than FFT but still practical; LightGlue is optimized for efficiency and can match or beat SuperPoint+SuperGlue speed; GPU preferred but CPU is possible with reduced keypoints.[^11][^14][^15] | ⚠️ Requires PyTorch, model weights, and some plumbing around keypoints, descriptor extraction, LightGlue invocation, and RANSAC; however, reference implementations and wrappers exist.[^14][^16][^17][^15] | ✅ Highly interpretable via matches, inlier sets, estimated homography, and residual statistics; scalar scores can be derived and calibrated as probabilities. | ✅ Extensive benchmarks on HPatches, MegaDepth, ScanNet and aerial/UAV mosaicking; LightGlue+SIFT or SuperPoint is reported as reliable for UAV image mosaicking, including low‑texture agricultural fields.[^10][^18][^19][^12][^13] |
| 3 | LoFTR / EfficientLoFTR | ✅ Detector‑free dense/semi‑dense matching built for challenging pose changes and low‑texture regions; well suited to scoring alignment by inlier ratios or residuals; naturally handles translation and rotation, moderate scale; good cross‑view robustness.[^20][^21][^22][^23] | ✅ Pretrained models available; no labels needed for core matching; a small labeled set can calibrate QA thresholds; no dense annotations needed.[^20][^24][^25] | ✅ Dense matching and global context give strong performance in low‑texture regions (fields, forests), and in wide‑baseline outdoor scenes; EfficientLoFTR improves both accuracy and robustness over original LoFTR.[^20][^11][^26][^24] | ⚠️ Original LoFTR is relatively heavy (≈116–121 ms for 640×480 on RTX 2080 Ti); EfficientLoFTR claims ≈2.5× speed‑up and can even surpass SuperPoint+LightGlue in speed, but both are still significantly slower than FFT on CPU.[^22][^26][^24][^25] | ⚠️ Requires PyTorch and more complex integration than sparse matchers; EfficientLoFTR models and HF processors simplify this somewhat but still more involved than SuperPoint+LightGlue.[^27][^28][^24][^25] | ✅ Produces dense or semi‑dense correspondence maps and confidence scores, which can be summarized into shift statistics, inlier heatmaps, and uncertainty measures. | ✅ Strong benchmarks on MegaDepth, ScanNet, and visual localization; forest‑environment VO experiments show that dense SuperPoint‑LoFTR achieves top pose accuracy but at higher computational cost than LightGlue‑based pipelines.[^20][^11][^29][^22][^23] |
| 4 | DISK / ALIKED (as SuperPoint drop‑ins) | ✅ Learned keypoint detectors/descriptors; when used with LightGlue or SuperGlue they are directly usable for image‑pair alignment scoring; applicable to translation+rotation+moderate scale; support cross‑domain to some extent.[^30][^19][^31][^14][^16] | ✅ Pretrained models exist; no labels needed beyond a small set to choose thresholds; no dense annotation required; however, best performance often assumes training on similar domains.[^30][^19][^31] | ✅ In aerial/remote‑sensing settings, ALIKED+LightGlue and SuperPoint+SuperGlue were found robust under strong radiometric changes (day/night), with ALIKED+LightGlue offering the best match quantity vs. efficiency balance; DISK often yields many matches and strong 3D recon performance but can introduce more outliers.[^32][^33][^30][^31] | ⚠️ Similar to SuperPoint in cost when used with LightGlue; some ALIKED variants are explicitly designed to be lighter than dense matchers; all are heavier than phase correlation but usable on CPU with care.[^30][^34][^14][^16] | ⚠️ Requires swapping detector in an existing pipeline and re‑tuning keypoint budgets and thresholds; however, LightGlue directly supports DISK and ALIKED features.[^14][^16] | ✅ Same interpretability as other sparse matchers: matches, inliers, homographies, residuals. | ✅ Recent work in day/night aerial photogrammetry highlights ALIKED+LightGlue as a robust choice; DISK+SuperGlue/LightGlue has been used to recover tie points in historical aerial imagery.[^32][^33][^31] |
| 5 | VLM zero‑shot (GPT‑4V / Gemini) | ⚠️ VLMs can reason over multiple images and qualitatively assess alignment, but they are not specifically designed for precise georeferencing QA; no native notion of pixel‑level shift and limited control over decision thresholds.[^35][^36] | ✅ Zero‑shot or few‑shot prompting; needs only textual task descriptions and a small set of in‑context exemplars; no pixel‑level labels required; however, cost per query is high and labels are noisy. | ⚠️ Earth observation benchmarks show good scene/landmark recognition accuracy (e.g., ~0.67 on aerial landmark recognition for GPT‑4V) but poor object localization (IoU and centroid errors large), indicating limited reliability for precise alignment scoring; robustness to partial tiles or large shifts is unproven.[^35] | ❌ Inference is remote and relatively slow/expensive; no local CPU deployment; subject to API latency and rate limits. | ⚠️ Implementation is simple at the API level but introduces dependency on external services, model versions, prompt maintenance, and billing; difficult to integrate into a tight real‑time QA loop. | ⚠️ Output is natural language plus possibly bounding boxes; probabilities must be inferred via heuristics (e.g., verbalized confidence scales), and calibration is weak. | ⚠️ Emerging benchmarks for EO show mixed performance, with good high‑level understanding but poor localization; there is no established benchmark specifically for UAV–OSM alignment QA yet.[^35][^36] |
| 6 | Siamese CNN / Contrastive Learning | ⚠️ Can be adapted to map an image pair to a similarity score and thus support QA, but not purpose‑built for geospatial alignment; will tend to learn appearance similarity more than precise geometric consistency unless carefully designed.[^37][^38] | ❌ Data‑hungry: Siamese networks for image matching typically train on thousands to millions of labeled pairs; landmark and remote‑sensing Siamese models use tens of thousands of pairs, even with transfer learning; your current 83 samples are far too few without heavy synthetic augmentation.[^37][^39][^40] | ⚠️ With good training data, Siamese descriptors can be robust to lighting and appearance changes and even cross‑modal (e.g., rendered vs. real images), but low‑texture and large‑shift robustness still depends on training distribution; limited evidence in UAV‑vs‑basemap settings.[^41][^37][^38] | ⚠️ Training usually requires GPU(s); inference per pair is cheap once trained; model size depends on backbone (e.g., ResNet‑18 vs. larger CNN). | ❌ Requires designing pairwise input pipelines, mining hard positive/negative examples, managing severe class imbalance, and avoiding collapse; substantially higher engineering and maintenance burden than classical matchers. | ⚠️ Produces a similarity score or distance; not directly interpretable as explicit shifts, though CAMs or gradient‑based attribution can give rough localization. | ⚠️ Evidence from general image matching and some cross‑domain tasks (e.g., ground vs. rendered UAV views) shows promise but uses far larger datasets; no specific benchmarks for UAV orthos vs. OSM at screenshot scale.[^41][^37][^38] |
| 7 | Fine‑tuned ViT binary classifier | ⚠️ Can in principle learn a direct "aligned vs. misaligned" classifier on concatenated ortho+basemap crops, but requires substantial training data or strong augment/reg; not inherently designed to output a shift estimate.[^42][^43][^44] | ❌ ViTs are data‑hungry; even with augmentation and regularization, best practices assume at least thousands of images or strong pretraining; small‑scale ViT training papers emphasize the need for self‑supervised pretraining and aggressive AugReg to make ViTs work on CIFAR‑scale datasets.[^42][^44] | ⚠️ A well‑trained ViT can learn invariance to color/lighting and some geometric jitter; however, it may struggle to distinguish subtle misalignments, and robustness to low‑texture scenes depends heavily on data diversity; no explicit mechanism for handling missing tiles beyond learning from examples.[^42][^43][^44] | ⚠️ Fine‑tuning small ViTs is feasible on a single GPU; inference on CPU is slower than CNNs but still reasonable for 1024×1024 with small models; heavier than FFT and typically heavier than shallow CNNs.[^42][^44] | ❌ Requires a full training and evaluation pipeline with careful augmentation, regularization, class‑imbalance handling, and potentially self‑supervised pretraining on your own ortho data; maintenance and re‑training as data grows. | ⚠️ Output is a class probability; ViT attention maps provide some interpretability but no explicit motion vector; probability calibration needs separate validation. | ⚠️ ViTs have been applied to remote‑sensing classification with advanced augmentation strategies (e.g., Quantitative Augmentation, CutMix/Label smoothing), but not specifically to georeferencing QA; small‑data ViT works focus on standard classification datasets.[^42][^43][^44] |


## Method-by-Method Analysis

### 1. Phase Correlation + FFT

Phase correlation estimates the relative translation between two images by computing the cross‑power spectrum in the Fourier domain and locating the peak of the inverse FFT, and is widely used for image registration. It is attractive for your task because it is unsupervised, very fast, and directly yields a shift vector and a peak signal measure that can be turned into a confidence score.[^1][^2]

However, simple phase correlation assumes that the two images share similar intensity structure, which is violated when comparing drone RGB orthos to stylized OSM vector tiles or to satellite imagery with different radiometry. Multisensor remote‑sensing work has addressed this by applying phase correlation not on raw intensities but on structural representations such as phase congruency maps, gradient magnitude images, or Log‑Gabor filtered spectra, significantly improving robustness under strong radiometric differences.[^45][^2][^3][^8]

Low‑texture regions and featureless scenes (e.g., clouds, sky, water, uniform canopy) are a known failure mode for basic phase correlation, producing noisy correlation peaks; this has motivated research into preprocessing and masking strategies, which show substantial gains for featureless imagery. Large shifts are not a problem in principle as long as there is sufficient overlap and aliasing is avoided, and work in atmospheric and sky‑image tracking reports stable performance even for displacements up to a large fraction of the image size after appropriate preprocessing.[^2][^5]

### Edge and structural representations

Several studies explicitly examine applying phase correlation to edge or structural images instead of raw intensity.
An "edge phase correlation" method using Canny edges demonstrates improved speed and robust registration (including angle estimation) by running phase correlation on edge maps rather than full images.[^46]
Work on multimodal registration uses Canny or phase‑congruency features combined with phase correlation to handle strong modality and illumination differences, including infrared vs. visible and optical vs. LiDAR data.[^3][^47][^2]

For your cross‑domain case (ortho vs. OSM streets / satellite tiles), running phase correlation on Canny or gradient‑magnitude maps of the streets overlay (e.g., ortho_streets vs. streets_only, ortho_satellite vs. satellite_only) is likely to be significantly more robust than raw RGB, because it emphasizes edges of roads, buildings, and other common structures while suppressing style differences.[^2][^46][^3]

### Computational profile and implementation

Phase correlation is dominated by a small number of FFTs on 1024×1024 arrays, which is extremely efficient on CPU and trivially available in NumPy, SciPy, OpenCV, and scikit‑image; enhanced methods add lightweight preprocessing and masking. Rotation/scale invariance can be obtained by performing phase correlation in log‑polar coordinates (Fourier–Mellin), or by combining coarse PC with local feature‑based registration.[^6][^7][^2]

Given your constraints, phase correlation (with edge or phase‑congruency preprocessing) is an excellent candidate as a fast, unsupervised pre‑filter that estimates a shift vector and a peak‑to‑sidelobe ratio (PSR) or normalized peak value for QA.


### 2. SuperPoint + LightGlue + RANSAC (current baseline)

SuperPoint is a learned keypoint detector/descriptor trained on synthetic homography adaptation; LightGlue is a transformer‑based feature matcher that adaptively prunes computation and has been shown to match or surpass SuperGlue in efficiency and accuracy.[^19][^14][^15]
In your current setup, SuperPoint+LightGlue+RANSAC already delivers ~94% test accuracy with a threshold on "good_probability", which aligns with the literature showing strong performance on planar and near‑planar scenes.[^10][^19]

Remote‑sensing and UAV photogrammetry work confirms that LightGlue can provide robust tie points and accurate mosaics under challenging conditions.
An ISPRS study on UAV image mosaicking found that SIFT+LightGlue outperformed traditional SIFT+BF and SIFT+FLANN as well as SuperPoint+LightGlue, particularly in low‑texture agricultural areas and under larger rotations, due to the rotational invariance of SIFT combined with LightGlue's contextual matching.[^12][^13]
Other work on forest environments shows that SuperPoint with LightGlue or SuperGlue achieves competitive pose estimation accuracy while being much faster than dense matchers like SuperPoint‑LoFTR.[^11]

### Strengths and limitations for this task

Sparse learned features like SuperPoint handle translation, moderate rotation, and moderate scale changes well, and can operate reasonably on cross‑domain aerial imagery vs. basemap tiles when structural features (roads, buildings) appear consistently.[^10][^19][^12]
However, in very low‑texture areas such as homogeneous crop fields or dense forest canopy, the number of repeatable keypoints drops, which can lead to unstable RANSAC fits and poor QA signals unless you aggregate across multiple zooms or crops or enforce minimum inlier counts.[^19][^11][^10]

From an engineering perspective, SuperPoint+LightGlue is relatively straightforward to use thanks to existing PyTorch implementations, pip packages, and wrappers.
LightGlue directly supports SuperPoint, DISK, ALIKED, and SIFT features and exposes match confidences that you can aggregate into a scalar score.[^14][^16][^15]
Because you already have this pipeline in place, it forms a strong backbone that can be augmented rather than replaced.


### 3. LoFTR and EfficientLoFTR

LoFTR is a detector‑free transformer‑based matcher that computes dense coarse‑to‑fine correspondences, designed to perform well under large viewpoint changes and in low‑texture regions where conventional keypoint detectors fail.[^20][^21]
The authors explicitly highlight its ability to aggregate global context via self‑ and cross‑attention, enabling matches in indistinctive or low‑texture areas that are difficult for sparse detectors.[^20]

Timing measurements in the original LoFTR paper report about 116–121 ms per 640×480 image pair on an RTX 2080 Ti, depending on the matching variant, indicating that LoFTR is substantially heavier than sparse matchers but still feasible for offline or batch processing.[^22][^23]
Follow‑up work such as ASpanFormer and EfficientLoFTR aims to retain dense matching quality while improving efficiency; EfficientLoFTR in particular reports being approximately 2.5× faster than LoFTR and even surpassing SuperPoint+LightGlue in speed on some benchmarks, while maintaining or improving accuracy.[^27][^26][^24][^25]

### Performance in low-texture outdoor scenes

In low‑texture forest environments, experiments in the ForestGlue work indicate that dense matchers (e.g., SuperPoint‑LoFTR) can achieve the highest pose estimation AUC (up to ≈0.916) but with significantly higher computational cost, whereas SuperPoint+LightGlue achieves comparable pose accuracy at much lower runtimes.[^11]
This aligns with the LoFTR design: it excels at recovering matches in weakly textured regions, which is beneficial for your forest and field cases, but the computational overhead is substantial compared to sparse pipelines.[^20][^11]

Given your CPU‑first constraint, original LoFTR is likely too heavy for per‑pair QA at scale, but EfficientLoFTR or other optimized variants specifically target low‑end devices and semi‑dense matching with "sparse‑like" speed and could be used either offline or as a fallback on difficult cases.[^48][^24][^25][^27]


### 4. DISK / ALIKED as SuperPoint replacements

DISK is a learned keypoint detector/descriptor optimized to produce many high‑quality matches, and has been shown to perform very well in multiview reconstruction tasks, often providing more matches than competitors such as ALIKE at the cost of more outliers.[^30]
ALIKED is a lighter keypoint and descriptor network that introduces a Sparse Deformable Descriptor Head (SDDH) to efficiently extract deformable, geometrically invariant descriptors on sparse keypoints rather than dense maps, leading to good performance with reduced computation.[^34][^30]

The ALIKED paper reports that ALIKED variants achieve excellent performance on image matching, 3D reconstruction, and stereo tasks, and that a tiny ALIKED model can match or surpass larger baselines such as DISK or ALIKE on some benchmarks despite being significantly smaller.[^30]
In stereo and multiview experiments, ALIKED‑N variants sometimes outperform DISK in metrics like mean Average Accuracy while using fewer matches, though DISK can deliver slightly better 3D reconstructions due to sheer match count.[^30]

### Evidence on aerial/satellite imagery

An ISPRS study on night‑and‑day aerial photogrammetry evaluated several deep feature matchers (ALIKED+LightGlue, SuperPoint+SuperGlue, LoFTR, etc.) on multi‑temporal day/night aerial datasets.
It found that feature extraction is highly sensitive to scale, and that only a subset of DL methods—especially ALIKED with LightGlue and SuperPoint with SuperGlue—remained robust under low illumination, with ALIKED+LightGlue providing the best balance between match quantity and computational efficiency.[^32][^33]
Another photogrammetric application, solving "cold cases" with historical aerial images, leverages DISK combined with SuperGlue/LightGlue to improve tiepoint extraction in degraded, low‑contrast aerial imagery, outperforming purely handcrafted methods.[^31]

There are no widely distributed ALIKED models trained exclusively on aerial/satellite data, but the above results indicate that ALIKED (and DISK) generalize reasonably well to such domains, and retraining or fine‑tuning on aerial pairs further improves robustness.[^33][^32][^31][^30]


### 5. Vision–Language Models (GPT‑4V, Gemini) in zero‑shot mode

Recent work benchmarks GPT‑4V and other VLMs on Earth observation tasks, including aerial landmark recognition, land‑cover classification, and object localization.[^35]
GPT‑4V reaches an overall accuracy of about 0.67 on aerial landmark recognition in the proposed benchmark, outperforming open models but still leaving substantial room for error.[^35]
However, the same study reports very poor object localization performance: on remote‑sensing object detection tasks, GPT‑4V achieves a Precision@0.5 of about 0.08, mean IoU of 0.16, and a mean centroid distance of 147 pixels, showing that current VLMs are not reliable for precise spatial localization.[^35]

A Google developers blog highlights that Gemini models can classify EuroSAT land‑cover images reasonably well and can be instructed via in‑context examples to handle more complex satellite data, but still misclassify challenging cases and require careful prompting.[^36]
There is currently no dedicated benchmark for UAV‑to‑basemap alignment QA using VLMs; any use would thus be exploratory and heuristic.

### Prompting strategies for QA

For your four‑image setup (ortho_streets, streets_only, ortho_satellite, satellite_only), a VLM could be prompted to answer questions like "Is the drone imagery correctly aligned with the map?" with step‑by‑step reasoning, perhaps aided by visual overlays or difference images.
Few‑shot prompts with examples of good vs. bad alignments and explanations would likely improve reliability.
However, given the lack of quantitative evidence and the models' poor localization metrics, VLMs are best treated as an auxiliary sanity check or annotation aid rather than a core QA engine.


### 6. Siamese CNN / Contrastive Learning

Siamese CNNs learn an embedding such that similar image pairs are close and dissimilar pairs are far, typically trained with contrastive or triplet losses.[^37]
For your task, one could feed concatenated ortho+basemap crops through a twin network and train a classifier/regressor on the embedding distance to predict "aligned" vs. "misaligned".

However, published Siamese approaches for general image matching and retrieval train on large datasets: a landmark matching study trains a Siamese CNN (sHybridCNN) on many thousands of positive and negative image pairs and reports AUC improvements of 5–11% over baselines, with unbalanced training sets where dissimilar pairs are 1.5× more frequent than similar ones.[^37]
Remote‑sensing change‑detection Siamese networks similarly rely on tens of thousands of patches or image chips for training.[^39][^40]

In contrast, your current dataset of ~83 screenshots with only 6 "bad" samples is orders of magnitude too small for effective supervised training, even with strong augmentation.
You would need to synthesize misaligned examples (see data strategy section) or dramatically expand your labeled dataset before Siamese CNNs become competitive with classical feature‑based methods.


### 7. Fine‑tuned ViT binary classifier

Vision Transformers have achieved strong performance on image classification and remote‑sensing tasks when pretrained on large datasets and tuned with heavy augmentation and regularization.[^43][^44]
However, they are known to be more data‑hungry than CNNs, and small‑data ViT results emphasize how critical both pretraining and aggressive augmentation are.

A BMVC paper on training ViTs on small datasets shows that self‑supervised pretraining on the target dataset using low‑resolution view prediction, followed by supervised fine‑tuning, substantially improves performance (up to +8% accuracy on CIFAR‑10/100) compared with training from scratch.[^42]
Another study on efficient ViT training demonstrates that combining strong "AugReg" (mixtures of RandAugment, Mixup, CutMix, label smoothing, dropout, and stochastic depth) with increased compute can match the performance of models trained on datasets an order of magnitude larger.[^44]

In remote sensing specifically, a recent paper introduces "Quantitative Augmentation" and shows that combining it with CutMix and Online Label Smoothing notably improves CNN and ViT performance, especially when labeled training samples are limited.[^43]
These results are promising but still assume hundreds to thousands of labeled images per class; your current dataset is far smaller, and the subtlety of misalignment vs. correctly aligned screenshots further complicates the task.

Consequently, a fine‑tuned ViT classifier is best seen as a long‑term option once you can generate or collect several hundred to a few thousand labeled good/bad examples and can invest in self‑supervised pretraining on your entire ortho/basemap corpus.


## Answers to Specific Questions

### LoFTR vs SuperPoint+LightGlue on low-texture outdoor scenes; lightweight variants

LoFTR's dense matching with transformer attention is explicitly designed to handle low‑texture regions by aggregating global and local context, and empirical results show that it outperforms sparse methods on challenging indoor and outdoor localization benchmarks.[^21][^29][^23][^20]
In forest and other low‑texture outdoor environments, experiments indicate that dense combinations like SuperPoint‑LoFTR can achieve the highest pose estimation accuracy (AUC ≈0.916), but sparse pipelines with SuperPoint+LightGlue or SuperGlue deliver similar accuracy with significantly lower runtime.[^11]

Original LoFTR is relatively heavy (~116–121 ms per 640×480 pair on RTX 2080 Ti), whereas EfficientLoFTR and related optimized variants claim about 2.5× speed‑ups while even surpassing SuperPoint+LightGlue in speed for some resolutions, making them more appealing where GPU is available.[^26][^24][^25][^27][^22]
There are also adaptations of LoFTR for low‑end devices that prune the transformer and train via knowledge distillation, demonstrating workable accuracy with significantly reduced model size and compute, though still primarily targeting GPUs rather than pure CPU inference.[^48]

For strictly CPU‑bound inference at 1024×1024, SuperPoint+LightGlue (with reduced keypoint counts) or ALIKED+LightGlue are generally more practical than LoFTR; LoFTR or EfficientLoFTR could be reserved for offline checks on particularly ambiguous cases.


### Phase Correlation + FFT failure modes on cross-domain pairs; benefits of Canny edges

On cross‑domain image pairs with strong radiometric and textural differences (e.g., optical vs. SAR, infrared vs. visible), basic phase correlation can produce weak or spurious peaks because intensity variations violate the assumptions of similar spectra, leading to degraded accuracy or registration failures.[^9][^3][^2]
Multisensor registration studies emphasize that phase correlation alone is insufficient and propose enhancements such as phase‑congruency structural maps, Log‑Gabor filtering, and robust masking to reduce the impact of modality‑induced radiometric differences.[^8][^3][^2]

Applying phase correlation on edge or structural images is a proven way to improve robustness.
An edge‑phase‑correlation algorithm using Canny edges demonstrates that running phase correlation on edge images improves speed and maintains accurate localization, and can even support rotation estimation when combined with additional processing.[^46]
Other remote‑sensing works similarly precompute structural or edge‑like representations (phase congruency, gradient norm, Log‑Gabor spectra) before performing phase correlation, explicitly to eliminate or reduce radiometric differences between modalities.[^4][^3][^8][^2]

For UAV orthos vs. OSM tiles, this suggests that computing Canny edges or gradient magnitude on both layers (ortho_streets vs. streets_only, ortho_satellite vs. satellite_only) and then performing phase correlation on those features will be more reliable than raw RGB, especially when OSM styling differs strongly from orthophoto appearance.


### DISK / ALIKED vs SuperPoint on aerial/satellite imagery; aerial-trained weights

ALIKED and DISK are both modern learned local feature extractors.
The ALIKED paper proposes a deformable descriptor head to extract geometrically invariant descriptors on sparse keypoints efficiently, and reports excellent performance across image matching, 3D reconstruction, and visual localization benchmarks, with tiny variants offering a good accuracy–speed trade‑off.[^34][^30]
In stereo and multiview 3D reconstruction, ALIKED‑N(16/32) variants outperform or match DISK in some metrics (e.g., reprojection accuracy) while being more efficient, though DISK often yields more matches and sometimes slightly better reconstruction due to additional constraints.[^30]

In aerial photogrammetry, a recent night‑and‑day study finds that ALIKED+LightGlue and SuperPoint+SuperGlue are among the few methods robust under extreme radiometric (day/night) changes, with ALIKED+LightGlue giving the best balance between match density and computational efficiency.[^32][^33]
Another study on UAV mosaicking concludes that SIFT+LightGlue is overall the most reliable across urban and low‑texture agricultural scenes, with SuperPoint+LightGlue and other learned features also performing strongly; DISK is included in broader deep‑image‑matching toolkits used in aerophotogrammetric pipelines.[^13][^16][^12][^10]

There are no widely used off‑the‑shelf DISK or ALIKED models advertised as being trained exclusively on aerial/satellite imagery, but both have been successfully applied to such data, and additional self‑supervised training on aerial datasets has been explored in related learned‑feature work.
Given their strong performance in challenging aerial conditions, ALIKED (with LightGlue) is a particularly promising alternative to SuperPoint for your cross‑domain screenshots.


### VLM zero-shot accuracy on geospatial alignment; prompts

Existing VLM benchmarks for Earth observation indicate that GPT‑4V is reasonably strong at high‑level tasks (scene/landmark recognition) but weak at precise localization and counting.
In one benchmark, GPT‑4V achieves about 0.67 accuracy on zero‑shot aerial landmark recognition, outperforming open models, but all evaluated models—including GPT‑4V—perform poorly on object localization tasks, with GPT‑4V obtaining Precision@0.5 of 0.08, mean IoU of 0.16, and large centroid errors.[^35]
These results suggest that while GPT‑4V can understand scenes globally, it is unreliable for pixel‑accurate alignment judgments.

Google's Gemini models show similar patterns: blog examples demonstrate good land‑cover classification on EuroSAT imagery and the ability to interpret multi‑spectral data via in‑context instructions, but they still misclassify challenging images and are not benchmarked on georeferencing QA tasks.[^36]

For prompting, strategies that seem most promising based on current EO VLM experiments include:
- Providing explicit task instructions ("You see four images: ... Decide if the drone imagery is aligned with the map.") and requesting step‑by‑step reasoning.
- Including a small number of labeled examples of aligned vs. misaligned pairs in the prompt (few‑shot in‑context learning).
- Asking for a numeric confidence score (e.g., 0–100) and a rationale, which you can post‑process.

Nonetheless, due to lack of robust benchmarks and the models' poor localization metrics, VLM use should be limited to auxiliary QA, annotation assistance, or tooling for rapid prototyping, not core automated QA decisions.


### Siamese CNN: dataset size and backbone choice

Siamese networks for image similarity and matching have demonstrated strong performance when trained on large datasets.
A landmark image‑matching study trains a Siamese CNN (sHybridCNN) on many thousands of image pairs and shows that the learned similarity measure outperforms features from classification CNNs by 5–11% AUC, but this training relies on considerable data and even then is limited by noisy labels.[^37]
Remote‑sensing Siamese and Siamese‑Transformer networks for change detection similarly use large datasets such as SECOND and LEVIR‑CD, with tens of thousands of patch pairs and extensive data augmentation.[^49][^50][^40][^39]

There is no single published "threshold" dataset size where Siamese CNNs begin to outperform classical feature‑based methods, but practice suggests that hundreds of positive/negative pairs are a bare minimum and thousands are preferable, especially when negative pairs must represent subtle misalignments rather than arbitrary unrelated images.[^40][^39][^37]
Given your current 83 samples (6 bad), Siamese CNNs are not advisable until you have either:
- Synthesized large numbers of misaligned examples from trusted aligned orthos, or
- Collected and labeled several hundred real misaligned cases.

For backbones under small‑to‑moderate data, light CNNs such as ResNet‑18 or MobileNet‑v2 are generally recommended over deeper nets, as they are easier to train and less prone to overfitting; this aligns with many small‑data contrastive learning and few‑shot Siamese works.[^51][^52][^37]


### Fine-tuned ViT: augmentations and self-supervised pretraining

For small datasets (<300 samples), ViTs require aggressive data augmentation and regularization.
Efficient ViT training studies highlight that combinations of strong image augmentations (RandAugment/AutoAugment), Mixup, CutMix, label smoothing, stochastic depth, and other regularizers ("AugReg") allow ViTs trained on medium‑sized datasets (e.g., ImageNet‑21k) to match or exceed models trained on much larger proprietary datasets.[^44]

The BMVC work on training ViTs on small datasets shows that self‑supervised pretraining via low‑resolution view prediction on the target dataset, followed by standard supervised fine‑tuning, yields substantial gains (up to about +8% accuracy on CIFAR‑10/100) compared with direct supervised training, and this approach is agnostic to the specific ViT architecture.[^42]
Remote‑sensing classification work introduces "Quantitative Augmentation" (QA) and associated modules that adjust feature distributions and couple QA with CutMix and Online Label Smoothing, significantly boosting CNN and ViT performance under limited labeled samples by aligning training and testing distributions.[^43]

Translating this to your setting, the most effective strategies for a ViT classifier would likely be:
- Pretraining a ViT on unlabeled ortho and basemap images using a self‑supervised objective (e.g., DINO, masked autoencoding, or low‑resolution view prediction),
- Strong augmentations (random crops, flips, color jitter, minor rotations, small random translations) applied symmetrically to both ortho and basemap inputs during training,
- Regularization via Mixup/CutMix on concatenated ortho+basemap images and label smoothing to mitigate overfitting and imbalance.

However, these approaches become attractive only once you have at least a few hundred labeled good/bad pairs; until then, feature‑based or unsupervised methods are more suitable.


### Ensembling classical (Phase Correlation) and deep (SuperPoint+LightGlue) methods

In multimodal remote‑sensing registration, several papers use phase correlation as a coarse stage combined with feature‑based methods for refinement.
For example, one multispectral registration scheme first uses phase correlation to estimate coarse offsets between band images, then uses SIFT feature matching constrained by the coarse offset to improve both accuracy and efficiency.[^45]
Another method for registering aerial images and LiDAR data combines structural features with 3D phase correlation to obtain control points, then refines camera parameters, yielding robust and fast registration across modalities.[^53][^6]

These works suggest a best‑practice pattern:
- Use fast, global, unsupervised phase correlation (possibly on structural/edge images) to estimate a candidate shift and a confidence measure.
- Use the estimated shift to restrict the search space or guide feature‑based matching (e.g., by limiting descriptor matching to neighborhoods or using it as a prior in RANSAC).
- Optionally, treat phase correlation as a pre‑filter or gating mechanism: if PC confidence is high and shift small, skip expensive deep inference; if PC confidence is low or shift large/inconsistent, invoke the deep matcher and/or flag the tile for manual review.

For your QA task, this translates naturally into an ensemble in which phase correlation provides a cheap, approximate QA score and candidate shift, and SuperPoint+LightGlue+RANSAC provides a more robust but expensive confirmation when needed.


## Ranked Recommendations for This Use Case

Given your constraints (small labeled dataset, CPU‑first, need to detect mainly translational shifts in cross‑domain screenshot pairs), the following ranking is recommended:

1. **Primary backbone: SuperPoint + LightGlue + RANSAC (refined and calibrated)**
   - Already works well and is well supported in literature for UAV and aerial imagery, including low‑texture conditions when properly tuned.[^12][^10][^11]
   - Provides rich geometric signals (inlier counts, homography, residuals) that can be collapsed into a calibrated "good_probability" using only a small labeled set.

2. **Fast unsupervised pre-filter: Phase Correlation on edge/structural images**
   - Extremely fast on CPU, no labels required, directly outputs a shift and confidence (PSR or normalized peak).
   - When applied to Canny edges or gradient magnitude of ortho/basemap pairs, becomes more robust to style and radiometric differences typical of OSM and satellite tiles.[^3][^2][^46]
   - Ideal to cheaply reject obviously bad or obviously good alignments before invoking deep matchers.

3. **Drop‑in feature upgrade: ALIKED (or DISK) + LightGlue**
   - ALIKED+LightGlue is documented as robust in aerial day/night photogrammetry while being efficient, and DISK+LightGlue/SuperGlue is strong on challenging aerial imagery.[^33][^31][^32][^30]
   - Swapping SuperPoint for ALIKED or DISK in your current pipeline is relatively low‑effort and may improve robustness, especially where SuperPoint struggles.

4. **Secondary / offline method: EfficientLoFTR (or similar dense matcher)**
   - Dense/semi‑dense matching is particularly helpful in low‑texture fields and forests; EfficientLoFTR reduces runtime while maintaining strong performance.[^24][^26][^20][^11]
   - Use as an offline validator or fallback when sparse pipelines yield ambiguous scores.

5. **Future options (once data grows)**
   - Siamese CNN and fine‑tuned ViT classifiers become attractive only after generating or collecting a few hundred to thousands of labeled misaligned examples.
   - VLMs (GPT‑4V/Gemini) are best kept as auxiliary tools for manual QA or annotation assistance, not automated QA.


## Implementation Roadmap for Top Methods

### A. Phase Correlation Pre-filter (edge-based)

**Libraries and tools**
- Python: `numpy`, `scipy.fft` or `skimage.registration.phase_cross_correlation`, and `opencv-python` for Canny edges.
- Runs natively on Windows/CPU.

**Key steps**
1. Load ortho and basemap images at 1024×1024.
2. Convert to grayscale and compute Canny edges (or gradient magnitude).
3. Optionally apply a window function (e.g., Hanning) to reduce edge artefacts.
4. Run phase correlation to obtain subpixel shift (dy, dx) and correlation peak.
5. Compute a PSR‑like confidence: peak height divided by standard deviation of non‑peak values.
6. Map |shift| and PSR into a preliminary good_probability_pc score via a heuristic or small calibration set.

**Sketch of `check_georeferencing` wrapper**

```python
import cv2
import numpy as np
from skimage.registration import phase_cross_correlation

def phase_corr_score(ortho_path, basemap_path):
    ortho = cv2.imread(ortho_path, cv2.IMREAD_GRAYSCALE)
    base = cv2.imread(basemap_path, cv2.IMREAD_GRAYSCALE)

    # Edge maps to reduce radiometric/style differences
    ortho_e = cv2.Canny(ortho, 100, 200)
    base_e = cv2.Canny(base, 100, 200)

    shift, error, _ = phase_cross_correlation(base_e, ortho_e, upsample_factor=10)
    dy, dx = shift  # note sign convention

    # Simple peak-based confidence proxy (1 - normalized error)
    conf = max(0.0, 1.0 - float(error))
    disp = np.linalg.norm(shift)

    # Example heuristic mapping (to be calibrated):
    # small displacement + high confidence → good
    # large displacement or low confidence → bad
    good_prob = np.clip(conf * np.exp(-(disp / 10.0) ** 2), 0.0, 1.0)

    return {
        "good_probability": float(good_prob),
        "shift": (float(dy), float(dx)),
        "pc_confidence": float(conf),
    }

def check_georeferencing(ortho_path, basemap_path):
    scores = phase_corr_score(ortho_path, basemap_path)
    return {"good_probability": scores["good_probability"]}
```

In practice, you would calibrate the mapping from `(disp, conf)` to `good_probability` using your labeled dataset (e.g., simple logistic regression or isotonic regression on top of PC features).


### B. Refined SuperPoint + LightGlue + RANSAC backbone

**Libraries and tools**
- `lightglue` GitHub implementation (or equivalent PyPI/Conda packages) which includes SuperPoint/DISK/ALIKED feature extractors and the LightGlue matcher.[^15][^14]
- `opencv-python` or `kornia_moons` for converting matches into homographies via RANSAC and computing residual statistics.[^54]

**Key steps**
1. Load ortho and basemap images and normalize to `[0, 1]` tensors.
2. Use SuperPoint (or ALIKED/DISK) to extract a fixed budget of keypoints and descriptors per image (e.g., 1024–2048 keypoints, possibly fewer on CPU).[^16][^14]
3. Run LightGlue to obtain matches with confidence scores.
4. Convert matches to coordinate arrays and fit a homography (or similarity) via RANSAC; compute:
   - inlier ratio,
   - median reprojection error,
   - estimated translation components from the homography.
5. Train a simple classifier (e.g., logistic regression) using a small labeled set of screenshots with features such as inlier ratio, log inlier count, translation magnitude, and perhaps phase‑correlation confidence, to output `good_probability`.

**Sketch of `check_georeferencing` (pseudo-code)**

```python
import torch
import numpy as np
import cv2
from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, rbd

# initialize once
device = "cuda" if torch.cuda.is_available() else "cpu"
extractor = SuperPoint(max_num_keypoints=1024).eval().to(device)
matcher = LightGlue(features="superpoint").eval().to(device)

# assume clf is a trained logistic regression model on match/geometric features

def match_and_score(ortho_path, basemap_path):
    img0 = load_image(ortho_path).to(device)
    img1 = load_image(basemap_path).to(device)

    with torch.inference_mode():
        feats0 = extractor.extract(img0)
        feats1 = extractor.extract(img1)
        matches01 = matcher({"image0": feats0, "image1": feats1})

    feats0, feats1, matches01 = [rbd(x) for x in (feats0, feats1, matches01)]
    kpts0 = feats0["keypoints"].cpu().numpy()
    kpts1 = feats1["keypoints"].cpu().numpy()
    matches = matches01["matches"].cpu().numpy()  # [M,2]

    if len(matches) < 8:
        # too few matches
        return {"good_probability": 0.0}

    src = kpts0[matches[:, 0]][:, ::-1]  # (x,y) for OpenCV
    dst = kpts1[matches[:, 1]][:, ::-1]

    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, ransacReprojThreshold=3.0)
    if H is None or inliers is None:
        return {"good_probability": 0.0}

    inliers = inliers.ravel().astype(bool)
    inlier_ratio = inliers.mean()
    inlier_count = inliers.sum()

    # approximate translation from H
    tx, ty = H[0, 2], H[1, 2]
    disp = np.hypot(tx, ty)

    # features for classifier
    X = np.array([[inlier_ratio, np.log1p(inlier_count), disp]])
    good_prob = float(clf.predict_proba(X)[0, 1])
    return {"good_probability": good_prob}


def check_georeferencing(ortho_path, basemap_path):
    return match_and_score(ortho_path, basemap_path)
```

On CPU, you can reduce `max_num_keypoints` and image resolution or downsample images before matching to keep runtimes acceptable.
You can also swap `SuperPoint` for `ALIKED` or `DISK` by changing the extractor and LightGlue configuration.[^14][^16]


### C. ALIKED + LightGlue as an alternative

**Motivation**
- ALIKED+LightGlue has demonstrated robustness to strong radiometric changes in aerial day/night datasets and a good balance between accuracy and efficiency.[^32][^33][^30]
- Switching to ALIKED is low‑effort in a LightGlue‑based pipeline.

**Implementation notes**
- Use ALIKED from the official repository or via wrappers such as `easy-local-features` or `deep-image-matching`, both of which support ALIKED and LightGlue.[^17][^16][^34][^30]
- Keep the rest of the pipeline (RANSAC, logistic classifier) unchanged; only the local feature statistics (e.g., inlier counts) need re‑calibration.

Pseudo‑code changes compared to the SuperPoint example are minimal, e.g.:

```python
from lightglue import LightGlue, ALIKED

extractor = ALIKED().eval().to(device)
matcher = LightGlue(features="aliked").eval().to(device)
```

After swapping, you would re‑compute features on your labeled dataset and retrain `clf` to map the new feature distributions to `good_probability`.


## Data Collection and "Bad" Class Growth Strategy

Your current dataset (~83 samples with only 6 bad) is far too small and imbalanced for data‑hungry methods like Siamese CNNs or ViT classifiers, but there are efficient strategies to grow the "bad" class in ways that reflect realistic misgeoreferencing.

### 1. Systematic synthetic misalignment from trusted good orthos

Given correctly georeferenced orthomosaics and their corresponding basemap screenshots, you can generate arbitrarily many misaligned examples by applying controlled geometric transformations to the ortho layer before overlaying onto the basemap, while keeping the basemap fixed.

Practical steps:
- Start from good examples (where your current pipeline and/or GCPs confirm alignment).
- Apply random translations in pixel space with magnitudes that reflect real failure modes (e.g., 10–300 px at zoom 17), including both axis‑aligned and diagonal directions.
- Optionally add small rotations (±2–5 degrees) and scale perturbations (±1–3%) to simulate more complex georeferencing errors.
- Render new browser‑style overlays or simulate them offline by compositing the warped ortho onto the basemap canvas and re‑capturing 1024×1024 crops.
- Label these synthesized examples as "bad" with known ground‑truth shifts.

This approach can cheaply yield hundreds or thousands of labeled bad samples while preserving the exact same visual and domain characteristics as your production data.
It is especially powerful because you control the distribution of misalignments and can generate curriculum‑style datasets (e.g., a spectrum from barely visible to obviously catastrophic shifts) for training and calibration.

### 2. Hard negative mining from production

As you deploy even a simple QA system (e.g., phase correlation + SuperPoint+LightGlue), you can log:
- Examples where the two methods disagree strongly (one says very good, the other says very bad).
- Examples near your decision boundary (good_probability near 0.5).
- Examples where phase correlation confidence is low or shift estimates are unstable across modalities (streets vs. satellite).

These cases are excellent candidates for human review and labeling as truly good or bad; they will concentrate labeling effort on informative examples that stress your models.
Over time, this yields a curated pool of realistic bad and borderline cases that can complement synthetic misalignments.

### 3. Structured sampling of challenging content

To avoid a bias towards urban, high‑texture scenes, explicitly target areas that are likely to be difficult:
- Large homogeneous agricultural fields.
- Dense forests and mountainous regions with repeating patterns.
- Water bodies and coastal zones (where orthos/basemaps may differ due to tides or acquisition time).
- Areas with partial or missing OSM/satellite tiles, or where OSM geometry is known to be inaccurate.

For each such region, acquire multiple orthos and basemap screenshots and, where possible, deliberately mis‑georeference some orthos in your processing pipeline (e.g., by perturbing camera poses or GCPs) to produce real bad cases.

### 4. Labeling protocol and metadata

When labeling examples, capture not just a binary good/bad label but also:
- Estimated shift magnitude and direction.
- Presence of rotation/scale errors.
- Dominant land‑cover type (urban, forest, agriculture, water, mixed).
- Source of misalignment (synthetic vs. real pipeline error).

This metadata will be invaluable for:
- Analyzing per‑content performance of QA methods.
- Training models with curriculum or multi‑task losses (e.g., predicting both good/bad and shift magnitude).
- Balancing datasets across content types and error severities.

### 5. Progressive model roadmap

With an expanded dataset, you can progressively adopt more data‑hungry methods:
- **Phase 1 (now):** Hard‑thresholded and calibrated phase correlation + feature‑based (SuperPoint/ALIKED+LightGlue) ensemble for QA, using synthetic misalignments only to calibrate thresholds.
- **Phase 2 (~few hundred bad):** Train shallow learned classifiers (logistic regression, small MLPs) on hand‑crafted QA features (PC shift/PSR, inlier ratios, residuals, land‑cover proxies) to improve decision boundaries.
- **Phase 3 (~thousands of bad):** Explore Siamese CNNs for pairwise QA and small ViTs with self‑supervised pretraining on your full unlabeled ortho/basemap archive, keeping feature‑based methods as interpretable baselines and fallbacks.

Throughout, classical and feature‑based methods remain valuable both as baselines and as rich feature generators for any higher‑level learned classifier.

---

## References

1. [Phase correlation](https://en.wikipedia.org/wiki/Phase_correlation)

2. [Robust Fine Registration of Multisensor Remote Sensing Images Based on Enhanced Subpixel Phase Correlation](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7435469/) - Automatic fine registration of multisensor images plays an essential role in many remote sensing app...

3. [A novel extended phase correlation algorithm based on Log-Gabor filtering for multimodal remote sensing image registration](https://www.origamesh.com/publication/papers/2019_anepca.pdf)

4. [Robust Phase-Correlation Based Registration of Airborne ...](https://infoscience.epfl.ch/server/api/core/bitstreams/facde84b-0ac3-4945-bdbc-cee662c6e9ee/content)

5. [Image preprocessing to enhance phase correlation of featureless ...](https://www.nature.com/articles/s41598-025-94176-x) - Based on the mathematical expressions, several image pre-processing approaches are proposed to impro...

6. [FAST AND ROBUST REGISTRATION OF AERIAL IMAGES AND ...](https://isprs-annals.copernicus.org/articles/V-2-2020/135/2020/) - To tackle the problem, this paper proposes an automatic registration method based on structural feat...

7. [Rotation Invariant Registration of 2D Aerial Images Using Local ...](http://www.diva-portal.org/smash/get/diva2:620266/FULLTEXT01) - Before phase correlation, the reference and sensed images are transformed from the Carte- sian coord...

8. [AUTOMATED MULTI-SOURCE REMOTE SENSING ...](https://d-nb.info/1148355324/34)

9. [[PDF] phase correlation based local illumination-invariant method for multi ...](https://pdfs.semanticscholar.org/16e1/c2c14caf73729e02e23d4e187c66425fe945.pdf) - This paper aims at image matching under significantly different illumination conditions, especially ...

10. [Robust UAV Image Mosaicking Using SIFT and LightGlue](https://d-nb.info/1380892244/34)

11. [ForestVO: Enhancing Visual Odometry in Forest Environments ...](https://arxiv.org/html/2504.01261v1) - ForestGlue enhances the SuperPoint feature detector through four configurations – grayscale, RGB, RG...

12. [Robust UAV Image Mosaicking Using SIFT and LightGlue](https://isprs-archives.copernicus.org/articles/XLVIII-2-W11-2025/169/2025/) - To construct a seamless and geometrically accurate mosaic from multiple overlapping UAV images, it i...

13. [[PDF] Robust UAV Image Mosaicking Using SIFT and LightGlue](https://isprs-archives.copernicus.org/articles/XLVIII-2-W11-2025/169/2025/isprs-archives-XLVIII-2-W11-2025-169-2025.pdf) - To achieve precise and gap-free mosaicking, it is critical to accurately interpret the imaging geome...

14. [LightGlue ⚡️ Local Feature Matching at Light Speed](https://huggingface.co/spaces/fantos/vidimatch/blob/98f8793ee7629fe436cc0d032c546dc32f3a8caa/third_party/LightGlue/README.md) - We’re on a journey to advance and democratize artificial intelligence through open source and open s...

15. [LightGlue: Local Feature Matching at Light Speed - GitHub](https://github.com/cvg/LightGlue) - We release pretrained weights of LightGlue with SuperPoint, DISK, ALIKED and SIFT local features. Th...

16. [deep-image-matching](https://pypi.org/project/deep-image-matching/) - Multiview matching with deep-learning and hand-crafted local features for COLMAP and other SfM softw...

17. [easy-local-features - PyPI](https://pypi.org/project/easy-local-features/) - Installation. pip install easy-local-features. Installing from source. pip install -e . Usage. Stabl...

18. [HPatches: A benchmark and evaluation of handcrafted and learned local descriptors](https://lirias.kuleuven.be/bitstream/handle/123456789/644032/haptches-early-access-08712555.pdf;jsessionid=C7A95687044A399330C22074AB0F811C?sequence=2)

19. [Local Feature Matching Using Deep Learning: A Survey - arXiv.org](https://arxiv.org/html/2401.17592v2) - The objective of this endeavor is to furnish a comprehensive overview of local feature matching meth...

20. [LoFTR: Detector-Free Local Feature Matching with Transformers](https://zju3dv.github.io/loftr/) - width=device-width, initial-scale=1

21. [Transformer-Based LoFTR for Dense Matching - Emergent Mind](https://www.emergentmind.com/topics/transformer-based-loftr) - This paradigm has set new standards in performance on various indoor and outdoor visual localization...

22. [[PDF] LoFTR: Detector-Free Local Feature Matching with Transformers](https://zju3dv.github.io/loftr/files/LoFTR-suppmat.pdf)

23. [LoFTR: Detector-Free Local Feature Matching with Transformers](https://ar5iv.labs.arxiv.org/html/2104.00680) - ... ms for a 640 × \times 480 image pair on an RTX 2080Ti. Under the optimal transport setup, we use...

24. [Efficient LoFTR: Semi-Dense Local Feature Matching with Sparse ...](https://arxiv.org/html/2403.04765v1) - As shown in Fig. 1, our approach achieves the best inference speed compared with recent image matchi...

25. [EfficientLoFTR - Hugging Face](https://huggingface.co/docs/transformers/en/model_doc/efficientloftr) - We’re on a journey to advance and democratize artificial intelligence through open source and open s...

26. [Semi-Dense Local Feature Matching with Sparse-Like Speed - Liner](https://liner.com/review/efficient-loftr-semidense-local-feature-matching-with-sparselike-speed) - Specifically, the method achieves the best inference speed among semi-dense matchers, being approxim...

27. [zju-community/efficientloftr - Hugging Face](https://huggingface.co/zju-community/efficientloftr) - The model is designed to be highly efficient, with its optimized version being approximately 2.5 tim...

28. [EfficientLoFTR](https://huggingface.co/docs/transformers/main/en/model_doc/efficientloftr) - We’re on a journey to advance and democratize artificial intelligence through open source and open s...

29. [Papers with Code - ASpanFormer: Detector-Free Image Matching with Adaptive Span Transformer](https://paperswithcode.com/paper/aspanformer-detector-free-image-matching-with) - Implemented in one code library.

30. [ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation](https://ar5iv.labs.arxiv.org/html/2304.03608) - Image keypoints and descriptors play a crucial role in many visual measurement tasks. In recent year...

31. [Solving photogrammetric cold cases using AI-based image matching](https://www.sciencedirect.com/science/article/pii/S0924271623003131) - We apply the two synergetic neural network methods SuperGlue and DISK, improving feature matching fo...

32. [Night and Day Aerial Photogrammetry](https://isprs-archives.copernicus.org/articles/XLVIII-1-W4-2025/81/2025/)

33. [[PDF] Night and Day Aerial Photogrammetry](https://isprs-archives.copernicus.org/articles/XLVIII-1-W4-2025/81/2025/isprs-archives-XLVIII-1-W4-2025-81-2025.pdf) - Co-registering day and night aerial images presents a significant challenge. Preliminary investigati...

34. [ALIKED: A Lighter Keypoint and Descriptor Extraction Network via ...](https://github.com/Shiaoming/ALIKED) - ALIKED is an improvement on ALIKE, which introduces the Sparse Deformable Descriptor Head (SDDH) to ...

35. [[PDF] Benchmarking GPT-4V on Earth Observation Data - CVF Open Access](https://openaccess.thecvf.com/content/CVPR2024W/EarthVision/papers/Zhang_Good_at_Captioning_Bad_at_Counting_Benchmarking_GPT-4V_on_Earth_CVPRW_2024_paper.pdf) - In this work, we propose a comprehen- sive benchmark to gauge the progress of VLMs toward being usef...

36. [Unlocking Multi-Spectral Data with Gemini - Google Developers Blog](https://developers.googleblog.com/unlocking-multi-spectral-data-with-gemini/) - You can start analyzing complex satellite data, right out of the box. What is Multi-Spectral Imagery...

37. [[PDF] Siamese Network Features for Image Matching](https://users.aalto.fi/~kannalj1/publications/icpr2016.pdf) - In this paper, we propose finding matching and non-matching pairs of images by representing them wit...

38. [Cross-Modal feature description for remote sensing image matching](https://www.sciencedirect.com/science/article/pii/S1569843222001583) - We construct a cross-modal feature description matching network (CM-Net) for remote sensing image ma...

39. [Semantic Change Detection with Asymmetric Siamese Networks](https://captain-whu.github.io/SCD/) - In this paper, we present an asymmetric siamese network (ASN) to locate and identify semantic change...

40. [Deep Siamese Network for annual change detection in Beijing using Landsat satellite data](https://oa-fund.ub.uni-muenchen.de/id/eprint/1451/1/1-s2.0-S1569843224002516-main.pdf)

41. [Learning to Match Ground Camera Image and UAV 3-D Model-Rendered Image Based on Siamese Network With Attention Mechanism](https://www.cs.nthu.edu.tw/~lai/pdf/publications/2020/Learning_to_Match_Ground_Camera_Image_and_UAV_3-D_Model-Rendered_Image_Based_on_Siamese_Network_With_Attention_Mechanism.pdf)

42. [[PDF] How to Train Vision Transformer on Small-scale Datasets?](https://bmvc2022.mpi-inf.mpg.de/0731.pdf) - Large-scale pre-training captures inductive biases from the data [13] and allows successful transfer...

43. [Pure data correction enhancing remote sensing image classification ...](https://www.nature.com/articles/s41598-025-89735-1) - This strategy effectively corrects feature distributions across remote sensing data, significantly i...

44. [Efficient ViT Training Strategies](https://www.emergentmind.com/papers/2106.10270) - This paper shows how data, augmentation, and regularization enable efficient Vision Transformer trai...

45. [A Registration Scheme for Multispectral Systems Using Phase Correlation and Scale Invariant Feature Matching](https://onlinelibrary.wiley.com/doi/10.1155/2016/3789570) - In the past few years, many multispectral systems which consist of several identical monochrome came...

46. [[PDF] Image Registration Method Based On Edge Phase Correlation ...](https://www.atlantis-press.com/article/25852406.pdf) - First of all, applying optimized edge phase correlation algorithm, this algorithm obtains target are...

47. [Highly robust thermal infrared and visible image registration with ...](https://www.sciencedirect.com/science/article/abs/pii/S0143816624005049) - This paper presents a novel algorithm, namely, Highly Robust Thermal Infrared and Visible Image Regi...

48. [Local Feature Matching with Transformers for low-end devices ...](https://ar5iv.labs.arxiv.org/html/2202.00770) - LoFTR [19] is an efficient deep learning method for finding appropriate local feature matches on ima...

49. [A Siamese Swin-Unet for image change detection | Scientific Reports](https://www.nature.com/articles/s41598-024-54096-8) - In this paper, we propose a network named Siam-Swin-Unet, which is a Siamesed pure Transformer with ...

50. [A Mamba-Based Siamese Network for Remote Sensing ...](https://openaccess.thecvf.com/content/WACV2025/papers/Paranjape_A_Mamba-Based_Siamese_Network_for_Remote_Sensing_Change_Detection_WACV_2025_paper.pdf)

51. [Crafting Better Contrastive Views for Siamese Representation Learning](https://arxiv.org/pdf/2202.03278.pdf)

52. [Siamese Transformer Networks for Few-shot Image Classification](https://arxiv.org/html/2408.01427v1) - We propose a novel approach based on the Siamese Transformer Network (STN). Our method employs two p...

53. [Fast and Robust Registration of Aerial Images and LiDAR data Based on Structrual Features and 3D Phase Correlation](https://arxiv.org/abs/2004.09811v1) - Co-Registration of aerial imagery and Light Detection and Ranging (LiDAR) data is quilt challenging ...

54. [Image matching example with LightGlue and DISK - Kornia](https://www.kornia.org/tutorials/nbs/image_matching_lightglue.html) - In this tutorial we are going to show how to perform image matching using a LightGlue algorithm with...

