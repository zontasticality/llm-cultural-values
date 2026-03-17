#set document(title: "LLM vs Human Cultural Values: Figure Analysis")
#set page(margin: 2.5cm)
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.")
#set par(justify: true)

#align(center)[
  #text(size: 18pt, weight: "bold")[LLM vs Human Cultural Values]
  #v(0.3em)
  #text(size: 14pt)[Detailed Figure Analysis]
  #v(0.3em)
  #text(size: 10pt, fill: gray)[Track 1: EVS Survey Elicitation from HPLT-2.15B, EuroLLM-22B, and Gemma-3-27B]
]

#v(1em)

= Overview

This report analyzes 7 main figures comparing language model (LLM) response distributions to real European Values Study (EVS) 2017 survey data across 21 countries. Three model configurations are compared:

- *HPLT-2.15B*: 22 monolingual models, each trained exclusively on text from a single European language. These are small (2.15B parameter) Gemma-3-based models from the HPLT project. Two prompt configurations are compared:
  - *HPLT-2.15B (cue)*: uses a native-language answer cue ("Answer with a number from 1 to N:")
  - *HPLT-2.15B (opt)*: uses an optimized format with English scale hint ("On a scale of 1 to N:") and native-language answer cue
- *EuroLLM-22B*: A single multilingual model trained on all 22 European languages jointly, roughly 10x larger at 22B parameters. Uses the optimized prompt format.
- *Gemma-3-27B* (base): A 27B-parameter base model from the Gemma 3 family. Same tokenizer family as the HPLT models but substantially larger and trained on a broader multilingual corpus. Uses the optimized prompt format.

Gemma-3-27B-IT (instruction-tuned) was evaluated but excluded from main figures due to near-zero P_valid across all languages---instruction tuning redirects probability mass away from digit tokens, making logprob elicitation ineffective. Its results appear in supplementary materials.

The core research question: do base language models---which have never seen survey data or been instruction-tuned---internalize cultural values from their training text that align with real population-level survey responses?

The EVS 2017 Integrated Dataset (ZA7500, $n = 59,438$) provides ground truth: weighted response distributions for 186 questions across 21 of our 22 target countries (Greece is absent). LLM distributions were extracted via next-token logprob elicitation with position-bias debiasing (forward + reversed option ordering averaging).

The primary distance metric is the *Jensen--Shannon distance* (JSD), defined as $sqrt("JSD"_"div")$ where $"JSD"_"div"$ is the Jensen--Shannon divergence. This is a proper metric ($0 <= "JSD" <= sqrt(ln 2) approx 0.83$). A value of 0 means identical distributions; $sqrt(ln 2)$ means maximally different. All JSD values in this report are JS distances (square root of the divergence), computed via `scipy.spatial.distance.jensenshannon`.

Aggregate JSD metrics use *bias-weighted* means: each question receives a weight of $w = 1 - "position\_bias\_magnitude"$ (clipped to $[0, 1]$), so questions with high position bias---where the forward/reversed debiasing is least reliable---contribute less to aggregate metrics.

Cultural clusters follow the official *Inglehart--Welzel World Cultural Map* classification: Protestant Europe (Danish, Finnish, Swedish, Dutch, German), Catholic Europe (French, Italian, Spanish, Portuguese, Czech, Hungarian, Polish, Slovak, Slovenian, Croatian), English-speaking (English), Orthodox (Bulgarian, Romanian, Greek), and Baltic (Estonian, Lithuanian, Latvian).

The report follows a narrative arc: *methodology validation* (can we trust the extracted distributions?) → *aggregate alignment* (how close are LLM and human distributions overall?) → *per-question analysis* (what drives the gaps?) → *cultural geography* (do LLMs recover real cultural structure?) → *established frameworks* (does the signal map onto recognized theory?).

= F1: Methodology Validation

== F1a: P_valid Heatmap

#align(center)[#image("main/F6_pvalid_heatmap.png", width: 100%)]

=== What it shows

A heatmap of mean P_valid (fraction of next-token probability on valid response digits) by model type and language. Green = high P_valid (model understands the response format); red = low P_valid (model does not produce valid digits).

=== How it was calculated

For each (model, language, question), the P_valid is the sum of probabilities assigned to the valid response digit tokens in the next-token distribution. The heatmap shows the mean P_valid across all questions for each (model, language) pair.

=== Key observations

- *EuroLLM-22B shows the highest P_valid* ($approx 0.82$--$0.96$), suggesting its tokenizer and multilingual training produce the strongest format compliance.
- *Gemma-3-27B base shows high P_valid* ($approx 0.77$--$0.92$) across most languages, confirming that the logprob elicitation prompt format works well for large base models.
- *HPLT-2.15B shows more variable P_valid.* The optimized variant (opt) ranges from $approx 0.71$--$0.94$, while the cue variant is lower ($approx 0.64$--$0.95$), with notable weakness for Croatian ($0.65$) and Lithuanian ($0.64$).
- *Some languages show systematically lower P_valid* (Croatian, Lithuanian, Latvian), which may indicate tokenizer coverage issues or culturally unfamiliar prompt formatting.

=== Why this figure comes first

P_valid is a prerequisite quality check: if the model assigns no probability to valid response digits, the extracted distribution is meaningless. The reader needs to trust the methodology before examining results. This figure validates that the logprob elicitation approach works for the target models and identifies any language-specific issues.

#v(1em)

== F1b: Position Bias

#align(center)[#image("main/F7_position_bias.png", width: 100%)]

=== What it shows

Violin plots of position bias magnitude by model type and by response type. Position bias measures how much the probability distribution changes when the order of response options is reversed. Gemma-3-27B-IT is excluded.

=== How it was calculated

For each (model, language, question), the prompt is presented twice: once with options in the original order and once reversed. Position bias magnitude is the maximum absolute probability difference across response options between the two orderings: $"bias" = max_i |p_"fwd"(x_i) - p_"rev"(x_i)|$.

The debiased distribution used in all other analyses is the average of forward and reversed: $p_"avg"(x_i) = (p_"fwd"(x_i) + p_"rev"(x_i))/2$.

=== Key observations

- *Gemma-3-27B base shows the lowest position bias* (mean $= 0.31$, median $= 0.27$), contributing to its overall superior performance.
- *HPLT-2.15B (cue) shows the highest position bias* (mean $= 0.64$, median $= 0.67$), likely reflecting both the smaller model size and the cue-style prompt being more ambiguous. The optimized variant (opt) is substantially lower (mean $= 0.43$, median $= 0.42$).
- *EuroLLM-22B falls in between* (mean $= 0.39$, median $= 0.36$).
- *Likert-3 shows the highest position bias by response type* (median $= 0.52$), followed by Likert-4 ($0.44$) and Likert-5 ($0.41$). Likert-10 is moderate ($0.35$) and categorical/frequency are lowest ($0.36$--$0.43$).

=== Why this matters

Position bias is a methodological concern for logprob-based survey elicitation. High bias means the forward and reversed distributions are very different, and their average may not reflect the model's "true" preference. Low bias increases confidence that the debiased distribution is meaningful. This completes the methodology validation: the reader now knows both that the models produce valid responses (P_valid) and that ordering effects are controlled (debiasing).

#pagebreak()

= F2: Aggregate Alignment

== F2a: JSD Heatmap

#align(center)[#image("main/F4_jsd_heatmap.png", width: 100%)]

=== What it shows

A heatmap with rows for each model type (excluding Gemma-3-27B-IT) and 21 columns (one per language). Cell color encodes *bias-weighted* mean JSD from LLM to human data, with green = better alignment and red = worse. Each cell is annotated with the weighted mean JSD and standard deviation ($"mean" plus.minus "std"$).

=== How it was calculated

For each (model, language) pair, a *bias-weighted* mean JSD across all shared questions is computed. Each question receives a weight of $w = 1 - "position\_bias\_magnitude"$ (clipped to $[0, 1]$), so questions with high position bias contribute less to the aggregate. The weighted mean is $overline("JSD")_w = (sum_q w_q dot "JSD"_q) / (sum_q w_q)$. Gemma-3-27B-IT is excluded due to its near-zero P_valid rendering the JSD values uninformative.

=== Key observations

- *Gemma-3-27B base shows the greenest row*, confirming its systematic advantage across languages.
- *Bias weighting slightly reduces aggregate JSD* for models with high position bias (especially HPLT-2.15B), since their worst JSD questions tend to also have the highest bias.
- *The standard deviations reveal within-language heterogeneity.* Some languages show high mean JSD with low std (consistently poor), while others show moderate mean with high std (mixed---some questions are well-captured, others not).
- *The green corridor* (Spanish, French, Italian, Hungarian, Danish) shows particularly strong alignment for Gemma-3-27B, with JSD values in the $0.22$--$0.24$ range.
- *Croatian stands out as the most challenging language* across all models, with JSD values reaching $0.41$ for HPLT (cue). Lithuanian and Latvian show moderate difficulty but are not as consistently poor as Croatian.

=== Why this figure matters

Now that methodology is established (F1), this provides the first results overview: a complete picture of model $times$ language performance. Bias-weighting ensures that questions where the debiasing methodology is least reliable are downweighted, providing a more honest aggregate. The $plus.minus "std"$ annotation reveals whether poor performance is uniform (all questions bad) or driven by a subset of difficult questions.

#v(1em)

== F2b: Expected Value Scatter (Raw and Z-Scored)

#align(center)[#image("main/F3_ev_scatter.png", width: 100%)]

=== What it shows

Two rows of scatter plots, one column per model type. *Top row*: raw expected values (LLM vs human) for all ordinal questions. *Bottom row*: z-score normalized expected values. Each point is one (language, question) pair, colored by cultural cluster. Dashed line = regression fit; dotted line = identity. Pearson $r$ and Spearman $rho$ are reported per panel.

=== How it was calculated

For each ordinal question and (model, language) pair, $EE[X] = sum_i x_i dot p(x_i)$ is computed for both the LLM's debiased distribution and the human survey distribution. For the z-scored row, each question's values are standardized using *human-only reference statistics*: $z = (x - mu_h) / sigma_h$ where $mu_h$ and $sigma_h$ are the mean and standard deviation of the human expected values for that question across countries. Both human and LLM values are mapped into this human-defined reference frame, allowing direct comparison of where each source falls relative to human cross-country variation.

=== Key observations

- *Raw correlations are moderate to strong* ($r approx 0.67$--$0.72$), with Gemma-3-27B ($r = 0.72$) and EuroLLM-22B ($r = 0.72$) outperforming HPLT-2.15B ($r approx 0.67$--$0.70$).
- *Z-scoring collapses the correlation to zero.* After normalizing each question using human-only statistics, $r approx 0.00$ ($-0.007$ to $+0.008$) and $rho approx -0.03$ across all models. This reveals that the entire raw correlation was driven by cross-question scale structure (e.g., both humans and LLMs agree that 1--10 scales produce higher means than 1--3 scales). There is no detectable within-question cultural signal: LLMs do not capture which countries score above or below the cross-country mean on any given question.
- *The identity line gap is visible in raw plots*: LLM points lie closer to the center than human points, reflecting the well-documented entropy compression where LLMs produce more uniform distributions.
- *The z-scored panels show extreme outliers* (LLM z-scores up to $20$+) for questions where human cross-country variance is small ($sigma_h approx 0$), inflating LLM z-scores. These outliers do not drive the near-zero correlation, which persists in the bulk of the data.

=== Why this figure matters

This is the most important diagnostic in the report. Raw correlation measures overall agreement including scale; z-scored correlation isolates within-question relative variation. The complete collapse from $r approx 0.70$ to $r approx 0.00$ upon z-scoring means that LLMs reproduce the overall statistical properties of each response scale (mean, range) but do not differentiate between countries _within_ any given question. The apparent cultural alignment in raw correlations is entirely an artifact of scale structure.

#pagebreak()

= F3: Per-Question Deep Dive

#align(center)[#image("main/F5_deepdive.png", width: 100%)]

== What it shows

A 3$times$3 grid of bar charts showing human vs LLM response distributions for 9 selected questions across 4 representative languages (English, Finnish, Polish, Romanian---covering Western, Nordic, Central, and Southeast clusters). Three categories of questions are shown:

- *IW questions* (blue titles): Key questions from the Inglehart--Welzel dimensions (importance of God, homosexuality justifiability, gay couples as parents)
- *Best-aligned questions* (green titles): Questions with the lowest mean JSD across all model-language pairs
- *Worst-aligned questions* (red titles): Questions with the highest mean JSD

For each question, human distributions are shown as thick outlined bars and the best-performing model's distributions as filled bars.

== How it was calculated

Questions are selected by: (1) the 3 IW questions from the Inglehart--Welzel dimension definitions, (2) the 3 lowest mean JSD questions not already in the IW set, and (3) the 3 highest mean JSD questions. For each question, the best-performing model (lowest mean JSD across languages, excluding Gemma-3-27B-IT) is used for the LLM distribution. The 4 languages span the cultural diversity of the dataset.

== Key observations

- *IW questions show meaningful cross-cultural variation.* Human distributions for "importance of God" differ dramatically between Finnish (low) and Polish/Romanian (high), and LLMs capture this direction.
- *Best-aligned questions* are typically binary or short-list items where getting the majority direction right yields low JSD.
- *Worst-aligned questions* show the "flattening" pattern: humans have a strong mode (50--80% on one option) while LLMs distribute mass more uniformly. This is the primary failure mode of logprob elicitation.
- *Cross-language variation within a question* is visible: the same question produces different distributions for different countries, and LLMs partially track these differences.

== Why this figure matters

Aggregate metrics (JSD, correlation) hide distributional shape. This figure shows _how_ LLM distributions differ from human ones at the question level, revealing the entropy compression mechanism and identifying which questions are most and least amenable to logprob elicitation. It bridges from aggregate results to the spatial analysis by introducing the concept that some questions carry cultural signal while others are dominated by methodological noise.

#pagebreak()

= F4: Per-Question JSD Distribution

#align(center)[#image("main/F4_jsd_distribution.png", width: 90%)]

== What it shows

A histogram of per-question mean JSD (averaged across all models and languages), with vertical lines marking filtering thresholds. In the cultural geography analysis (F5), questions are filtered by a combination of JSD and position bias thresholds to remove those where the methodology is least reliable.

== How it was calculated

For each of the 149 ordinal questions, the mean JSD is computed across all non-excluded (model, language) pairs. The histogram shows the distribution of these per-question means, revealing which questions are well-captured by the methodology and which are not.

== Key observations

- *The distribution has a main body centered around 0.25--0.32*, with a sparse tail extending to $approx 0.52$. Most questions fall in a moderate JSD range.
- *The 0.35 threshold* sits at the transition from the main body to the tail, retaining 158 of 186 questions overall.
- *The 0.25 threshold* cuts into the main body, retaining only the 57 best-aligned questions of 186 overall.

== Why this figure matters

This figure establishes the rationale for the progressive filtering in the UMAP analysis (F5). Rather than choosing thresholds arbitrarily, the histogram shows a natural break point in the data. The thresholds partition questions into three meaningful groups: the tail (methodology failures), the body (moderate alignment), and the core (strong alignment). The reader can now interpret the UMAP filtering as principled rather than cherry-picked.

#pagebreak()

= F5: Cultural Geography --- PCA + UMAP

These panels show the cultural geography of LLM and human data using a *PCA-first* approach: PCA is fitted on the 21-country human expected-value matrix, LLM data is projected into the same space, per-model-family variance matching is applied, and UMAP reduces the top 5 PCA components to 2D for visualization. Panels are shown at progressive quality cutoffs combining JSD threshold (distributional distance) and position-bias threshold (prompt insensitivity).

Human points are large circles with bold labels; LLM points are smaller model-specific markers. Gray lines connect LLM points to their corresponding human country. Points are colored by official Inglehart--Welzel cultural cluster.

#align(center)[#image("main/F5_umap_legend.png", width: 90%)]

== No filter (149 questions)
#align(center)[#image("main/F5_pca_nofilter.png", width: 85%)]

== Mild quality filter (JSD < 0.40, Bias < 0.55)
#align(center)[#image("main/F5_pca_q1.png", width: 85%)]

== Moderate quality filter (JSD < 0.35, Bias < 0.50)
#align(center)[#image("main/F5_pca_q2.png", width: 85%)]

== Strict quality filter (JSD < 0.30, Bias < 0.45)
#align(center)[#image("main/F5_pca_q3.png", width: 85%)]

== How they were calculated

For each source (human country or LLM model-language pair) and each ordinal question, the expected value $EE[X] = sum_i x_i dot p(x_i)$ is computed, yielding a feature vector per source. Missing values are imputed (mean strategy, fit on human data).

*Human-fitted centering*: human points remain at their natural coordinates (defining the reference frame), while each LLM model family is shifted so its centroid aligns with the human centroid: $x'_("LLM",i) = x_("LLM",i) - overline(x)_m + overline(x)_"human"$.

*PCA fitted on human data*: PCA is fitted on the 21 human country vectors. This yields a coordinate system defined by human cultural variation. The top 5 components capture $approx 80%$ of variance (see scree plot below). LLM data is projected into this human-defined space using the same PCA transformation.

*Per-family variance matching*: LLM models produce compressed distributions, resulting in artificially small variance in PCA space. For each model family independently, the LLM projections are rescaled per-axis so their mean and standard deviation match the human statistics: $x'_j = (x_j - mu_("family",j)) dot sigma_("human",j) / sigma_("family",j) + mu_("human",j)$. This removes the compression artifact while preserving relative structure within each model family.

*UMAP* (`n_neighbors=10, min_dist=0.3`) is applied to the variance-matched PCA-5 coordinates. Trustworthiness $T$ (from `sklearn.manifold.trustworthiness`, $k=10$) quantifies how well local neighborhoods are preserved.

*Quality filtering* excludes questions exceeding both a JSD threshold (high LLM--human distributional distance) and a position-bias threshold (high sensitivity to option ordering). Both criteria indicate unreliable methodology for that question, independent of whether the underlying cultural signal exists.

== Scree plot
#align(center)[#image("main/F5_pca_scree.png", width: 70%)]

The first principal component captures $approx 55%$ of human cultural variance. The top 5 components capture $approx 81%$, providing a reasonable low-dimensional summary.

== Key observations

- *The PCA-first approach is necessary.* Direct UMAP on the full 149-dimensional expected-value space places LLM and human data in separate regions regardless of centering, because the per-question cultural signal is near zero ($r approx 0.00$ after z-scoring) and too noisy for UMAP to detect in high dimensions. PCA preprocessing extracts the dominant human cultural dimensions, and variance matching corrects for LLM entropy compression.
- *Human cultural cluster structure is visible.* In most panels, Protestant European countries (Danish, Swedish, Finnish, Dutch, German) cluster together, and Orthodox/traditional countries (Bulgarian, Romanian, Croatian, Polish) form a separate group, consistent with the Inglehart--Welzel cultural map.
- *LLM points do not consistently land near their corresponding human countries.* While LLM model families form their own clusters in the variance-matched space, individual language points are often far from their human counterpart. This is consistent with the z-scored correlation finding ($r approx 0.00$): there is no detectable per-question cultural signal, so country-level spatial correspondence is not expected.
- *Stricter quality filtering modestly improves structure.* Trustworthiness increases from $T = 0.915$ (no filter) to $T = 0.943$ (strict), but the improvement is gradual rather than transformative.
- *Formal alignment tests (Procrustes, RSA Mantel test) do not reach statistical significance.* With only 21 countries and 5 PCA dimensions, neither rotation-based alignment nor distance-matrix correlation yields $p < 0.05$ for any model family or quality filter level. Given the absence of within-question cultural signal ($r approx 0.00$), this null result is expected.

#pagebreak()

= F6: Inglehart--Welzel Cultural Map

#align(center)[#image("main/F2_inglehart_welzel.png", width: 90%)]

== What it shows

A scatter plot reproducing the classic Inglehart--Welzel World Cultural Map layout, with the x-axis representing Survival→Self-Expression values and the y-axis representing Traditional→Secular-Rational values. Both human EVS data (large circles) and LLM distributions (smaller model-specific markers) are plotted, with gray lines connecting LLM points to their corresponding human country.

== How it was calculated

Following Inglehart and Welzel's methodology, composite scores are computed from 10 EVS questions (5 per dimension):

*Traditional→Secular-Rational*: importance of God (v63, flipped), religion importance (v6), abortion justifiability (v154), obedience in children (v95), independence in children (v86, flipped).

*Survival→Self-Expression*: life satisfaction (v39), interpersonal trust (v31, flipped), gay couples as parents (v82, flipped), homosexuality justifiability (v153), petition signing (v98, flipped).

For each question, the expected value is computed from the probability distribution, polarity-flipped where needed (so higher = more secular/self-expression), then *quantile-normalized* across all sources (human + LLM) jointly: values are ranked and mapped to uniform quantiles on $[0, 1]$. This replaces z-scoring to prevent LLM answer banding from compressing the composite scores. Each dimension's score is the mean of its 5 quantile-normalized items (requiring $>=3$ of 5 to be present), yielding "mean quantile rank" scores where 0 = lowest across all sources, 0.5 = median, and 1 = highest.

== Key observations

- *The human IW map matches the reference.* Protestant European countries appear in the top-right (secular-rational, self-expression), Orthodox countries in the bottom-left (traditional, survival), validating the composite construction. Cultural clusters follow the official Inglehart--Welzel World Cultural Map classification (Protestant Europe, Catholic Europe, English-speaking, Orthodox, Baltic). Note: Greek is absent due to missing EVS coverage.
- *LLM points cluster tightly near the center of the plot.* Due to entropy compression, LLM expected values have less cross-country variance than humans, and the quantile normalization maps this compressed range to intermediate ranks. The between-language variation in LLM composites is small relative to the human spread.
- *Within the LLM cluster, there is modest directional structure.* Models trained on Protestant European languages sit slightly higher/right-er than those trained on Orthodox/traditional languages, preserving the broad rank ordering of cultural clusters. However, the effect size is small.
- *Gray lines radiate outward from the LLM center to the human periphery*, reflecting entropy compression (all LLM points pulled toward the center) rather than a consistent directional shift.
- *Quantile normalization inflates apparent alignment.* Because values are ranked jointly across human and LLM sources, even LLMs with near-constant expected values receive spread-out quantile ranks. This prevents assessment of absolute effect size.

== Why this figure matters

After the data-driven UMAP (F5), this figure tests whether LLMs reproduce cultural structure along _interpretable, validated dimensions_. The human IW map is well-reproduced, and LLM points show weak but directionally correct rank ordering across cultural clusters. However, the result should be interpreted cautiously: the quantile normalization guarantees some spread, and the LLM-to-human gap (entropy compression) is much larger than the between-language LLM variation. This figure demonstrates that LLMs preserve a coarse _rank ordering_ of cultural clusters on curated questions, but not the magnitude of cultural differentiation.

#pagebreak()

= Summary of Conclusions

+ *The apparent cultural signal is driven entirely by scale structure.* Raw Pearson $r = 0.67$--$0.72$ between LLM and human expected values (F2b) drops to $r approx 0.00$ (range $-0.007$ to $+0.008$) after z-scoring with human-only reference statistics. This means the raw correlation reflects cross-question scale agreement (e.g., 1--10 scales produce higher means than 1--3 scales) rather than genuine within-question cultural sensitivity. LLMs do not capture cross-country variation within individual questions---they merely reproduce the overall shape of each scale.

+ *The methodology works but has clear limitations.* EuroLLM-22B shows the highest P_valid ($approx 0.82$--$0.96$), followed by Gemma-3-27B ($approx 0.78$--$0.92$) and HPLT-2.15B ($approx 0.64$--$0.95$, with the cue variant showing more weakness). Position bias is substantial: HPLT-2.15B (cue) has the highest (median $= 0.67$), while Gemma-3-27B has the lowest (median $= 0.27$). Instruction-tuned models (Gemma-3-27B-IT) fail entirely with near-zero P_valid. Note: all JSD values in this report are Jensen--Shannon _distances_ ($sqrt("JSD"_"div")$), bounded by $sqrt(ln 2) approx 0.83$.

+ *Model scale dominates over language specificity.* Gemma-3-27B (27B params, general multilingual) achieves the lowest mean JSD across nearly every language (F2a), outperforming both the European-specialized EuroLLM-22B and the language-specific HPLT-2.15B monolingual models. This suggests that model capacity for understanding the question format matters more than language-specific training data for this methodology.

+ *LLMs compress distributions.* The primary failure mode is entropy compression: LLMs produce more uniform distributions than humans, who often show strong consensus peaks. This compression dominates the JSD and manifests as systematically reduced variance in PCA space, requiring explicit variance matching for spatial analysis (F5).

+ *Country-level cultural geography does not reach statistical significance.* While human points in the PCA-based UMAP show recognizable Inglehart--Welzel cluster structure (Protestant Europe, Catholic Europe, Orthodox), LLM points do not consistently land near their corresponding human countries (F5). Formal tests (Procrustes alignment, RSA Mantel test) fail to reach significance ($p > 0.05$) for any model family. Given that the within-question cultural signal is effectively zero ($r approx 0.00$ after z-scoring), the absence of country-level geometry is expected rather than a power issue.

+ *The Inglehart--Welzel composite provides partial spatial structure.* By aggregating across only 10 carefully chosen questions with known cultural valence (F6), the IW composite scores show LLM points broadly tracking the recognized cultural map: Nordic countries appear more secular/self-expressive and Southeast European countries appear more traditional/survival-oriented. However, LLM points cluster tightly in the center of the plot due to entropy compression, and quantile normalization (used to counteract this compression) inflates apparent alignment by spreading near-constant values across the rank scale. This result demonstrates that LLMs preserve _rank ordering_ of countries on curated cultural dimensions, but not the magnitude of cultural differences.

+ *Question type matters.* Binary/categorical questions are best aligned (low JSD); Likert-4 and Likert-10 scales are the most challenging and show the highest position bias. Likert-3 and Likert-5 are comparable to categorical questions. The per-question JSD distribution (F4) reveals a natural break between well-captured and poorly-captured questions.

+ *Language resource level predicts alignment, driven mainly by Croatian.* High-resource languages (French, Spanish, Italian) show consistently lower JSD. Croatian stands out as the most challenging language across all models. Lithuanian and Latvian show moderate difficulty but are not as consistently poor. This pattern is consistent across model families, suggesting it reflects training data coverage rather than model architecture.
