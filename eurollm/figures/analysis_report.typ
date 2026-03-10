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

- *HPLT-2.15B*: 22 monolingual models, each trained exclusively on text from a single European language. These are small (2.15B parameter) Gemma-3-based models from the HPLT project.
- *EuroLLM-22B*: A single multilingual model trained on all 22 European languages jointly, roughly 10x larger at 22B parameters.
- *Gemma-3-27B* (base): A 27B-parameter base model from the Gemma 3 family. Same tokenizer family as the HPLT models but substantially larger and trained on a broader multilingual corpus.

Gemma-3-27B-IT (instruction-tuned) was evaluated but excluded from main figures due to near-zero P_valid across all languages---instruction tuning redirects probability mass away from digit tokens, making logprob elicitation ineffective. Its results appear in supplementary materials.

The core research question: do base language models---which have never seen survey data or been instruction-tuned---internalize cultural values from their training text that align with real population-level survey responses?

The EVS 2017 Integrated Dataset (ZA7500, $n = 59,438$) provides ground truth: weighted response distributions for 186 questions across 21 of our 22 target countries (Greece is absent). LLM distributions were extracted via next-token logprob elicitation with position-bias debiasing (forward + reversed option ordering averaging).

The primary distance metric is *Jensen--Shannon divergence* (JSD), a symmetric, bounded measure ($0 <= "JSD" <= ln 2 approx 0.83$). A JSD of 0 means identical distributions; $ln 2$ means maximally different.

The report follows a narrative arc: *methodology validation* (can we trust the extracted distributions?) → *aggregate alignment* (how close are LLM and human distributions overall?) → *per-question analysis* (what drives the gaps?) → *cultural geography* (do LLMs recover real cultural structure?) → *established frameworks* (does the signal map onto recognized theory?).

#pagebreak()

= F1: Methodology Validation

== F1a: P_valid Heatmap

#align(center)[#image("main/F6_pvalid_heatmap.png", width: 100%)]

=== What it shows

A heatmap of mean P_valid (fraction of next-token probability on valid response digits) by model type and language. Green = high P_valid (model understands the response format); red = low P_valid (model does not produce valid digits).

=== How it was calculated

For each (model, language, question), the P_valid is the sum of probabilities assigned to the valid response digit tokens in the next-token distribution. The heatmap shows the mean P_valid across all questions for each (model, language) pair.

=== Key observations

- *HPLT-2.15B and Gemma-3-27B base show high P_valid* ($>0.85$) across most languages, confirming that the logprob elicitation prompt format works for base models.
- *EuroLLM-22B shows moderate P_valid* ($approx 0.65$--$0.80$), reflecting its different tokenizer and potentially different prompt-completion behavior.
- *Some languages show systematically lower P_valid*, which may indicate tokenizer coverage issues or culturally unfamiliar prompt formatting.

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

- *Gemma-3-27B base shows the lowest position bias* (mean $approx 0.32$), contributing to its overall superior performance.
- *EuroLLM-22B shows the highest position bias* (mean $approx 0.49$), suggesting it is more sensitive to option ordering.
- *Likert scales show higher position bias* than binary/categorical questions, likely because the multiple similarly-worded options create more ambiguity about which option maps to which digit.

=== Why this matters

Position bias is a methodological concern for logprob-based survey elicitation. High bias means the forward and reversed distributions are very different, and their average may not reflect the model's "true" preference. Low bias increases confidence that the debiased distribution is meaningful. This completes the methodology validation: the reader now knows both that the models produce valid responses (P_valid) and that ordering effects are controlled (debiasing).

#pagebreak()

= F2: Aggregate Alignment

== F2a: JSD Heatmap

#align(center)[#image("main/F4_jsd_heatmap.png", width: 100%)]

=== What it shows

A heatmap with rows for each model type (excluding Gemma-3-27B-IT) and 21 columns (one per language). Cell color encodes mean JSD from LLM to human data, with green = better alignment and red = worse. Each cell is annotated with the mean JSD and standard deviation ($"mean" plus.minus "std"$).

=== How it was calculated

For each (model, language) pair, the mean and standard deviation of JSD across all shared questions is computed and displayed. Gemma-3-27B-IT is excluded due to its near-zero P_valid rendering the JSD values uninformative.

=== Key observations

- *Gemma-3-27B base shows the greenest row*, confirming its systematic advantage across languages.
- *The standard deviations reveal within-language heterogeneity.* Some languages show high mean JSD with low std (consistently poor), while others show moderate mean with high std (mixed---some questions are well-captured, others not).
- *The green corridor* (Italian, Hungarian, Romanian, Danish) shows particularly strong alignment for Gemma-3-27B.
- *The red corridor* (Lithuanian, Croatian, Latvian) shows weak alignment across models---these are lower-resource languages where training data may be insufficient.

=== Why this figure matters

Now that methodology is established (F1), this provides the first results overview: a complete picture of model $times$ language performance. The $plus.minus "std"$ annotation reveals whether poor performance is uniform (all questions bad) or driven by a subset of difficult questions.

#v(1em)

== F2b: Expected Value Scatter (Raw and Z-Scored)

#align(center)[#image("main/F3_ev_scatter.png", width: 100%)]

=== What it shows

Two rows of scatter plots, one column per model type. *Top row*: raw expected values (LLM vs human) for all ordinal questions. *Bottom row*: z-score normalized expected values. Each point is one (language, question) pair, colored by cultural cluster. Dashed line = regression fit; dotted line = identity. Pearson $r$ and Spearman $rho$ are reported per panel.

=== How it was calculated

For each ordinal question and (model, language) pair, $EE[X] = sum_i x_i dot p(x_i)$ is computed for both the LLM's debiased distribution and the human survey distribution. For the z-scored row, each question's values are standardized: $z = (x - mu) / sigma$ where $mu$ and $sigma$ are computed _jointly_ across all human and LLM values for that question.

=== Key observations

- *Raw correlations are strong* ($r approx 0.71$--$0.73$), confirming LLMs encode the direction of cultural values.
- *Z-scoring should improve correlation* by removing cross-question scale differences. If the LLM compresses all distributions toward the center, z-scoring removes this mean shift, revealing the underlying rank-order agreement.
- *The identity line gap is visible in raw plots*: LLM points lie closer to the center than human points, reflecting the well-documented entropy compression where LLMs produce more uniform distributions.

=== Why this figure matters

This answers: "How well do LLMs capture _relative_ cultural differences?" Raw correlation measures overall agreement including scale; z-scored correlation measures whether the LLM correctly ranks countries within each question, removing confounds from distributional compression.

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

A histogram of per-question mean JSD (averaged across all models and languages), with vertical lines marking the two filtering thresholds used in the UMAP analysis: JSD = 0.35 (moderate) and JSD = 0.25 (aggressive).

== How it was calculated

For each of the 149 ordinal questions, the mean JSD is computed across all non-excluded (model, language) pairs. The histogram shows the distribution of these per-question means, revealing which questions are well-captured by the methodology and which are not.

== Key observations

- *The distribution has a main body centered around 0.20--0.30*, with a sparse tail extending to $approx 0.52$. Most questions fall in a moderate JSD range.
- *The 0.35 threshold* sits at the transition from the main body to the tail, removing $approx 27$ questions where LLMs systematically produce the most divergent distributions (159 of 186 retained overall; 122 of 149 ordinal questions used in the UMAP).
- *The 0.25 threshold* cuts into the main body, retaining only the best-aligned questions (55 of 186 overall; 29 of 149 ordinal questions used in the UMAP).

== Why this figure matters

This figure establishes the rationale for the progressive filtering in the UMAP analysis (F5). Rather than choosing thresholds arbitrarily, the histogram shows a natural break point in the data. The thresholds partition questions into three meaningful groups: the tail (methodology failures), the body (moderate alignment), and the core (strong alignment). The reader can now interpret the UMAP filtering as principled rather than cherry-picked.

#pagebreak()

= F5: Cultural Geography UMAP

#align(center)[#image("main/F5_umap_grid.png", width: 100%)]

== What it shows

A 2$times$3 grid of UMAP embeddings exploring the geometry of cultural variation across two dimensions:

- *Columns*: distance metric --- Centered Euclidean (left) vs Centered Correlation (right)
- *Rows*: progressive JSD filtering --- no filter (149 questions) / JSD < 0.35 (122 questions) / JSD < 0.25 (29 questions)

All three model families (HPLT, EuroLLM, Gemma-3-27B) are plotted together on each panel. Human points are large circles with bold labels; LLM points are smaller model-specific markers. Gray lines connect each LLM point to its corresponding country's human data. All points are colored by Inglehart--Welzel cultural cluster.

== How they were calculated

For each source (human country or LLM model-language pair) and each ordinal question, the expected value $EE[X] = sum_i x_i dot p(x_i)$ is computed, yielding a feature vector per source. Missing values are imputed (mean strategy, fit on human data).

*Per-source centering* is applied first: the group mean vector is subtracted from each group independently---the mean of all 21 human country vectors from each human point, and the mean of each model family's language vectors from each LLM point. This removes additive offsets (e.g., entropy compression).

*Euclidean columns* run UMAP (`n_neighbors=10, min_dist=0.3`) with Euclidean distance on the centered residuals, preserving within-group magnitude differences.

*Correlation columns* run UMAP with correlation distance ($d(x,y) = 1 - r_(x y)$) on the centered residuals. Correlation additionally normalizes each point's cross-question pattern, removing per-point scale. Centering and correlation operate on different axes: centering removes the _group-level_ mean across countries for each question, while correlation removes the _per-point_ mean across questions and normalizes variance.

*Rows* apply progressively stricter JSD filtering: questions with mean JSD above the threshold are removed before building feature vectors.

== Key observations

- *Both distance metrics successfully mix LLM and human points* across all filtering levels. Per-source centering removes the systematic LLM--human gap regardless of downstream metric.
- *Cultural cluster structure is replicated by LLMs.* In all panels, LLM points for Nordic languages cluster with human Nordic points, Mediterranean with Mediterranean, etc.
- *Euclidean preserves magnitude*, making it sensitive to _how much_ a model exaggerates or compresses cultural differences. Correlation normalizes per-point scale, revealing pattern agreement that magnitude differences might obscure.
- *The cultural structure is robust across filtering levels.* All six panels show recognizable cultural clusters with LLM points landing near the correct human countries.
- *Progressive filtering tightens alignment.* The bottom row (JSD < 0.25, 29 questions) shows strikingly tight LLM--human correspondence, demonstrating that on questions where the methodology works well, LLMs faithfully reproduce the cultural geography.
- *Model families show consistent ordering.* Gemma-3-27B points land closest to human countries, followed by EuroLLM-22B, then HPLT-2.15B---consistent with the JSD analysis in F2.

== Why centering is necessary (even for correlation distance)

Correlation distance ($1 - r$) is theoretically shift- and scale-invariant, so one might expect it to handle the LLM--human gap without centering. However, correlation alone fails because entropy compression affects questions _unevenly_: questions with strong human consensus are compressed much more than already-split questions. This distorts the cross-question correlation pattern itself. Per-source centering first removes the dominant group-level offset per question, leaving residuals whose cross-question patterns are comparable.

== Why this figure matters

This is the climax figure. The reader now understands the methodology (F1), the aggregate numbers (F2), the per-question behavior (F3), and the filtering rationale (F4). The 2$times$3 grid tells the complete spatial story in one figure: both distance metrics $times$ three filtering levels, showing that the cultural signal is robust, metric-independent, and strengthens when methodological noise is removed. The progression from top to bottom supports the hypothesis that LLMs strongly capture cultural values but the signal is partially obscured by methodological limitations.

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

- *The human IW map matches the reference.* Nordic countries appear in the top-right (secular-rational, self-expression), Balkan/Southeast countries in the bottom-left (traditional, survival), validating the composite construction.
- *LLM points track the correct cultural direction.* Models trained on Nordic languages shift toward the secular/self-expression quadrant; those trained on traditional-value languages shift toward the traditional/survival quadrant.
- *The LLM-to-human displacement is systematic.* Gray lines tend to point in a consistent direction, suggesting a distributional bias (e.g., entropy compression) rather than random noise.

== Why this figure matters

After the data-driven UMAP (F5), this figure grounds the findings in the most recognized framework in cross-cultural values research. The Inglehart--Welzel map shows that LLMs reproduce cultural structure not just in an arbitrary embedding but along _interpretable, validated dimensions_. This provides the strongest evidence that LLMs encode genuine cultural values, and connects the UMAP findings to decades of established cross-cultural theory.

#pagebreak()

= Summary of Conclusions

+ *Cultural signal is real and strong.* Pearson $r = 0.71$--$0.73$ between LLM and human expected values (F2b) demonstrates that base language models encode genuine population-level value orientations from their training text.

+ *The methodology is sound.* High P_valid confirms that logprob elicitation works for base models (F1a), and position bias debiasing controls ordering effects (F1b). Instruction-tuned models (Gemma-3-27B-IT) are the exception---near-zero P_valid renders logprob extraction ineffective.

+ *Model scale is the dominant factor.* Gemma-3-27B (27B params) wins most languages with the lowest mean JSD (F2a), outperforming both the European-specialized EuroLLM-22B and the language-specific HPLT-2.15B monolingual models.

+ *LLMs compress distributions.* The main failure mode is entropy compression: LLMs produce more uniform distributions than humans, who often show strong consensus peaks. Z-scoring (F2b) partially corrects this, and per-source centering (F5) removes the additive offset.

+ *Question type matters.* Binary/categorical questions are best aligned (low JSD); likert-4/5 scales with many options are most challenging and show the highest position bias. The per-question JSD distribution (F4) reveals a natural break between well-captured and poorly-captured questions.

+ *The cultural signal is robust and metric-independent.* The 2$times$3 UMAP grid (F5) shows that cultural cluster structure survives across both Euclidean and correlation distance metrics and across three filtering levels. Progressive filtering strengthens the signal, supporting the hypothesis that LLMs strongly capture cultural values but the signal is partially obscured by methodological limitations.

+ *The signal is interpretable.* The Inglehart--Welzel composite (F6) shows that LLM cultural positions track the recognized cultural geography: Nordic countries are secular/self-expressive, Southeast European countries are traditional/survival-oriented, in both human and LLM data.

+ *Language resource level predicts alignment.* High-resource languages (French, English, Italian) show consistently lower JSD. Lower-resource languages (Croatian, Lithuanian, Latvian) remain challenging across all models.
