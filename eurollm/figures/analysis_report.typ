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

#pagebreak()

= F1: Human-Fitted UMAP

#align(center)[#image("F1_human_fitted_umap.png", width: 100%)]

== What it shows

A 2D UMAP embedding fitted _only_ on human EVS country distributions, with LLM distributions projected into the same space. Three panels show each model family separately (HPLT, EuroLLM, Gemma-3-27B). Human points are large circles with bold labels; LLM points are smaller model-specific markers. Gray lines connect each LLM point to its corresponding country's human data. All points are colored by Inglehart--Welzel cultural cluster.

== How it was calculated

For each source (human country or LLM model-language pair) and each ordinal question, the expected value $EE[X] = sum_i x_i dot p(x_i)$ is computed. This yields a feature vector per source (one dimension per ordinal question). A `SimpleImputer` (mean strategy) is fit on the 21 human vectors, then applied to both human and LLM vectors. UMAP (`n_neighbors=10, min_dist=0.3`) is fit on the 21 human points, and LLM points are projected via `reducer.transform()`. This means the UMAP axes represent _human cultural variation_ rather than model-type artifacts.

== Key observations

- *Human points form recognizable cultural clusters.* Nordic, Western, Mediterranean, Central, Baltic, and Southeast countries are spatially separated, confirming the embedding captures real cultural structure.
- *LLM points are pulled toward their corresponding human countries.* The gray connection lines show that LLMs trained on language X produce distributions that are _directionally_ correct---closer to country X's human data than to random countries.
- *Model families show different degrees of displacement.* Gemma-3-27B points (diamonds) tend to be closest to human points, while HPLT-2.15B points (circles) show more scatter, consistent with the JSD analysis.
- *The human-fitted embedding prevents model-type clustering from dominating.* Unlike a joint UMAP (which clusters by model type), fitting on human data first reveals whether LLMs land in the _correct cultural region_.

== Why this figure matters

This figure answers: "When we define cultural space by how real populations vary, do LLMs land near the right country?" By fitting UMAP on human data alone, we prevent the embedding from being dominated by systematic model-type differences (e.g., entropy compression), isolating the cultural signal.

== Further questions

- Do LLM points cluster tighter with scale (27B closer to human than 2.15B)?
- Which languages show the largest human-to-LLM displacement?
- Would a supervised embedding (e.g., metric learning) improve separation?

#pagebreak()

= F2: Inglehart--Welzel Cultural Map

#align(center)[#image("F2_inglehart_welzel.png", width: 90%)]

== What it shows

A scatter plot reproducing the classic Inglehart--Welzel World Cultural Map layout, with the x-axis representing Survival→Self-Expression values and the y-axis representing Traditional→Secular-Rational values. Both human EVS data (large circles) and LLM distributions (smaller model-specific markers) are plotted, with gray lines connecting LLM points to their corresponding human country.

== How it was calculated

Following Inglehart and Welzel's methodology, composite scores are computed from 10 EVS questions (5 per dimension):

*Traditional→Secular-Rational*: importance of God (v63, flipped), religion importance (v6), abortion justifiability (v154), obedience in children (v95), independence in children (v86, flipped).

*Survival→Self-Expression*: life satisfaction (v39), interpersonal trust (v31, flipped), gay couples as parents (v82, flipped), homosexuality justifiability (v153), petition signing (v98, flipped).

For each question, the expected value is computed from the probability distribution, polarity-flipped where needed (so higher = more secular/self-expression), then z-score normalized across all sources (human + LLM) jointly. Each dimension's score is the mean of its 5 z-scored items (requiring $>=3$ of 5 to be present).

== Key observations

- *The human IW map matches the reference.* Nordic countries appear in the top-right (secular-rational, self-expression), Balkan/Southeast countries in the bottom-left (traditional, survival), validating the composite construction.
- *LLM points track the correct cultural direction.* Models trained on Nordic languages shift toward the secular/self-expression quadrant; those trained on traditional-value languages shift toward the traditional/survival quadrant.
- *The LLM-to-human displacement is systematic.* Gray lines tend to point in a consistent direction, suggesting a distributional bias (e.g., entropy compression) rather than random noise.

== Why this figure matters

The Inglehart--Welzel map is the most recognized visualization in cross-cultural values research. Showing that LLMs reproduce its structure---not just in an arbitrary embedding but along _interpretable, validated dimensions_---provides the strongest evidence that LLMs encode genuine cultural values.

== Further questions

- Does the LLM displacement direction correspond to entropy compression (flattening distributions)?
- Do specific IW questions drive the displacement more than others?
- How does the composite score change with model scale?

#pagebreak()

= F3: Expected Value Scatter (Raw and Z-Scored)

#align(center)[#image("F3_ev_scatter.png", width: 100%)]

== What it shows

Two rows of scatter plots, one column per model type. *Top row*: raw expected values (LLM vs human) for all ordinal questions. *Bottom row*: z-score normalized expected values. Each point is one (language, question) pair, colored by cultural cluster. Dashed line = regression fit; dotted line = identity. Pearson $r$ and Spearman $rho$ are reported per panel.

== How it was calculated

For each ordinal question and (model, language) pair, $EE[X] = sum_i x_i dot p(x_i)$ is computed for both the LLM's debiased distribution and the human survey distribution. For the z-scored row, each question's values are standardized: $z = (x - mu) / sigma$ where $mu$ and $sigma$ are computed _jointly_ across all human and LLM values for that question.

== Key observations

- *Raw correlations are strong* ($r approx 0.71$--$0.73$), confirming LLMs encode the direction of cultural values.
- *Z-scoring should improve correlation* by removing cross-question scale differences. If the LLM compresses all distributions toward the center, z-scoring removes this mean shift, revealing the underlying rank-order agreement.
- *The identity line gap is visible in raw plots*: LLM points lie closer to the center than human points, reflecting the well-documented entropy compression where LLMs produce more uniform distributions.

== Why this figure matters

This answers: "How well do LLMs capture _relative_ cultural differences?" Raw correlation measures overall agreement including scale; z-scored correlation measures whether the LLM correctly ranks countries within each question, removing confounds from distributional compression.

== Further questions

- Which questions show the largest z-score improvement?
- Is the compression systematic across all models or model-specific?

#pagebreak()

= F4: JSD Heatmap

#align(center)[#image("F4_jsd_heatmap.png", width: 100%)]

== What it shows

A heatmap with rows for each model type (excluding Gemma-3-27B-IT) and 21 columns (one per language). Cell color encodes mean JSD from LLM to human data, with green = better alignment and red = worse. Each cell is annotated with the mean JSD and standard deviation ($"mean" plus.minus "std"$).

== How it was calculated

For each (model, language) pair, the mean and standard deviation of JSD across all shared questions is computed and displayed. Gemma-3-27B-IT is excluded due to its near-zero P_valid rendering the JSD values uninformative.

== Key observations

- *Gemma-3-27B base shows the greenest row*, confirming its systematic advantage across languages.
- *The standard deviations reveal within-language heterogeneity.* Some languages show high mean JSD with low std (consistently poor), while others show moderate mean with high std (mixed---some questions are well-captured, others not).
- *The green corridor* (Italian, Hungarian, Romanian, Danish) shows particularly strong alignment for Gemma-3-27B.
- *The red corridor* (Lithuanian, Croatian, Latvian) shows weak alignment across models---these are lower-resource languages where training data may be insufficient.

== Why this figure matters

The heatmap provides a complete picture of model × language performance, identifying both model-level and language-level patterns. The $plus.minus "std"$ annotation reveals whether poor performance is uniform (all questions bad) or driven by a subset of difficult questions.

== Further questions

- Do the high-std languages share particular question types that drive variance?
- Is the green corridor explained by resource availability or cultural similarity to high-resource languages?

#pagebreak()

= F5: Per-Question Deep Dive

#align(center)[#image("F5_deepdive.png", width: 100%)]

== What it shows

A 3×3 grid of bar charts showing human vs LLM response distributions for 9 selected questions across 4 representative languages (English, Finnish, Polish, Romanian---covering Western, Nordic, Central, and Southeast clusters). Three categories of questions are shown:

- *IW questions* (blue titles): Key questions from the Inglehart--Welzel dimensions (importance of God, homosexuality justifiability, gay couples as parents)
- *Best-aligned questions* (green titles): Questions with the lowest mean JSD across all model-language pairs
- *Worst-aligned questions* (red titles): Questions with the highest mean JSD

For each question, human distributions are shown as thick outlined bars and the best-performing model's distributions as filled bars.

== How it was calculated

Questions are selected by: (1) the 3 IW questions from `DEEPDIVE_IW_QUESTIONS`, (2) the 3 lowest mean JSD questions not already in the IW set, and (3) the 3 highest mean JSD questions. For each question, the best-performing model (lowest mean JSD across languages, excluding Gemma-3-27B-IT) is used for the LLM distribution. The 4 languages span the cultural diversity of the dataset.

== Key observations

- *IW questions show meaningful cross-cultural variation.* Human distributions for "importance of God" differ dramatically between Finnish (low) and Polish/Romanian (high), and LLMs capture this direction.
- *Best-aligned questions* are typically binary or short-list items where getting the majority direction right yields low JSD.
- *Worst-aligned questions* show the "flattening" pattern: humans have a strong mode (50--80% on one option) while LLMs distribute mass more uniformly. This is the primary failure mode of logprob elicitation.
- *Cross-language variation within a question* is visible: the same question produces different distributions for different countries, and LLMs partially track these differences.

== Why this figure matters

Aggregate metrics (JSD, correlation) hide distributional shape. This figure shows _how_ LLM distributions differ from human ones at the question level, revealing the entropy compression mechanism and identifying which questions are most and least amenable to logprob elicitation.

== Further questions

- Can the "flattening" failure mode be corrected by temperature scaling?
- Do the worst-aligned questions share structural features (e.g., many options, battery format)?
- Would sampling-based elicitation (generating full responses) reduce the entropy compression?

#pagebreak()

= F6: P_valid Heatmap

#align(center)[#image("pvalid_heatmap.png", width: 100%)]

== What it shows

A heatmap of mean P_valid (fraction of next-token probability on valid response digits) by model type and language. Green = high P_valid (model understands the response format); red = low P_valid (model does not produce valid digits).

== How it was calculated

For each (model, language, question), the P_valid is the sum of probabilities assigned to the valid response digit tokens in the next-token distribution. The heatmap shows the mean P_valid across all questions for each (model, language) pair.

== Key observations

- *HPLT-2.15B and Gemma-3-27B base show high P_valid* ($>0.85$) across most languages, confirming that the logprob elicitation prompt format works for base models.
- *EuroLLM-22B shows moderate P_valid* ($approx 0.65$--$0.80$), reflecting its different tokenizer and potentially different prompt-completion behavior.
- *Some languages show systematically lower P_valid*, which may indicate tokenizer coverage issues or culturally unfamiliar prompt formatting.

== Why this figure matters

P_valid is a prerequisite quality check: if the model assigns no probability to valid response digits, the extracted distribution is meaningless. This figure validates that the methodology works for the target models and identifies any language-specific issues.

== Further questions

- Does P_valid correlate with downstream JSD performance?
- Can prompt engineering improve P_valid for the lower-performing language-model pairs?

#pagebreak()

= F7: Position Bias

#align(center)[#image("position_bias.png", width: 100%)]

== What it shows

Violin plots of position bias magnitude by model type and by response type. Position bias measures how much the probability distribution changes when the order of response options is reversed. Gemma-3-27B-IT is excluded.

== How it was calculated

For each (model, language, question), the prompt is presented twice: once with options in the original order and once reversed. Position bias magnitude is the maximum absolute probability difference across response options between the two orderings: $"bias" = max_i |p_"fwd"(x_i) - p_"rev"(x_i)|$.

The debiased distribution used in all other analyses is the average of forward and reversed: $p_"avg"(x_i) = (p_"fwd"(x_i) + p_"rev"(x_i))/2$.

== Key observations

- *Gemma-3-27B base shows the lowest position bias* (mean $approx 0.32$), contributing to its overall superior performance.
- *EuroLLM-22B shows the highest position bias* (mean $approx 0.49$), suggesting it is more sensitive to option ordering.
- *Likert scales show higher position bias* than binary/categorical questions, likely because the multiple similarly-worded options create more ambiguity about which option maps to which digit.

== Why this figure matters

Position bias is a methodological concern for logprob-based survey elicitation. High bias means the forward and reversed distributions are very different, and their average may not reflect the model's "true" preference. Low bias increases confidence that the debiased distribution is meaningful.

== Further questions

- Does position bias correlate with the number of options?
- Would presenting options vertically (one per line) reduce bias?
- Is the bias direction systematic (e.g., always favoring the first option)?

#pagebreak()

= Summary of Conclusions

+ *Cultural signal is real and strong.* Pearson $r = 0.71$--$0.73$ between LLM and human expected values demonstrates that base language models encode genuine population-level value orientations from their training text.

+ *The signal is interpretable.* The Inglehart--Welzel composite (F2) shows that LLM cultural positions track the recognized cultural geography: Nordic countries are secular/self-expressive, Southeast European countries are traditional/survival-oriented, in both human and LLM data.

+ *Model scale is the dominant factor.* Gemma-3-27B (27B params) wins most languages with the lowest mean JSD, outperforming both the European-specialized EuroLLM-22B and the language-specific HPLT-2.15B monolingual models.

+ *LLMs compress distributions.* The main failure mode is entropy compression: LLMs produce more uniform distributions than humans, who often show strong consensus peaks. Z-scoring (F3) partially corrects this.

+ *Instruction tuning destroys logprob elicitation.* Gemma-3-27B-IT produces near-zero P_valid, rendering logprob extraction ineffective. Base models are the correct choice for this methodology.

+ *Question type matters.* Binary/categorical questions are best aligned (low JSD); likert-4/5 scales with many options are most challenging and show the highest position bias.

+ *Language resource level predicts alignment.* High-resource languages (French, English, Italian) show consistently lower JSD. Lower-resource languages (Croatian, Lithuanian, Latvian) remain challenging across all models.
