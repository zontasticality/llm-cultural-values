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
  #text(size: 10pt, fill: gray)[Track 1: EVS Survey Elicitation from HPLT-2.15B, EuroLLM-22B, Gemma-3-27B, and Gemma-3-27B-IT]
]

#v(1em)

= Overview

This report analyzes figures produced by comparing language model (LLM) response distributions to real European Values Study (EVS) 2017 survey data across 21 countries. Four model configurations are compared:

- *HPLT-2.15B*: 22 monolingual models, each trained exclusively on text from a single European language. These are small (2.15B parameter) Gemma-3-based models from the HPLT project.
- *EuroLLM-22B*: A single multilingual model trained on all 22 European languages jointly, roughly 10x larger at 22B parameters.
- *Gemma-3-27B* (base): A 27B-parameter base model from the Gemma 3 family. Same tokenizer family as the HPLT models but substantially larger and trained on a broader multilingual corpus.
- *Gemma-3-27B-IT* (instruction-tuned): The instruction-tuned variant of Gemma-3-27B, prompted via chat template. Included to test whether instruction tuning helps or hinders cultural value elicitation via logprob extraction.

Qwen3-235B results are pending (int4 quantization jobs in queue) and will be added in a subsequent update.

The core research question is whether base language models---which have never seen survey data or been instruction-tuned---internalize cultural values from their training text that align with real population-level survey responses in the corresponding country.

The EVS 2017 Integrated Dataset (ZA7500, $n = 59,438$) provides ground truth: weighted response distributions for 186 questions across 21 of our 22 target countries (Greece is absent from this dataset version). LLM distributions were extracted via next-token logprob elicitation with position-bias debiasing (forward + reversed option ordering averaging).

The primary distance metric throughout is the *Jensen--Shannon divergence* (JSD), a symmetric, bounded measure ($0 <= "JSD" <= ln 2 approx 0.83$). A JSD of 0 means identical distributions; $ln 2$ means maximally different. For reference, two independent random 4-option distributions have an expected JSD of approximately 0.15--0.25.

#pagebreak()

= Figure 1: Expected Value Scatter Plot

#align(center)[#image("human_scatter_ev.png", width: 100%)]

== What it shows

Each point represents one (language, question) pair, plotting the *human expected value* (x-axis) against the *LLM expected value* (y-axis) for ordinal questions only (likert3, likert4, likert5, likert10, and frequency scales). Four panels show EuroLLM-22B, Gemma-3-27B-IT, Gemma-3-27B, and HPLT-2.15B respectively. Points are colored by Inglehart--Welzel cultural cluster (Nordic, Western, Mediterranean, Central, Baltic, Southeast). The dashed line is a linear regression fit; the faint diagonal is the identity line ($y = x$).

== How it was calculated

For each ordinal question and each (model, language) pair, the expected value is the probability-weighted mean of the response options: $EE[X] = sum_i x_i dot p(x_i)$, where $p(x_i)$ is either the human survey proportion or the LLM's position-bias-debiased probability for option $x_i$.

The Pearson $r$ measures linear correlation; Spearman $rho$ measures rank correlation (more robust to outliers). Both are reported in the panel titles.

== Key observations

*Strong positive correlation across all models.* All four models show strong agreement with human values. Gemma-3-27B base achieves the highest correlation ($r = 0.727$, $rho = 0.683$), followed by Gemma-3-27B-IT ($r = 0.718$, $rho = 0.661$), HPLT-2.15B ($r = 0.715$, $rho = 0.615$), and EuroLLM-22B ($r = 0.709$, $rho = 0.639$). This demonstrates that *base LMs encode real cultural signal*, and that the 27B Gemma model captures it most faithfully.

*Instruction tuning does not help logprob elicitation.* Gemma-3-27B-IT has near-zero P_valid (median 0.000 across all languages), meaning the chat-template-wrapped prompts produce almost no probability mass on valid response digits. Despite this, it still achieves reasonable correlation because the tiny valid mass that exists is directionally correct. However, the distributions are essentially degenerate---instruction tuning redirects the model's probability mass toward verbose chat responses rather than bare digit tokens.

*Gemma-3-27B base outperforms models trained specifically for European languages.* Despite not being European-specialized, the 27B base model beats both the 22B European multilingual model and the 2.15B dedicated monolingual models, confirming that model scale is a primary driver of cultural value encoding.

#pagebreak()

= Figure 2: JSD by Question Type

#align(center)[#image("human_jsd_by_qtype.png", width: 85%)]

== What it shows

Box plots of JSD (LLM vs human) grouped by response type, with separate boxes for each model type. Each box shows the median (horizontal line), interquartile range (box), whiskers ($1.5 times "IQR"$), and outliers (dots). Lower JSD means better alignment with human data.

== How it was calculated

For each (model_type, language, question_id) triple, the LLM's debiased probability distribution and the human country's weighted response distribution are aligned over shared response values and compared using Jensen--Shannon divergence. The resulting JSD values are then grouped by the question's response type (categorical, frequency, likert3, likert4, likert5, likert10).

== Key observations

*Categorical questions are best aligned* (median JSD $approx 0.23$). These are typically binary (yes/no) or short-list questions with 2--3 options. The low JSD indicates LLMs correctly pick up the _direction_ of consensus.

*Likert-4 and likert-5 scales show highest JSD* (median $approx 0.37$). These are the workhorses of values surveys. Position bias is most damaging here, as it redistributes mass across options that are close in probability.

*Gemma-3-27B base (green) shows consistently lower JSD across all question types* except where Gemma-3-27B-IT's degenerate distributions happen to coincidentally match. The base model's advantage is not limited to particular question formats.

*Gemma-3-27B-IT performs worst overall* (mean JSD 0.396), reflecting its near-zero P_valid. The instruction-tuned model's chat-oriented training makes it fundamentally unsuited for raw logprob elicitation.

#pagebreak()

= Figure 3: Model Comparison

#align(center)[#image("human_model_comparison.png", width: 100%)]

== What it shows

*Panel (a)*: Grouped bar chart showing mean JSD to human data for each language, with bars for each model type. Lower bars mean closer to human ground truth.

*Panel (b)*: Win-count bar chart showing, for each model, the number of languages (out of 21) where that model achieves the lowest mean JSD to human data.

== How it was calculated

For each (model_type, language) pair, the mean JSD across all available questions is computed. This yields 21 values per model (one per language, excluding Greece). The comparison asks: for the same language, which model produces distributions closest to the real country's survey data?

== Key observations

*Gemma-3-27B base dominates, winning 19 of 21 languages.* The only exceptions are English and French, where HPLT-2.15B achieves the lowest JSD. For every other language, the 27B base model outperforms both the European-specific multilingual model and the language-specific monolingual models.

*The Gemma-3-27B advantage is substantial.* Its overall mean JSD of 0.293 compares to 0.330 for HPLT and 0.333 for EuroLLM---a gap of 0.037--0.040. The largest advantages appear for Hungarian ($Delta = -0.077$ vs HPLT), Danish ($-0.065$), and Finnish ($-0.019$).

*Scale continues to trump specialization.* Gemma-3-27B was not designed specifically for European languages, yet it outperforms both a European-specialized multilingual model (EuroLLM-22B) and 22 dedicated monolingual models (HPLT-2.15B). The 27B parameter count provides enough capacity to encode cultural nuances that smaller models cannot represent.

*French and English remain the best-aligned languages overall.* French (HPLT: 0.247, Gemma-3-27B: 0.264) and English (HPLT: 0.279, Gemma-3-27B: 0.286) consistently produce the lowest JSD, reflecting their status as high-resource languages.

*Croatian remains the hardest language.* Croatian shows the highest JSD for most models ($approx 0.32$--$0.37$), suggesting genuine difficulty in cultural value elicitation.

#pagebreak()

= Figure 4: JSD Heatmap

#align(center)[#image("human_jsd_heatmap.png", width: 100%)]

== What it shows

A heatmap with rows for each model type and 21 columns (one per language, alphabetical). Cell color encodes mean JSD from LLM to human data, with green indicating better alignment (lower JSD) and red indicating worse alignment (higher JSD). Exact values are printed in each cell.

== How it was calculated

Identical to the per-language means in Figure 3: for each (model, language) pair, the mean JSD across all shared questions is computed and displayed.

== Key observations

*The Gemma-3-27B row is visibly greener.* The base model row shows a uniformly greener hue than the HPLT and EuroLLM rows, confirming its systematic advantage across languages.

*Gemma-3-27B-IT is visibly redder.* The instruction-tuned variant's row is the most red, confirming that instruction tuning degrades logprob-based survey elicitation.

*The green corridor.* Italian (0.247), Hungarian (0.241), Romanian (0.243), and Danish (0.259) show particularly strong alignment for Gemma-3-27B---languages where neither HPLT nor EuroLLM performed as well.

*The red corridor.* Lithuanian ($approx 0.35$--$0.41$), Croatian ($approx 0.32$--$0.38$), and Latvian ($approx 0.30$--$0.47$) show the weakest alignment across models.

#pagebreak()

= Figure 5: Best and Worst Aligned Questions

#align(center)[#image("human_best_worst.png", width: 100%)]

== What it shows

Six bar charts comparing human (blue) and LLM (orange) response distributions for specific questions, using English/HPLT as the representative pair. The top row shows the three _best_-aligned questions (lowest mean JSD across all model-language pairs); the bottom row shows the three _worst_-aligned.

== How it was calculated

Mean JSD is computed per question across all (model, language) pairs. Questions are ranked, and the top 3 and bottom 3 are selected. For each, the human distribution and the HPLT English model's debiased distribution are plotted side by side.

== Best-aligned questions

*Pattern*: The best-aligned questions are binary categorical items. With only two options, the LLM essentially needs to get the _direction_ of opinion right, and it does.

== Worst-aligned questions

*Pattern*: The worst questions share a key feature: humans have a *strong unimodal peak* (one option chosen by 50--80% of respondents), while the LLM distributes mass more uniformly. This "flattening" or "hedging" behavior is consistent with base LMs having high entropy distributions.

#pagebreak()

= Figure 6: UMAP Cultural Map (LLM Only)

#align(center)[#image("umap_cultural_map.png", width: 100%)]

== What it shows

A 2D UMAP embedding of all 89 LLM model-language pairs, where each point represents one (model, language) combination. Points are colored by Inglehart--Welzel cultural cluster and shaped by model type. Proximity in UMAP space indicates similar ordinal response distributions across the 149 ordinal survey questions.

== How it was calculated

For each (model, language) pair, the expected value $EE[X]$ is computed for each of the 149 ordinal questions, yielding a 149-dimensional feature vector. Missing values are imputed with column means. UMAP (n_neighbors=15, min_dist=0.2, random_state=42) projects these vectors to 2D while preserving local and some global structure.

== Key observations

*Model type dominates the embedding structure.* The UMAP reveals clear model-type clusters rather than language clusters. Gemma-3-27B-IT points (pentagons) form a distinct, tight cluster separated from the base models---consistent with their degenerate P_valid producing a qualitatively different distribution pattern. The base models (HPLT circles, EuroLLM triangles, Gemma-3-27B diamonds) intermix more, but still show partial model-type grouping.

*Within-model language variation is smaller than between-model variation.* Languages from the same cultural cluster do not consistently cluster together across models, suggesting that model architecture and training regime have a larger effect on the output distribution than the cultural signal encoded for any particular language.

*Gemma-3-27B base points occupy an intermediate position* between the HPLT/EuroLLM cluster and the Gemma-3-27B-IT outlier cluster, reflecting its shared architecture with the IT variant but functionally different (base vs instruction-tuned) behavior.

#pagebreak()

= Figure 7: Combined UMAP --- Human Survey Data + LLM Distributions

#align(center)[#image("umap_combined.png", width: 100%)]

== What it shows

A joint UMAP embedding of 110 points: 89 LLM model-language pairs (small markers) and 21 human country distributions from the EVS (large stars). Gray lines connect each LLM point to its corresponding country's human data. Both human and LLM points are colored by cultural cluster.

== How it was calculated

Human expected values are computed identically to the LLM expected values: for each country and ordinal question, $EE[X] = sum_i x_i dot p_"human"(x_i)$ using weighted EVS response proportions. The 149-dimensional human and LLM feature vectors are concatenated into a single matrix (110 $times$ 149) and jointly embedded with UMAP.

== Key observations

*Human and LLM distributions occupy distinct regions.* The most striking feature is the clear separation between the human cluster (stars, left) and the LLM cluster (markers, right). This confirms quantitatively what the JSD analysis shows: while LLMs capture the _relative ordering_ of cultural values (high Pearson $r$), there is a systematic distributional shift between how humans and LLMs respond. LLMs compress the response range and distribute mass more uniformly.

*Human points cluster tightly by cultural geography.* The 21 human country distributions form a compact, culturally coherent cluster with visible sub-structure: Nordic, Western, Mediterranean, Central, Baltic, and Southeast countries are partially separable. This reflects genuine cross-national variation in values documented in the political science literature.

*LLM points are more dispersed than human points.* The LLM cluster spans a wider region, driven primarily by model-type differences (architecture, scale, training data) rather than by language-specific cultural encoding. This suggests that model choice is a larger source of variation than the cultural signal itself.

*The gray connection lines are long and roughly parallel.* The lines connecting each LLM point to its corresponding country's human data all point in a similar direction (left-to-right), confirming a systematic shift rather than random noise. The shift direction corresponds to the LLM tendency to produce more uniform, higher-entropy distributions compared to humans' peaked distributions.

*Implication for methodology.* The human--LLM gap visible in UMAP suggests that raw logprob distributions should not be interpreted as direct proxies for human survey responses. However, the strong _correlation_ between human and LLM expected values (Figure 1) means that relative comparisons---which country's LLM distribution is most like which country's human data---remain valid and informative.

#pagebreak()

= Figure 8: Human-Only UMAP vs Inglehart--Welzel World Cultural Map

#align(center)[#image("umap_human_only.png", width: 85%)]

== What it shows

A UMAP embedding of the 21 EVS country distributions _without any LLM data_. Each point represents one country's human survey responses, colored by Inglehart--Welzel cultural cluster. This isolates whether the EVS survey data alone---processed through our ordinal expected-value pipeline---recovers the cultural structure documented in the political science literature.

== How it was calculated

For each country and each of the 149 ordinal questions, the human expected value $EE[X] = sum_i x_i dot p_"human"(x_i)$ is computed from weighted EVS response proportions. The resulting $21 times 149$ matrix (with mean imputation for missing values) is embedded via UMAP with `n_neighbors=10` (must be less than the 21 data points) and `min_dist=0.3` (increased for readability).

== Reference: Inglehart--Welzel World Cultural Map

#align(center)[#image("inglehart_welzel_reference.png", width: 85%)]
#align(center)[#text(size: 9pt, fill: gray)[Source: World Values Survey, Cultural Map 2023 (worldvaluessurvey.org)]]

The Inglehart--Welzel World Cultural Map positions countries along two dimensions---Traditional vs Secular-Rational values, and Survival vs Self-Expression values---based on decades of survey data from the World Values Survey and European Values Study. Our human-only UMAP should recover similar cultural groupings if the ordinal expected-value representation preserves the underlying value structure.

== Key observations

*Nordic and Western clusters are clearly separated.* The Nordic countries (Finland, Sweden, Denmark) cluster in the upper-left, and the Western European countries (German, French, English, Dutch) form a coherent group on the left side. This matches the Inglehart--Welzel map's placement of these regions in the high secular-rational, high self-expression quadrant.

*Baltic and Southeast countries cluster on the right.* The Baltic states (Estonian, Lithuanian, Latvian) and Southeast European countries (Bulgarian, Croatian, Romanian) occupy the right side of the map, reflecting their shared post-communist heritage and the more traditional/survival-oriented values documented in the Inglehart--Welzel framework.

*Central European countries span the middle.* Czech, Slovak, Slovenian, Hungarian, and Polish countries spread across the central-to-right region, consistent with their intermediate position between Western and post-Soviet cultural zones in the Inglehart--Welzel map.

*The projection validates our methodology.* The fact that a simple UMAP of ordinal expected values---computed from the same EVS questions we use for LLM elicitation---recovers recognizable cultural geography confirms that our survey processing pipeline preserves meaningful cultural signal. This is a prerequisite for interpreting the LLM--human comparisons in preceding figures.

#pagebreak()

= Key Finding: Instruction Tuning Degrades Logprob Elicitation

The Gemma-3-27B-IT results reveal a critical methodological insight: *instruction-tuned models are fundamentally unsuited for survey elicitation via next-token logprob extraction*.

All 22 Gemma-3-27B-IT language runs produced median P_valid of 0.000, meaning effectively zero probability mass lands on valid response digit tokens. The chat template wrapping redirects the model's probability mass toward verbose natural language responses (e.g., "I would choose option...") rather than bare digit tokens. The model has been trained to produce helpful, conversational output---not to assign high probability to isolated digits.

This has important implications for the research methodology:
- *Base models are the correct choice* for logprob-based survey elicitation
- *Instruction-tuned models require a different elicitation strategy* (e.g., sampling-based approaches, or constrained generation)
- The Gemma-3-27B-IT data should be excluded from quantitative comparisons due to its degenerate P_valid

#pagebreak()

= Summary of Conclusions

+ *Cultural signal is real and strong.* Pearson $r = 0.71$--$0.73$ between LLM and human expected values demonstrates that base language models encode genuine population-level value orientations from their training text. Gemma-3-27B base achieves the strongest correlation ($r = 0.727$, $rho = 0.683$).

+ *Model scale is the dominant factor.* Gemma-3-27B (27B params) wins 19 of 21 languages with mean JSD = 0.293, compared to 0.330 for HPLT-2.15B and 0.333 for EuroLLM-22B. The advantage holds even for languages it was not specifically trained on.

+ *Gemma-3-27B has the lowest position bias.* Mean bias magnitude of 0.322 versus 0.381 (HPLT) and 0.487 (EuroLLM). Lower position bias contributes to better alignment with human data.

+ *Instruction tuning destroys logprob elicitation.* Gemma-3-27B-IT produces near-zero P_valid across all languages (median 0.000), rendering the logprob extraction methodology ineffective. Chat-template prompting redirects probability mass away from digit tokens.

+ *Signal strength depends on question type.* Binary/categorical questions show strong alignment (median JSD $approx 0.23$); likert-4/5 scales show moderate alignment (median JSD $approx 0.37$). This hierarchy is consistent across all models.

+ *LLMs compress the response range.* All models produce less extreme distributions than human populations. They capture the _direction_ of opinion (positive correlation) but underestimate the _magnitude_ of consensus.

+ *High-resource Western languages align best.* French, English, Italian, and Spanish consistently show the lowest JSD across base models. Croatian and Lithuanian remain the most challenging languages.

+ *Qwen3-235B results pending.* The Qwen3-235B-A22B MoE model (int4 quantization) is currently in the SLURM queue and will be added to a future update.
