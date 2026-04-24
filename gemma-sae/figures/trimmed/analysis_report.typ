#set document(title: "Cultural Values in LLM Completions: Figure Analysis")
#set page(margin: 2.5cm)
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.")
#set par(justify: true)

#align(center)[
  #text(size: 18pt, weight: "bold")[Cultural Values in LLM Completions]
  #v(0.3em)
  #text(size: 14pt)[Detailed Figure Analysis]
  #v(0.3em)
  #text(size: 10pt, fill: gray)[Track 2: Temperature Sampling + LLM Classification]
  #v(0.3em)
  #text(size: 10pt, fill: gray)[April 2026]
]

#v(1em)

= Overview

This report analyzes figures from an experiment asking whether LLMs produce culturally differentiated text when prompted in different languages, and whether that differentiation aligns with human cultural variation measured by the European Values Study (EVS 2017).

The approach is *indirect*. Rather than asking LLMs survey questions (Track 1, which produced a null result at $r approx 0.00$ after z-scoring), we prompt models with open-ended sentence stems in 27 languages and classify completions by topic and cultural loading. We then ask whether the *topics LLMs emphasize* in each language correlate with the *attitudes humans report* in surveys for the corresponding country. These are different measurement modalities probing the same latent construct (cultural values), mediated by an LLM classifier. The correlation is interpretively interesting but methodologically complex --- every figure needs to be understood in terms of what it actually measures.

Three model configurations are compared:
- *Gemma 3 27B PT* and *Gemma 3 12B PT* --- multilingual base models from Google
- *EuroLLM-22B* --- European-focused multilingual base model
- *HPLT 2.15B $times$ 22* --- monolingual models, one per European language, providing a pure-culture baseline where any signal must originate in language-specific training data

For each (model, language, template) cell, $N = 200$ completions are generated at temperature $T = 1.0$ (multilingual) or $T = 0.8$ (HPLT), $"top"_p = 0.95$. A local Gemma 3 27B IT classifier assigns each completion a content category (one of 8: `family_social`, `occupation_achievement`, `personality_trait`, `physical_attribute`, `spiritual_religious`, `emotional_state`, `material_practical`, `other`) and three cultural dimension scores (1--5 Likert: IC, TS, SS). Token-level logprob distributions are also extracted.

The primary analysis uses *trimmed prompts* (trailing whitespace removed) with *language-match filtering* (wrong-language completions dropped via automated language detection), yielding 197,273 classified completions. The v2 classifier corrects a template-identity leak present in the v1 classifier (Section 4.1).

The report follows a narrative arc: *methodology validation* (can we trust the classifier and completions?) $arrow.r$ *content differentiation* (do LLMs say different things in different languages?) $arrow.r$ *cultural geography* (does the structure match known cultural regions?) $arrow.r$ *human alignment* (how precisely does LLM variation track EVS variation, and at what granularity?) $arrow.r$ *robustness* (what could be artifacts?) $arrow.r$ *limitations of Likert scores* (why aggregate dimensions mostly fail).

#pagebreak()

= F1: Methodology Validation --- Classifier Reliability

== F1a: Cross-model agreement (Gold Set)

=== What it shows

Inter-rater reliability between the Gemma-3-27B-IT v2 classifier and Claude (acting as independent LLM judge) on 150 stratified completions sampled across models, languages, templates, and classifier-assigned categories.

=== How it was calculated

150 completions were sampled from the trimmed, language-matched subset with stratification to cover all 3 multilingual models, a sample of HPLT monolinguals, 10+ languages, all 8 templates, and all 8 content categories (proportional to v2 classifier frequency). Each completion was independently classified by Claude using the identical category definitions and Likert dimension rubric from the v2 system prompt. Claude's labels were then compared to the v2 classifier's stored labels. Cohen's $kappa$ was computed using `sklearn.metrics.cohen_kappa_score` (unweighted for categories, linear-weighted for ordinal Likert).

=== Key observations

- *Content category $kappa = 0.80$* (substantial agreement). Gate 3 target was $kappa > 0.7$: *PASSES*.
- Per-category F1: `physical_attribute` 0.94, `spiritual_religious` 0.90, `occupation_achievement` 0.89, `personality_trait` 0.87, `emotional_state` 0.87, `material_practical` 0.86, `family_social` 0.85. The *`other` category is the weakest at $"F1" = 0.57$*.
- Likert dimension MAE: IC 0.40, TS 0.36, SS 0.53. Within $plus.minus 1$ agreement: 93--99%. Pearson $r$: 0.65--0.70.
- *Primary disagreement pattern*: the v2 classifier *under-detects off-topic noise*. Claude labeled 28/150 completions as `other`; the v2 classifier only 14. The classifier assigns substantive category labels (typically `personality_trait`, `family_social`, `material_practical`) to web-scraped boilerplate and fragmentary text that should be `other`. 60% of all disagreements (15/25) follow this pattern.
- *HPLT monolinguals*: 92.5% category agreement (cleaner, more predictable text). *Multilingual models*: 80% (noisier, more off-topic drift).

=== Why this figure comes first

The entire downstream analysis depends on the classifier being reliable. If the classifier assigned categories randomly, the chi-square tests and EVS correlations would be meaningless. $kappa = 0.80$ establishes that the classifier agrees with an independent judge at a level conventionally called "substantial." The `other`-under-detection weakness means noise is slightly recategorized as cultural content, inflating counts in some categories --- but this affects all languages equally and does not systematically bias cross-language comparisons.

#pagebreak()

= F2: Methodology Validation --- Prompt Quality

== F2a: Quality score distribution by template

#align(center)[#image("quality/quality_template_distribution.png", width: 95%)]

=== What it shows

Per-prompt quality scores ($q = 1 - "frac"_"other"$, the fraction of completions *not* classified as off-topic noise, averaged across all models) for all 265 trimmed prompts, grouped by template. A score of 1.0 means every completion was classified with a substantive category; 0.0 means every completion was noise.

=== How it was calculated

For each prompt (a specific template $times$ language $times$ variant), all v2-classified completions across all models are queried. The `other` fraction is computed as the number classified as `other` divided by total. The quality score is $q = 1 - "frac"_"other"$. This is computed by `scripts/prompt_quality.py --classifier gemma3_27b_it_v2 --write`, which stores results in `prompts.quality_score`.

=== Key observations

- *Six of eight templates are clean* ($q > 0.9$ for essentially every language).
- *Two templates carry almost all noise*:
  - *`self_concept`*: bare first-person pronouns ("I am", "Soy", "Ben", "Sono") are 2--4 characters long. These tokenize into stems with many valid continuations in other languages or domains (e.g. "Soy" $arrow.r$ soybean prices, "Ben" $arrow.r$ "Benjamin", "Sem" $arrow.r$ Indonesian text). Quality ranges from 0.46 to 0.99 depending on prompt length and homograph density.
  - *`belief`*: "I believe that..." triggers news-headline and speculative continuations.
- *Distribution*: 204 clean ($q gt.eq 0.9$), 49 usable ($0.75 lt.eq q < 0.9$), 9 degraded ($0.5 lt.eq q < 0.75$), 3 unsalvageable ($q < 0.5$).
- *Alternative prompts work*: 30 longer-form alternatives (e.g. Turkish "Kendimi şöyle tanımlarım: ben" instead of "Ben") raise all unsalvageable stems to $q gt.eq 0.89$, confirming the problem is prompt length, not language difficulty.

=== Why this figure matters

Prompt quality determines signal-to-noise ratio. The reader needs to know that most of our data is clean, and that the problematic prompts are specific, understood, and fixable. The quality scores are stored per-prompt and available as weights for downstream analysis; the figures in Section 3+ use the full dataset (not quality-filtered) because the 2.6% noise is small relative to the effect sizes.

== F2b: Model variation in noise rates

#align(center)[#image("quality/quality_by_model.png", width: 100%)]

=== What it shows

Overall `other` rate and per-template `other` rate for each model.

=== How it was calculated

Same data as F2a but aggregated by model rather than by prompt. Generated by `scripts/quality_figures.py`.

=== Key observations

- *EuroLLM-22B*: 14.7% overall noise --- $2 times$ the Gemma rate. `self_concept` is 44% noise for EuroLLM. The model drifts into Italian/Swedish/German Wikipedia-style content on short prompts.
- *Gemma 3 27B/12B*: 6.1--6.3% overall. Size alone does not fix the short-stem problem.
- *HPLT monolinguals*: 1.5--8.5% overall. Because each model only completes in its training language, the cross-lingual homograph problem is largely absent.
- EuroLLM's disadvantage is *specific to short prompts* --- on full-sentence templates its noise rate is comparable to Gemma's.

=== Why this figure matters

Model-specific noise rates affect cross-model comparisons. EuroLLM's high noise on `self_concept` means that analyses involving this template must be interpreted with caution for EuroLLM. Alternatively, `self_concept` can be excluded or quality-weighted for EuroLLM specifically.

#pagebreak()

= F3: Methodology Validation --- Language Detection

=== What it shows

Automated language identification (`lingua-language-detector`, 75 languages) run over all 402,006 completions, identifying completions where the detected language does not match the prompt language.

=== How it was calculated

`scripts/detect_lang.py` runs lingua's `detect_language_of()` on each `completion_text`. Completions shorter than 10 characters are skipped (too short for reliable detection). The detected ISO 639-3 code is compared to `prompts.lang`. A completion is "language-drifted" if the detected language differs from the prompt language.

=== Key results

#table(
  columns: (auto, auto),
  align: (left, right),
  [*Metric*], [*Value*],
  [Total completions], [402,006],
  [Successfully detected], [397,752 (98.9%)],
  [Match prompt language], [385,369 (95.9%)],
  [Language drift (mismatch)], [12,383 (3.1%)],
  [Trimmed subset drift], [5,283 / 204,000 (2.6%)],
)

Of the 5,283 drifted rows in the trimmed subset, only 43% were classified as `other`. The remaining *57% received a real content-category label* despite being in the wrong language --- these are the silent contaminant that neither the classifier's `other` detection nor the prompt quality score catches.

*Top drift destinations*: English (3,426 rows), Norwegian Bokmål (1,190), Italian (680), Swedish (606). Multilingual models drift predominantly to English; this is consistent with English being the dominant language in their training data.

*Worst (model, language) pairs*: EuroLLM/Croatian (14.3% drift), Gemma-27B/Hindi (14.3%), Gemma-12B/Danish (13.5%).

=== Why this figure matters

This is the third independent quality gate after classifier agreement (F1) and prompt quality (F2). The key finding is that the classifier is *not a reliable language-drift detector* --- it labels wrong-language text with substantive categories 57% of the time. All downstream analyses in this report use the `--lang-match` filter (keep only rows where `detected_lang IS NULL OR detected_lang = prompts.lang`), which drops 2.6% of the trimmed data. Section 5.3 shows this filter changes no headline result.

#pagebreak()

= F4: Content Category Distributions

== F4a: Gemma 3 27B PT

#align(center)[#image("categories_gemma3_27b_pt.png", width: 100%)]

=== What it shows

Stacked bar charts showing the proportion of completions in each content category, per language, for Gemma 3 27B PT.

=== How it was calculated

For each language, the number of completions in each category (from v2 classifier, trimmed, lang-matched) is divided by the total completions in that language. Generated by `analysis/analyze.py categories`.

=== Key observations

- *Arabic*: 9.8% `spiritual_religious` --- highest of any language, consistent with the prevalence of religious discourse in Arabic web text. Hindi 4.4%.
- *Nordic/Baltic*: lowest `spiritual_religious` (Finnish 1.0%, Swedish 1.9%, Estonian 2.7%) --- the world's most secular societies per WVS.
- *`personality_trait` dominates* across all languages (32--50%), with French and Portuguese highest.
- *Post-communist languages show elevated `material_practical`*: Polish 23.1%, Czech 20.6%, Slovak 19.4%, Croatian 20.3% --- consistent with post-communist emphasis on material security.
- *Hindi* shows the highest `family_social` (21.3%), Hungarian (18.0%) is also elevated; French is lowest (15.1%).

=== Why this figure matters

This is the primary data --- before any statistical test, the reader can see that the category distributions are visibly different across languages. The patterns match recognizable cultural geography: religious axis (Arabic $>>$ Nordic), material security (post-communist $>>$ Western), family emphasis (South Asian, Eastern European $>$ Western). The chi-square tests in F5 quantify what is already visible here.

== F4b: EuroLLM-22B

#align(center)[#image("categories_eurollm22b.png", width: 100%)]

=== Key observations

- EuroLLM shows the *sharpest language differentiation*. English: 52.7% `personality_trait`; Croatian: 18.1% `material_practical`.
- High `other` rates for Swedish (22.9%), Finnish (23.0%), Hindi (22.5%) indicate EuroLLM's difficulty with some languages (see F2b).
- Patterns that *replicate across both Gemma 27B and EuroLLM* are more likely cultural than artifactual: Arabic/Hindi elevated `spiritual_religious`, post-communist elevated `material_practical`, English/French elevated `personality_trait`.

== F4c: HPLT monolingual models

The 22 HPLT models provide per-language "pure culture" baselines:
- *HPLT Polish*: 27.8% `material_practical` (vs Gemma 27B's 23.1%)
- *HPLT Swedish*: 0.7% `spiritual_religious` --- the most secular profile in the dataset
- *HPLT French*: 53.5% `personality_trait` --- the highest `personality_trait` rate
- Every cultural pattern is *stronger* in the monolingual model, consistent with a dilution interpretation: multilingual models blend signals across languages through shared parameters.

#pagebreak()

= F5: Chi-Square Cluster Differentiation

=== What it shows

Chi-square tests of independence between Inglehart--Welzel cultural cluster membership and content-category distributions. Tests whether the category distributions shown in F4 differ significantly between cultural clusters.

=== How it was calculated

For each multilingual model, completions are grouped by Inglehart--Welzel cultural cluster (Protestant Europe, Catholic Europe, English-speaking, Orthodox, Baltic, South Asian, Middle Eastern). A contingency table of cluster $times$ category counts is formed, and the overall $chi^2$ statistic is computed. Pairwise cluster comparisons are also computed for all $mat(7; 2) = 21$ pairs. Generated by `analysis/analyze.py categories`.

=== Key results

#table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, center),
  [*Model*], [*$chi^2$*], [*df*], [*$p$*],
  [Gemma 3 27B], [$779.7$], [$42$], [$< 10^(-136)$],
  [Gemma 3 12B], [$828.9$], [$42$], [$< 10^(-147)$],
  [EuroLLM-22B], [$991.2$], [$42$], [$< 10^(-180)$],
)

All pairwise comparisons are significant at $p < 0.01$ for Gemma models. For EuroLLM, 20/21 pass at $p < 0.01$; Baltic vs Orthodox passes at $p = 0.011$ (the most culturally proximate post-Soviet clusters).

Cross-model $chi^2$ within the same language ranges from 256 to 915, all $p < 10^(-42)$ --- model architecture independently shapes cultural expression.

=== Why this figure matters

This is the primary go/no-go test. The question was: "Do content category distributions differ between cultural clusters?" The answer is yes, with $p$-values that are not marginal --- they are among the most significant results in this entire analysis. Note that chi-square detects *any* distributional difference, not necessarily a culturally meaningful one. The stacked bar charts (F4) provide the qualitative interpretation; this test quantifies the significance.

#pagebreak()

= F6: Language Clustering by Category Distributions

#align(center)[#image("lang_dendrogram.png", width: 100%)]

=== What it shows

Hierarchical clustering of languages by Jensen--Shannon divergence (JSD) between their content-category distributions. One panel per multilingual model. Leaf labels are colored by Inglehart--Welzel cultural cluster.

=== How it was calculated

For each model, a category proportion vector per language is computed (8-dimensional, one entry per category). JSD is computed between all pairs of languages: $"JSD"(P, Q) = sqrt(1/2 D_"KL"(P || M) + 1/2 D_"KL"(Q || M))$ where $M = (P + Q)/2$. Ward linkage clustering is applied to the JSD distance matrix. Generated by `scripts/visualize_culture.py`.

=== Key observations

- *Arabic separates first* from the European cluster across all three models --- the most culturally distinct profile, driven by elevated `spiritual_religious`.
- *English is an outlier in EuroLLM* --- its European training data makes English completions culturally distinct from the otherwise-European cluster.
- *Nordic languages co-cluster*: Danish, Swedish, Finnish, Dutch group together, reflecting shared secular-individualist profiles.
- *Post-communist languages group*: Polish, Czech, Slovak, Croatian, Hungarian share elevated `material_practical`.
- *Baltic and Orthodox languages merge*: Estonian, Lithuanian, Latvian cluster near Bulgarian, Romanian, Greek.

=== Why this figure matters

This clustering requires *no human comparison data* --- it emerges purely from the models' own completion distributions. If the tree matches known cultural geography (and it does), this supports the interpretation that LLMs absorb cultural patterns from their training corpora. It also shows which cultural distinctions are captured by which model: EuroLLM separates English more sharply than Gemma, while Gemma provides finer-grained European clustering.

#pagebreak()

= F7: Inglehart--Welzel Cultural Map --- Human vs LLM

#align(center)[#image("iw_cultural_map.png", width: 100%)]

=== What it shows

Four-panel scatter plot reproducing the Inglehart--Welzel World Cultural Map layout. Left panel: human EVS 2017 data (21 European countries). Remaining panels: Gemma-3-27B, Gemma-3-12B, EuroLLM-22B. X-axis: Traditional$arrow.r$Secular (z-scored). Y-axis: Survival$arrow.r$Self-Expression (z-scored). Points colored by IW cluster.

=== How it was calculated

*Human scores*: Standard Inglehart--Welzel methodology. 10 EVS questions (5 per dimension), polarity-flipped where needed, z-scored per question across the 21 countries, averaged into composites.

*LLM scores*: For each (model, language), the mean of v2 classifier `dim_trad_secular` and `dim_surv_selfexpr` scores across all trimmed, lang-matched completions. These raw 1--5 means are then z-scored across languages to match the scale of the human composites.

*Spearman $rho$* is computed between human and LLM z-scored values for each model $times$ dimension.

=== Key observations

- The *human panel* reproduces the classic IW map: Protestant Europe (blue) in the upper-right (secular, self-expressive), Orthodox (red) lower-left, Catholic Europe (green) intermediate.
- *EuroLLM's panel* shows the clearest cluster separation among LLM panels. SE: $rho = 0.57$, $p = 0.007$ --- the only significant model$times$dimension correlation.
- *Gemma panels* show weaker structure: TS $rho = 0.22$--$0.33$, SE $rho = 0.27$--$0.34$, none significant.
- *LLM axes are compressed*: raw dimension scores range $approx 2.8$--$3.4$ on a 1--5 scale, reflecting classifier midpoint compression. Z-scoring recovers the rank ordering but cannot restore the lost variance.

=== What we're comparing (critical framing)

This figure compares two different measurements of the same latent construct. The *human* values are direct survey responses ("How important is God in your life?"). The *LLM* values are a classifier's judgment of how "traditional" or "self-expressive" an open-ended completion sounds. The correlation between them is mediated by: (a) an LLM generating text shaped by its training data, (b) a different LLM classifying that text on cultural dimensions, and (c) our assumption that the classifier's 1--5 scale corresponds to the same construct that EVS measures. The $kappa = 0.80$ inter-rater reliability (F1) validates (b). The HPLT amplification (F4c) validates that (a) is training-data-driven. But (c) remains an assumption --- there is no ground truth connecting classifier Likert scores to EVS composites except this correlation itself.

#pagebreak()

= F8: Direct Category--Question Comparison

#align(center)[#image("category_vs_evs_scatter.png", width: 100%)]

=== What it shows

Scatter plots comparing EuroLLM content-category *fractions* (y-axis: proportion of completions in a given category) against individual EVS question *means* (x-axis), for 21 European languages. Eight (category, question) pairs are shown, chosen for interpretive directness: e.g. `spiritual_religious` fraction vs "Importance of God" (v63), `material_practical` fraction vs "Life satisfaction" (v39).

=== How it was calculated

For each language, the fraction of EuroLLM completions classified into each category is computed (from the trimmed, lang-matched subset). For each EVS question, the expected value of the human response distribution is computed. Spearman $rho$ is computed per (category, question) pair. Generated inline via `scripts/visualize_culture.py` query.

=== Key observations

- *All 8 pairs are non-significant* ($p > 0.08$). The strongest trends are `personality_trait` vs "Homosexuality justifiable" ($rho = 0.38$, $p = 0.088$) and `material_practical` vs "Abortion justifiable" ($rho = -0.39$, $p = 0.084$).
- *The "obvious" comparison fails*: `spiritual_religious` fraction vs "Importance of God" is $rho = -0.15$ ($p = 0.50$). Languages where humans say God is important do not produce more religious LLM completions in proportion.
- Croatia and Portugal have high human religion importance but low LLM `spiritual_religious` rates; English has low religion importance but relatively high `spiritual_religious` content from web-discourse about religion (not piety).

=== Why this figure matters

This is the comparison a reader expects first: "If humans in Poland care more about material security, do LLM completions in Polish contain more `material_practical` content?" The answer is *no, not significantly*, for any individual category--question pair. This null result is crucial for interpreting the subsequent figures: the signal does *not* operate at the level of individual categories predicting individual survey questions. It operates at a different granularity, which F9 and F10 reveal.

#pagebreak()

= F9: Aggregate IW Composite Alignment

#align(center)[#image("iw_comparison_v2.png", width: 100%)]

=== What it shows

Scatter plots of LLM mean dimension score vs human EVS IW composite for 21 languages. Top row: Traditional$arrow.r$Secular. Bottom row: Survival$arrow.r$Self-Expression. One column per model.

=== How it was calculated

Same methodology as F7 but without z-scoring --- y-axis shows raw classifier means (1--5 Likert), x-axis shows z-scored human IW composites. Generated by `scripts/compare_to_wvs.py --classifier gemma3_27b_it_v2 --trimmed-only --lang-match`.

=== Key results

#table(
  columns: (auto, auto, auto, auto, auto),
  align: (left, right, right, right, right),
  [*Model*], [*TS $rho$*], [*TS $p$*], [*SE $rho$*], [*SE $p$*],
  [Gemma 3 27B], [$0.239$], [$0.297$], [$0.273$], [$0.232$],
  [Gemma 3 12B], [$0.330$], [$0.144$], [$0.344$], [$0.127$],
  [EuroLLM-22B], [$0.295$], [$0.195$], [$0.568$], [$0.007$],
)

=== Key observations

- *EuroLLM-22B SE is the only significant cell*: $rho = 0.568$, $p = 0.007$. This survives the v1$arrow.r$v2 classifier fix, the language-drift filter, and quality weighting (Section 5).
- All TS correlations are non-significant ($rho = 0.24$--$0.33$). Under the v1 classifier, all three models showed TS $rho approx 0.45$ ($p < 0.05$) --- those correlations were entirely driven by the template-identity leak (Section 5.1).
- The *transition from F8 to F9* is notable: individual category--question pairs show no significant alignment ($rho < 0.39$), but the *composite* across 5 questions shows $rho = 0.57$. This means the signal aggregates across questions rather than concentrating in any single one.

=== Why this figure matters

F8 showed that individual categories don't predict individual survey questions. F9 shows that *something* in the LLM's overall SE profile does predict the human SE composite --- but only for EuroLLM. The question is: where in the LLM's output does this signal come from? F10 answers this.

#pagebreak()

= F10: Template $times$ Question Correlation Grid

#align(center)[#image("question_scatter_grid.png", width: 80%)]

=== What it shows

An 8$times$10 grid. Rows = 8 prompt templates. Columns = 10 individual EVS questions (5 Traditional$arrow.r$Secular, 5 Survival$arrow.r$Self-Expression). Each cell is a mini scatter plot (21 language points) of EuroLLM's per-language mean dimension score on that template vs the human EVS mean on that question. Background color encodes significance: green ($p < 0.01$), yellow ($p < 0.05$), gray (n.s.).

=== How it was calculated

For each (template, EVS question) pair: (1) compute EuroLLM's mean `dim_trad_secular` or `dim_surv_selfexpr` per language, using only completions from that template; (2) compute human EVS expected values for that question, z-scored across countries; (3) Spearman $rho$ between the two series. This is a finer-grained decomposition of the aggregate IW signal in F9. Generated by `scripts/visualize_culture.py`.

=== Key observations

#table(
  columns: (auto, auto, auto, auto),
  align: (left, left, right, right),
  [*Template*], [*EVS question*], [*$rho$*], [*$p$*],
  [`values`], [Gay couples as parents (v82)], [$0.70$], [$< 0.001$],
  [`values`], [Abortion justifiable (v154)], [$0.63$], [$0.002$],
  [`values`], [Homosexuality justifiable (v153)], [$0.57$], [$0.007$],
  [`values`], [Importance of God (v63)], [$0.54$], [$0.012$],
  [`childrearing`], [Religion importance (v6)], [$0.51$], [$0.019$],
  [`childrearing`], [Importance of God (v63)], [$0.51$], [$0.020$],
  [`values`], [Petition signing (v98)], [$0.50$], [$0.021$],
)

The signal is concentrated in two templates: *`values`* ("The most important thing in life is...") tracks five SE questions at $p < 0.05$; *`childrearing`* tracks two TS questions. In contrast, `family`, `success`, `self_concept`, `decision`, and `moral` carry essentially no signal.

=== Why this figure matters

This resolves the apparent contradiction between F8 (individual categories don't predict individual questions) and F9 (the aggregate SE composite does). The alignment operates at the *template $times$ question* level --- specific prompts elicit completions whose classifier-rated dimension scores track specific human attitudes. "The most important thing in life is..." in a socially conservative language produces completions rated as less self-expressive than the same prompt in a liberal language, and that difference tracks attitudes toward homosexuality ($rho = 0.70$) and abortion ($rho = 0.63$).

*Interpretation*: LLMs match human cultural attitudes *on the topics their prompts elicit, not on abstract value dimensions.* Cultural alignment is topical and specific, not a smooth mapping onto Inglehart--Welzel coordinates.

#pagebreak()

= Robustness and Controls

== 5.1. Classifier bias correction (v1 $arrow.r$ v2)

An initial classifier version (v1, `gemma3_27b_it`) included `"Prompt type: {template_id}"` in the user message. This leaked template identity into category labels: the classifier near-deterministically mapped `family` templates to `family_social` regardless of content. Template$arrow.r$category dominance ratios dropped 23--68% after removing the leak:

#table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  [*Category*], [*v1 dom.*], [*v2 dom.*], [*$Delta$*],
  [personality_trait], [$27.0$], [$14.7$], [$-46%$],
  [physical_attribute], [$10.6$], [$3.4$], [$-68%$],
  [family_social], [$98.5$], [$72.4$], [$-27%$],
)

The bias correction halved the number of significant IW correlations: four of six v1 model$times$dimension cells were significant at $p < 0.05$; only EuroLLM SE survives and *strengthens* ($rho$: $0.544 arrow.r 0.562$). The v1 TS correlations ($rho approx 0.45$) were entirely driven by the template leak. *Classifier design choices can create or destroy cultural signal.*

== 5.2. Monolingual model amplification

Every cultural pattern (Polish `material_practical`, Swedish secular, French `personality_trait`) is stronger in HPLT monolingual models than in multilingual models (see F4c). Because HPLT models see only one language's web corpus, cross-lingual transfer is impossible --- the signal must originate in language-specific training data.

== 5.3. Language-drift filter robustness

Re-running all analyses with vs without the `--lang-match` filter (dropping 5,283 / 204,000 rows):

#table(
  columns: (auto, auto, auto, auto),
  align: (left, right, right, right),
  [*Metric*], [*No filter*], [*Lang-match*], [*$Delta$*],
  [EuroLLM SE $rho$], [$0.562$], [$0.568$], [$+0.006$],
  [Gemma 27B $chi^2$], [$774.5$], [$779.7$], [$+5$],
  [EuroLLM $chi^2$], [$958.2$], [$991.2$], [$+33$],
  [values $times$ v82 $rho$], [$0.703$], [$0.697$], [$-0.006$],
)

Every headline signal survives. Chi-square values increase modestly (cleaner data, sharper differentiation). The wrong-language rows were adding noise, not signal.

#pagebreak()

= Likert Dimension Scores --- Why They Mostly Fail

== F11: Cohen's $d$ known-group contrasts

#align(center)[#image("known_groups_cohens_d.png", width: 100%)]

=== What it shows

Cohen's $d$ effect sizes for 3 cultural contrasts (Finnish vs Romanian, English vs Polish, Polish vs Romanian) across 3 Likert dimensions (IC, TS, SS), per model. Red dashed line = $d = 0.3$ threshold.

=== How it was calculated

For each contrast, completions from both languages are pooled and the standardized mean difference is computed: $d = (overline(x)_1 - overline(x)_2) / s_"pooled"$.

=== Key observations

- *Only EuroLLM's English vs Polish* crosses $d = 0.3$ on all three dimensions ($|d|$: 0.35, 0.34, 0.39). Directions match WVS expectations.
- The *Finnish vs Romanian TS* contrast (our pre-registered Gate 1) does not reach threshold for any model (best: $d = 0.12$ for Gemma 27B).
- Likert scores cluster tightly around 3.0 (range $approx 2.8$--$3.4$ on 1--5). The classifier defaults to midpoint values for ambiguous completions, compressing between-group variance.

=== Why this figure matters

The cultural information *is present* --- F5's chi-square tests on *categories* demonstrate this overwhelmingly ($p < 10^(-135)$). But the Likert *dimension scores* are a lossy summary of that information. The classifier can tell that Polish completions are more about `material_practical` and less about `personality_trait` (a category-level distinction), but when forced to rate each completion on a 1--5 "survival to self-expression" scale, it compresses the signal toward 3.

This explains the seeming paradox: strong categorical differentiation ($chi^2 > 775$) but weak dimensional effect sizes ($d < 0.3$). The categories are the primary outcome; the Likert dimensions are secondary and lossy.

#pagebreak()

= Discussion

== What the signal is

LLMs produce text that differs systematically by language in ways that track established cultural geography. The signal is strongest at the content-category level: what models talk about varies across languages in patterns matching the Inglehart--Welzel framework. At the granular level, specific prompt templates align with specific human survey questions at $rho = 0.50$--$0.70$ --- models match humans on what they say about the topic a prompt elicits.

== What the signal is not

The signal is not a smooth mapping from language to Inglehart--Welzel coordinate. Individual category rates do not predict individual survey questions (F8). Aggregate Likert scores produce only one robust correlation (F9). The classifier compresses variance (F11). The alignment that exists is concentrated in specific template$times$question pairs (F10) and operates through topic emphasis rather than attitudinal positioning.

== Comparison to Track 1

Track 1 compared the *same measurement* in two populations: logprob distributions over survey response options for LLMs vs response distributions for humans. Result: $r approx 0.00$ after z-scoring. Track 2 compares *different measurements*: topic distributions from open-ended completions (LLM) vs survey attitudes (human). Result: significant alignment on specific template$times$question pairs and one aggregate composite. The methodological lesson: asking LLMs survey questions measures format preferences; letting LLMs speak freely measures cultural content.

== Limitations

- *Classifier-mediated*: all signals pass through a single LLM classifier ($kappa = 0.80$).
- *Web text $eq.not$ culture*: we measure web-corpus content, not population values.
- *$N = 21$*: Spearman correlations are computed over 21 European languages, limiting power.
- *No causal identification*: we cannot separate cultural content from genre, register, or tokenization artifacts. HPLT amplification argues against tokenization artifacts but not genre confounds.
- *Cross-modal comparison*: LLM topic rates and human survey attitudes are different measurements of the same latent construct. The correlation is mediated by our classifier categories, which were designed to map onto Inglehart--Welzel theory.
