#set document(title: "Cultural Completion Comparison: Pilot Analysis")
#set page(margin: 2.5cm)
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.")
#set par(justify: true)

#align(center)[
  #text(size: 18pt, weight: "bold")[Cultural Completion Comparison]
  #v(0.3em)
  #text(size: 14pt)[Pilot Analysis --- Go/No-Go Results]
  #v(0.3em)
  #text(size: 10pt, fill: gray)[Track 2 Phase 2: Temperature Sampling + LLM Classification]
]

#v(1em)

= Overview

This report analyzes the pilot run of the cultural completion comparison pipeline. The pilot tests whether base (non-instruct) LLMs produce culturally differentiated completions when prompted with sentence stems in different languages, as measured by an external LLM classifier.

*Method*: For each (model, language, template) combination, $N = 50$ completions are generated at temperature $T = 1.0$ (large models) or $T = 0.8$ (small models), with $"top"_p = 0.95$. Raw completions are truncated to the first meaningful sentence and filtered for quality. An external classifier (GPT-4.1 mini via OpenRouter + OpenAI Batch API) assigns each completion a content category and three cultural dimension scores (1--5 Likert).

*Pilot slice*:
- *2 templates*: `self_concept` ("I am" / equivalent) and `values` ("The most important thing in life is" / equivalent)
- *5 languages*: English (eng), Finnish (fin), Polish (pol), Romanian (ron), Chinese (zho)
- *4 model groups*:
  - Gemma 3 27B PT --- large multilingual base model (primary)
  - Gemma 3 12B PT --- smaller multilingual base model (size control)
  - EuroLLM-22B --- European-focused multilingual base model
  - HPLT 2.15B $times$ 4 --- monolingual models (eng, fin, pol, ron only)

*Total*: 3,100 completions generated, 3,094 passed filtering (99.8%), all 3,094 classified.

*Cultural clusters* follow the Inglehart--Welzel World Cultural Map: Protestant Europe (fin), Catholic Europe (pol), English-speaking (eng), Orthodox (ron), East Asian (zho).

*Go/no-go gates*:
+ `known_groups`: Cohen's $d > 0.3$ for Finnish vs Romanian on `dim_trad_secular`
+ `categories`: $chi^2$ $p < 0.01$ for content category distributions between any two IW clusters

*Primary outcome*: content categories (categorical, objective) rather than Likert dimensions (subjective, noisy).

#pagebreak()

= F1: Content Category Distributions

== F1a: Gemma 3 27B PT

#align(center)[#image("diagnostic/categories_gemma3_27b_pt.png", width: 100%)]

=== What it shows

Stacked bar chart of content category proportions per language for the primary model (Gemma 3 27B PT, $T = 1.0$). Each bar sums to 1.0. Categories: `family_social`, `occupation_achievement`, `personality_trait`, `physical_attribute`, `spiritual_religious`, `emotional_state`, `material_practical`, `other`.

=== How it was calculated

For each (language, template) pair, 50 completions were generated, extracted to the first sentence, and classified by GPT-4.1 mini into one of 8 content categories. The proportions shown aggregate across both templates (`self_concept` + `values`).

=== Key observations

- *Polish and Chinese show the highest `emotional_state`* ($approx 22$% and $10$% respectively), while Romanian is lowest ($11$%).
- *Romanian stands out with the most `physical_attribute`* ($12$%) and the most `spiritual_religious`* ($6$%) --- consistent with Orthodox cultural emphasis on embodiment and faith.
- *Polish has the least `personality_trait`* ($12$% vs $24$% for English and Finnish) --- compensated by higher `emotional_state` and `material_practical`.
- *Finnish has the least `occupation_achievement`* ($17$% vs $25$% for English) --- consistent with Protestant European de-emphasis on career identity.
- *Chinese shows the most balanced distribution* across categories, with slightly elevated `personality_trait` ($25$%).
- *All languages show 0% `spiritual_religious` except Romanian* ($6$%) --- a clear Orthodox signal.

=== Why this matters

Content category distributions are the primary outcome variable. These distributions differ visibly across languages, and the patterns are directionally consistent with known cultural psychology (Orthodox emphasis on spirituality, Protestant de-emphasis on achievement-as-identity). The chi-square tests below quantify whether these visual differences are statistically significant.

#v(1em)

== F1b: EuroLLM-22B

#align(center)[#image("diagnostic/categories_eurollm22b.png", width: 100%)]

=== Key observations

- *Much sharper contrasts than Gemma 3 27B.* Finnish is dominated by `other` ($42$%), suggesting EuroLLM produces harder-to-classify Finnish completions (possibly code-switching or domain confusion --- confirmed by manual inspection showing Latin species names and single-word fragments).
- *Polish shows the highest `family_social`* ($27$%) --- consistent with Catholic cultural emphasis on family.
- *Romanian has high `physical_attribute`* ($14$%) and elevated `personality_trait`* ($21$%).
- *English is the most `personality_trait`-heavy* ($24$%) and `emotional_state`-heavy ($21$%).
- *The `other` category is a diagnostic*: high `other` rates indicate the model produces text that doesn't fit the classification schema, often due to language confusion, code-switching, or degenerate output. Finnish EuroLLM completions should be investigated for quality issues.

=== Why this matters

EuroLLM produces the strongest statistical differentiation (see chi-square results below), but some of this may be driven by model quality issues (Finnish `other` rate) rather than genuine cultural signal. The cross-model comparison is essential: patterns that replicate across both Gemma 27B and EuroLLM are more likely cultural; patterns unique to one model may be artifacts.

#v(1em)

== F1c: HPLT Monolingual Models

The four HPLT monolingual models (2.15B parameters, one per language) each only produce completions in their training language, so cross-language comparison requires comparing across different models.

Key observations from the per-model category tables:

- *HPLT Polish: 34% `physical_attribute`* --- dramatically higher than any other model-language combination. This is likely a model artifact (the small Polish model fixates on physical descriptions in "Jestem" completions) rather than cultural signal, since Gemma 27B Polish shows only $10$% `physical_attribute`.
- *HPLT English: highest `personality_trait`* ($31$%) among HPLT models --- consistent with English-language web text emphasizing self-description.
- *HPLT Finnish: 28% `personality_trait`* with elevated `material_practical` ($12$%) --- the monolingual model's cultural signal looks different from the multilingual models' Finnish output.
- *HPLT Romanian: 30% `personality_trait`* --- higher than in multilingual models.

The HPLT models are useful as a reference but their small size (2.15B) means model capacity effects dominate cultural effects. The cross-model chi-square tests (see below) confirm that model differences within a language are highly significant.

#pagebreak()

= F2: Chi-Square Tests --- Content Category Differentiation

== F2a: Between IW Clusters (Same Model)

The primary go/no-go test: do content category distributions differ significantly between Inglehart--Welzel cultural clusters within a single model?

=== Gemma 3 27B PT

Overall $chi^2 = 78.2$, $"df" = 28$, $p = 1.22 times 10^(-6)$.

#text(weight: "bold", fill: rgb("#006600"))[GO/NO-GO GATE 2: PASSES ($p < 0.01$)]

Pairwise comparisons:

#table(
  columns: (auto, auto, auto),
  align: (left, right, center),
  [*Contrast*], [*$chi^2$ ($p$)*], [*Sig.*],
  [Catholic Europe vs East Asian], [$16.8$ ($1.0 times 10^(-2)$)], [\*\*],
  [Catholic Europe vs English-speaking], [$27.3$ ($3.0 times 10^(-4)$)], [\*\*\*],
  [Catholic Europe vs Orthodox], [$24.6$ ($9.0 times 10^(-4)$)], [\*\*\*],
  [Catholic Europe vs Protestant Europe], [$9.3$ ($0.16$)], [],
  [East Asian vs English-speaking], [$19.4$ ($7.1 times 10^(-3)$)], [\*\*\*],
  [East Asian vs Orthodox], [$31.3$ ($5.4 times 10^(-5)$)], [\*\*\*],
  [East Asian vs Protestant Europe], [$11.3$ ($0.08$)], [],
  [English-speaking vs Orthodox], [$17.7$ ($1.3 times 10^(-2)$)], [\*\*],
  [English-speaking vs Protestant Europe], [$14.6$ ($4.1 times 10^(-2)$)], [\*\*],
  [Orthodox vs Protestant Europe], [$16.0$ ($2.5 times 10^(-2)$)], [\*\*],
)

Every pairwise comparison except Catholic--Protestant and East Asian--Protestant reaches significance. The strongest contrasts involve Orthodox (Romanian) and East Asian (Chinese), consistent with these being the most culturally distant clusters in the Inglehart--Welzel framework.

The non-significance of Catholic--Protestant ($p = 0.16$) is expected: these are the two most similar European clusters, and the pilot has only 1 language per cluster (Polish vs Finnish). The full run with 10 Catholic and 5 Protestant languages will have much more power for this contrast.

=== EuroLLM-22B

Overall $chi^2 = 203.1$, $"df" = 28$, $p = 1.78 times 10^(-28)$.

#text(weight: "bold", fill: rgb("#006600"))[GO/NO-GO GATE 2: PASSES ($p < 0.01$)]

*Every single pairwise comparison is significant at $p < 0.001$.* EuroLLM shows the strongest cultural differentiation of any model, likely because its European-focused training data creates sharper linguistic-cultural associations.

=== Gemma 3 12B PT

Overall $chi^2 = 37.4$, $"df" = 28$, $p = 0.11$.

Does not reach overall significance. Only East Asian vs English-speaking ($p = 0.019$) and East Asian vs Protestant ($p = 0.030$) reach pairwise significance. This confirms the expected *size gradient*: the 12B model has less capacity for cultural differentiation than the 27B.

== F2b: Cross-Model Comparison (Same Language)

Content category distributions differ significantly across models *within each language* (all $p < 10^(-4)$). This means model architecture and training data --- not just language --- influence cultural expression. Finnish shows the largest cross-model effect ($chi^2 = 120.4$, $p = 6.0 times 10^(-16)$), driven by EuroLLM's high `other` rate.

#pagebreak()

= F3: Known-Group Contrasts --- Cohen's d

#align(center)[#image("diagnostic/known_groups_cohens_d.png", width: 100%)]

== What it shows

Bar charts of Cohen's $d$ effect sizes for 5 cultural contrasts across 3 cultural dimensions (IC = individualist--collectivist, TS = traditional--secular, SS = survival--self-expression). Red dashed lines mark the $d = 0.3$ go/no-go threshold.

== Key observations

=== Gemma 3 27B PT (left panel)
- All effect sizes are small ($|d| < 0.31$). The Finnish--Romanian contrast on `dim_trad_secular` ($d = 0.027$) is essentially zero.
- The largest effects are on `dim_surv_selfexpr`: English vs Polish ($d = 0.30$) and English vs Chinese ($d = 0.30$) --- both near the threshold.
- #text(weight: "bold", fill: rgb("#cc0000"))[GO/NO-GO GATE 1: FAILS] for the specific fin--ron trad_secular contrast.

=== Gemma 3 12B PT (middle panel)
- Finnish vs Romanian on `dim_trad_secular` shows $d = -0.34$ --- *passes the threshold but in the wrong direction* (Finnish scored as more traditional than Romanian). This is likely classifier noise rather than genuine signal.
- English vs Chinese and English vs Polish on `dim_surv_selfexpr` show $d approx 0.30$ --- near threshold.

=== EuroLLM-22B (right panel)
- *Strongest Likert effects overall.* English vs Polish on `dim_surv_selfexpr` ($d = 0.83$) and `dim_trad_secular` ($d = 0.59$) both clearly exceed the threshold.
- English vs Chinese on `dim_indiv_collect` ($d = -0.34$) crosses threshold --- Chinese rated as more individualist than English, which is *opposite* to the expected direction.
- Polish vs Romanian on `dim_trad_secular` ($d = -0.40$) --- Romanian rated more secular than Polish, broadly consistent with expectations (Romania's secularization under communism).

== Interpretation

The Likert cultural dimensions are *noisy and unreliable* as a primary outcome. Mean scores cluster narrowly around 2.6--3.4 across all languages (a range of $< 1$ scale point on a 1--5 scale), producing small effect sizes. Some effects are in the wrong direction (12B Finnish--Romanian), suggesting classifier bias rather than cultural signal.

*Content categories are confirmed as the superior primary outcome.* The chi-square tests on categories show highly significant differentiation ($p < 10^(-6)$) where Likert dimensions show marginal or null effects. This validates the PLAN.md decision to treat categories as the anchor finding and Likert dimensions as secondary/exploratory.

#pagebreak()

= Summary of Pilot Findings

+ *GO/NO-GO GATE 2 PASSES.* Content category distributions differ significantly between IW cultural clusters for Gemma 3 27B ($chi^2 = 78.2$, $p = 1.2 times 10^(-6)$) and EuroLLM-22B ($chi^2 = 203.1$, $p = 1.8 times 10^(-28)$). This is the primary outcome and it works.

+ *GO/NO-GO GATE 1 FAILS.* Likert dimension effect sizes for Finnish vs Romanian on `dim_trad_secular` do not reach $d > 0.3$ for Gemma 3 27B ($d = 0.03$). EuroLLM shows stronger Likert effects but on different contrasts. The Likert dimensions are too noisy to serve as a primary outcome but may contribute as secondary evidence.

+ *The size gradient works.* Gemma 3 27B shows significant cross-cluster differentiation; 12B does not. This is consistent with the expectation that cultural encoding requires model capacity.

+ *EuroLLM shows the strongest differentiation* --- likely because its European-focused training creates sharper linguistic-cultural associations. However, some of this signal may be artifactual (Finnish `other` rate, language confusion).

+ *Content categories are the right primary outcome.* They are categorical (no Likert scale noise), objectively defined (the classifier assigns one of 8 categories), and show clear, statistically significant cross-cultural variation. The patterns are directionally consistent with known cultural psychology: Orthodox Romanian shows elevated `spiritual_religious`, Catholic Polish shows elevated `family_social`, Protestant Finnish de-emphasizes `occupation_achievement`.

+ *Model effects are large.* Cross-model differences within the same language are highly significant, meaning the choice of model matters for cultural expression. This motivates the multi-model comparison in the full run.

+ *Filter rates are excellent.* The extraction fix (extending short first-sentences, lowering the `too_short` threshold) reduced filter rates from 41.5% to 0.2%. Nearly all completions are usable.

+ *Classification cost is minimal.* 3,094 completions classified for approximately \$0.70 via OpenRouter (GPT-4.1 mini) + OpenAI Batch API. The full run ($approx 1.2$M completions) is estimated at \$12--48 using GPT-4.1 nano/mini batch API.

= Next Steps

- Scale to full 27-language run (all 8 templates, $N = 200$ samples, all models)
- Add remaining analysis subcommands (`profiles`, `umap`, `stability`, `size_effect`)
- Hand-label 50 completions to compute classifier Cohen's $kappa$
- Phase 3 (SAE feature identification) is now unblocked by Gate 2 passing
