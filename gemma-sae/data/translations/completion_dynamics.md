# Completion Dynamics Analysis — Pre-Registered Predictions

This document serves as pre-registered expectations for Phase 2 pilot results. Deviations from these distributions — especially when they contradict the structural-confound direction — constitute stronger evidence of genuine cultural signal.

## 1. Validation Summary

### Rating Table (Naturalness / Semantic Equivalence / Completion Openness, each 1-5)

| template_id | eng | fin | pol | ron | zho | Flags |
|---|---|---|---|---|---|---|
| self_concept | 5/5/5 | 4/4/4 | 4/4/4 | 4/4/4 | 4/4/4 | zho "我是一个" changed to "我是"; ron "Eu sunt" changed to "Sunt" |
| values | 5/5/4 | 5/5/4 | 5/5/4 | 5/5/4 | 5/5/4 | Highly parallel across all languages |
| moral | 4/5/4 | 4/5/4 | 4/5/4 | 4/5/4 | 3/4/3 | zho "那种" framing diverges slightly |
| childrearing | 4/5/4 | 4/5/4 | 4/4/3 | 4/5/4 | 4/4/3 | pol genitive constraint; zho "从小学会" adds semantics |
| family | 5/5/4 | 4/4/4 | 5/5/4 | 5/5/4 | 4/4/4 | fin "meidän" and zho "我们家" use "our" not "my" |
| success | 5/5/5 | 5/5/5 | 5/5/5 | 5/5/5 | 5/5/5 | Best parallel template — use as primary benchmark |
| decision | 4/5/4 | 4/5/4 | 4/5/4 | 4/5/4 | 4/5/4 | ron dangling "eu" removed |
| belief | 5/5/5 | 5/5/5 | 5/5/5 | 5/5/5 | 5/5/5 | pol "wierzę" has stronger religious priming than equivalents |

### Applied Fixes (Priority 1)

1. **Chinese self_concept**: "我是一个" → "我是" — classifier was forcing nominal completions, eliminating adjective pathway
2. **Romanian self_concept**: "Eu sunt" → "Sunt" — emphatic pronoun removed (pro-drop is neutral form)
3. **Romanian decision**: Removed dangling "eu" at end
4. **Romanian belief**: "Eu cred că" → "Cred că" — emphatic pronoun removed

### Remaining Known Confounds (document, don't fix)

- **Polish "wierzę"** has stronger religious priming than equivalents (same verb as Nicene Creed). Some religious inflation on belief template is expected as a lexical artifact.
- **Chinese childrearing** "从小学会" adds "from a young age" + "master" semantics absent from English. Biases toward developmental/achievement completions.
- **Finnish/Chinese family** use "our" instead of "my" — reflects genuine usage but shifts framing toward collective norms.

### Open Questions for User Decision (Priority 2)

- Consider removing "从小" from Chinese childrearing → "孩子应该学会"
- Consider changing Finnish family to "Perheessäni tärkeintä on" (my family) — naturalness cost
- Consider changing Chinese family to "在我家，最重要的是" (my family) — actually also natural
- Consider adding Polish "Myślę, że" (I think that) as secular-framing parallel to "Wierzę, że"

---

## 2. Per-Template Cross-Language Analysis

### 2.1 SELF_CONCEPT: "I am"

| Language | Stem | Grammatical constraints | Top expected completions |
|---|---|---|---|
| eng | I am | Maximally open: adj, NP, PP, clause | a student, happy, from X, tired, a developer |
| fin | Olen | Pro-drop copula, same openness as eng | opettaja, iloinen, kotoisin Helsingistä, 35-vuotias |
| pol | Jestem | Instrumental case for nouns, gender on adj | studentem, szczęśliwy/a, z Polski, katolikiem |
| ron | Sunt | Nominative predicates, adj agreement | student, fericit/ă, din România, sigur/ă |
| zho | 我是 | Copula, takes NP or adj+的 | 一个学生, 中国人, 一个普通人, 开心的 |

**Confound risk**: LOW across all (after fixes). Chinese still slightly favors NP over bare adj.

**IW predictions**: fin/eng → personality traits, occupations (secular-individual). pol → Catholic identity markers possible. ron → national identity. zho → relational roles, group membership, humility ("普通人").

**Cross-language equivalence**: eng ≈ fin > pol ≈ ron > zho

### 2.2 VALUES: "The most important thing in life is"

| Language | Stem | Top expected completions |
|---|---|---|
| eng | The most important thing in life is | family, love, happiness, health, being true to yourself |
| fin | Elämässä tärkeintä on | terveys, rakkaus, perhe, onnellisuus, se että... |
| pol | Najważniejszą rzeczą w życiu jest | rodzina, zdrowie, miłość, wiara, szczęście |
| ron | Cel mai important lucru în viață este | familia, sănătatea, dragostea, credința, fericirea |
| zho | 人生中最重要的是 | 健康, 家人, 快乐, 自由, 找到自己 |

**Confound risk**: LOW across all. Best-matched template after success.

**IW predictions**: fin/eng → post-materialist (happiness, purpose, authenticity). pol/ron → traditional (family, faith, health). zho → pragmatic-collectivist (family, education, effort). **This template should produce the clearest Inglehart-axis separation.**

**Predicted category distributions**:

| lang | fam_soc | occ_ach | pers | mat | abs_phi | rel_spi | nat_civ | other |
|---|---|---|---|---|---|---|---|---|
| eng | 25% | 10% | 5% | 10% | 30% | 10% | 2% | 8% |
| fin | 20% | 8% | 5% | 8% | 40% | 3% | 2% | 14% |
| pol | 30% | 8% | 3% | 8% | 20% | 18% | 3% | 10% |
| ron | 30% | 7% | 3% | 10% | 18% | 20% | 3% | 9% |
| zho | 25% | 15% | 3% | 8% | 30% | 5% | 4% | 10% |

### 2.3 MORAL: "A good person is someone who"

| Language | Stem | Top expected completions |
|---|---|---|
| eng | A good person is someone who | helps others, is honest, does the right thing, respects others |
| fin | Hyvä ihminen on sellainen, joka | auttaa muita, on rehellinen, välittää toisista, kunnioittaa |
| pol | Dobry człowiek to ktoś, kto | pomaga innym, jest uczciwy, szanuje innych, kocha bliźniego |
| ron | Un om bun este cel care | ajută pe alții, este cinstit, respectă, se teme de Dumnezeu |
| zho | 一个好人，是那种 | 会为别人着想的人, 善良且真诚, 有责任感, 说到做到 |

**Confound risk**: LOW-MEDIUM. Chinese "那种" pushes toward behavioral typification + "的人" suffix structure. Minor divergence.

**IW predictions**: fin/eng → care/fairness moral foundations. pol/ron → care + loyalty/purity + religious moral vocabulary. zho → relational/Confucian virtues (仁义信), group harmony. **Childrearing and moral are the most diagnostic templates for autonomy-conformity axis.**

### 2.4 CHILDREARING: "Children should learn to"

| Language | Stem | Top expected completions |
|---|---|---|
| eng | Children should learn to | be independent, respect others, read, think for themselves |
| fin | Lasten pitäisi oppia | kunnioittamaan muita, itsenäisiksi, sietämään pettymyksiä |
| pol | Dzieci powinny uczyć się | szacunku, samodzielności, odpowiedzialności, posłuszeństwa |
| ron | Copiii ar trebui să învețe să | respecte, fie independenți, citească, asculte |
| zho | 孩子应该从小学会 | 独立, 尊重他人, 吃苦, 感恩, 自律 |

**Confound risk**: MEDIUM for Chinese (从小学会 adds developmental/mastery semantics). LOW for others.

**IW predictions**: **Most diagnostic template.** fin/eng → independence, critical thinking, emotional regulation ("sietämään pettymyksiä" = tolerating disappointments is characteristically Nordic). pol/ron → respect for elders, obedience, faith. zho → 吃苦 (endure hardship), 感恩 (gratitude), 孝顺 (filial piety).

**Key culturally diagnostic completions to watch for**:
- Finnish "sietämään pettymyksiä" (tolerate disappointments) — Nordic resilience education
- Polish "posłuszeństwa" (obedience) — Catholic-traditional conformity
- Chinese "吃苦" (eat bitterness / endure hardship) — Confucian childrearing icon
- Romanian "asculte" (listen/obey) — Orthodox respect hierarchy

### 2.5 FAMILY: "In my family, the most important thing is"

| Language | Stem | Framing note | Top expected completions |
|---|---|---|---|
| eng | In my family... | "my" | love, togetherness, respect, communication |
| fin | Meidän perheessä... | "our" | rakkaus, yhdessäolo, luottamus, avoimuus |
| pol | W mojej rodzinie... | "my" | miłość, szacunek, wiara, zdrowie |
| ron | În familia mea... | "my" | dragostea, respectul, sănătatea, credința |
| zho | 在我们家... | "our" | 和睦, 健康, 团圆, 孝顺, 教育 |

**Confound risk**: LOW-MEDIUM. Finnish and Chinese "our" may inflate consensus-oriented completions.

**IW predictions**: fin/eng → emotional bonds, egalitarianism, communication. pol/ron → duty, faith, tradition, intergenerational respect. zho → 和谐/和睦 (harmony), 教育 (education), 孝顺 (filial piety). Chinese "和睦" (harmony) as top completion would be a distinctive cultural marker.

### 2.6 SUCCESS: "Success means"

**Best parallel template — use as primary cross-language benchmark.**

| Language | Stem | Top expected completions |
|---|---|---|
| eng | Success means | being happy, achieving goals, money, different things to different people |
| fin | Menestys tarkoittaa | onnellisuutta, hyvää elämää, taloudellista vapautta |
| pol | Sukces to | ciężka praca, szczęście, pieniądze, osiąganie celów |
| ron | Succesul înseamnă | muncă, fericire, bani, sacrificiu |
| zho | 成功就是 | 做自己喜欢的事, 努力, 实现目标, 有钱 |

**Confound risk**: LOW across all five. Highest structural equivalence of any template.

**IW predictions**: fin/eng → post-materialist (happiness, purpose, work-life balance). pol/ron → materialist + effort (money, hard work, sacrifice). zho → mixed (material aspiration + philosophical contentment). Romanian "sacrificiu" (sacrifice) would be a survival-axis marker. Finnish anti-ostentatious framing expected.

### 2.7 DECISION: "When making a difficult decision, I"

| Language | Stem | Top expected completions |
|---|---|---|
| eng | When making a difficult decision, I | think carefully, consider options, pray, trust my gut |
| fin | Kun teen vaikean päätöksen, | mietin tarkkaan, kuuntelen itseäni, kysyn neuvoa |
| pol | Podejmując trudną decyzję, | zastanawiam się, modlę się, rozmawiam z rodziną |
| ron | Când trebuie să iau o decizie dificilă, | mă gândesc mult, mă rog, cer sfatul familiei |
| zho | 面对艰难的选择时，我 | 会仔细考虑, 问家人意见, 相信自己的判断 |

**IW predictions**: fin/eng → autonomous, rational deliberation, self-trust. pol/ron → prayer, family consultation, values-guided. zho → family/elder consultation, practical deliberation. **Captures autonomy-embeddedness axis.**

**Key diagnostic**: Whether "pray" appears is a strong religious/secular discriminator. Polish "modlę się" and Romanian "mă rog" vs. Finnish/Chinese where prayer is unlikely.

### 2.8 BELIEF: "I believe that"

| Language | Stem | Top expected completions |
|---|---|---|
| eng | I believe that | everyone deserves..., God..., everything happens for a reason |
| fin | Uskon, että | kaikki järjestyy, jokainen ihminen..., elämällä on tarkoitus |
| pol | Wierzę, że | Bóg istnieje, wszystko będzie dobrze, każdy ma szansę |
| ron | Cred că | Dumnezeu există, totul se întâmplă cu un motiv, viitorul... |
| zho | 我相信 | 未来会更好, 努力就会有回报, 一切都会好的 |

**Confound risk**: MEDIUM for Polish ("wierzę" has stronger religious loading than equivalents).

**IW predictions**: fin/zho → secular propositions (humanistic, meritocratic, future-oriented). pol/ron → theological propositions (God exists, divine providence). **Largest religious/secular gap expected here.** Some Polish religious inflation is a lexical artifact of "wierzę."

---

## 3. Critical Confounds Summary (Ranked)

| Rank | Confound | Templates | Direction of bias | Severity |
|---|---|---|---|---|
| 1 | Polish "wierzę" religious priming | belief | Inflates religious completions for Polish | MEDIUM-HIGH |
| 2 | Chinese "从小学会" extra semantics | childrearing | Inflates achievement/mastery for Chinese | MEDIUM |
| 3 | Finnish/Chinese "our" vs "my" family | family | Inflates consensus-oriented completions | MEDIUM |
| 4 | Chinese "那种" typification frame | moral | Pushes toward behavioral vs trait descriptions | LOW-MEDIUM |
| 5 | Polish genitive requirement after "uczyć się" | childrearing | Favors abstract nouns over concrete skills | LOW |
| 6 | Gender marking on Polish adjective predicates | self_concept | Reveals model gender default (not cultural confound) | LOW |

---

## 4. Predicted Category Distributions (Pre-Registration Table)

### All 40 (language, template) pairs

| template | lang | fam_soc | occ_ach | pers_tr | mat_phy | abs_phi | rel_spi | nat_civ | other |
|---|---|---|---|---|---|---|---|---|---|
| self_concept | eng | 10 | 30 | 25 | 5 | 5 | 3 | 2 | 20 |
| self_concept | fin | 10 | 30 | 25 | 5 | 5 | 2 | 3 | 20 |
| self_concept | pol | 15 | 25 | 20 | 5 | 5 | 8 | 5 | 17 |
| self_concept | ron | 12 | 22 | 20 | 5 | 5 | 8 | 10 | 18 |
| self_concept | zho | 15 | 30 | 18 | 3 | 4 | 2 | 15 | 13 |
| values | eng | 25 | 10 | 5 | 10 | 30 | 10 | 2 | 8 |
| values | fin | 20 | 8 | 5 | 8 | 40 | 3 | 2 | 14 |
| values | pol | 30 | 8 | 3 | 8 | 20 | 18 | 3 | 10 |
| values | ron | 30 | 7 | 3 | 10 | 18 | 20 | 3 | 9 |
| values | zho | 25 | 15 | 3 | 8 | 30 | 5 | 4 | 10 |
| moral | eng | 15 | 5 | 35 | 2 | 25 | 10 | 3 | 5 |
| moral | fin | 12 | 5 | 35 | 2 | 30 | 3 | 3 | 10 |
| moral | pol | 15 | 5 | 30 | 2 | 20 | 18 | 5 | 5 |
| moral | ron | 15 | 5 | 28 | 2 | 20 | 20 | 5 | 5 |
| moral | zho | 15 | 8 | 30 | 2 | 25 | 5 | 5 | 10 |
| childrearing | eng | 15 | 20 | 25 | 5 | 20 | 5 | 5 | 5 |
| childrearing | fin | 10 | 20 | 30 | 5 | 25 | 2 | 3 | 5 |
| childrearing | pol | 15 | 15 | 25 | 5 | 15 | 12 | 5 | 8 |
| childrearing | ron | 15 | 15 | 20 | 5 | 15 | 15 | 5 | 10 |
| childrearing | zho | 15 | 25 | 20 | 5 | 15 | 3 | 5 | 12 |
| family | eng | 50 | 3 | 5 | 5 | 15 | 10 | 2 | 10 |
| family | fin | 50 | 3 | 5 | 5 | 20 | 2 | 2 | 13 |
| family | pol | 45 | 3 | 5 | 5 | 12 | 20 | 3 | 7 |
| family | ron | 45 | 3 | 5 | 5 | 10 | 22 | 3 | 7 |
| family | zho | 45 | 10 | 3 | 5 | 15 | 5 | 5 | 12 |
| success | eng | 10 | 35 | 10 | 15 | 20 | 3 | 2 | 5 |
| success | fin | 10 | 25 | 10 | 10 | 35 | 2 | 2 | 6 |
| success | pol | 10 | 35 | 8 | 18 | 15 | 5 | 4 | 5 |
| success | ron | 10 | 35 | 5 | 20 | 15 | 5 | 5 | 5 |
| success | zho | 8 | 35 | 5 | 18 | 22 | 3 | 4 | 5 |
| decision | eng | 20 | 5 | 15 | 2 | 30 | 15 | 3 | 10 |
| decision | fin | 15 | 5 | 20 | 2 | 40 | 3 | 2 | 13 |
| decision | pol | 20 | 5 | 15 | 2 | 25 | 20 | 3 | 10 |
| decision | ron | 20 | 5 | 15 | 2 | 25 | 20 | 3 | 10 |
| decision | zho | 25 | 5 | 10 | 2 | 35 | 5 | 5 | 13 |
| belief | eng | 10 | 8 | 5 | 3 | 35 | 25 | 5 | 9 |
| belief | fin | 8 | 8 | 5 | 3 | 50 | 10 | 5 | 11 |
| belief | pol | 8 | 5 | 3 | 3 | 25 | 40 | 8 | 8 |
| belief | ron | 8 | 5 | 3 | 3 | 25 | 38 | 8 | 10 |
| belief | zho | 8 | 12 | 5 | 3 | 45 | 8 | 10 | 9 |

### Aggregated predictions by language (averaged across all 8 templates)

| Language | fam_soc | occ_ach | pers_tr | mat_phy | abs_phi | rel_spi | nat_civ | other |
|---|---|---|---|---|---|---|---|---|
| eng | 19.4 | 14.5 | 15.6 | 5.9 | 22.5 | 10.1 | 3.0 | 9.0 |
| fin | 16.9 | 13.0 | 16.9 | 5.4 | 30.6 | 3.4 | 2.8 | 11.5 |
| pol | 19.8 | 12.6 | 13.6 | 4.8 | 17.0 | 17.6 | 4.5 | 8.8 |
| ron | 19.4 | 12.1 | 12.4 | 5.3 | 16.5 | 18.5 | 5.3 | 8.6 |
| zho | 19.5 | 17.5 | 11.4 | 5.5 | 24.5 | 4.5 | 6.6 | 10.3 |

**Key predicted differences**:
- **Religious/spiritual**: pol (17.6%) ≈ ron (18.5%) >> eng (10.1%) > zho (4.5%) ≈ fin (3.4%)
- **Abstract/philosophical**: fin (30.6%) > zho (24.5%) > eng (22.5%) >> pol/ron (~17%)
- **Occupation/achievement**: zho (17.5%) > eng (14.5%) > fin/pol/ron (~12-13%)
- **Personality trait**: fin (16.9%) ≈ eng (15.6%) > pol (13.6%) > ron (12.4%) > zho (11.4%)

These aggregate differences, if they appear in actual model outputs, constitute the primary behavioral finding of Phase 2.

---

## 5. Methodological Recommendations

1. **Use the success template as primary cross-language benchmark** — highest structural equivalence, most open completion space.

2. **Treat childrearing as the most diagnostic template** for Inglehart autonomy-conformity — but note Chinese structural confound.

3. **Report confound-adjusted results alongside raw results** for templates with known structural non-equivalences.

4. **Consider running both original and fixed versions** of Chinese self_concept (我是 vs 我是一个) to empirically measure the confound effect.

5. **Flag Polish belief template** religious content as potentially inflated by the "wierzę" lexical confound.
