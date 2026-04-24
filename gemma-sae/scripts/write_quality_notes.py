"""Write quality_notes with failure patterns + alternative prompts for low-quality prompts.

Covers prompts with quality_score < 0.8 (all short-pronoun self_concepts plus the
noisy belief prompts). For each, stores JSON:
  {"pattern": "...", "alternatives": ["...", "..."]}

Alternatives are longer, disambiguated variants of the same semantic slot that
should cut the homograph/short-stem noise we see in self_concept. They are
NOT inserted as new prompt rows — that's a decision for the next sampling run.

Usage:
    PYTHONPATH=. python scripts/write_quality_notes.py --db data/culture.db
"""
import argparse
import json
import sqlite3

# Keyed by prompt_id. Each entry: (pattern, alternatives)
NOTES: dict[int, tuple[str, list[str]]] = {
    # ─── Unsalvageable (<0.5) ────────────────────────────────────
    487: (  # tur self_concept "Ben"
        "Short pronoun; 'Ben' is also a common English/Dutch name and a stem "
        "for many non-Turkish completions (Benvenuti, Benjamín, Benötigen). "
        "EuroLLM drifts to Italian/German/Dutch scraped web text 83% of the time. "
        "Classifier under-filters named-entity completions ('Ben is a Partner at...').",
        [
            "Kendimi şöyle tanımlarım: ben",
            "Bir insan olarak ben",
        ],
    ),
    455: (  # slk self_concept "Som"
        "Short verb stem; 'som' is also the English word 'some' prefix, Hungarian 'som', "
        "and stems to 'Somali', 'Somige' (Dutch), etc. Cross-lingual noise 56%; "
        "even gemma3 models drift to English/Dutch/German content.",
        [
            "Ako človek som",
            "Keď hovorím o sebe, som",
        ],
    ),
    463: (  # slv self_concept "Sem"
        "Short verb form; 'sem' collides with English 'semester/seminar', "
        "Latvian genitive, Portuguese 'sem' (without), etc. "
        "EuroLLM 77% noise. Non-other slots also heavily polluted by scraped text.",
        [
            "Kot oseba sem",
            "O sebi lahko rečem, da sem",
        ],
    ),

    # ─── Degraded (0.5-0.75) ─────────────────────────────────────
    374: (  # hin self_concept "मैं"
        "Hindi first-person pronoun alone; EuroLLM has weak Hindi coverage (72% noise). "
        "Gemma3 models handle it better (14-22%). Prompt is acceptable for Gemma-only analysis.",
        [
            "एक व्यक्ति के रूप में मैं",
            "अपने बारे में कहूं तो मैं",
        ],
    ),
    446: (  # ron self_concept "Sunt"
        "'Sunt' stems to 'Suntikan' (Indonesian injection), and all models drift to "
        "news headlines ('Sunt deja la jumătatea lunii'). hplt2c_ron handles it (17%) "
        "but multilingual models are 35-55% noise.",
        [
            "Ca persoană, sunt",
            "Despre mine pot spune că sunt",
        ],
    ),
    398: (  # ita self_concept "Sono"
        "'Sono' means 'I am' but also 'they are' and 'sound'; common news framing "
        "('Sono 68 le persone...', 'Sono stati resi noti...'). Even hplt2c_ita is 41% noise.",
        [
            "Come persona sono",
            "Parlando di me, sono",
        ],
    ),
    471: (  # spa self_concept "Soy"
        "'Soy' completes as soybean/soymilk (English), Soyuz (Russian space program), "
        "or French 'soyez'. 31% noise overall; 43% on gemma3_27b_pt. "
        "hplt2c_spa is fine (2%) because it's monolingual.",
        [
            "Como persona, soy",
            "Sobre mí puedo decir que soy",
        ],
    ),
    390: (  # hun self_concept "Én"
        "'Én' alone; EuroLLM 66% noise, drifts to French 'Énergie'. "
        "Other models OK (17-20%) but still noisier than full-sentence prompts.",
        [
            "Mint ember, én",
            "Magamról annyit mondhatok, hogy én",
        ],
    ),
    349: (  # fin self_concept "Olen"
        "EuroLLM catastrophic (93% noise — drifts to Swedish Wikipedia beetle articles). "
        "Other models fine (5-7%). Likely unusable for EuroLLM; OK for Gemma/HPLT.",
        [
            "Ihmisenä olen",
            "Itsestäni voin sanoa, että olen",
        ],
    ),
    389: (  # hrv belief "Vjerujem da"
        "EuroLLM 50% noise on belief prompt. Classifier also under-filters: "
        "many 'non-other' slots are fragments that are not actual belief statements.",
        [
            "Osobno vjerujem da",
            "Moje je uvjerenje da",
        ],
    ),
    324: (  # eng self_concept "I am"
        "Captures Stack Overflow / web boilerplate ('I am trying to...', 'I am having a problem'). "
        "26% other; the classifier catches most but these are coherent English, so "
        "residual mis-categorization is likely.",
        [
            "As a person, I am",
            "Speaking about myself, I am",
        ],
    ),
    406: (  # lit self_concept "Aš esu"
        "EuroLLM 70% noise (Wikipedia fragments). Other models fine (7-13%).",
        [
            "Kaip žmogus, aš esu",
            "Apie save galiu pasakyti, kad aš esu",
        ],
    ),
    268: (  # ara self_concept "أنا"
        "EuroLLM 68% noise on Arabic pronoun. Gemma3 models are fine (2-4%). "
        "Use Gemma-only for this prompt in analysis.",
        [
            "كشخص، أنا",
            "عن نفسي، أنا",
        ],
    ),
    479: (  # swe self_concept "Jag är"
        "EuroLLM 69% noise (Swedish Wikipedia song/film articles). Other models fine (5-10%).",
        [
            "Som person är jag",
            "Om mig själv: jag är",
        ],
    ),
    502: (  # tur belief "Bence"
        "'Bence-Jones proteins' medical term pulls completions off-topic. "
        "23% noise evenly across models.",
        [
            "Bence hayatta en önemli şey",
            "Bana göre",
        ],
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="data/culture.db")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    n_written = 0
    for pid, (pattern, alternatives) in NOTES.items():
        row = conn.execute(
            "SELECT lang, template_id, prompt_text, quality_score "
            "FROM prompts WHERE prompt_id=?", (pid,),
        ).fetchone()
        if row is None:
            print(f"  skip {pid}: not found")
            continue
        lang, tmpl, text, qs = row
        payload = json.dumps({
            "pattern": pattern,
            "alternatives": alternatives,
        }, ensure_ascii=False)
        conn.execute(
            "UPDATE prompts SET quality_notes=? WHERE prompt_id=?",
            (payload, pid),
        )
        print(f"  [{pid}] {lang}/{tmpl} (q={qs:.2f}) '{text}' -> {len(alternatives)} alts")
        n_written += 1
    conn.commit()
    conn.close()
    print(f"\nWrote notes for {n_written} prompts.")


if __name__ == "__main__":
    main()
