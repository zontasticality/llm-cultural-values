"""Quick comparison of Gemma vs GPT-4.1-mini classification on same completions."""
import asyncio, os, sqlite3, sys
from openai import AsyncOpenAI
sys.path.insert(0, os.path.dirname(__file__))
from classify.prompts import CLASSIFIER_SYSTEM, make_classifier_prompt, parse_classification

def main():
    db = sqlite3.connect('data/culture.db')
    db.row_factory = sqlite3.Row
    rows = db.execute("""
        SELECT c.completion_id, c.completion_text, p.lang, p.template_id
        FROM completions c JOIN prompts p ON c.prompt_id = p.prompt_id
        WHERE c.filter_status = 'ok'
          AND p.lang IN ('eng', 'fin', 'pol', 'zho', 'ron')
          AND p.template_id IN ('self_concept', 'values', 'moral', 'belief')
        GROUP BY p.lang, p.template_id ORDER BY p.lang, p.template_id
    """).fetchall()
    print(f"Comparing {len(rows)} completions", flush=True)

    or_client = AsyncOpenAI(base_url='https://openrouter.ai/api/v1', api_key=os.environ['OPENROUTER_API_KEY'])
    oai_client = AsyncOpenAI()  # uses OPENAI_API_KEY
    sem = asyncio.Semaphore(5)

    async def call(client, model, r, max_tokens=200):
        msg = make_classifier_prompt(r['completion_text'], r['lang'], r['template_id'])
        async with sem:
            for attempt in range(5):
                try:
                    resp = await client.chat.completions.create(
                        model=model, max_tokens=max_tokens,
                        messages=[{'role':'system','content':CLASSIFIER_SYSTEM},{'role':'user','content':msg}],
                        temperature=0, response_format={'type':'json_object'})
                    return parse_classification(resp.choices[0].message.content)
                except Exception as e:
                    if '429' in str(e) and attempt < 4:
                        await asyncio.sleep(2 ** attempt + 1)
                    else:
                        print(f"  FAIL {model}: {e}", flush=True)
                        return None

    async def run():
        print("Running Gemma (OpenRouter)...", flush=True)
        gr = await asyncio.gather(*[call(or_client, 'google/gemma-3-27b-it', r) for r in rows])
        print(f"Gemma: {sum(1 for x in gr if x)}/{len(gr)} ok", flush=True)

        print("Running GPT-4.1-mini (OpenAI)...", flush=True)
        pr = await asyncio.gather(*[call(oai_client, 'gpt-4.1-mini', r) for r in rows])
        print(f"GPT:   {sum(1 for x in pr if x)}/{len(pr)} ok\n", flush=True)

        agree = 0
        td, id_, sd = [], [], []
        for r, g, p in zip(rows, gr, pr):
            if not g or not p:
                continue
            t = r['completion_text'][:40]
            tag = f"{r['lang']}:{r['template_id']}"
            eq = '==' if g['content_category'] == p['content_category'] else '!='
            if eq == '==':
                agree += 1
            td.append(abs(g['dim_trad_secular'] - p['dim_trad_secular']))
            id_.append(abs(g['dim_indiv_collect'] - p['dim_indiv_collect']))
            sd.append(abs(g['dim_surv_selfexpr'] - p['dim_surv_selfexpr']))
            print(f"{tag:<18s} {t:<40s} | {g['content_category']:>17s} {g['dim_indiv_collect']}/{g['dim_trad_secular']}/{g['dim_surv_selfexpr']}  {eq} {p['content_category']:>17s} {p['dim_indiv_collect']}/{p['dim_trad_secular']}/{p['dim_surv_selfexpr']}")

        n = len(td)
        if n == 0:
            print("No valid pairs!")
            return
        print(f"\nCategory agreement: {agree}/{n} ({agree/n*100:.0f}%)")
        print(f"Mean |diff|: IC={sum(id_)/n:.2f}  TS={sum(td)/n:.2f}  SS={sum(sd)/n:.2f}")
        for nm in ['IC', 'TS', 'SS']:
            k = {'TS': 'dim_trad_secular', 'IC': 'dim_indiv_collect', 'SS': 'dim_surv_selfexpr'}[nm]
            dg = {v: sum(1 for x in gr if x and x[k] == v) for v in range(1, 6)}
            dp = {v: sum(1 for x in pr if x and x[k] == v) for v in range(1, 6)}
            print(f"  {nm} Gemma: {dg}  GPT: {dp}")

    asyncio.run(run())

if __name__ == '__main__':
    main()
