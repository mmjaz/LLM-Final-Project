import os
import json
from tqdm import tqdm
from openai import OpenAI


def extract_triplets(llm, ctx):
    query = f"""Extract triplets informative from the text following the examples. Make sure the triplet texts are only directly from the given text! Complete directly and strictly following the instructions without any additional words, line break nor space!
--------------------
Text: قاسم حاجی‌زاده (زادهٔ ۱۲ فروردین ۱۳۲۶) نقاش و کارگردان ایرانی مقیم فرانسه است. حاجی‌زاده در لنگرود بزرگ شد و در هنرستان هنرهای تجسمی تهران تحصیل کرد. او از پیشگامان هنر مردمی در ایران است.
Triplets:<قاسم حاجی‌زاده##متولد##1326>$$<قاسم حاجی‌زاده##ملیت##ایرانی>$$<قاسم حاجی‌زاده##شعل##کارگردان>$$<قاسم حاجی‌زاده##شغل##نقاش>$$$$<قاسم حاجی‌زاده##محل_تحصیل##هنرستان هنرهای تجسمی تهران>$$<قاسم حاجی‌زاده##محل_اقامت##فرانسه>$$<قاسم حاجی‌زاده##محل_رشد##لنگرود>
--------------------
Text: قاتلان ماه گل که با نام قاتلان ماه کامل نیز شناخته می‌شود، فیلم درام جنایی وسترن حماسی آمریکایی محصول ۲۰۲۳ به کارگردانی و تهیه‌کنندگی مارتین اسکورسیزی است که فیلم‌نامه را مشترکاً با اریک راث بر اساس کتابی به همین نام نوشتهٔ دیوید گرن چاپ ۲۰۱۷ به نگارش درآورد. داستان فیلم به قتل‌های ایالات اکلاهما در مردمان اوسیج طی دههٔ ۱۹۲۰ می‌پردازد که پس از کشف نفت در زمین‌های قبیله‌ای رخ داد. لئوناردو دی‌کاپریو، رابرت دنیرو و لیلی گلداستون گروه اصلی بازیگران را تشکیل می‌دهند
Triplets:<قاتلان ماه گل##نام_جایگزین##قاتلان ماه کامل>$$<قاتلان ماه گل##نوع##فیلم درام جنایی وسترن حماسی>$$<قاتلان ماه گل##کشور_سازنده##آمریکا>$$<قاتلان ماه گل##سال_تولید##2023>$$<قاتلان ماه گل##کارگردان##مارتین اسکورسیزی>$$<قاتلان ماه گل##تهیه‌کننده##مارتین اسکورسیزی>$$<مارتین اسکورسیزی##همکار_نویسنده##اریک راث>$$<کتاب قاتلان ماه گل##نویسنده##دیوید گرن>$$<کتاب قاتلان ماه گل##سال_چاپ##2017>$$<قتل‌های ایالات اکلاهما##قربانیان##مردمان اوسیج>$$<قتل‌های ایالات اکلاهما##زمان##دههٔ ۱۹۲۰>$$<قتل‌های ایالات اکلاهما##علت##کشف نفت در زمین‌های قبیله‌ای>$$<قاتلان ماه گل##بازیگر##لئوناردو دی‌کاپریو>$$<قاتلان ماه گل##بازیگر##رابرت دنیرو>$$<قاتلان ماه گل##بازیگر##لیلی گلداستون>$$
--------------------
Text: {ctx}
Triplets:"""
    resp = llm.completions.create(prompt=query, model="gemini-2.5-flash-lite-preview-06-17")
    # resp = resp.content.strip()
    resp = resp.choices[0].text.strip()
    triplets = set()
    triplet_texts = resp.split('$$')
    for triplet_text in triplet_texts:
        if len(triplet_text) <= 6:
            continue
        triplet_text = triplet_text[1:-1]
        tokens = triplet_text.split('##')
        if not len(tokens) == 3:
            continue
        h = tokens[0].strip()
        r = tokens[1].strip()
        t = tokens[2].strip()
        if ('no ' in h) or ('no ' in t) or ('نامشخص' in h) or ('unknown' in t) or ('No ' in h) or ('No ' in t) or ('Unknown' in h) or ('Unknown' in t) or ('null' in h) or ('null' in t) or ('Null' in h) or ('Null' in t) or ('NULL' in h) or ('NULL' in t) or ('NO' in h) or ('NO' in r) or ('NO' in t) or (h==t):
            continue
        if (r not in ctx) and (t not in ctx):
            continue

        triplets.add((h, r, t))
    triplets = [[h,r,t] for (h,r,t) in triplets]
    return triplets

# data_path = '../../data/hotpotqa/hotpot_dev_distractor_v1.json'
data_path = '../../data/persianmhqa/all_persian_mhqa.json'
with open(data_path, 'r', encoding="utf-8") as f:
    data = json.load(f)

triplets = {}
# llm = OpenAI(base_url="http://127.0.0.1:19002", api_key="mmm")
llm = OpenAI(base_url="https://api.avalai.ir/v1", api_key="")

# out_dir = '../../data/hotpotqa/kgs/extract_subkgs'
out_dir = '../../data/persianmhqa/kgs/extract_subkgs'
count = 0

for sample in tqdm(data):
    question = sample['question']
    answer = sample['answer']
    ctxs = sample['context']
    for ctx in ctxs:
        ent = ctx[0]
        if ent in triplets:
            continue
        ent = ent.strip()
        out_path = os.path.join(out_dir,f'{ent.replace("/","_")}.json')
        if os.path.exists(out_path):
            continue
        else:
            os.makedirs(out_dir, exist_ok=True)
        triplets[ent] = {}
        for i in range(len(ctx[1])):
            if not i==0:
                ctx_text = f'{ent}: {ctx[1][i]}'
            else:
                ctx_text = ctx[1][i]
            ext_triplets = extract_triplets(llm,ctx_text)
            if len(ext_triplets)==0:
                continue
            triplets[ent][i] = ext_triplets
        with open(out_path,'w', encoding="utf-8") as f:
            json.dump(triplets[ent],f, ensure_ascii=False, )
            count += 1

print(f'Newly extract entity KGs number: {count}')