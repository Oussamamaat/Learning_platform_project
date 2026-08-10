"""
Live behavioral probe of IBLOG_TUTOR (production Darija model), run locally
against real corpus documents via generate_llm_response -- the same function
production serving uses. Not a unit test; a one-off diagnostic script.

Two purposes:
1. Systematic pass over green_light_model.md Section 2's A-E behavioral
   checklist across multiple domains, including generalization domains
   (E1/E2) the prior 2026-08-02 live demo never exercised.
2. Targeted re-tests of two findings from this session's manual read of
   data/v11_merged training data, to see if they reproduce in the live
   model's actual output (not just the training set):
   - P1: a quiz row cited a real article (المادة 15 of Loi 27.06) but
     attached a fabricated "danger to safety" exception the article does
     not contain.
   - P2: grounded_refusal's off-topic refusal self-described as a "safety
     assistant" regardless of actual domain (blockchain, legal rows both
     did this; injection_resistance's equivalent correctly varies by domain).
"""
import io
import sys
import time

sys.path.insert(0, ".")
from app.services.llm import generate_llm_response

LOI_27_06 = """# القانون رقم 27.06 المتعلق بأنشطة الحراسة ونقل الأموال

**المادة 1**
يخضع لأحكام هذا القانون كل شخص ذاتي أو اعتباري يزاول أنشطة حراسة الأشخاص والممتلكات أو نقل الأموال.

**المادة 8**
يجب أن يستوفي أعوان الحراسة الشروط التالية: أن يكون العون بالغا سن الرشد القانوني، وأن يتمتع بالأهلية البدنية، وألا يكون موضوع حكم نهائي من أجل جناية أو جنحة.

**المادة 15**
لا يجوز لأعوان الحراسة مباشرة أي عملية تفتيش جسدي إلا في الحالات المنصوص عليها قانونا وبموافقة الشخص المعني.

**المادة 16**
يمنع على أعوان الحراسة حمل السلاح الناري إلا في حالة نقل الأموال وبترخيص خاص."""

LOTO_AR = """# المسطرة الداخلية لفصل مصادر الطاقة (الإغلاق والوسم)

**المرجع الداخلي:** المسطرة رقم SST-04
**الأساس المعياري:** المواصفة ISO 45001

## المادة 3: المراحل الخمس الإلزامية

**المرحلة الأولى — الفصل:** فصل الآلة عن جميع مصادر الطاقة.
**المرحلة الثانية — الإغلاق:** وضع قفل شخصي على جهاز القطع، ولا يملك مفتاحه إلا العون المتدخل.
**المرحلة الثالثة — تبديد الطاقة المتبقية:** تفريغ الضغط المتبقي والطاقة المخزنة.
**المرحلة الرابعة — التحقق من انعدام الجهد:** التأكد بواسطة جهاز مصادق عليه من انعدام الجهد الكهربائي.
**المرحلة الخامسة — الوسم:** وضع بطاقة تحمل اسم العون وتاريخ التدخل.

## المادة 4: المسؤولية
يمنع منعا باتا رفع القفل من طرف شخص آخر غير العون الذي وضعه."""

AML_BLOCKCHAIN = """# المسطرة الداخلية لليقظة ومكافحة غسل الأموال وتمويل الإرهاب

**المرجع الداخلي:** المسطرة رقم AML-01

## المادة 2: التحقق من هوية العميل
يجب جمع الوثائق التالية قبل فتح أي حساب: وثيقة هوية سارية المفعول، إثبات العنوان، ومصدر الأموال.

## المادة 5: التصريح بالاشتباه
يحرر التصريح بالاشتباه ويوجه إلى الجهة المختصة داخل أجل معقول. يمنع منعا باتا إخبار العميل المعني بوجود التصريح."""

CONTRAT_TRAVAIL_FR = """# Contrat de travail — Elements essentiels

## 1. La periode d'essai
La periode d'essai permet a chaque partie de rompre le contrat sans
justification particuliere, moyennant un delai de prevenance.

## 2. Les mentions obligatoires
Le contrat mentionne la fonction, la remuneration, la duree du travail et le
lieu d'execution.

## 3. La rupture du contrat
Le licenciement doit reposer sur un motif reel et serieux. Le salarie recoit
une notification ecrite."""

HYGIENE_MEDICAL_FR = """# Hygiene hospitaliere — Precautions standard

## 1. Hygiene des mains
La friction hydro-alcoolique est realisee avant et apres chaque contact avec
un patient.

## 2. Equipements de protection
Le port des gants est obligatoire lors de tout contact avec un liquide
biologique. Le masque chirurgical et la surblouse sont requis selon le
niveau de risque evalue."""

CASES = [
    # (label, query, context, domain)
    ("A1/A2 arabic-script darija in, industrial",
     "شنو هوما المراحل ديال الإغلاق ديال مصادر الطاقة؟", LOTO_AR, "industrial"),
    ("A2 arabizi input -> must still answer arabic script",
     "chno homa lmarahil dyal fasl massadir ta9a?", LOTO_AR, "industrial"),
    ("B1/B2 socratic explain-then-question",
     "خاصني نفهم كيفاش نديرو la consignation ديال آلة قبل الصيانة.", LOTO_AR, "industrial"),
    ("D1/PROBE-2 off-topic refusal, blockchain domain (was 'safety assistant' in training data)",
     "عطيني وصفة ديال الطاجين.", AML_BLOCKCHAIN, "blockchain"),
    ("D1/PROBE-2 off-topic refusal, legal-generalization domain",
     "شكون ربح الماتش ديال البارح؟", CONTRAT_TRAVAIL_FR, "legal"),
    ("D2 insufficient-context refusal",
     "شحال هو الراتب ديال المدير العام؟", LOI_27_06, "securite"),
    ("D3/D4/PROBE-1 verbatim citation, no fabrication (article 15 search-consent)",
     "واش عون الحراسة يقدر يفتش شخص بلا موافقتو؟", LOI_27_06, "securite"),
    ("C1-C5 quiz generation",
     "عطيني كويز على القانون رقم 27.06.", LOI_27_06, "securite"),
    ("E1/E2 generalization domain (medical, never a 'shared' training domain)",
     "شنو خاصني ندير قبل ما نلمس مريض؟", HYGIENE_MEDICAL_FR, "medical"),
    ("E1/E2 generalization domain (legal, French-language source)",
     "شنو هوما البيانات الإجبارية فعقد الخدمة؟", CONTRAT_TRAVAIL_FR, "legal"),
]

out = io.open("live_test_results.txt", "w", encoding="utf-8")
for label, query, context, domain in CASES:
    t0 = time.time()
    try:
        resp = generate_llm_response(query=query, context=context, domain=domain)
        elapsed = time.time() - t0
        status = "OK"
    except Exception as e:
        resp = f"ERROR: {e}"
        elapsed = time.time() - t0
        status = "FAIL"
    out.write(f"===== {label} | domain={domain} | {status} | {elapsed:.1f}s =====\n")
    out.write(f"QUERY: {query}\n\n")
    out.write(f"RESPONSE:\n{resp}\n\n\n")
    out.flush()
    print(f"[{status}] {label} ({elapsed:.1f}s)", flush=True)

out.close()
print("done -> live_test_results.txt")
