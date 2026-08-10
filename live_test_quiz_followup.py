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

**المادة 16**
يمنع على أعوان الحراسة حمل السلاح الناري إلا في حالة نقل الأموال وبترخيص خاص.

**المادة 17**
تسجل جميع الحوادث الأمنية في سجل خاص يوضع رهن إشارة السلطات المختصة."""

LOTO_AR = """# المسطرة الداخلية لفصل مصادر الطاقة (الإغلاق والوسم)

**المرجع الداخلي:** المسطرة رقم SST-04

## المادة 4: المسؤولية
يمنع منعا باتا رفع القفل من طرف شخص آخر غير العون الذي وضعه.

## المادة 3: المراحل الخمس الإلزامية
**المرحلة الثانية — الإغلاق:** وضع قفل شخصي على جهاز القطع، ولا يملك مفتاحه إلا العون المتدخل."""

CASES = [
    ("quiz on article 16 (firearm restriction) - same law, different fact",
     "عطيني كويز على المادة 16 ديال القانون 27.06.", LOI_27_06, "securite"),
    ("quiz on article 8 (age/capacity requirement) - same law, different fact",
     "عطيني كويز على شروط أعوان الحراسة.", LOI_27_06, "securite"),
    ("quiz on unrelated industrial LOTO doc - different domain entirely",
     "عطيني كويز على مسؤولية رفع القفل فالconsignation.", LOTO_AR, "industrial"),
]

out = io.open("live_test_quiz_followup_results.txt", "w", encoding="utf-8")
for label, query, context, domain in CASES:
    t0 = time.time()
    resp = generate_llm_response(query=query, context=context, domain=domain)
    elapsed = time.time() - t0
    out.write(f"===== {label} | domain={domain} | {elapsed:.1f}s =====\n")
    out.write(f"QUERY: {query}\n\nRESPONSE:\n{resp}\n\n\n")
    out.flush()
    print(f"[OK] {label} ({elapsed:.1f}s)", flush=True)
out.close()
print("done")
