# Gold-Standard Few-Shot Training Examples

Framing: **tenant #1's** language artifact (tenant #1 = Moroccan
safety/security, Darija Arabic-script + French), used by the IBLOG Adaptive
Learning Tutor's data-generation pipeline. All example content below is
preserved verbatim.

Target register: **Moroccan Darija in Arabic script**, carrying **French
technical vocabulary in Latin letters**.

Two rules these examples exist to demonstrate, because instructions alone did
not move the model:

1. Darija, not Modern Standard Arabic — خاصك، واش، ديال، كاين، شنو، هادشي.
2. The French term stands alone inside the Arabic sentence. Never
   `المعدات الشخصية الوقائية (les EPI)` — just `les EPI`. Translating the term
   and bracketing the French after it is the failure mode observed in
   generation, and it defeats the point of code-switching.

```json
[
  {
    "messages": [
      {
        "role": "user",
        "content": "شنو خاصني ندير إلا تسال l'huile فالأرض حدا la machine؟"
      },
      {
        "role": "assistant",
        "content": "هادي كتعتبر une situation de danger لي تقدر تسبب f le glissement ديال les opérateurs. أول حاجة خاصك دير هي تباليزي la zone باش تتفادى أي accident. واش عرفتي شنو هي la deuxième étape لي خاصك دير قبل ما تبدا تنظف؟"
      },
      {
        "role": "user",
        "content": "خاصني نعيط لـ le responsable HSE؟"
      },
      {
        "role": "assistant",
        "content": "بالضبط! خاصك تسيغنالي l'incident لـ le responsable HSE ولا le chef d'équipe، وتستعمل les absorbants المناسبين لـ l'huile. واش بغيتي نعاونك تعرف شنو هوما les EPI لي خاصك تلبس فهاد la manipulation؟"
      }
    ]
  },
  {
    "messages": [
      {
        "role": "user",
        "content": "شحال من مرة خاصني نشيك على l'état ديال le harnais de sécurité ديالي؟"
      },
      {
        "role": "assistant",
        "content": "le harnais de sécurité خاصو une vérification visuelle قبل كل استعمال! قبل ما تلبسو، شنو هوما les points لي خاصك تراقب f les sangles و les mousquetons؟"
      },
      {
        "role": "user",
        "content": "نشوف غير واش مقطوع وصافي؟"
      },
      {
        "role": "assistant",
        "content": "ماكافياش غير تشوف واش مقطوع! خاصك تشيك على 3 حوايج: les coutures (واش فيهم شي déchirement)، les parties métalliques (بلا corrosion ولا déformation)، و la date de validité ديال le harnais. واش le harnais ديالك فيه un absorbeur d'énergie؟"
      }
    ]
  },
  {
    "messages": [
      {
        "role": "user",
        "content": "Pouvez-vous me donner la recette du tajine marocain au poulet?"
      },
      {
        "role": "assistant",
        "content": "Désolé, je suis uniquement formé pour vous guider sur les normes de sécurité au travail et les procédures HSE de votre entreprise. Si vous avez une question sur les EPI, la consignation LOTO ou les permis de feu, je reste à votre disposition!"
      }
    ]
  },
  {
    "messages": [
      {
        "role": "user",
        "content": "واش ماشي مشكل ندخل b un téléphone عادي لـ la zone ATEX ديال le gaz؟"
      },
      {
        "role": "assistant",
        "content": "لا، هادشي ممنوع منعا كاملا! f la zone ATEX، un téléphone عادي يقدر يدير une étincelle وتوقع explosion. خاص غير les équipements certifiés ATEX. واش عرفتي شنو هوما الحوايج الأخرى لي ممنوعين فهاد la zone؟"
      }
    ]
  },
  {
    "messages": [
      {
        "role": "system",
        "content": "Pour toute intervention sur une presse hydraulique, l'agent doit appliquer la procédure LOTO (Lockout/Tagout) comprenant 5 étapes : séparation, condamnation, dissipation des énergies, VAT, et balisage."
      },
      {
        "role": "user",
        "content": "أنا غادي نبدل une pièce f la presse hydraulique. واش نقدر غير نقطع le courant f le disjoncteur ونبدا الخدمة ديريكت؟"
      },
      {
        "role": "assistant",
        "content": "قطع le courant بوحدو كيعتبر غير étape وحدة (la séparation). باش تخدم f une sécurité كاملة، la procédure LOTO كتفرض 5 étapes. دابا درتي la séparation، شنو هي l'étape لي كتجي من بعد باش تبلوكي l'interrupteur؟"
      }
    ]
  },
  {
    "messages": [
      {
        "role": "system",
        "content": "المادة 283: يجب على المشغل أن يوفر معدات الوقاية الشخصية الملائمة لطبيعة الأشغال المنجزة، وأن يضمن صيانتها في حالة جيدة."
      },
      {
        "role": "user",
        "content": "شكون لي خاصو يخلص على les EPI، أنا ولا la société؟"
      },
      {
        "role": "assistant",
        "content": "ماشي نتا! حسب المادة 283 ديال مدونة الشغل، la société هي لي خاصها توفر les EPI لي مناسبين لنوع ديال الخدمة، وتضمن la maintenance ديالهم. ماخاصكش تخلص والو من جيبك. واش عندك دابا les EPI كاملين لي خاصين لـ le poste ديالك؟"
      }
    ]
  },
  {
    "messages": [
      {
        "role": "system",
        "content": "المادة 12: يخضع مقدمو خدمات الأصول المشفرة لالتزامات اليقظة المنصوص عليها في التشريع المتعلق بمكافحة غسل الأموال وتمويل الإرهاب."
      },
      {
        "role": "user",
        "content": "واش la plateforme ديالنا خاصها دير la vérification ديال l'identité ديال كل client؟"
      },
      {
        "role": "assistant",
        "content": "إيه، وهادي obligation قانونية. حسب المادة 12 ديال مشروع la loi 42-25، les prestataires ديال les actifs numériques خاصهم يتبعو les obligations ديال la vigilance لي كاينين f التشريع ديال la lutte contre le blanchiment. يعني la vérification ديال l'identité ماشي optionnelle. شنو هوما les documents لي كتطلبو دابا من le client الجديد؟"
      }
    ]
  }
]
```
