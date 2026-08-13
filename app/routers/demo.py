from fastapi import APIRouter
from pydantic import BaseModel

from app.config import get_settings
from app.services.llm import _call_ollama_generate

router = APIRouter(prefix="/api/v1/demo", tags=["demo"])

class DemoRequest(BaseModel):
    language: str  # "fr" or "darija"
    turn: int  # 1-3 for multi-turn

DEMO_CONVERSATIONS = {
    "fr": [
        {
            "turn": 1,
            "context": "Contexte: Le port des équipements de protection individuelle (EPI) est obligatoire sur les postes à risque, conformément à l'article 281 du Code du travail marocain. Les EPI incluent les casques de sécurité, les gants, et les chaussures de sécurité.",
            "user_question": "Explique-moi pourquoi le port des EPI est obligatoire ?",
            "system": "Tu es un tuteur d'entreprise expert, spécialisé en sécurité industrielle. Réponds en français avec une méthode socratique. Le contexte ci-dessous peut être rédigé en arabe : traduis-le et explique en français. Réponds en français même si le contexte est en arabe. N'utilise pas l'écriture arabe, sauf pour citer une référence légale mot pour mot. Cite les références légales telles quelles, mot pour mot, exactement comme elles apparaissent dans le document source. Fonde toutes tes réponses strictement sur le contexte fourni."
        },
        {
            "turn": 2,
            "context": "Contexte: Le port des équipements de protection individuelle (EPI) est obligatoire sur les postes à risque, conformément à l'article 281 du Code du travail marocain. Les EPI incluent les casques de sécurité, les gants, et les chaussures de sécurité.",
            "user_question": "D'accord. Mais comment je peux adapter les EPI selon le type de poste ?",
            "system": "Tu es un tuteur d'entreprise expert, spécialisé en sécurité industrielle. Réponds en français avec une méthode socratique. Adapte ta réponse au niveau de compréhension de l'utilisateur. Pose des questions pour aider l'apprenant à découvrir la réponse par lui-même."
        },
        {
            "turn": 3,
            "context": "Contexte: Les postes à risque élevé nécessitent une protection complète (casque, gants renforcés, chaussures de sécurité). Les postes à risque modéré nécessitent une protection partielle.",
            "user_question": "Et si un employé refuse de porter les EPI ?",
            "system": "Tu es un tuteur d'entreprise expert, spécialisé en sécurité industrielle. Réponds de manière socratique et pédagogique."
        }
    ],
    "darija": [
        {
            "turn": 1,
            "context": "السياق: يجب على العمال ارتداء معدات الحماية الشخصية في الوظائف الخطرة، وفقاً للمادة 281 من قانون العمل المغربي. معدات الحماية تشمل خوذات الأمان والقفازات وأحذية السلامة.",
            "user_question": "شنو هي فايدة لبس المعدات الديال السلامة فالشغل ؟",
            "system": "أنت معلم متخصص في السلامة الصناعية. أجب بطريقة سقراطية. أسس جميع إجاباتك بشكل صارم على السياق المقدم."
        },
        {
            "turn": 2,
            "context": "السياق: معدات الحماية تحمي العمال من الإصابات والأمراض المهنية. الخوذة تحمي الرأس، والقفازات تحمي اليدين، والأحذية تحمي القدمين.",
            "user_question": "واخا فهمت. كيفاش نتأكد بلي كاع الموضفين ديالنا لابسين المعدات ؟",
            "system": "أنت معلم متخصص في السلامة الصناعية. أجب بطريقة سقراطية وتعليمية. اطرح أسئلة لمساعدة المتعلم على اكتشاف الإجابة بنفسه."
        },
        {
            "turn": 3,
            "context": "السياق: التفتيش المنتظم ضروري للتأكد من الامتثال. يجب توثيق أي حالات عدم امتثال.",
            "user_question": "واش كاين عقوبة إذا ما لبس الموضف المعدات ؟",
            "system": "أنت معلم متخصص في السلامة الصناعية. أجب بناءً على السياق المقدم فقط."
        }
    ]
}

@router.post("/conversation")
def get_demo_conversation(request: DemoRequest):
    """Generate a demo multi-turn conversation."""
    language = request.language  # "fr" or "darija"
    turn = request.turn  # 1, 2, or 3

    if language not in DEMO_CONVERSATIONS:
        return {"error": f"Language {language} not supported. Use 'fr' or 'darija'"}

    if turn < 1 or turn > 3:
        return {"error": "Turn must be 1, 2, or 3"}

    demo_turn = DEMO_CONVERSATIONS[language][turn - 1]
    settings = get_settings()
    model = settings.ollama_model_fr if language == "fr" else settings.ollama_model

    # Generate response from Ollama
    try:
        response = _call_ollama_generate(
            model, demo_turn['user_question'], demo_turn['system'] + "\n\n" + demo_turn['context'], timeout=60
        )
    except Exception as e:
        response = f"Error: {str(e)}"

    return {
        "turn": turn,
        "language": language,
        "model": model,
        "user_question": demo_turn['user_question'],
        "assistant_response": response,
        "context": demo_turn['context']
    }
