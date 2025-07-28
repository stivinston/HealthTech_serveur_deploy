from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
from typing import Dict, Optional
import pygame
import time
from langchain.schema import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from huggingface_hub import InferenceClient
from langchain_core.language_models import LLM
from typing import List
import pyttsx3
import os
from gtts import gTTS

app = FastAPI(title="Health Tech Assistant API")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for favicon
app.mount("/static", StaticFiles(directory="static"), name="static")

# --- Root Endpoint ---
@app.get("/")
async def root():
    return {"message": "Welcome to the Health Tech Assistant API. Visit /docs for API documentation."}

# --- Data Loading ---
def load_data():
    """Load patient data from CSV file"""
    try:
        return pd.read_csv("clinical_summaries.csv")
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Fichier clinical_summaries.csv introuvable")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du chargement des données: {str(e)}")

df = load_data()

# --- Custom LLM for HuggingFace ---
class LlamaLLM(LLM):
    def _call(self, prompt: str, stop: List[str] = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        response = InferenceClient(
            model="m42-health/Llama3-Med42-8B",
            api_key="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"
        ).chat.completions.create(messages=messages)
        return response.choices[0].message.content.strip()

    @property
    def _llm_type(self) -> str:
        return "llama_custom"

llm = LlamaLLM()

# --- Prompt Templates ---
desc_prompt = PromptTemplate.from_template("""
Tu es un médecin expérimenté et empathique. Tu vas expliquer les informations médicales suivantes à un patient de façon claire, humaine et rassurante, comme si vous étiez en consultation réelle.

Voici les données du patient :
{contexte}

Explique-lui brièvement son état de santé, avec des mots simples mais précis. Rassure-le si nécessaire, et adapte ton ton en fonction de ce que les données révèlent (ex : fièvre, tension normale ou élevée, diagnostic, etc.).
Termine toujours ta réponse en l'invitant naturellement à poser toutes ses questions via le chat.

**Répond en la langue fourni par l'utilisateur**

Langue : {langue}
""")

llm_prompt = PromptTemplate.from_template("""
Tu es un médecin expérimenté, bienveillant et compétent.  
Tu as déjà consulté le patient et tu disposes de ses données médicales.  
Ta mission est de répondre à toutes ses préoccupations avec clarté, empathie et autorité médicale.  
Utilise un ton rassurant et humain, mais reste concis et médicalement rigoureux.  
Ne propose pas d'aller consulter un autre médecin : tu es celui qui l’a vu.  
Exprime-toi dans la même langue que le patient.  

**Limite ta réponse à 3 à 5 phrases maximum.**  
**N’inclus pas d’excuses, ni de phrases vagues. Sois précis, respectueux, et professionnel.**  

{history}  
Patient: {question}  
Contexte médical : {context}  
Réponse :

""")

# --- Memory Setup ---
def create_persistent_memory(session_id: str, db_path: str = "chat_memory.sqlite"):
    """Create persistent memory for chat sessions using SQLite"""
    message_history = SQLChatMessageHistory(session_id=session_id, connection_string=f"sqlite:///{db_path}")
    memory = ConversationBufferMemory(
        chat_memory=message_history,
        memory_key="history",
        input_key="question",
        return_messages=True
    )
    return memory

# --- Cache pour les descriptions des patients ---
patient_descriptions_cache: Dict[str, str] = {}

# --- Helper Functions ---
def get_patient_data(summary_id: str):
    """Get patient data by summary_id from CSV"""
    if summary_id not in df["summary_id"].values:
        raise HTTPException(status_code=404, detail="ID introuvable. Veuillez vérifier.")
    
    # Vérifier d'abord dans le cache
    if summary_id in patient_descriptions_cache:
        return patient_descriptions_cache[summary_id]
    
    patient_info = df[df["summary_id"] == summary_id].fillna("Inconnu")
    patient_data = patient_info.to_dict(orient="records")[0]
    
    # Mettre en cache la description
    patient_descriptions_cache[summary_id] = patient_data
    
    return patient_data

# --- Pydantic Models for Request/Response ---
class PatientRequest(BaseModel):
    summary_id: str
    language: str

class ChatRequest(BaseModel):
    summary_id: str
    question: str

class AudioRequest(BaseModel):
    text: str
    language: str

# --- API Endpoints ---
@app.post("/patient-description", response_model=dict)
async def get_patient_description(request: PatientRequest):
    """Get patient description based on summary_id"""
    try:
        context_dict = get_patient_data(request.summary_id)
        
        desc_chain = desc_prompt | llm | StrOutputParser()
        description = desc_chain.invoke({"contexte": context_dict, "langue": request.language})
        
        return {"description": description}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de la description: {str(e)}")

@app.post("/chat-response", response_model=dict)
async def get_chat_response(request: ChatRequest):
    """Get chat response for patient questions"""
    try:
        context_dict = get_patient_data(request.summary_id)
        
        memory = create_persistent_memory(session_id=request.summary_id)
        
        rag_chain = (
            {
                "question": lambda x: x["question"],
                "context": lambda x: context_dict,
                "history": lambda x: memory.load_memory_variables({"question": x["question"]})["history"]
            }
            | llm_prompt
            | llm
            | StrOutputParser()
        )
        
        response = rag_chain.invoke({"question": request.question})
        memory.save_context({"question": request.question}, {"answer": response})
        
        return {"response": response}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la génération de la réponse: {str(e)}")

@app.post("/text-to-speech", response_model=dict)
async def text_to_speech(request: AudioRequest):
    """Convert text to speech"""
    try:
        # Determine language code for gTTS
        lang_code = "fr" if request.language.lower() in ["french", "français", "fr"] else "en"
        
        # Generate TTS
        tts = gTTS(text=request.text, lang=lang_code)
        output_file = f"output_{hash(request.text)}.mp3"
        
        # Create static directory if it doesn't exist
        os.makedirs("static", exist_ok=True)
        output_path = os.path.join("static", output_file)
        
        tts.save(output_path)
        
        # Lire le fichier audio généré avec pygame
        try:
            pygame.mixer.init()
            pygame.mixer.music.load(output_path)
            pygame.mixer.music.play()
            # Attendre que la lecture soit terminée
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"Erreur lors de la lecture audio: {str(e)}")
        
        return {"response": f"/static/{output_file}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur audio : {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "data_loaded": len(df) > 0 if 'df' in globals() else False}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=4000)
