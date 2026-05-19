#api/main.py
#API FastAPI pour Sensante - Assistant pre-diagnostic medical 

from fastapi import FastAPI

#application Fast
app = FastAPI(
    title = "SenSante API",
    description = "Assistant pre-diagnostic medical pour le Senegal",
    version = "0.2.0"
)

#Route de base : verifier que l'API fonctionne 
@app.get("/health")
def health_check():
    return{
        "status": "ok",
        "message": "SenSante API is healthy and running"
    }


from pydantic import BaseModel , Field

#Schemas Pydantic
class PatientsInput(BaseModel):
    age: int = Field(..., ge=0 , le=120, description="Age en annees")
    sexe: str = Field(..., description="Sexe : M ou F")
    temperature: float = Field(..., ge=35.0 , le=42.0,  description="Temperature corporelle en degres Celsius")
    tension_sys: int = Field(...,ge=60 , le=250 , description="Tension systolique ")
    toux:bool = Field(...,description="Presence de toux")
    fatigue:bool = Field(...,description="Presence de fatigue")
    maux_tete:bool = Field(...,description="Presence de maux de tete")
    region: str = Field(..., description="Region du Senegal")

class DiagnosticOutput(BaseModel):
    diagnostic:str = Field(...,description="Diagnostic predit")
    probabilite: float = Field(...,description="Probabilite du diagnostic")
    confiance: str = Field(...,description="Niveau de confiance : faible, moyen, eleve")
    message: str=Field(...,description="Recommandation")

import joblib
import numpy as np

#Charger le modele et les encodeurs au demarrage 
print("Chargement du modele...")
model = joblib.load("notebooks/models/model.pkl")
le_sexe = joblib.load("notebooks/models/encoder_sexe.pkl")
la_region = joblib.load("notebooks/models/encoder_region.pkl")
feature_cols = joblib.load("notebooks/models/feature_cols.pkl")
print(f"Modele charge: {type(model).__name__}")
print(f"Classes : {list(model.classes_)}")



@app.post("/predict" , response_model=DiagnosticOutput)
def predict ( patients : PatientsInput ) :

    try:
        sexe_enc = le_sexe . transform ([ patients . sexe ]) [0]
    except ValueError :
        return DiagnosticOutput (
            diagnostic ="erreur",
            probabilite =0.0 ,
            confiance =" aucune ",
            message = f" Sexe invalide : { patients . sexe }. Utiliser M ou F."
            )
    try:
        region_enc = la_region . transform ([ patients . region ]) [0]
    except ValueError :
        return DiagnosticOutput (
            diagnostic =" erreur ",
            probabilite =0.0 ,
            confiance =" aucune ",
            message = f" Region inconnue : { patients . region }"
            )
    
    # 2. Construire le vecteur de features
    features = np . array ([[
        patients . age ,
        sexe_enc ,
        patients . temperature ,
        patients . tension_sys ,
        int( patients . toux ) ,
        int( patients . fatigue ) ,
        int( patients . maux_tete ) ,
        region_enc
    ]])
    # 3. Predire
    diagnostic = model . predict ( features ) [0]
    probas = model . predict_proba ( features ) [0]
    proba_max = float ( probas . max () )
    # 4. Determiner le niveau de confiance
    if proba_max >= 0.7:
            confiance = "haute"
    elif proba_max >= 0.4:
            confiance = "moyenne"
    else :
            confiance = "faible"
    # 5. Generer la recommandation
    messages = {
            "palu": " Suspicion de paludisme . Consultez un medecin rapidement .",
            "grippe": " Suspicion de grippe . Repos et hydratation recommandes .",
            "typh": " Suspicion de typhoide . Consultation medicale necessaire .",
            "sain": "Pas de pathologie detectee . Continuez a surveiller ."
    }
    # 6. Renvoyer le resultat
    return DiagnosticOutput (
        diagnostic = diagnostic ,
        probabilite = round ( proba_max , 2) ,
        confiance = confiance ,
        message = messages . get ( diagnostic , "Consultez un medecin.")
    )



@app.get("/model-info" )
def model_info () :
     return {
          "type": type(model).__name__,
          "nombre_arbres":model.n_estimators,
          "classes_possibles": model.classes_.tolist(),
          "nombre_features": model.n_features_in_
     }


from fastapi.middleware.cors import CORSMiddleware


#Autoriser les requetes depuis le frontend
app.add_middleware (
     CORSMiddleware,
     allow_origins=["*"],
     allow_credentials=True,
     allow_methods=["*"],
     allow_headers=["*"],
)