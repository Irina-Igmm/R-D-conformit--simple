import json
from typing import Optional, Dict, Any
from google.cloud import aiplatform


class ClassificationDocument:
    def __init__(
        self,
        json_file_path: str,
        service_account_path: str,
        location: str,
    ):
        self.documents_info = self.load_json(json_file_path)
        with open(service_account_path, "r") as f:
            service_account = json.load(f)
            project_id = service_account["project_id"]

        # Initialisation de Vertex AI avec le compte de service
        self.initialize_vertex_ai(service_account_path, project_id, location)

    def initialize_vertex_ai(
        self, service_account_path: str, project_id: str, location: str
    ):
        """Initialise Vertex AI avec un compte de service."""
        aiplatform.init(
            credentials=service_account_path,
            project=project_id,
            location=location,
        )

    def load_json(self, json_file_path: str) -> list:
        """Charge les informations du fichier JSON."""
        try:
            with open(json_file_path, "r", encoding="utf-8") as file:
                data = json.load(file)
            print("JSON file loaded successfully.")
            return data["documents"]
        except FileNotFoundError:
            raise Exception(f"File '{json_file_path}' not found.")
        except json.JSONDecodeError:
            raise Exception("JSON decoding error: Check the file format.")

    def create_prompt(self, document_content: str) -> str:
        """Crée un prompt basé sur le contenu du document."""
        prompt = (
            "Vous êtes un assistant d'IA expert en classification de documents. "
            "Votre tâche est de classer le document donné dans l'une des catégories suivantes :\n\n"
        )

        for doc in self.documents_info:
            prompt += f"## {doc['type_document']}\n"
            prompt += f"{doc['description']}\n"
            prompt += "**Champs clés:**\n"
            for champ in doc["champs"]:
                prompt += f"- {champ['nom']}\n"
            prompt += "\n"

        prompt += "## Contenu du document:\n"
        prompt += f"{document_content}\n\n"

        prompt += f"""Analysez le document fourni et classez-le dans l'une des catégories disponibles. 
        - Fournissez une justification détaillée de votre classification, en expliquant le raisonnement derrière votre choix. 
        - Incluez toute information supplémentaire pertinente dérivée du document. 
        - Utilisez *uniquement* le contenu et les catégories du document fournis pour votre analyse.

        Contenu du document:
        {document_content}

        Catégories disponibles:
        { [d['type_document'] for d in json.loads(self.documents_info)]}

        Répondez au format suivant:

        TYPE: [Catégorie sélectionnée]
        JUSTIFICATION: [Explication détaillée des raisons pour lesquelles cette catégorie a été choisie, en faisant référence à des éléments spécifiques du contenu du document.]
        INFORMATIONS ADDITIONNELLES: [Toute autre information pertinente trouvée dans le document.]
            """
        return prompt

    def classify(self, document_content: str) -> Dict[str, Any]:
        """Utilise Google Gemini Flash pour classer le document."""
        prompt = self.create_prompt(document_content)

        try:
            # Crée une requête de classification via Vertex AI
            response = aiplatform.TextGenerationModel.from_pretrained(
                "text-bison@001"
            ).predict(
                prompt,
                temperature=0.3,  # Moins de variabilité dans les réponses
                max_output_tokens=1000,
            )

            result = response.text.strip()

            # Analyse la réponse structurée
            classification_result = self._parse_response(result)

            # Validation du type de document
            if classification_result["type"].lower() not in [
                doc["type_document"].lower() for doc in self.documents_info
            ]:
                classification_result["type"] = "Non classifiable"

            return classification_result

        except Exception as e:
            print(f"Error during classification: {str(e)}")
            return {
                "type": "Erreur",
                "justification": f"Une erreur est survenue: {str(e)}",
                "informations_additionnelles": None,
            }

    def _parse_response(self, response: str) -> Dict[str, str]:
        """Analyse la réponse retournée par le modèle."""
        lines = response.split("\n")
        result = {
            "type": "Non classifiable",
            "justification": "",
            "informations_additionnelles": "",
        }

        current_section = None
        for line in lines:
            line = line.strip()
            if "TYPE:" in line:
                current_section = "type"
                result["type"] = line.split("TYPE:")[1].strip()
            elif "JUSTIFICATION:" in line:
                current_section = "justification"
                result["justification"] = line.split("JUSTIFICATION:")[1].strip()
            elif "INFORMATIONS ADDITIONNELLES:" in line:
                current_section = "informations_additionnelles"
                result["informations_additionnelles"] = line.split(
                    "INFORMATIONS ADDITIONNELLES:"
                )[1].strip()
            elif current_section and line:
                result[current_section] += " " + line

        return result
