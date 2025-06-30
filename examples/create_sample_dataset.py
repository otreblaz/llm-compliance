import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    LLMComplianceToolkit, ModelMetadata, QAPair, 
    ModelType, AccessType, ComplianceLevel
)

def create_sample_dataset():
    """Crea un dataset de ejemplo con dos modelos LLM"""

    toolkit = LLMComplianceToolkit()

    gpt4_metadata = ModelMetadata(
        model_id="gpt4",
        name="GPT-4",
        version="v1.0",
        provider="OpenAI",
        release_date=datetime(2023, 3, 14),
        model_type=ModelType.GENERATIVE,
        access_type=AccessType.API_ONLY,
        description="Modelo de lenguaje grande generativo con capacidades avanzadas de razonamiento y comprensión",
        training_data_description="Entrenado con datos públicos de internet hasta septiembre de 2021",
        parameters_count=1500000000000,  # 1.5 trillion (estimado)
        context_window=8192,
        environmental_impact={
            "carbon_emissions_training": "Información no publicada",
            "energy_consumption": "Información no publicada"
        },
        documentation_url="https://openai.com/gpt-4",
        papers_url=[
            "https://arxiv.org/abs/2303.08774"
        ],
        additional_metadata={
            "fine_tuning_available": False,
            "moderation_system": True,
            "content_filtering": True,
            "known_limitations": "Puede generar información incorrecta, alucinaciones, o respuestas sesgadas"
        }
    )
    
    toolkit.save_model_metadata(gpt4_metadata)
    
    llama2_metadata = ModelMetadata(
        model_id="llama2",
        name="Llama 2",
        version="70B",
        provider="Meta",
        release_date=datetime(2023, 7, 18),
        model_type=ModelType.GENERATIVE,
        access_type=AccessType.OPEN_SOURCE,
        description="Modelo de lenguaje grande optimizado para diálogo y con licencia abierta para investigación y uso comercial",
        training_data_description="Entrenado con datos disponibles públicamente y filtrados para mayor seguridad",
        parameters_count=70000000000,  # 70 billion
        context_window=4096,
        environmental_impact={
            "carbon_emissions_training": "Información proporcionada en la documentación",
            "energy_consumption": "1,015 MWh"
        },
        documentation_url="https://ai.meta.com/llama/",
        papers_url=[
            "https://arxiv.org/abs/2307.09288"
        ],
        additional_metadata={
            "fine_tuning_available": True,
            "moderation_system": False,
            "content_filtering": True,
            "license": "Llama 2 Community License",
            "known_limitations": "Puede mostrar sesgos o generar contenido inexacto"
        }
    )
    
    toolkit.save_model_metadata(llama2_metadata)
    
    gpt4_qa_pairs = [
        # Legitimidad y competencia
        QAPair(
            question="¿Quién autorizó el desarrollo de este modelo?",
            response="El desarrollo de GPT-4 fue autorizado por la junta directiva de OpenAI y supervisado por un equipo de investigadores líderes en IA con amplia experiencia en modelos de lenguaje grandes.",
            category="Legitimidad y competencia"
        ),
        QAPair(
            question="¿Se realizaron evaluaciones éticas y legales antes de su despliegue?",
            response="Sí, OpenAI realizó evaluaciones éticas y legales extensas antes del despliegue, incluyendo evaluaciones de riesgo, análisis de sesgos, y pruebas de seguridad con usuarios alfa y beta.",
            category="Legitimidad y competencia"
        ),
        
        # Minimización de daños
        QAPair(
            question="¿Cómo se identifican y mitigan los sesgos en el modelo?",
            response="Los sesgos se identifican mediante evaluaciones sistemáticas en múltiples dimensiones como género, raza, religión y política. Se mitigan a través de técnicas RLHF (aprendizaje por refuerzo con retroalimentación humana) y sistemas de filtrado de contenido inapropiado.",
            category="Minimización de daños"
        ),
        QAPair(
            question="¿Se realizan evaluaciones de impacto regularmente?",
            response="Sí, OpenAI realiza evaluaciones de impacto regulares y tiene un equipo dedicado a la seguridad y alineación que monitorea constantemente el comportamiento del modelo en producción.",
            category="Minimización de daños"
        ),
        
        # Transparencia
        QAPair(
            question="¿Existe documentación clara sobre los conjuntos de datos y métodos usados?",
            response="OpenAI proporciona información general sobre los métodos de entrenamiento, pero no ha publicado detalles específicos sobre los conjuntos de datos utilizados, citando consideraciones de seguridad y competitividad.",
            category="Transparencia"
        ),
        QAPair(
            question="¿Se permite la verificación independiente del funcionamiento del modelo?",
            response="La verificación independiente es limitada ya que el modelo solo está disponible a través de API y no hay acceso abierto a los pesos. Algunos investigadores externos pueden tener acceso bajo acuerdos específicos.",
            category="Transparencia"
        ),
        
        # Limitación de impactos ambientales
        QAPair(
            question="¿El sistema informa sobre sus impactos ambientales y emisiones de carbono?",
            response="No, OpenAI no ha publicado información detallada sobre el impacto ambiental o las emisiones de carbono asociadas con el entrenamiento o la operación de GPT-4.",
            category="Limitación de impactos ambientales"
        )
    ]
    
    toolkit.save_qa_pairs("gpt4", gpt4_qa_pairs)
    
    llama2_qa_pairs = [
        # Legitimidad y competencia
        QAPair(
            question="¿Quién autorizó el desarrollo de este modelo?",
            response="El desarrollo de Llama 2 fue autorizado por Meta AI y su equipo directivo, con supervisión de investigadores con experiencia en modelos de lenguaje de gran escala y aprendizaje profundo.",
            category="Legitimidad y competencia"
        ),
        QAPair(
            question="¿Se realizaron evaluaciones éticas y legales antes de su despliegue?",
            response="Sí, Meta realizó evaluaciones éticas y legales completas, documentadas en el paper 'Llama 2: Open Foundation and Fine-Tuned Chat Models'. El modelo se sometió a evaluaciones de seguridad, pruebas adversariales y análisis de riesgos antes de su liberación.",
            category="Legitimidad y competencia"
        ),
        
        # Minimización de daños
        QAPair(
            question="¿Cómo se identifican y mitigan los sesgos en el modelo?",
            response="Meta implementó un proceso de refinamiento iterativo que incluye anotación humana, filtrado de datos tóxicos, y optimización supervisada para identificar y reducir sesgos. Mantienen equipos de red team para evaluar constantemente el modelo.",
            category="Minimización de daños"
        ),
        QAPair(
            question="¿Se realizan evaluaciones de impacto regularmente?",
            response="Meta indica que realizó evaluaciones de impacto antes del lanzamiento, pero la información sobre evaluaciones continuas posteriores al despliegue es limitada debido a su naturaleza de código abierto y a que el mantenimiento depende tanto de Meta como de la comunidad.",
            category="Minimización de daños"
        ),
        
        # Transparencia
        QAPair(
            question="¿Existe documentación clara sobre los conjuntos de datos y métodos usados?",
            response="Sí, Meta ha publicado documentación detallada y un paper técnico que describe los métodos de entrenamiento y tipos de datos utilizados. Sin embargo, no se proporcionan listas exhaustivas de todas las fuentes de datos específicas.",
            category="Transparencia"
        ),
        QAPair(
            question="¿Se permite la verificación independiente del funcionamiento del modelo?",
            response="Sí, al ser un modelo de código abierto, Llama 2 permite verificación independiente completa. Los pesos del modelo están disponibles para descarga y análisis, permitiendo a terceros auditarlo y validar su funcionamiento.",
            category="Transparencia"
        ),
        
        # Limitación de impactos ambientales
        QAPair(
            question="¿El sistema informa sobre sus impactos ambientales y emisiones de carbono?",
            response="Meta ha proporcionado información limitada sobre el consumo energético del entrenamiento de Llama 2, mencionando en su documentación que utilizaron aproximadamente 1,015 MWh de electricidad. Sin embargo, no hay un desglose detallado o estimaciones de emisiones de carbono.",
            category="Limitación de impactos ambientales"
        )
    ]

    toolkit.save_qa_pairs("llama2", llama2_qa_pairs)

    for model_id in ["gpt4", "llama2"]:
        compliance = toolkit.evaluate_model_compliance(model_id)
        toolkit.save_compliance_report(compliance)

        print(f"\nInforme de conformidad para {model_id}:")
        print(compliance.summary)
        print("\nEvaluaciones por principio:")
        for eval in compliance.evaluations:
            print(f"- {eval.principle_name}: {eval.compliance_level}")

    questions_by_principle = toolkit.generate_all_evaluation_questions()

    questions_file = os.path.join(toolkit.data_dir, "evaluation_questions.json")
    with open(questions_file, 'w', encoding='utf-8') as f:
        json.dump(questions_by_principle, f, ensure_ascii=False, indent=2)
    
    print(f"\nSe ha creado un dataset de ejemplo con dos modelos: GPT-4 y Llama 2")
    print(f"Los datos se han guardado en: {toolkit.data_dir}")
    print(f"Preguntas de evaluación guardadas en: {questions_file}")

if __name__ == "__main__":
    create_sample_dataset() 