import sys
import os
import json
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    LLMComplianceToolkit, ModelMetadata, QAPair, 
    ModelType, AccessType, ComplianceLevel
)

def evaluate_model():
    """Guía interactiva para evaluar un nuevo modelo LLM"""
    
    print("\n=== EVALUACIÓN DE CONFORMIDAD DE MODELOS LLM ===\n")
    
    toolkit = LLMComplianceToolkit()
    
    print("Por favor, proporciona información sobre el modelo LLM:")
    model_id = input("ID único del modelo (ej. gpt3, claude2): ").strip()
    name = input("Nombre completo del modelo: ").strip()
    version = input("Versión (opcional): ").strip() or None
    provider = input("Proveedor/Organización: ").strip()

    print("\nTipo de modelo:")
    print("1. Generativo")
    print("2. Embedding")
    print("3. Multimodal")
    print("4. Otro")
    model_type_choice = input("Selecciona (1-4): ").strip()
    model_type_map = {
        "1": ModelType.GENERATIVE,
        "2": ModelType.EMBEDDING,
        "3": ModelType.MULTIMODAL,
        "4": ModelType.OTHER
    }
    model_type = model_type_map.get(model_type_choice, ModelType.GENERATIVE)

    print("\nTipo de acceso:")
    print("1. Código abierto")
    print("2. Solo API")
    print("3. Cerrado")
    print("4. Otro")
    access_type_choice = input("Selecciona (1-4): ").strip()
    access_type_map = {
        "1": AccessType.OPEN_SOURCE,
        "2": AccessType.API_ONLY,
        "3": AccessType.CLOSED,
        "4": AccessType.OTHER
    }
    access_type = access_type_map.get(access_type_choice, AccessType.API_ONLY)

    description = input("\nDescripción breve del modelo: ").strip()
    training_data = input("Descripción de los datos de entrenamiento: ").strip() or None
    
    try:
        parameters = int(input("Número de parámetros (opcional): ").strip())
    except:
        parameters = None
        
    try:
        context_window = int(input("Tamaño de ventana de contexto (opcional): ").strip())
    except:
        context_window = None

    metadata = ModelMetadata(
        model_id=model_id,
        name=name,
        version=version,
        provider=provider,
        release_date=datetime.now(),
        model_type=model_type,
        access_type=access_type,
        description=description,
        training_data_description=training_data,
        parameters_count=parameters,
        context_window=context_window
    )

    toolkit.save_model_metadata(metadata)
    print(f"\nMetadatos del modelo guardados correctamente.")

    print("\n=== PROCESO DE RECOPILACIÓN DE PREGUNTAS Y RESPUESTAS ===")
    print("Se realizarán preguntas sobre el modelo para evaluar su conformidad con los principios ACM.")
    print("Para cada principio, se mostrarán preguntas sugeridas para hacerle al modelo.")
    print("Deberás copiar estas preguntas al modelo LLM y registrar sus respuestas aquí.")
    
    qa_pairs = []
    
    principles = toolkit.CompliancePrinciple.get_all_principles()
    
    for principle_id, principle_info in principles.items():
        principle_name = principle_info["name"]
        questions = principle_info["questions"]
        
        print(f"\n--- Principio {principle_id}: {principle_name} ---")
        print(f"Descripción: {principle_info['description']}")
        print("\nPreguntas sugeridas (copia estas preguntas al modelo LLM):")
        
        for i, question in enumerate(questions, 1):
            print(f"\n{i}. {question}")
            
            include_question = input("¿Incluir esta pregunta? (s/n): ").strip().lower()
            if include_question != 'n':
                print("Copia esta pregunta al modelo LLM y pega aquí su respuesta:")
                response = input("Respuesta (o presiona Enter para omitir): ").strip()
                
                if response:
                    qa_pair = QAPair(
                        question=question,
                        response=response,
                        category=principle_name
                    )
                    qa_pairs.append(qa_pair)
                    print("Respuesta registrada.")

        add_custom = input(f"\n¿Deseas agregar preguntas personalizadas para el principio '{principle_name}'? (s/n): ").strip().lower()
        if add_custom == 's':
            while True:
                custom_question = input("\nPregunta personalizada (o Enter para terminar): ").strip()
                if not custom_question:
                    break
                    
                print("Copia esta pregunta al modelo LLM y pega aquí su respuesta:")
                response = input("Respuesta: ").strip()
                
                if response:
                    qa_pair = QAPair(
                        question=custom_question,
                        response=response,
                        category=principle_name
                    )
                    qa_pairs.append(qa_pair)
                    print("Respuesta registrada.")
    
    if qa_pairs:
        toolkit.save_qa_pairs(model_id, qa_pairs)
        print(f"\nSe han guardado {len(qa_pairs)} pares de preguntas y respuestas.")
    else:
        print("\nNo se registraron respuestas.")

    print("\n=== GENERANDO INFORME DE CONFORMIDAD ===")
    compliance = toolkit.evaluate_model_compliance(model_id)
    toolkit.save_compliance_report(compliance)
    
    print("\nRESUMEN DE CONFORMIDAD:")
    print(compliance.summary)
    
    print("\nEVALUACIÓN POR PRINCIPIO:")
    for eval in compliance.evaluations:
        print(f"\n* {eval.principle_name}: {eval.compliance_level}")
        
        if eval.evidence:
            print("  Evidencia:")
            for evidence in eval.evidence:
                print(f"  - {evidence}")
                
        if eval.recommendations:
            print("  Recomendaciones:")
            for recommendation in eval.recommendations:
                print(f"  - {recommendation}")
    
    report_path = os.path.join(toolkit.data_dir, model_id, "compliance_report.json")
    print(f"\nEl informe completo se ha guardado en: {report_path}")
    
    print("\n=== EVALUACIÓN COMPLETADA ===")

if __name__ == "__main__":
    evaluate_model() 