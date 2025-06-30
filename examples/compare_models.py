import sys
import os
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import (
    LLMComplianceToolkit, ModelMetadata, PrincipleEvaluation,
    ModelCompliance, ComplianceLevel
)

def compare_models(model_ids: List[str]):
    """Compara la conformidad entre varios modelos LLM
    
    Args:
        model_ids: Lista de IDs de modelos para comparar
    """
    toolkit = LLMComplianceToolkit()

    for model_id in model_ids:
        model_dir = os.path.join(toolkit.data_dir, model_id)
        if not os.path.exists(model_dir):
            print(f"Error: El modelo '{model_id}' no existe. Por favor, evalúalo primero.")
            return

    reports = {}
    for model_id in model_ids:
        report = toolkit.load_compliance_report(model_id)
        if report:
            reports[model_id] = report
        else:
            print(f"Error: No se encontró informe de conformidad para el modelo '{model_id}'.")
            return
    
    print(f"\n=== COMPARACIÓN DE CONFORMIDAD ENTRE MODELOS ===")
    print(f"Modelos: {', '.join(model_ids)}")

    comparison_data = []

    for principle_id in toolkit.CompliancePrinciple.PRINCIPLES:
        principle_name = toolkit.CompliancePrinciple.get_principle(principle_id)["name"]
        row = {"Principio": principle_name}
        
        # Añadir nivel de conformidad de cada modelo
        for model_id, report in reports.items():
            eval = next((e for e in report.evaluations if e.principle_id == principle_id), None)
            if eval:
                row[model_id] = eval.compliance_level
            else:
                row[model_id] = ComplianceLevel.UNKNOWN
        
        comparison_data.append(row)

    overall_row = {"Principio": "CONFORMIDAD GENERAL"}
    for model_id, report in reports.items():
        overall_row[model_id] = report.overall_compliance
    
    comparison_data.append(overall_row)

    df = pd.DataFrame(comparison_data)

    print("\nCOMPARACIÓN POR PRINCIPIO:")
    print(df.to_string(index=False))
    
    try:
        compliance_values = {
            ComplianceLevel.COMPLIANT: 3,
            ComplianceLevel.PARTIALLY_COMPLIANT: 2,
            ComplianceLevel.NON_COMPLIANT: 1,
            ComplianceLevel.UNKNOWN: 0
        }

        heatmap_data = []
        principles = []

        for row in comparison_data[:-1]:
            principles.append(row["Principio"])
            model_values = []
            for model_id in model_ids:
                compliance_level = row[model_id]
                model_values.append(compliance_values[compliance_level])
            heatmap_data.append(model_values)
        
        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap_data, cmap='RdYlGn', aspect='auto')

        plt.yticks(range(len(principles)), principles)
        plt.xticks(range(len(model_ids)), model_ids)

        for i in range(len(principles)):
            for j in range(len(model_ids)):
                text = list(compliance_values.keys())[list(compliance_values.values()).index(heatmap_data[i][j])]
                plt.text(j, i, text, ha="center", va="center", color="black")
        
        plt.colorbar(ticks=[0, 1, 2, 3], 
                    label='Nivel de Conformidad')
        plt.colorbar().set_ticklabels(['Desconocido', 'No Conforme', 'Parcialmente Conforme', 'Conforme'])
        
        plt.title('Comparación de Conformidad por Principio')
        plt.tight_layout()

        output_dir = os.path.join(toolkit.data_dir, "comparisons")
        os.makedirs(output_dir, exist_ok=True)
        
        models_str = "_vs_".join(model_ids)
        output_file = os.path.join(output_dir, f"comparison_{models_str}.png")
        plt.savefig(output_file)
        print(f"\nGráfico de comparación guardado en: {output_file}")
        
    except Exception as e:
        print(f"\nNo se pudo generar visualización: {str(e)}")

    comparison_json = {
        "models": model_ids,
        "date": datetime.now().isoformat(),
        "principles": comparison_data
    }
    
    output_json = os.path.join(toolkit.data_dir, "comparisons", f"comparison_{models_str}.json")
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(comparison_json, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"Datos de comparación guardados en: {output_json}")
    print("\n=== COMPARACIÓN COMPLETADA ===")

if __name__ == "__main__":
    print("\n=== COMPARACIÓN DE MODELOS LLM ===")
    print("Este script compara la conformidad de diferentes modelos LLM con los principios ACM.")

    toolkit = LLMComplianceToolkit()
    model_dirs = [d for d in os.listdir(toolkit.data_dir) 
                if os.path.isdir(os.path.join(toolkit.data_dir, d))
                and d != "comparisons"]
    
    if not model_dirs:
        print("No hay modelos disponibles para comparar. Evalúa algunos modelos primero.")
        sys.exit(1)
    
    print("\nModelos disponibles para comparar:")
    for i, model_id in enumerate(model_dirs, 1):
        print(f"{i}. {model_id}")
    
    while True:
        selection = input("\nSelecciona los modelos a comparar (números separados por coma): ").strip()
        try:
            indices = [int(idx.strip()) - 1 for idx in selection.split(",")]
            selected_models = [model_dirs[idx] for idx in indices if 0 <= idx < len(model_dirs)]
            
            if len(selected_models) >= 2:
                break
            else:
                print("Por favor, selecciona al menos 2 modelos válidos.")
        except:
            print("Entrada inválida. Ingresa números separados por coma.")
    
    compare_models(selected_models) 