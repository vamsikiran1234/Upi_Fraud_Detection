"""
UPI Fraud Detection System - Architecture Diagram Generator
Creates visual architecture diagrams using Python libraries
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_system_architecture():
    """Create comprehensive system architecture diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 14)
    ax.axis('off')
    
    # Color scheme
    colors = {
        'ingestion': '#E3F2FD',
        'processing': '#F3E5F5', 
        'storage': '#E8F5E8',
        'serving': '#FFF3E0',
        'monitoring': '#FCE4EC',
        'external': '#F5F5F5'
    }
    
    # Title
    ax.text(10, 13.5, 'UPI Fraud Detection System Architecture', 
            fontsize=20, fontweight='bold', ha='center')
    
    # Layer 1: External Systems
    upi_gateway = FancyBboxPatch((0.5, 11.5), 3, 1.5, 
                                boxstyle="round,pad=0.1", 
                                facecolor=colors['external'],
                                edgecolor='black', linewidth=2)
    ax.add_patch(upi_gateway)
    ax.text(2, 12.25, 'UPI Gateway\n(External)', ha='center', va='center', fontweight='bold')
    
    # Layer 2: Ingestion
    kafka = FancyBboxPatch((5, 11.5), 3, 1.5,
                          boxstyle="round,pad=0.1",
                          facecolor=colors['ingestion'],
                          edgecolor='blue', linewidth=2)
    ax.add_patch(kafka)
    ax.text(6.5, 12.25, 'Apache Kafka\n(Message Broker)', ha='center', va='center', fontweight='bold')
    
    ingestion_api = FancyBboxPatch((9, 11.5), 3, 1.5,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['ingestion'],
                                  edgecolor='blue', linewidth=2)
    ax.add_patch(ingestion_api)
    ax.text(10.5, 12.25, 'Ingestion API\n(FastAPI)', ha='center', va='center', fontweight='bold')
    
    # Layer 3: Stream Processing
    spark = FancyBboxPatch((2, 9), 4, 1.5,
                          boxstyle="round,pad=0.1",
                          facecolor=colors['processing'],
                          edgecolor='purple', linewidth=2)
    ax.add_patch(spark)
    ax.text(4, 9.75, 'Spark Streaming\n(Feature Engineering)', ha='center', va='center', fontweight='bold')
    
    gnn = FancyBboxPatch((7, 9), 4, 1.5,
                        boxstyle="round,pad=0.1",
                        facecolor=colors['processing'],
                        edgecolor='purple', linewidth=2)
    ax.add_patch(gnn)
    ax.text(9, 9.75, 'GNN Service\n(Collusion Detection)', ha='center', va='center', fontweight='bold')
    
    # Layer 4: Storage
    redis = FancyBboxPatch((1, 6.5), 3, 1.5,
                          boxstyle="round,pad=0.1",
                          facecolor=colors['storage'],
                          edgecolor='green', linewidth=2)
    ax.add_patch(redis)
    ax.text(2.5, 7.25, 'Redis\n(Feature Cache)', ha='center', va='center', fontweight='bold')
    
    postgres = FancyBboxPatch((5, 6.5), 3, 1.5,
                             boxstyle="round,pad=0.1",
                             facecolor=colors['storage'],
                             edgecolor='green', linewidth=2)
    ax.add_patch(postgres)
    ax.text(6.5, 7.25, 'PostgreSQL\n(Historical Data)', ha='center', va='center', fontweight='bold')
    
    model_store = FancyBboxPatch((9, 6.5), 3, 1.5,
                                boxstyle="round,pad=0.1",
                                facecolor=colors['storage'],
                                edgecolor='green', linewidth=2)
    ax.add_patch(model_store)
    ax.text(10.5, 7.25, 'Model Store\n(ML Models)', ha='center', va='center', fontweight='bold')
    
    # Layer 5: Serving
    fraud_api = FancyBboxPatch((3, 4), 4, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['serving'],
                              edgecolor='orange', linewidth=2)
    ax.add_patch(fraud_api)
    ax.text(5, 4.75, 'Fraud Detection API\n(FastAPI + SHAP)', ha='center', va='center', fontweight='bold')
    
    decision_engine = FancyBboxPatch((8, 4), 4, 1.5,
                                    boxstyle="round,pad=0.1",
                                    facecolor=colors['serving'],
                                    edgecolor='orange', linewidth=2)
    ax.add_patch(decision_engine)
    ax.text(10, 4.75, 'Decision Engine\n(Rules + ML)', ha='center', va='center', fontweight='bold')
    
    # Layer 6: Monitoring & UI
    dashboard = FancyBboxPatch((1, 1.5), 3, 1.5,
                              boxstyle="round,pad=0.1",
                              facecolor=colors['monitoring'],
                              edgecolor='red', linewidth=2)
    ax.add_patch(dashboard)
    ax.text(2.5, 2.25, 'Dashboard\n(React UI)', ha='center', va='center', fontweight='bold')
    
    prometheus = FancyBboxPatch((5, 1.5), 3, 1.5,
                               boxstyle="round,pad=0.1",
                               facecolor=colors['monitoring'],
                               edgecolor='red', linewidth=2)
    ax.add_patch(prometheus)
    ax.text(6.5, 2.25, 'Prometheus\n(Metrics)', ha='center', va='center', fontweight='bold')
    
    grafana = FancyBboxPatch((9, 1.5), 3, 1.5,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['monitoring'],
                            edgecolor='red', linewidth=2)
    ax.add_patch(grafana)
    ax.text(10.5, 2.25, 'Grafana\n(Visualization)', ha='center', va='center', fontweight='bold')
    
    # External Services
    analyst = FancyBboxPatch((14, 11.5), 3, 1.5,
                            boxstyle="round,pad=0.1",
                            facecolor=colors['external'],
                            edgecolor='black', linewidth=2)
    ax.add_patch(analyst)
    ax.text(15.5, 12.25, 'Fraud Analysts\n(Human Review)', ha='center', va='center', fontweight='bold')
    
    # Add arrows for data flow
    arrows = [
        # UPI Gateway -> Kafka
        ((3.5, 12.25), (5, 12.25)),
        # Kafka -> Spark
        ((6.5, 11.5), (4, 10.5)),
        # Spark -> Redis
        ((3, 9), (2.5, 8)),
        # Spark -> PostgreSQL  
        ((4.5, 9), (6, 8)),
        # Redis -> Fraud API
        ((2.5, 6.5), (4, 5.5)),
        # PostgreSQL -> Fraud API
        ((6.5, 6.5), (5.5, 5.5)),
        # Model Store -> Fraud API
        ((9.5, 6.5), (6, 5.5)),
        # Fraud API -> Decision Engine
        ((7, 4.75), (8, 4.75)),
        # GNN -> PostgreSQL
        ((8.5, 9), (7, 8)),
        # Dashboard connections
        ((2.5, 3), (2.5, 4)),
        ((6.5, 3), (5, 4)),
    ]
    
    for start, end in arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=20, fc="black", alpha=0.7)
        ax.add_patch(arrow)
    
    # Add legend
    legend_elements = [
        patches.Patch(color=colors['ingestion'], label='Data Ingestion'),
        patches.Patch(color=colors['processing'], label='Stream Processing'),
        patches.Patch(color=colors['storage'], label='Data Storage'),
        patches.Patch(color=colors['serving'], label='Model Serving'),
        patches.Patch(color=colors['monitoring'], label='Monitoring & UI'),
        patches.Patch(color=colors['external'], label='External Systems')
    ]
    
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    plt.tight_layout()
    plt.savefig('upi_fraud_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_data_flow_diagram():
    """Create detailed data flow diagram"""
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 12))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Title
    ax.text(9, 11.5, 'UPI Fraud Detection - Data Flow', 
            fontsize=18, fontweight='bold', ha='center')
    
    # Step boxes with numbers
    steps = [
        (1, 10, "Transaction\nEvent"),
        (3.5, 10, "Kafka\nIngestion"),
        (6, 10, "Feature\nExtraction"),
        (8.5, 10, "Feature\nStore"),
        (11, 10, "Model\nInference"),
        (13.5, 10, "Decision\nEngine"),
        (16, 10, "Response"),
        
        (1, 7, "Raw Data\nValidation"),
        (4, 7, "Stream\nProcessing"),
        (7, 7, "Real-time\nCache"),
        (10, 7, "Ensemble\nML Models"),
        (13, 7, "Business\nRules"),
        (16, 7, "Action\nExecution"),
        
        (2.5, 4, "GNN Graph\nConstruction"),
        (5.5, 4, "Collusion\nDetection"),
        (8.5, 4, "Risk\nScoring"),
        (11.5, 4, "Alert\nGeneration"),
        (14.5, 4, "Analyst\nReview"),
        
        (4, 1, "Model\nRetraining"),
        (8, 1, "Performance\nMonitoring"),
        (12, 1, "Feedback\nLoop")
    ]
    
    # Draw step boxes
    for i, (x, y, text) in enumerate(steps):
        if i < 7:  # Main flow
            color = '#E3F2FD'
        elif i < 13:  # Processing flow
            color = '#F3E5F5'
        elif i < 18:  # Analysis flow
            color = '#E8F5E8'
        else:  # Feedback flow
            color = '#FFF3E0'
            
        box = FancyBboxPatch((x-0.7, y-0.4), 1.4, 0.8,
                            boxstyle="round,pad=0.1",
                            facecolor=color,
                            edgecolor='black', linewidth=1)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Add flow arrows
    main_flow_arrows = [
        ((1.7, 10), (2.8, 10)),
        ((4.2, 10), (5.3, 10)),
        ((6.7, 10), (7.8, 10)),
        ((9.2, 10), (10.3, 10)),
        ((11.7, 10), (12.8, 10)),
        ((14.2, 10), (15.3, 10)),
    ]
    
    for start, end in main_flow_arrows:
        arrow = ConnectionPatch(start, end, "data", "data",
                               arrowstyle="->", shrinkA=5, shrinkB=5,
                               mutation_scale=15, fc="blue", ec="blue")
        ax.add_patch(arrow)
    
    # Add timing annotations
    ax.text(9, 8.5, 'Real-time Flow (< 200ms)', ha='center', fontsize=12, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
    
    ax.text(9, 5.5, 'Batch Analysis (30 min cycles)', ha='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
    
    ax.text(8, 2.5, 'Feedback & Learning (Weekly)', ha='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('upi_fraud_dataflow.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Generating UPI Fraud Detection Architecture Diagrams...")
    
    # Create system architecture diagram
    create_system_architecture()
    print("✓ System architecture diagram saved as 'upi_fraud_architecture.png'")
    
    # Create data flow diagram
    create_data_flow_diagram()
    print("✓ Data flow diagram saved as 'upi_fraud_dataflow.png'")
    
    print("\nDiagrams generated successfully!")
    print("Use these diagrams for:")
    print("- System documentation")
    print("- Architecture reviews") 
    print("- Stakeholder presentations")
    print("- Technical onboarding")
