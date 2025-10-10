"""
UPI Fraud Detection - Graph Visualization
Visualization tools for transaction graphs and collusion patterns
"""

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from pathlib import Path

from graph_builder import TransactionGraphBuilder
from collusion_detector import CollusionDetectionPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GraphVisualizer:
    """Visualization tools for transaction graphs and fraud patterns"""
    
    def __init__(self, output_dir: str = "./visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_graph_statistics(self, graph_data, save_path: Optional[str] = None):
        """Plot basic graph statistics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Node counts
        node_counts = {node_type: graph_data[node_type].num_nodes for node_type in graph_data.node_types}
        axes[0, 0].bar(node_counts.keys(), node_counts.values())
        axes[0, 0].set_title('Node Counts by Type')
        axes[0, 0].set_ylabel('Number of Nodes')
        
        # Edge counts
        edge_counts = {}
        for edge_type in graph_data.edge_types:
            edge_name = f"{edge_type[0]}-{edge_type[2]}"
            edge_counts[edge_name] = graph_data[edge_type].edge_index.size(1)
        
        axes[0, 1].bar(range(len(edge_counts)), list(edge_counts.values()))
        axes[0, 1].set_xticks(range(len(edge_counts)))
        axes[0, 1].set_xticklabels(list(edge_counts.keys()), rotation=45)
        axes[0, 1].set_title('Edge Counts by Type')
        axes[0, 1].set_ylabel('Number of Edges')
        
        # User feature distribution (if available)
        if 'user' in graph_data.node_types and hasattr(graph_data['user'], 'x'):
            user_features = graph_data['user'].x.numpy()
            axes[1, 0].hist(user_features[:, 0], bins=30, alpha=0.7, label='Transaction Count')
            axes[1, 0].hist(user_features[:, 1], bins=30, alpha=0.7, label='Total Amount')
            axes[1, 0].set_title('User Feature Distribution')
            axes[1, 0].set_xlabel('Feature Value')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
        
        # Merchant feature distribution (if available)
        if 'merchant' in graph_data.node_types and hasattr(graph_data['merchant'], 'x'):
            merchant_features = graph_data['merchant'].x.numpy()
            axes[1, 1].hist(merchant_features[:, 0], bins=30, alpha=0.7, label='Transaction Count')
            axes[1, 1].hist(merchant_features[:, 2], bins=30, alpha=0.7, label='Average Amount')
            axes[1, 1].set_title('Merchant Feature Distribution')
            axes[1, 1].set_xlabel('Feature Value')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Graph statistics saved to {save_path}")
        
        plt.show()
    
    def plot_networkx_graph(self, nx_graph: nx.Graph, 
                           node_colors: Optional[Dict] = None,
                           save_path: Optional[str] = None):
        """Plot NetworkX graph with custom styling"""
        plt.figure(figsize=(20, 16))
        
        # Create layout
        pos = nx.spring_layout(nx_graph, k=1, iterations=50)
        
        # Separate nodes by type
        user_nodes = [n for n, d in nx_graph.nodes(data=True) if d.get('type') == 'user']
        merchant_nodes = [n for n, d in nx_graph.nodes(data=True) if d.get('type') == 'merchant']
        device_nodes = [n for n, d in nx_graph.nodes(data=True) if d.get('type') == 'device']
        
        # Draw nodes
        if user_nodes:
            nx.draw_networkx_nodes(nx_graph, pos, nodelist=user_nodes, 
                                 node_color='lightblue', node_size=100, 
                                 alpha=0.8, label='Users')
        
        if merchant_nodes:
            nx.draw_networkx_nodes(nx_graph, pos, nodelist=merchant_nodes,
                                 node_color='lightcoral', node_size=150,
                                 alpha=0.8, label='Merchants')
        
        if device_nodes:
            nx.draw_networkx_nodes(nx_graph, pos, nodelist=device_nodes,
                                 node_color='lightgreen', node_size=80,
                                 alpha=0.8, label='Devices')
        
        # Draw edges
        nx.draw_networkx_edges(nx_graph, pos, alpha=0.3, width=0.5)
        
        # Add labels for important nodes
        important_nodes = {n: n.split('_')[1] for n in list(nx_graph.nodes())[:20]}
        nx.draw_networkx_labels(nx_graph, pos, important_nodes, font_size=8)
        
        plt.title('UPI Transaction Network Graph', size=16)
        plt.legend()
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Network graph saved to {save_path}")
        
        plt.show()
    
    def plot_collusion_rings(self, detection_results: Dict[str, Any], 
                           save_path: Optional[str] = None):
        """Visualize detected collusion rings"""
        if not detection_results['collusion_rings']:
            logger.info("No collusion rings to visualize")
            return
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Collusion Ring Sizes', 'Risk Level Distribution', 
                          'User Risk Scores', 'Ring Network'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "histogram"}, {"type": "scatter"}]]
        )
        
        # Ring sizes
        ring_sizes = [ring['ring_size'] for ring in detection_results['collusion_rings']]
        fig.add_trace(
            go.Bar(x=list(range(len(ring_sizes))), y=ring_sizes, name='Ring Size'),
            row=1, col=1
        )
        
        # Risk level distribution
        risk_levels = [ring['risk_level'] for ring in detection_results['collusion_rings']]
        risk_counts = pd.Series(risk_levels).value_counts()
        fig.add_trace(
            go.Pie(labels=risk_counts.index, values=risk_counts.values, name='Risk Levels'),
            row=1, col=2
        )
        
        # User risk scores
        if detection_results['suspicious_users']:
            risk_scores = [user['fraud_probability'] for user in detection_results['suspicious_users']]
            fig.add_trace(
                go.Histogram(x=risk_scores, name='Risk Scores', nbinsx=20),
                row=2, col=1
            )
        
        # Ring network visualization
        if len(detection_results['collusion_rings']) > 0:
            # Create a simple network of rings
            ring_ids = list(range(len(detection_results['collusion_rings'])))
            ring_sizes = [ring['ring_size'] for ring in detection_results['collusion_rings']]
            
            fig.add_trace(
                go.Scatter(
                    x=ring_ids, y=ring_sizes,
                    mode='markers+text',
                    marker=dict(size=[s*10 for s in ring_sizes], color=ring_sizes, 
                              colorscale='Reds', showscale=True),
                    text=[f"Ring {i}" for i in ring_ids],
                    textposition="middle center",
                    name='Collusion Rings'
                ),
                row=2, col=2
            )
        
        fig.update_layout(height=800, showlegend=True, 
                         title_text="Collusion Detection Analysis")
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Collusion analysis saved to {save_path}")
        
        fig.show()
    
    def plot_temporal_patterns(self, transactions_df: pd.DataFrame, 
                             detection_results: Dict[str, Any],
                             save_path: Optional[str] = None):
        """Plot temporal patterns in fraud and collusion"""
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Transaction Volume Over Time', 
                          'Suspicious Activity Timeline',
                          'Hourly Risk Distribution'),
            vertical_spacing=0.08
        )
        
        # Transaction volume over time
        transactions_df['hour'] = pd.to_datetime(transactions_df['timestamp']).dt.hour
        hourly_counts = transactions_df.groupby('hour').size()
        
        fig.add_trace(
            go.Scatter(x=hourly_counts.index, y=hourly_counts.values, 
                      mode='lines+markers', name='Transaction Volume'),
            row=1, col=1
        )
        
        # Suspicious activity timeline (if we have timestamps)
        if detection_results['suspicious_users']:
            # Create synthetic timeline for demonstration
            hours = np.random.randint(0, 24, len(detection_results['suspicious_users']))
            risk_scores = [user['fraud_probability'] for user in detection_results['suspicious_users']]
            
            fig.add_trace(
                go.Scatter(x=hours, y=risk_scores, mode='markers',
                          marker=dict(size=10, color=risk_scores, colorscale='Reds'),
                          name='Suspicious Users'),
                row=2, col=1
            )
        
        # Hourly risk distribution
        if 'user_risk_distribution' in detection_results.get('risk_scores', {}):
            # Create synthetic hourly risk data
            hours = list(range(24))
            risk_by_hour = np.random.beta(2, 5, 24)  # Synthetic risk distribution
            
            fig.add_trace(
                go.Bar(x=hours, y=risk_by_hour, name='Average Risk by Hour'),
                row=3, col=1
            )
        
        fig.update_layout(height=900, showlegend=True,
                         title_text="Temporal Fraud Patterns")
        fig.update_xaxes(title_text="Hour of Day", row=3, col=1)
        fig.update_yaxes(title_text="Transaction Count", row=1, col=1)
        fig.update_yaxes(title_text="Risk Score", row=2, col=1)
        fig.update_yaxes(title_text="Average Risk", row=3, col=1)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Temporal patterns saved to {save_path}")
        
        fig.show()
    
    def create_interactive_dashboard(self, detection_results: Dict[str, Any],
                                   transactions_df: pd.DataFrame,
                                   save_path: Optional[str] = None):
        """Create comprehensive interactive dashboard"""
        
        # Create main dashboard with multiple tabs
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('Network Overview', 'Risk Distribution', 'Collusion Rings',
                          'Temporal Patterns', 'Geographic Distribution', 'Alert Summary'),
            specs=[[{"type": "scatter"}, {"type": "histogram"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "scatter"}, {"type": "table"}]]
        )
        
        # Network overview (simplified)
        if detection_results['suspicious_users']:
            user_indices = list(range(len(detection_results['suspicious_users'])))
            risk_scores = [user['fraud_probability'] for user in detection_results['suspicious_users']]
            
            fig.add_trace(
                go.Scatter(
                    x=user_indices, y=risk_scores,
                    mode='markers',
                    marker=dict(size=10, color=risk_scores, colorscale='Reds', 
                              showscale=True, colorbar=dict(title="Risk Score")),
                    name='Users',
                    text=[f"User {user['user_hash'][:8]}" for user in detection_results['suspicious_users']],
                    hovertemplate='<b>%{text}</b><br>Risk: %{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
        
        # Risk distribution
        if detection_results['suspicious_users']:
            risk_scores = [user['fraud_probability'] for user in detection_results['suspicious_users']]
            fig.add_trace(
                go.Histogram(x=risk_scores, nbinsx=20, name='Risk Distribution'),
                row=1, col=2
            )
        
        # Collusion rings
        if detection_results['collusion_rings']:
            ring_sizes = [ring['ring_size'] for ring in detection_results['collusion_rings']]
            ring_labels = [f"Ring {ring['ring_id']}" for ring in detection_results['collusion_rings']]
            
            fig.add_trace(
                go.Bar(x=ring_labels, y=ring_sizes, name='Ring Sizes'),
                row=1, col=3
            )
        
        # Add more visualizations...
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title_text="UPI Fraud Detection Dashboard",
            title_x=0.5
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Interactive dashboard saved to {save_path}")
        
        fig.show()
    
    def export_report_visualizations(self, detection_results: Dict[str, Any],
                                   transactions_df: pd.DataFrame,
                                   graph_data=None):
        """Export all visualizations for reporting"""
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        viz_dir = self.output_dir / f"fraud_report_{timestamp}"
        viz_dir.mkdir(exist_ok=True)
        
        logger.info(f"Exporting visualizations to {viz_dir}")
        
        # Graph statistics
        if graph_data:
            self.plot_graph_statistics(graph_data, viz_dir / "graph_statistics.png")
        
        # Collusion rings
        self.plot_collusion_rings(detection_results, viz_dir / "collusion_rings.html")
        
        # Temporal patterns
        self.plot_temporal_patterns(transactions_df, detection_results, 
                                   viz_dir / "temporal_patterns.html")
        
        # Interactive dashboard
        self.create_interactive_dashboard(detection_results, transactions_df,
                                        viz_dir / "fraud_dashboard.html")
        
        logger.info(f"All visualizations exported to {viz_dir}")
        return viz_dir

def main():
    """Example usage of visualization tools"""
    from graph_builder import create_sample_graph_data, TransactionGraphBuilder, GraphConfig
    
    # Create sample data
    transactions_df = create_sample_graph_data()
    
    # Build graph
    config = GraphConfig()
    builder = TransactionGraphBuilder(config)
    graph_data = builder.build_heterogeneous_graph(transactions_df)
    
    # Create sample detection results
    detection_results = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'total_users': graph_data['user'].num_nodes,
        'total_merchants': graph_data['merchant'].num_nodes,
        'total_devices': graph_data['device'].num_nodes,
        'suspicious_users': [
            {'user_hash': f'user_{i}', 'fraud_probability': np.random.beta(2, 3), 'risk_level': 'high'}
            for i in range(20)
        ],
        'collusion_rings': [
            {'ring_id': i, 'ring_size': np.random.randint(3, 8), 'risk_level': 'critical'}
            for i in range(5)
        ],
        'risk_scores': {
            'user_risk_distribution': {
                'mean': 0.3, 'std': 0.2, 'max': 0.9, 'suspicious_count': 20
            }
        }
    }
    
    # Create visualizer
    visualizer = GraphVisualizer()
    
    # Generate all visualizations
    visualizer.export_report_visualizations(detection_results, transactions_df, graph_data)

if __name__ == "__main__":
    main()
