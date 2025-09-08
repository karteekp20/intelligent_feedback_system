"""
Analytics components for the Streamlit dashboard.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import numpy as np

from config.settings import OUTPUT_DIR


class AnalyticsComponent:
    """Analytics visualization component for the dashboard."""
    
    def __init__(self):
        """Initialize analytics component."""
        self.output_dir = OUTPUT_DIR
    
    def render_overview_charts(self, data: Dict[str, Any]):
        """Render overview analytics charts."""
        col1, col2 = st.columns(2)
        
        with col1:
            self.render_category_distribution(data.get("category_distribution", {}))
        
        with col2:
            self.render_priority_distribution(data.get("priority_distribution", {}))
        
        # Timeline chart
        self.render_processing_timeline(data.get("timeline_data", []))
        
        # Performance metrics
        self.render_performance_metrics(data.get("performance_data", {}))
    
    def render_category_distribution(self, category_data: Dict[str, int]):
        """Render feedback category distribution pie chart."""
        if not category_data:
            st.info("No category data available")
            return
        
        # Prepare data
        categories = list(category_data.keys())
        counts = list(category_data.values())
        
        # Create pie chart
        fig = px.pie(
            values=counts,
            names=categories,
            title="📊 Feedback Category Distribution",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )
        
        fig.update_layout(
            showlegend=True,
            height=400,
            font=dict(size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_priority_distribution(self, priority_data: Dict[str, int]):
        """Render priority distribution bar chart."""
        if not priority_data:
            st.info("No priority data available")
            return
        
        # Prepare data with custom order
        priority_order = ["Critical", "High", "Medium", "Low"]
        priorities = []
        counts = []
        colors = []
        
        color_map = {
            "Critical": "#dc2626",  # Red
            "High": "#ea580c",      # Orange
            "Medium": "#ca8a04",    # Yellow
            "Low": "#16a34a"        # Green
        }
        
        for priority in priority_order:
            if priority in priority_data:
                priorities.append(priority)
                counts.append(priority_data[priority])
                colors.append(color_map[priority])
        
        # Create bar chart
        fig = px.bar(
            x=priorities,
            y=counts,
            title="🚨 Priority Distribution",
            color=priorities,
            color_discrete_map=color_map,
            text=counts
        )
        
        fig.update_traces(
            texttemplate='%{text}',
            textposition='outside',
            hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
        )
        
        fig.update_layout(
            showlegend=False,
            height=400,
            xaxis_title="Priority Level",
            yaxis_title="Number of Tickets",
            font=dict(size=12)
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_processing_timeline(self, timeline_data: List[Dict[str, Any]]):
        """Render processing timeline chart."""
        if not timeline_data:
            st.info("No timeline data available")
            return
        
        st.markdown("### 📈 Processing Timeline")
        
        # Convert to DataFrame
        df = pd.DataFrame(timeline_data)
        
        if df.empty:
            st.info("No timeline data to display")
            return
        
        # Ensure datetime column
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Create line chart
        fig = px.line(
            df,
            x='date',
            y='count',
            title="Processing Activity Over Time",
            markers=True,
            color_discrete_sequence=['#3b82f6']
        )
        
        fig.update_traces(
            mode='lines+markers',
            hovertemplate='<b>Date:</b> %{x}<br><b>Items Processed:</b> %{y}<extra></extra>'
        )
        
        fig.update_layout(
            height=400,
            xaxis_title="Date",
            yaxis_title="Items Processed",
            font=dict(size=12),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_metrics(self, performance_data: Dict[str, Any]):
        """Render performance metrics dashboard."""
        if not performance_data:
            st.info("No performance data available")
            return
        
        st.markdown("### ⚡ Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_time = performance_data.get("avg_processing_time", 0)
            st.metric(
                label="⏱️ Avg Processing Time",
                value=f"{avg_time:.2f}s",
                delta=performance_data.get("time_delta", None)
            )
        
        with col2:
            success_rate = performance_data.get("success_rate", 0)
            st.metric(
                label="✅ Success Rate",
                value=f"{success_rate:.1%}",
                delta=performance_data.get("success_rate_delta", None)
            )
        
        with col3:
            throughput = performance_data.get("throughput", 0)
            st.metric(
                label="🚀 Throughput",
                value=f"{throughput:.1f}/min",
                delta=performance_data.get("throughput_delta", None)
            )
        
        with col4:
            confidence = performance_data.get("avg_confidence", 0)
            st.metric(
                label="🎯 Avg Confidence",
                value=f"{confidence:.2f}",
                delta=performance_data.get("confidence_delta", None)
            )
    
    def render_detailed_analytics(self, tickets_df: Optional[pd.DataFrame]):
        """Render detailed analytics page."""
        if tickets_df is None or tickets_df.empty:
            st.warning("No ticket data available for detailed analytics")
            return
        
        st.markdown("## 📊 Detailed Analytics")
        
        # Confidence analysis
        self.render_confidence_analysis(tickets_df)
        
        # Source analysis
        self.render_source_analysis(tickets_df)
        
        # Temporal analysis
        self.render_temporal_analysis(tickets_df)
        
        # Quality metrics
        self.render_quality_metrics(tickets_df)
    
    def render_confidence_analysis(self, tickets_df: pd.DataFrame):
        """Render confidence score analysis."""
        st.markdown("### 🎯 Confidence Score Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Confidence histogram
            fig = px.histogram(
                tickets_df,
                x="agent_confidence",
                nbins=20,
                title="Confidence Score Distribution",
                color_discrete_sequence=['#3b82f6']
            )
            
            fig.update_layout(
                xaxis_title="Confidence Score",
                yaxis_title="Number of Tickets",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Confidence by category
            fig = px.box(
                tickets_df,
                x="category",
                y="agent_confidence",
                title="Confidence by Category",
                color="category"
            )
            
            fig.update_layout(
                xaxis_title="Category",
                yaxis_title="Confidence Score",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Confidence statistics
        confidence_stats = tickets_df["agent_confidence"].describe()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean Confidence", f"{confidence_stats['mean']:.3f}")
        with col2:
            st.metric("Median Confidence", f"{confidence_stats['50%']:.3f}")
        with col3:
            st.metric("Min Confidence", f"{confidence_stats['min']:.3f}")
        with col4:
            st.metric("Max Confidence", f"{confidence_stats['max']:.3f}")
    
    def render_source_analysis(self, tickets_df: pd.DataFrame):
        """Render source type analysis."""
        st.markdown("### 📱 Source Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Source distribution
            source_counts = tickets_df["source_type"].value_counts()
            
            fig = px.pie(
                values=source_counts.values,
                names=source_counts.index,
                title="Distribution by Source Type",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Category by source
            crosstab = pd.crosstab(tickets_df["source_type"], tickets_df["category"])
            
            fig = px.bar(
                crosstab,
                title="Categories by Source Type",
                barmode="group"
            )
            
            fig.update_layout(
                xaxis_title="Source Type",
                yaxis_title="Count",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_temporal_analysis(self, tickets_df: pd.DataFrame):
        """Render temporal analysis of ticket creation."""
        if "created_at" not in tickets_df.columns:
            st.info("No temporal data available")
            return
        
        st.markdown("### 📅 Temporal Analysis")
        
        # Convert to datetime
        tickets_df["created_at"] = pd.to_datetime(tickets_df["created_at"])
        
        # Group by date
        daily_counts = tickets_df.groupby(tickets_df["created_at"].dt.date).size()
        
        fig = px.line(
            x=daily_counts.index,
            y=daily_counts.values,
            title="Tickets Created Over Time",
            markers=True
        )
        
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Tickets Created",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Hour of day analysis
        if len(tickets_df) > 0:
            hourly_counts = tickets_df.groupby(tickets_df["created_at"].dt.hour).size()
            
            fig = px.bar(
                x=hourly_counts.index,
                y=hourly_counts.values,
                title="Tickets by Hour of Day",
                color_discrete_sequence=['#10b981']
            )
            
            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Number of Tickets",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_quality_metrics(self, tickets_df: pd.DataFrame):
        """Render quality metrics analysis."""
        st.markdown("### 📝 Quality Metrics")
        
        # Title length analysis
        tickets_df["title_length"] = tickets_df["title"].str.len()
        tickets_df["description_length"] = tickets_df["description"].str.len()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                tickets_df,
                x="title_length",
                title="Title Length Distribution",
                nbins=20,
                color_discrete_sequence=['#f59e0b']
            )
            
            fig.update_layout(
                xaxis_title="Title Length (characters)",
                yaxis_title="Number of Tickets",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                tickets_df,
                x="description_length",
                title="Description Length Distribution",
                nbins=20,
                color_discrete_sequence=['#8b5cf6']
            )
            
            fig.update_layout(
                xaxis_title="Description Length (characters)",
                yaxis_title="Number of Tickets",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def render_agent_performance(self, metrics_df: Optional[pd.DataFrame]):
        """Render agent performance metrics."""
        if metrics_df is None or metrics_df.empty:
            st.info("No agent performance data available")
            return
        
        st.markdown("### 🤖 Agent Performance")
        
        # Filter agent metrics
        agent_metrics = metrics_df[
            metrics_df["metric_name"].str.contains("agent_", na=False)
        ].copy()
        
        if agent_metrics.empty:
            st.info("No agent-specific metrics found")
            return
        
        # Parse agent names and metric types
        agent_metrics["agent_name"] = agent_metrics["metric_name"].str.extract(r"agent_([^_]+)_")
        agent_metrics["metric_type"] = agent_metrics["metric_name"].str.extract(r"agent_[^_]+_(.+)")
        
        # Performance by agent
        col1, col2 = st.columns(2)
        
        with col1:
            # Processing time by agent
            time_metrics = agent_metrics[
                agent_metrics["metric_type"] == "average_processing_time"
            ]
            
            if not time_metrics.empty:
                fig = px.bar(
                    time_metrics,
                    x="agent_name",
                    y="value",
                    title="Average Processing Time by Agent",
                    color="agent_name"
                )
                
                fig.update_layout(
                    xaxis_title="Agent",
                    yaxis_title="Processing Time (seconds)",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Success rate by agent
            success_metrics = agent_metrics[
                agent_metrics["metric_type"] == "success_rate"
            ]
            
            if not success_metrics.empty:
                fig = px.bar(
                    success_metrics,
                    x="agent_name",
                    y="value",
                    title="Success Rate by Agent",
                    color="agent_name",
                    color_discrete_sequence=px.colors.qualitative.Set2
                )
                
                fig.update_layout(
                    xaxis_title="Agent",
                    yaxis_title="Success Rate",
                    height=400,
                    showlegend=False
                )
                
                st.plotly_chart(fig, use_container_width=True)
    
    def render_comparison_analysis(self, tickets_df: pd.DataFrame, expected_df: Optional[pd.DataFrame]):
        """Render comparison with expected classifications."""
        if expected_df is None or expected_df.empty:
            st.info("No expected classifications data for comparison")
            return
        
        st.markdown("### 🎯 Accuracy Analysis")
        
        # Merge with expected results
        comparison_df = tickets_df.merge(
            expected_df,
            left_on="source_id",
            right_on="source_id",
            how="inner",
            suffixes=("_actual", "_expected")
        )
        
        if comparison_df.empty:
            st.warning("No matching data for comparison")
            return
        
        # Calculate accuracy metrics
        category_accuracy = (
            comparison_df["category_actual"] == comparison_df["category_expected"]
        ).mean()
        
        priority_accuracy = (
            comparison_df["priority_actual"] == comparison_df["priority_expected"]
        ).mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Category Accuracy", f"{category_accuracy:.1%}")
        
        with col2:
            st.metric("Priority Accuracy", f"{priority_accuracy:.1%}")
        
        # Confusion matrix for categories
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        import matplotlib.pyplot as plt
        
        # Category confusion matrix
        cm = confusion_matrix(
            comparison_df["category_expected"],
            comparison_df["category_actual"]
        )
        
        categories = sorted(comparison_df["category_expected"].unique())
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=categories,
            yticklabels=categories,
            ax=ax,
            cmap="Blues"
        )
        
        ax.set_title("Category Classification Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        
        st.pyplot(fig)
    
    def create_executive_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create executive summary of analytics."""
        summary = {
            "total_items_processed": data.get("total_processed", 0),
            "success_rate": data.get("success_rate", 0),
            "average_confidence": data.get("avg_confidence", 0),
            "processing_speed": data.get("items_per_second", 0),
            "top_categories": [],
            "quality_score": 0,
            "recommendations": []
        }
        
        # Top categories
        if "category_distribution" in data:
            category_dist = data["category_distribution"]
            sorted_categories = sorted(
                category_dist.items(),
                key=lambda x: x[1],
                reverse=True
            )
            summary["top_categories"] = sorted_categories[:3]
        
        # Quality score (based on confidence)
        if summary["average_confidence"] > 0.8:
            summary["quality_score"] = "High"
        elif summary["average_confidence"] > 0.6:
            summary["quality_score"] = "Medium"
        else:
            summary["quality_score"] = "Low"
        
        # Recommendations
        recommendations = []
        
        if summary["success_rate"] < 0.9:
            recommendations.append("Consider reviewing failed processing cases")
        
        if summary["average_confidence"] < 0.7:
            recommendations.append("Review and tune classification thresholds")
        
        if summary["processing_speed"] < 10:
            recommendations.append("Consider optimizing processing pipeline")
        
        summary["recommendations"] = recommendations
        
        return summary
    
    def export_analytics_report(self, data: Dict[str, Any], format: str = "pdf"):
        """Export analytics report in specified format."""
        # This would implement report generation
        # For now, return a placeholder
        
        if format == "pdf":
            # Generate PDF report
            report_content = self._generate_pdf_report(data)
        elif format == "excel":
            # Generate Excel report
            report_content = self._generate_excel_report(data)
        else:
            # Generate CSV report
            report_content = self._generate_csv_report(data)
        
        return report_content
    
    def _generate_pdf_report(self, data: Dict[str, Any]) -> bytes:
        """Generate PDF report (placeholder)."""
        # Would use libraries like reportlab or weasyprint
        return b"PDF report placeholder"
    
    def _generate_excel_report(self, data: Dict[str, Any]) -> bytes:
        """Generate Excel report (placeholder)."""
        # Would use pandas to_excel or openpyxl
        return b"Excel report placeholder"
    
    def _generate_csv_report(self, data: Dict[str, Any]) -> str:
        """Generate CSV report."""
        summary = self.create_executive_summary(data)
        
        # Create summary CSV
        report_lines = [
            "Metric,Value",
            f"Total Items Processed,{summary['total_items_processed']}",
            f"Success Rate,{summary['success_rate']:.1%}",
            f"Average Confidence,{summary['average_confidence']:.3f}",
            f"Processing Speed,{summary['processing_speed']:.1f} items/sec",
            f"Quality Score,{summary['quality_score']}",
            "",
            "Top Categories,Count"
        ]
        
        for category, count in summary['top_categories']:
            report_lines.append(f"{category},{count}")
        
        report_lines.append("")
        report_lines.append("Recommendations")
        for rec in summary['recommendations']:
            report_lines.append(f'"{rec}"')
        
        return "\n".join(report_lines)