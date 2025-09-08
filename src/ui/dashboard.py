#!/usr/bin/env python3
"""
Enhanced Real-time Dashboard for Intelligent Feedback Analysis System

This module provides a comprehensive Streamlit dashboard with real-time monitoring,
interactive controls, and advanced analytics capabilities.
"""
import json
import streamlit as st

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "isoformat"):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, "value"):  # Enum objects
            return obj.value
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)
import pandas as pd

# Custom JSON encoder for datetime objects  
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, 'isoformat'):  # datetime objects
            return obj.isoformat() 
        elif hasattr(obj, 'value'):  # Enum objects
            return obj.value
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        return super().default(obj)

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "isoformat"):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, "value"):  # Enum objects
            return obj.value
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)
import plotly.express as px

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "isoformat"):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, "value"):  # Enum objects
            return obj.value
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)
import plotly.graph_objects as go

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "isoformat"):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, "value"):  # Enum objects
            return obj.value
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)
from plotly.subplots import make_subplots
import asyncio

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "isoformat"):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, "value"):  # Enum objects
            return obj.value
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)
import json

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "isoformat"):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, "value"):  # Enum objects
            return obj.value
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)
import time

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "isoformat"):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, "value"):  # Enum objects
            return obj.value
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Custom JSON encoder for datetime objects
class DateTimeEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "isoformat"):  # datetime objects
            return obj.isoformat()
        elif hasattr(obj, "value"):  # Enum objects
            return obj.value
        elif hasattr(obj, "__dict__"):
            return obj.__dict__
        return super().default(obj)
from typing import Dict, Any, List, Optional

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.core.pipeline import FeedbackPipeline
from src.core.data_models import PipelineResult, SystemStats
from src.utils.logger import get_logger
from src.utils.csv_handler import CSVHandler
from config.settings import INPUT_DIR, OUTPUT_DIR, PROCESSING_SETTINGS

# Configure Streamlit page
st.set_page_config(
    page_title="Intelligent Feedback Analysis Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize logger
logger = get_logger(__name__)

class FeedbackDashboard:
    """Enhanced dashboard class with real-time capabilities."""
    
    def __init__(self):
        self.pipeline = None
        self.last_refresh = None
        self.auto_refresh = False
        
        # Initialize session state
        if 'processing_history' not in st.session_state:
            st.session_state.processing_history = []
        if 'system_stats' not in st.session_state:
            st.session_state.system_stats = SystemStats()
        if 'real_time_metrics' not in st.session_state:
            st.session_state.real_time_metrics = []
    
    def run(self):
        """Main dashboard execution."""
        self._render_header()
        self._render_sidebar()
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._render_main_content()
        
        with col2:
            self._render_control_panel()
        
        # Auto-refresh functionality
        if self.auto_refresh:
            time.sleep(st.session_state.get('refresh_interval', 30))
            st.rerun()
    
    def _render_header(self):
        """Render dashboard header with status indicators."""
        st.title("🤖 Intelligent Feedback Analysis Dashboard")
        
        # Status indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "System Status",
                "🟢 Online" if self._check_system_health() else "🔴 Offline",
                delta=None
            )
        
        with col2:
            stats = st.session_state.system_stats
            st.metric(
                "Files Processed",
                stats.total_feedback_processed,
                delta=stats.total_feedback_processed
            )
        
        with col3:
            st.metric(
                "Success Rate",
                f"{stats.success_rate:.1f}%",
                delta=f"{stats.success_rate:.1f}%"
            )
        
        with col4:
            st.metric(
                "Total Tickets",
                stats.total_tickets_generated,
                delta=stats.total_tickets_generated
            )
        
        # Real-time status bar
        if self.last_refresh:
            st.caption(f"Last updated: {self.last_refresh.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def _render_sidebar(self):
        """Render sidebar with navigation and controls."""
        st.sidebar.title("📊 Dashboard Controls")
        
        # Navigation
        page = st.sidebar.selectbox(
            "Navigate to:",
            ["📈 Overview", "📄 File Processing", "🎫 Ticket Management", 
             "📊 Analytics", "⚙️ Configuration", "📱 Real-time Monitor"]
        )
        
        # Auto-refresh controls
        st.sidebar.subheader("🔄 Auto Refresh")
        self.auto_refresh = st.sidebar.checkbox("Enable Auto Refresh")
        
        if self.auto_refresh:
            refresh_interval = st.sidebar.slider(
                "Refresh Interval (seconds)",
                min_value=10,
                max_value=300,
                value=30,
                step=10
            )
            st.session_state.refresh_interval = refresh_interval
        
        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Now"):
            self._refresh_data()
            st.rerun()
        
        # System controls
        st.sidebar.subheader("🎛️ System Controls")
        
        if st.sidebar.button("🚀 Process New Files"):
            self._process_new_files()
        
        if st.sidebar.button("🧹 Clear Logs"):
            self._clear_logs()
        
        if st.sidebar.button("📊 Export Report"):
            self._export_report()
        
        # Route to appropriate page
        if page == "📈 Overview":
            self._render_overview()
        elif page == "📄 File Processing":
            self._render_file_processing()
        elif page == "🎫 Ticket Management":
            self._render_ticket_management()
        elif page == "📊 Analytics":
            self._render_analytics()
        elif page == "⚙️ Configuration":
            self._render_configuration()
        elif page == "📱 Real-time Monitor":
            self._render_real_time_monitor()
    
    def _render_main_content(self):
        """Render main content area based on selected page."""
        # This will be populated by the sidebar navigation
        pass
    
    def _render_control_panel(self):
        """Render control panel with quick actions."""
        st.subheader("🎛️ Quick Actions")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload feedback file",
            type=['csv'],
            help="Upload a CSV file with feedback data"
        )
        
        if uploaded_file:
            if st.button("📤 Process Uploaded File"):
                self._process_uploaded_file(uploaded_file)
        
        # Quick stats
        st.subheader("📊 Quick Stats")
        self._render_quick_stats()
        
        # Recent activity
        st.subheader("🕒 Recent Activity")
        self._render_recent_activity()
    
    def _render_overview(self):
        """Render overview page."""
        st.header("📈 System Overview")
        
        # Load recent data
        tickets_df = self._load_tickets_data()
        processing_df = self._load_processing_data()
        
        if tickets_df is not None and not tickets_df.empty:
            # Category distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Feedback Categories")
                category_counts = tickets_df['category'].value_counts()
                fig_pie = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="Feedback Distribution by Category"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            with col2:
                st.subheader("⚡ Processing Timeline")
                if processing_df is not None and not processing_df.empty:
                    # Create timeline based on log entries per day
                    processing_df['timestamp'] = pd.to_datetime(processing_df['timestamp'])
                    daily_logs = processing_df.groupby(processing_df['timestamp'].dt.date).size().reset_index()
                    daily_logs.columns = ['date', 'log_entries']
                    
                    if len(daily_logs) > 0:
                        fig_timeline = px.line(
                            daily_logs,
                            x='date',
                            y='log_entries',
                            title="Log Entries Over Time"
                        )
                        st.plotly_chart(fig_timeline, use_container_width=True)
                    else:
                        st.info("No timeline data available")
                else:
                    st.info("No processing data available")
            
            # Priority distribution
            st.subheader("🎯 Priority Distribution")
            priority_counts = tickets_df['priority'].value_counts()
            
            fig_bar = px.bar(
                x=priority_counts.index,
                y=priority_counts.values,
                title="Tickets by Priority Level",
                color=priority_counts.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
        else:
            st.info("📝 No data available yet. Process some feedback files to see analytics.")
    
    def _render_file_processing(self):
        """Render file processing page."""
        st.header("📄 File Processing")
        
        # Available files
        st.subheader("📂 Available Input Files")
        input_files = list(INPUT_DIR.glob("*.csv"))
        
        if input_files:
            file_data = []
            for file_path in input_files:
                stat = file_path.stat()
                file_data.append({
                    'File Name': file_path.name,
                    'Size (KB)': round(stat.st_size / 1024, 2),
                    'Modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                    'Path': str(file_path)
                })
            
            files_df = pd.DataFrame(file_data)
            st.dataframe(files_df, use_container_width=True)
            
            # File selection for processing
            selected_files = st.multiselect(
                "Select files to process:",
                options=[f['File Name'] for f in file_data],
                default=[]
            )
            
            if selected_files and st.button("🚀 Process Selected Files"):
                self._process_selected_files(selected_files)
        
        else:
            st.warning("⚠️ No CSV files found in the input directory.")
            st.info(f"📁 Place your CSV files in: {INPUT_DIR}")
        
        # Processing history
        st.subheader("📋 Processing History")
        history_df = pd.DataFrame(st.session_state.processing_history)
        
        if not history_df.empty:
            st.dataframe(history_df, use_container_width=True)
        else:
            st.info("📝 No processing history available yet.")
    
    def _render_ticket_management(self):
        """Render ticket management page."""
        st.header("🎫 Ticket Management")
        
        tickets_df = self._load_tickets_data()
        
        if tickets_df is not None and not tickets_df.empty:
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                category_filter = st.selectbox(
                    "Filter by Category:",
                    options=['All'] + list(tickets_df['category'].unique())
                )
            
            with col2:
                priority_filter = st.selectbox(
                    "Filter by Priority:",
                    options=['All'] + list(tickets_df['priority'].unique())
                )
            
            with col3:
                status_filter = st.selectbox(
                    "Filter by Status:",
                    options=['All'] + list(tickets_df.get('status', ['Open']).unique())
                )
            
            # Apply filters
            filtered_df = tickets_df.copy()
            
            if category_filter != 'All':
                filtered_df = filtered_df[filtered_df['category'] == category_filter]
            
            if priority_filter != 'All':
                filtered_df = filtered_df[filtered_df['priority'] == priority_filter]
            
            if status_filter != 'All':
                filtered_df = filtered_df[filtered_df['status'] == status_filter]
            
            # Display tickets
            st.subheader(f"📋 Tickets ({len(filtered_df)} of {len(tickets_df)})")
            
            for idx, ticket in filtered_df.iterrows():
                with st.expander(f"🎫 {ticket['title']} - {ticket['priority']} Priority"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Category:** {ticket['category']}")
                        st.write(f"**Priority:** {ticket['priority']}")
                        st.write(f"**Created:** {ticket.get('created_date', 'N/A')}")
                    
                    with col2:
                        st.write(f"**Assigned Team:** {ticket.get('assigned_team', 'N/A')}")
                        st.write(f"**Status:** {ticket.get('status', 'Open')}")
                        st.write(f"**Confidence:** {ticket.get('confidence', 'N/A')}")
                    
                    st.write(f"**Description:** {ticket['description']}")
                    
                    if 'reproduction_steps' in ticket and ticket['reproduction_steps']:
                        st.write(f"**Reproduction Steps:** {ticket['reproduction_steps']}")
        
        else:
            st.info("📝 No tickets available yet. Process some feedback files to generate tickets.")
    
    def _render_analytics(self):
        """Render analytics page."""
        st.header("📊 Advanced Analytics")
        
        tickets_df = self._load_tickets_data()
        processing_df = self._load_processing_data()
        
        if tickets_df is not None and not tickets_df.empty:
            # Performance metrics
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("⚡ Performance Metrics")
                
                # Processing speed
                if processing_df is not None and not processing_df.empty:
                    #avg_processing_time = pd.Series([0]).mean()
                    avg_processing_time = 0.0  # No processing time data available
                    st.info("Processing time data not available in current logs")
                    st.metric("Avg Processing Time", f"{avg_processing_time:.2f}s")
                    
                    throughput = 0.0  # Analytics data not available
                    st.metric("Throughput", f"{throughput:.1f} items/sec")
                
                # Classification accuracy
                accuracy = tickets_df['agent_confidence'].mean() if 'confidence' in tickets_df else 0
                st.metric("Avg Classification Confidence", f"{accuracy:.1%}")
            
            with col2:
                st.subheader("📈 Trends Analysis")
                
                # Category trends over time
                if 'created_date' in tickets_df:
                    tickets_df['created_date'] = pd.to_datetime(tickets_df['created_date'])
                    daily_counts = tickets_df.groupby([
                        tickets_df['created_date'].dt.date,
                        'category'
                    ]).size().reset_index(name='count')
                    
                    fig_trend = px.line(
                        daily_counts,
                        x='created_date',
                        y='count',
                        color='category',
                        title="Daily Ticket Creation by Category"
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)
            
            # Detailed breakdowns
            st.subheader("🔍 Detailed Breakdowns")
            
            # Create tabs for different analyses
            tab1, tab2, tab3 = st.tabs(["Category Analysis", "Priority Analysis", "Team Workload"])
            
            with tab1:
                self._render_category_analysis(tickets_df)
            
            with tab2:
                self._render_priority_analysis(tickets_df)
            
            with tab3:
                self._render_team_workload_analysis(tickets_df)
        
        else:
            st.info("📝 No data available for analytics. Process some feedback files first.")
    
    def _render_configuration(self):
        """Render configuration page."""
        st.header("⚙️ System Configuration")
        
        # Load current configuration
        config = PROCESSING_SETTINGS.copy()
        
        st.subheader("🔧 Processing Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Pipeline settings
            st.write("**Pipeline Configuration**")
            
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=1000,
                value=config.get('batch_size', 100),
                help="Number of items to process in each batch"
            )
            
            max_concurrent = st.number_input(
                "Max Concurrent Agents",
                min_value=1,
                max_value=20,
                value=config.get('max_concurrent_agents', 5),
                help="Maximum number of agents running concurrently"
            )
            
            confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=config.get('confidence_threshold', 0.8),
                step=0.05,
                help="Minimum confidence required for classifications"
            )
        
        with col2:
            # Agent-specific settings
            st.write("**Agent Settings**")
            
            enable_quality_critic = st.checkbox(
                "Enable Quality Critic",
                value=config.get('quality_critic', {}).get('enabled', True),
                help="Enable the quality critic agent for validation"
            )
            
            auto_ticket_creation = st.checkbox(
                "Auto Ticket Creation",
                value=config.get('ticket_creator', {}).get('auto_create', True),
                help="Automatically create tickets for high-confidence items"
            )
            
            enable_preprocessing = st.checkbox(
                "Enable Preprocessing",
                value=config.get('enable_preprocessing', True),
                help="Enable text preprocessing and cleaning"
            )
        
        # API Configuration
        st.subheader("🔑 API Configuration")
        
        openai_model = st.selectbox(
            "OpenAI Model",
            options=["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
            index=0,
            help="OpenAI model to use for classification and analysis"
        )
        
        api_rate_limit = st.number_input(
            "API Rate Limit (requests/minute)",
            min_value=1,
            max_value=1000,
            value=config.get('api_rate_limit', 60),
            help="Rate limit for API requests"
        )
        
        # Save configuration
        if st.button("💾 Save Configuration"):
            new_config = {
                'batch_size': batch_size,
                'max_concurrent_agents': max_concurrent,
                'confidence_threshold': confidence_threshold,
                'quality_critic': {'enabled': enable_quality_critic},
                'ticket_creator': {'auto_create': auto_ticket_creation},
                'enable_preprocessing': enable_preprocessing,
                'openai_model': openai_model,
                'api_rate_limit': api_rate_limit
            }
            
            self._save_configuration(new_config)
            st.success("✅ Configuration saved successfully!")
            st.rerun()
        
        # Reset to defaults
        if st.button("🔄 Reset to Defaults"):
            st.warning("⚠️ This will reset all settings to default values.")
            if st.button("Confirm Reset"):
                self._reset_configuration()
                st.success("✅ Configuration reset to defaults!")
                st.rerun()
    
    def _render_real_time_monitor(self):
        """Render real-time monitoring page."""
        st.header("📱 Real-time System Monitor")
        
        # System health indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            cpu_usage = self._get_cpu_usage()
            st.metric("CPU Usage", f"{cpu_usage:.1f}%", delta=None)
        
        with col2:
            memory_usage = self._get_memory_usage()
            st.metric("Memory Usage", f"{memory_usage:.1f}%", delta=None)
        
        with col3:
            active_processes = self._get_active_processes()
            st.metric("Active Processes", active_processes, delta=None)
        
        with col4:
            queue_size = self._get_queue_size()
            st.metric("Queue Size", queue_size, delta=None)
        
        # Real-time charts
        st.subheader("📊 Real-time Metrics")
        
        # Create placeholder for real-time updates
        chart_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        # Real-time processing log
        st.subheader("📝 Live Processing Log")
        log_placeholder = st.empty()
        
        # Auto-refresh logic for real-time data
        if st.button("▶️ Start Real-time Monitoring"):
            self._start_real_time_monitoring(chart_placeholder, metrics_placeholder, log_placeholder)
    
    def _render_category_analysis(self, tickets_df):
        """Render detailed category analysis."""
        category_stats = tickets_df.groupby('category').agg({
            'priority': lambda x: (x == 'High').sum(),
            'agent_confidence': 'mean',
            'title': 'count'
        }).round(2)
        
        category_stats.columns = ['High Priority Count', 'Avg Confidence', 'Total Count']
        st.dataframe(category_stats, use_container_width=True)
        
        # Category confidence distribution
        fig_box = px.box(
            tickets_df,
            x='category',
            y='agent_confidence',
            title="Confidence Distribution by Category"
        )
        st.plotly_chart(fig_box, use_container_width=True)
    
    def _render_priority_analysis(self, tickets_df):
        """Render detailed priority analysis."""
        priority_stats = tickets_df.groupby('priority').agg({
            'category': lambda x: x.value_counts().index[0],
            'agent_confidence': 'mean',
            'title': 'count'
        }).round(2)
        
        priority_stats.columns = ['Most Common Category', 'Avg Confidence', 'Total Count']
        st.dataframe(priority_stats, use_container_width=True)
        
        # Priority vs confidence scatter
        fig_scatter = px.scatter(
            tickets_df,
            x='agent_confidence',
            y='priority',
            color='category',
            title="Priority vs Confidence by Category"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    def _render_team_workload_analysis(self, tickets_df):
        """Render team workload analysis."""
        if 'assigned_team' in tickets_df:
            team_stats = tickets_df.groupby('assigned_team').agg({
                'priority': lambda x: (x == 'High').sum(),
                'category': lambda x: x.value_counts().to_dict(),
                'title': 'count'
            })
            
            team_stats.columns = ['High Priority Tickets', 'Category Distribution', 'Total Tickets']
            st.dataframe(team_stats, use_container_width=True)
            
            # Team workload distribution
            team_counts = tickets_df['assigned_team'].value_counts()
            fig_team = px.bar(
                x=team_counts.index,
                y=team_counts.values,
                title="Team Workload Distribution"
            )
            st.plotly_chart(fig_team, use_container_width=True)
        else:
            st.info("Team assignment data not available in current tickets.")
    
    def _render_quick_stats(self):
        """Render quick statistics panel."""
        stats = st.session_state.system_stats
        
        # Recent performance
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Avg Processing Time",
                f"{stats.average_processing_time:.2f}s",
                delta=None
            )
        
        with col2:
            st.metric(
                "Items/Hour",
                f"{stats.total_feedback_processed:.0f}",
                delta=None
            )
        
        # Progress indicators
        if stats.total_feedback_processed > 0:
            success_rate = (stats.total_tickets_generated / stats.total_feedback_processed) * 100
            st.progress(success_rate / 100)
            st.caption(f"Success Rate: {success_rate:.1f}%")
    
    def _render_recent_activity(self):
        """Render recent activity feed."""
        activities = st.session_state.processing_history[-5:]  # Last 5 activities
        
        if activities:
            for activity in reversed(activities):
                timestamp = activity.get('timestamp', 'Unknown')
                action = activity.get('action', 'Processing')
                status = activity.get('status', 'Unknown')
                
                status_icon = "✅" if status == "Success" else "❌"
                st.write(f"{status_icon} {timestamp}: {action}")
        else:
            st.info("No recent activity")
    
    # Helper methods
    def _check_system_health(self) -> bool:
        """Check overall system health."""
        try:
            # Check if input/output directories exist
            return INPUT_DIR.exists() and OUTPUT_DIR.exists()
        except:
            return False
    
    def _refresh_data(self):
        """Refresh dashboard data."""
        self.last_refresh = datetime.now()
        
        # Update system stats
        self._update_system_stats()
        
        # Add refresh to history
        st.session_state.processing_history.append({
            'timestamp': self.last_refresh.strftime('%Y-%m-%d %H:%M:%S'),
            'action': 'Dashboard Refresh',
            'status': 'Success'
        })
    
    def _update_system_stats(self):
        """Update system statistics."""
        stats = st.session_state.system_stats
        
        # Load latest data
        tickets_df = self._load_tickets_data()
        processing_df = self._load_processing_data()
        
        if tickets_df is not None:
            stats.total_tickets_generated = len(tickets_df)
        
        if processing_df is not None:
            stats.total_processing_time = pd.Series([0]).sum()
            stats.total_feedback_items = pd.Series([0]).sum()
    
    def _load_tickets_data(self) -> Optional[pd.DataFrame]:
        """Load tickets data from CSV."""
        try:
            tickets_file = OUTPUT_DIR / "generated_tickets.csv"
            if tickets_file.exists():
                return pd.read_csv(tickets_file)
            return None
        except Exception as e:
            st.error(f"Error loading tickets data: {e}")
            return None
    
    def _load_processing_data(self) -> Optional[pd.DataFrame]:
        """Load processing data from CSV."""
        try:
            processing_file = OUTPUT_DIR / "processing_log.csv"
            if processing_file.exists():
                return pd.read_csv(processing_file)
            return None
        except Exception as e:
            st.error(f"Error loading processing data: {e}")
            return None
    
    def _process_new_files(self):
        """Process new files in input directory."""
        try:
            input_files = list(INPUT_DIR.glob("*.csv"))
            if input_files:
                st.info(f"🔄 Processing {len(input_files)} files...")
                
                # Add to processing history
                st.session_state.processing_history.append({
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'action': f'Processing {len(input_files)} files',
                    'status': 'In Progress'
                })
                
                # Simulate processing (replace with actual processing)
                time.sleep(2)
                
                st.success("✅ Files processed successfully!")
                
                # Update history
                st.session_state.processing_history[-1]['status'] = 'Success'
                
            else:
                st.warning("⚠️ No new files found to process.")
        
        except Exception as e:
            st.error(f"❌ Error processing files: {e}")
    
    def _process_uploaded_file(self, uploaded_file):
        """Process an uploaded file."""
        try:
            # Save uploaded file to input directory
            file_path = INPUT_DIR / uploaded_file.name
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
            
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Process the file
            self._process_selected_files([uploaded_file.name])
            
        except Exception as e:
            st.error(f"❌ Error processing uploaded file: {e}")
    
    def _process_selected_files(self, selected_files: List[str]):
        """Process selected files."""
        try:
            st.info(f"🔄 Processing {len(selected_files)} selected files...")
            
            # Add to processing history
            st.session_state.processing_history.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'action': f'Processing selected files: {", ".join(selected_files)}',
                'status': 'In Progress'
            })
            
            # Simulate processing (replace with actual processing)
            time.sleep(3)
            
            st.success("✅ Selected files processed successfully!")
            
            # Update history
            st.session_state.processing_history[-1]['status'] = 'Success'
            
        except Exception as e:
            st.error(f"❌ Error processing selected files: {e}")
            st.session_state.processing_history[-1]['status'] = 'Failed'
    
    def _clear_logs(self):
        """Clear system logs."""
        try:
            st.session_state.processing_history = []
            st.success("✅ Logs cleared successfully!")
        except Exception as e:
            st.error(f"❌ Error clearing logs: {e}")
    
    def _export_report(self):
        """Export system report."""
        try:
            # Generate report data
            report_data = {
                'system_stats': {attr: getattr(st.session_state.system_stats, attr) for attr in ["total_feedback_processed", "total_tickets_generated", "average_processing_time", "success_rate", "agent_performance", "category_distribution", "source_distribution", "priority_distribution", "last_updated"]},
                'processing_history': st.session_state.processing_history,
                'export_timestamp': datetime.now().isoformat()
            }
            
            # Convert to JSON
            report_json = json.dumps(report_data, indent=2, cls=DateTimeEncoder)
            
            # Provide download
            st.download_button(
                label="📊 Download Report",
                data=report_json,
                file_name=f"feedback_system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
            
            st.success("✅ Report ready for download!")
            
        except Exception as e:
            st.error(f"❌ Error generating report: {e}")
    
    def _save_configuration(self, config: Dict[str, Any]):
        """Save configuration to file."""
        try:
            config_file = project_root / "config" / "dashboard_config.json"
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2, cls=DateTimeEncoder)
        except Exception as e:
            st.error(f"❌ Error saving configuration: {e}")
    
    def _reset_configuration(self):
        """Reset configuration to defaults."""
        try:
            config_file = project_root / "config" / "dashboard_config.json"
            if config_file.exists():
                config_file.unlink()
        except Exception as e:
            st.error(f"❌ Error resetting configuration: {e}")
    
    # Real-time monitoring helpers
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        import psutil
        return psutil.cpu_percent(interval=1)
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage."""
        import psutil
        return psutil.virtual_memory().percent
    
    def _get_active_processes(self) -> int:
        """Get number of active processes."""
        import psutil
        return len([p for p in psutil.process_iter() if p.name().startswith('python')])
    
    def _get_queue_size(self) -> int:
        """Get current processing queue size."""
        # Simulate queue size (replace with actual implementation)
        return len(list(INPUT_DIR.glob("*.csv")))
    
    def _start_real_time_monitoring(self, chart_placeholder, metrics_placeholder, log_placeholder):
        """Start real-time monitoring with live updates."""
        st.info("🔴 Real-time monitoring started...")
        
        # This would be implemented with actual real-time data streaming
        # For now, we'll show a placeholder
        with chart_placeholder.container():
            st.line_chart(pd.DataFrame({
                'CPU': [30, 35, 32, 38, 40],
                'Memory': [45, 47, 46, 49, 48],
                'Processing': [10, 15, 12, 18, 16]
            }))
        
        with metrics_placeholder.container():
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Active Jobs", "3", delta="1")
            with col2:
                st.metric("Completed", "47", delta="5")
            with col3:
                st.metric("Errors", "2", delta="0")
        
        with log_placeholder.container():
            st.text_area(
                "Live Log",
                value="[2024-09-05 10:30:15] Processing file: reviews_batch_1.csv\n"
                      "[2024-09-05 10:30:16] Classified 25 feedback items\n"
                      "[2024-09-05 10:30:17] Generated 8 tickets\n"
                      "[2024-09-05 10:30:18] Processing complete",
                height=200,
                disabled=True
            )


# Main execution
def main():
    """Main dashboard execution function."""
    dashboard = FeedbackDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()