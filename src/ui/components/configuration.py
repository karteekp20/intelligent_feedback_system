"""
Configuration Component for the Streamlit dashboard.
Provides comprehensive system configuration management interface.
"""

import streamlit as st
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from ...core.data_models import FeedbackCategory, Priority, FeedbackSource
from ...utils.logger import get_logger
from config.settings import (
    DEFAULT_CONFIG, 
    OPENAI_MODELS, 
    PROCESSING_SETTINGS,
    OUTPUT_DIR,
    INPUT_DIR
)


class ConfigurationManager:
    """Manages system configuration with persistence and validation."""
    
    def __init__(self):
        self.logger = get_logger(self.__class__.__name__)
        self.config_file = "config/user_settings.json"
        self.backup_dir = "config/backups"
        self.current_config = self.load_config()
        
        # Ensure backup directory exists
        Path(self.backup_dir).mkdir(parents=True, exist_ok=True)
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from file or return defaults."""
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                # Merge with defaults
                config = DEFAULT_CONFIG.copy()
                config.update(user_config)
                return config
            else:
                return DEFAULT_CONFIG.copy()
        except Exception as e:
            self.logger.error(f"Error loading config: {str(e)}")
            return DEFAULT_CONFIG.copy()
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """Save configuration to file with backup."""
        try:
            # Create backup of current config
            if os.path.exists(self.config_file):
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = f"{self.backup_dir}/config_backup_{timestamp}.json"
                with open(self.config_file, 'r') as src, open(backup_path, 'w') as dst:
                    dst.write(src.read())
            
            # Save new config
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2, default=str)
            
            self.current_config = config
            self.logger.info("Configuration saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving config: {str(e)}")
            return False
    
    def reset_to_defaults(self) -> bool:
        """Reset configuration to defaults."""
        return self.save_config(DEFAULT_CONFIG.copy())
    
    def export_config(self) -> str:
        """Export current configuration as JSON string."""
        return json.dumps(self.current_config, indent=2, default=str)
    
    def import_config(self, config_json: str) -> bool:
        """Import configuration from JSON string."""
        try:
            config = json.loads(config_json)
            if self.validate_config(config):
                return self.save_config(config)
            else:
                self.logger.error("Invalid configuration format")
                return False
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON: {str(e)}")
            return False
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure and values."""
        try:
            # Check required top-level keys
            required_keys = ["openai", "processing", "ui", "agents", "data_sources"]
            for key in required_keys:
                if key not in config:
                    self.logger.error(f"Missing required config key: {key}")
                    return False
            
            # Validate OpenAI config
            if "api_key" not in config["openai"]:
                self.logger.warning("OpenAI API key not configured")
            
            if config["openai"].get("model") not in OPENAI_MODELS:
                self.logger.warning(f"Unknown OpenAI model: {config['openai'].get('model')}")
            
            # Validate processing settings
            processing = config["processing"]
            if not isinstance(processing.get("batch_size"), int) or processing.get("batch_size") <= 0:
                self.logger.error("Invalid batch_size in processing config")
                return False
            
            if not isinstance(processing.get("max_concurrent_agents"), int) or processing.get("max_concurrent_agents") <= 0:
                self.logger.error("Invalid max_concurrent_agents in processing config")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating config: {str(e)}")
            return False


def render_configuration_ui():
    """Render the configuration management interface."""
    st.header("⚙️ System Configuration")
    
    # Initialize configuration manager
    if 'config_manager' not in st.session_state:
        st.session_state.config_manager = ConfigurationManager()
    
    config_manager = st.session_state.config_manager
    config = config_manager.current_config
    
    # Configuration tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🤖 AI & Models", 
        "⚡ Processing", 
        "🎨 User Interface", 
        "🔧 Agents", 
        "📊 Data Sources",
        "💾 Management"
    ])
    
    # AI & Models Configuration
    with tab1:
        st.subheader("AI Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**OpenAI Settings**")
            
            # API Key configuration
            current_api_key = config.get("openai", {}).get("api_key", "")
            api_key_display = "***" + current_api_key[-4:] if len(current_api_key) > 4 else ""
            
            st.text_input(
                "API Key", 
                value=api_key_display,
                help="Your OpenAI API key",
                disabled=True,
                key="api_key_display"
            )
            
            if st.button("🔑 Update API Key"):
                with st.expander("Enter New API Key", expanded=True):
                    new_api_key = st.text_input(
                        "New API Key",
                        type="password",
                        help="Enter your OpenAI API key"
                    )
                    
                    col_a, col_b = st.columns(2)
                    with col_a:
                        if st.button("Save API Key"):
                            if new_api_key:
                                config["openai"]["api_key"] = new_api_key
                                if config_manager.save_config(config):
                                    st.success("API Key updated successfully!")
                                    st.rerun()
                                else:
                                    st.error("Failed to save API Key")
                            else:
                                st.warning("Please enter an API key")
                    
                    with col_b:
                        if st.button("Cancel"):
                            st.rerun()
            
            # Model selection
            current_model = config.get("openai", {}).get("model", "gpt-3.5-turbo")
            model = st.selectbox(
                "Model",
                options=list(OPENAI_MODELS.keys()),
                index=list(OPENAI_MODELS.keys()).index(current_model) if current_model in OPENAI_MODELS else 0,
                help="Select the OpenAI model to use"
            )
            config["openai"]["model"] = model
            
            # Model parameters
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=2.0,
                value=float(config.get("openai", {}).get("temperature", 0.7)),
                step=0.1,
                help="Controls randomness in responses"
            )
            config["openai"]["temperature"] = temperature
            
            max_tokens = st.number_input(
                "Max Tokens",
                min_value=100,
                max_value=4000,
                value=int(config.get("openai", {}).get("max_tokens", 1000)),
                help="Maximum tokens per response"
            )
            config["openai"]["max_tokens"] = max_tokens
        
        with col2:
            st.write("**Rate Limiting**")
            
            requests_per_minute = st.number_input(
                "Requests per Minute",
                min_value=1,
                max_value=100,
                value=int(config.get("openai", {}).get("rate_limit", {}).get("requests_per_minute", 20)),
                help="API rate limit"
            )
            
            tokens_per_minute = st.number_input(
                "Tokens per Minute",
                min_value=1000,
                max_value=100000,
                value=int(config.get("openai", {}).get("rate_limit", {}).get("tokens_per_minute", 40000)),
                help="Token rate limit"
            )
            
            config["openai"]["rate_limit"] = {
                "requests_per_minute": requests_per_minute,
                "tokens_per_minute": tokens_per_minute
            }
            
            st.write("**Retry Configuration**")
            
            max_retries = st.number_input(
                "Max Retries",
                min_value=0,
                max_value=10,
                value=int(config.get("openai", {}).get("max_retries", 3)),
                help="Number of retry attempts"
            )
            config["openai"]["max_retries"] = max_retries
            
            retry_delay = st.number_input(
                "Retry Delay (seconds)",
                min_value=1,
                max_value=60,
                value=int(config.get("openai", {}).get("retry_delay", 5)),
                help="Delay between retries"
            )
            config["openai"]["retry_delay"] = retry_delay
    
    # Processing Configuration
    with tab2:
        st.subheader("Processing Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Performance Settings**")
            
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=1000,
                value=int(config.get("processing", {}).get("batch_size", 50)),
                help="Number of items processed in each batch"
            )
            config["processing"]["batch_size"] = batch_size
            
            max_concurrent_agents = st.number_input(
                "Max Concurrent Agents",
                min_value=1,
                max_value=20,
                value=int(config.get("processing", {}).get("max_concurrent_agents", 5)),
                help="Maximum number of agents running simultaneously"
            )
            config["processing"]["max_concurrent_agents"] = max_concurrent_agents
            
            timeout_seconds = st.number_input(
                "Agent Timeout (seconds)",
                min_value=30,
                max_value=600,
                value=int(config.get("processing", {}).get("timeout_seconds", 120)),
                help="Timeout for individual agent operations"
            )
            config["processing"]["timeout_seconds"] = timeout_seconds
            
            enable_caching = st.checkbox(
                "Enable Caching",
                value=bool(config.get("processing", {}).get("enable_caching", True)),
                help="Cache results to improve performance"
            )
            config["processing"]["enable_caching"] = enable_caching
        
        with col2:
            st.write("**Quality Settings**")
            
            enable_validation = st.checkbox(
                "Enable Data Validation",
                value=bool(config.get("processing", {}).get("enable_validation", True)),
                help="Validate data at each processing step"
            )
            config["processing"]["enable_validation"] = enable_validation
            
            min_confidence_threshold = st.slider(
                "Minimum Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(config.get("processing", {}).get("min_confidence_threshold", 0.7)),
                step=0.05,
                help="Minimum confidence required for automatic processing"
            )
            config["processing"]["min_confidence_threshold"] = min_confidence_threshold
            
            auto_retry_failed = st.checkbox(
                "Auto-retry Failed Items",
                value=bool(config.get("processing", {}).get("auto_retry_failed", True)),
                help="Automatically retry failed processing items"
            )
            config["processing"]["auto_retry_failed"] = auto_retry_failed
            
            max_retries = st.number_input(
                "Max Processing Retries",
                min_value=0,
                max_value=5,
                value=int(config.get("processing", {}).get("max_retries", 2)),
                help="Maximum retry attempts for failed items"
            )
            config["processing"]["max_retries"] = max_retries
    
    # User Interface Configuration
    with tab3:
        st.subheader("User Interface Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Display Settings**")
            
            theme = st.selectbox(
                "Theme",
                options=["light", "dark", "auto"],
                index=["light", "dark", "auto"].index(config.get("ui", {}).get("theme", "auto")),
                help="UI theme preference"
            )
            config["ui"]["theme"] = theme
            
            items_per_page = st.number_input(
                "Items per Page",
                min_value=10,
                max_value=100,
                value=int(config.get("ui", {}).get("items_per_page", 25)),
                help="Number of items displayed per page"
            )
            config["ui"]["items_per_page"] = items_per_page
            
            auto_refresh_interval = st.number_input(
                "Auto-refresh Interval (seconds)",
                min_value=0,
                max_value=300,
                value=int(config.get("ui", {}).get("auto_refresh_interval", 30)),
                help="Auto-refresh interval (0 to disable)"
            )
            config["ui"]["auto_refresh_interval"] = auto_refresh_interval
            
            show_confidence_scores = st.checkbox(
                "Show Confidence Scores",
                value=bool(config.get("ui", {}).get("show_confidence_scores", True)),
                help="Display confidence scores in results"
            )
            config["ui"]["show_confidence_scores"] = show_confidence_scores
        
        with col2:
            st.write("**Chart Settings**")
            
            default_chart_height = st.number_input(
                "Default Chart Height",
                min_value=200,
                max_value=800,
                value=int(config.get("ui", {}).get("chart_settings", {}).get("default_height", 400)),
                help="Default height for charts"
            )
            
            color_scheme = st.selectbox(
                "Color Scheme",
                options=["plotly", "viridis", "plasma", "blues", "reds"],
                index=["plotly", "viridis", "plasma", "blues", "reds"].index(
                    config.get("ui", {}).get("chart_settings", {}).get("color_scheme", "plotly")
                ),
                help="Color scheme for charts"
            )
            
            show_grid = st.checkbox(
                "Show Grid Lines",
                value=bool(config.get("ui", {}).get("chart_settings", {}).get("show_grid", True)),
                help="Show grid lines in charts"
            )
            
            config["ui"]["chart_settings"] = {
                "default_height": default_chart_height,
                "color_scheme": color_scheme,
                "show_grid": show_grid
            }
            
            st.write("**Notification Settings**")
            
            enable_notifications = st.checkbox(
                "Enable Notifications",
                value=bool(config.get("ui", {}).get("notifications", {}).get("enabled", True)),
                help="Show system notifications"
            )
            
            notification_duration = st.number_input(
                "Notification Duration (seconds)",
                min_value=1,
                max_value=30,
                value=int(config.get("ui", {}).get("notifications", {}).get("duration", 5)),
                help="How long notifications are displayed"
            )
            
            config["ui"]["notifications"] = {
                "enabled": enable_notifications,
                "duration": notification_duration
            }
    
    # Agents Configuration
    with tab4:
        st.subheader("Agent Configuration")
        
        # Agent-specific settings
        agent_tabs = st.tabs([
            "CSV Reader", "Classifier", "Bug Analysis", 
            "Feature Extractor", "Ticket Creator", "Quality Critic"
        ])
        
        agent_configs = config.get("agents", {})
        
        with agent_tabs[0]:  # CSV Reader
            st.write("**CSV Reader Agent**")
            
            csv_batch_size = st.number_input(
                "CSV Batch Size",
                min_value=100,
                max_value=10000,
                value=int(agent_configs.get("csv_reader", {}).get("batch_size", 1000)),
                help="Number of CSV rows processed at once"
            )
            
            validate_data = st.checkbox(
                "Validate CSV Data",
                value=bool(agent_configs.get("csv_reader", {}).get("validate_data", True)),
                help="Validate CSV data during reading"
            )
            
            handle_encoding_errors = st.checkbox(
                "Handle Encoding Errors",
                value=bool(agent_configs.get("csv_reader", {}).get("handle_encoding_errors", True)),
                help="Automatically handle file encoding issues"
            )
            
            agent_configs["csv_reader"] = {
                "batch_size": csv_batch_size,
                "validate_data": validate_data,
                "handle_encoding_errors": handle_encoding_errors
            }
        
        with agent_tabs[1]:  # Classifier
            st.write("**Feedback Classifier Agent**")
            
            classification_confidence = st.slider(
                "Classification Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=float(agent_configs.get("feedback_classifier", {}).get("confidence_threshold", 0.8)),
                step=0.05,
                help="Minimum confidence for classification"
            )
            
            use_few_shot_learning = st.checkbox(
                "Use Few-shot Learning",
                value=bool(agent_configs.get("feedback_classifier", {}).get("use_few_shot", True)),
                help="Use examples to improve classification"
            )
            
            enable_category_weights = st.checkbox(
                "Enable Category Weights",
                value=bool(agent_configs.get("feedback_classifier", {}).get("enable_weights", False)),
                help="Apply weights to different categories"
            )
            
            agent_configs["feedback_classifier"] = {
                "confidence_threshold": classification_confidence,
                "use_few_shot": use_few_shot_learning,
                "enable_weights": enable_category_weights
            }
        
        with agent_tabs[2]:  # Bug Analysis
            st.write("**Bug Analysis Agent**")
            
            extract_stack_traces = st.checkbox(
                "Extract Stack Traces",
                value=bool(agent_configs.get("bug_analysis", {}).get("extract_stack_traces", True)),
                help="Automatically extract stack traces from bug reports"
            )
            
            analyze_error_patterns = st.checkbox(
                "Analyze Error Patterns",
                value=bool(agent_configs.get("bug_analysis", {}).get("analyze_patterns", True)),
                help="Look for common error patterns"
            )
            
            severity_auto_assignment = st.checkbox(
                "Auto-assign Severity",
                value=bool(agent_configs.get("bug_analysis", {}).get("auto_severity", True)),
                help="Automatically assign bug severity"
            )
            
            agent_configs["bug_analysis"] = {
                "extract_stack_traces": extract_stack_traces,
                "analyze_patterns": analyze_error_patterns,
                "auto_severity": severity_auto_assignment
            }
        
        with agent_tabs[3]:  # Feature Extractor
            st.write("**Feature Extractor Agent**")
            
            extract_use_cases = st.checkbox(
                "Extract Use Cases",
                value=bool(agent_configs.get("feature_extractor", {}).get("extract_use_cases", True)),
                help="Extract use cases from feature requests"
            )
            
            estimate_complexity = st.checkbox(
                "Estimate Complexity",
                value=bool(agent_configs.get("feature_extractor", {}).get("estimate_complexity", True)),
                help="Automatically estimate feature complexity"
            )
            
            identify_dependencies = st.checkbox(
                "Identify Dependencies",
                value=bool(agent_configs.get("feature_extractor", {}).get("identify_dependencies", False)),
                help="Identify potential feature dependencies"
            )
            
            agent_configs["feature_extractor"] = {
                "extract_use_cases": extract_use_cases,
                "estimate_complexity": estimate_complexity,
                "identify_dependencies": identify_dependencies
            }
        
        with agent_tabs[4]:  # Ticket Creator
            st.write("**Ticket Creator Agent**")
            
            use_templates = st.checkbox(
                "Use Templates",
                value=bool(agent_configs.get("ticket_creator", {}).get("use_templates", True)),
                help="Use predefined templates for tickets"
            )
            
            auto_assign_teams = st.checkbox(
                "Auto-assign Teams",
                value=bool(agent_configs.get("ticket_creator", {}).get("auto_assign_teams", True)),
                help="Automatically assign tickets to teams"
            )
            
            generate_acceptance_criteria = st.checkbox(
                "Generate Acceptance Criteria",
                value=bool(agent_configs.get("ticket_creator", {}).get("generate_acceptance_criteria", True)),
                help="Automatically generate acceptance criteria"
            )
            
            agent_configs["ticket_creator"] = {
                "use_templates": use_templates,
                "auto_assign_teams": auto_assign_teams,
                "generate_acceptance_criteria": generate_acceptance_criteria
            }
        
        with agent_tabs[5]:  # Quality Critic
            st.write("**Quality Critic Agent**")
            
            min_approval_score = st.slider(
                "Minimum Approval Score",
                min_value=0.0,
                max_value=100.0,
                value=float(agent_configs.get("quality_critic", {}).get("min_approval_score", 75.0)),
                step=5.0,
                help="Minimum quality score for ticket approval"
            )
            
            detailed_analysis = st.checkbox(
                "Detailed Analysis",
                value=bool(agent_configs.get("quality_critic", {}).get("detailed_analysis", True)),
                help="Perform detailed quality analysis"
            )
            
            auto_fix_enabled = st.checkbox(
                "Auto-fix Issues",
                value=bool(agent_configs.get("quality_critic", {}).get("auto_fix_enabled", False)),
                help="Automatically fix minor quality issues"
            )
            
            strict_mode = st.checkbox(
                "Strict Mode",
                value=bool(agent_configs.get("quality_critic", {}).get("strict_mode", False)),
                help="Apply stricter quality criteria"
            )
            
            agent_configs["quality_critic"] = {
                "min_approval_score": min_approval_score,
                "detailed_analysis": detailed_analysis,
                "auto_fix_enabled": auto_fix_enabled,
                "strict_mode": strict_mode
            }
        
        config["agents"] = agent_configs
    
    # Data Sources Configuration
    with tab5:
        st.subheader("Data Sources Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**File Paths**")
            
            # Input file paths
            app_store_file = st.text_input(
                "App Store Reviews File",
                value=config.get("data_sources", {}).get("app_store_reviews", "data/input/app_store_reviews.csv"),
                help="Path to app store reviews CSV file"
            )
            
            support_emails_file = st.text_input(
                "Support Emails File",
                value=config.get("data_sources", {}).get("support_emails", "data/input/support_emails.csv"),
                help="Path to support emails CSV file"
            )
            
            expected_classifications_file = st.text_input(
                "Expected Classifications File",
                value=config.get("data_sources", {}).get("expected_classifications", "data/input/expected_classifications.csv"),
                help="Path to expected classifications CSV file"
            )
            
            # Output paths
            output_tickets_file = st.text_input(
                "Output Tickets File",
                value=config.get("data_sources", {}).get("output_tickets", "data/output/generated_tickets.csv"),
                help="Path for generated tickets output"
            )
            
            config["data_sources"] = {
                "app_store_reviews": app_store_file,
                "support_emails": support_emails_file,
                "expected_classifications": expected_classifications_file,
                "output_tickets": output_tickets_file
            }
        
        with col2:
            st.write("**Data Processing Options**")
            
            # File format settings
            csv_delimiter = st.selectbox(
                "CSV Delimiter",
                options=[",", ";", "\t", "|"],
                index=[",", ";", "\t", "|"].index(config.get("data_sources", {}).get("csv_delimiter", ",")),
                help="CSV file delimiter"
            )
            
            csv_encoding = st.selectbox(
                "CSV Encoding",
                options=["utf-8", "utf-8-sig", "latin-1", "cp1252"],
                index=["utf-8", "utf-8-sig", "latin-1", "cp1252"].index(
                    config.get("data_sources", {}).get("csv_encoding", "utf-8")
                ),
                help="CSV file encoding"
            )
            
            skip_empty_lines = st.checkbox(
                "Skip Empty Lines",
                value=bool(config.get("data_sources", {}).get("skip_empty_lines", True)),
                help="Skip empty lines in CSV files"
            )
            
            max_file_size_mb = st.number_input(
                "Max File Size (MB)",
                min_value=1,
                max_value=1000,
                value=int(config.get("data_sources", {}).get("max_file_size_mb", 100)),
                help="Maximum file size for processing"
            )
            
            config["data_sources"].update({
                "csv_delimiter": csv_delimiter,
                "csv_encoding": csv_encoding,
                "skip_empty_lines": skip_empty_lines,
                "max_file_size_mb": max_file_size_mb
            })
            
            # Data validation settings
            st.write("**Data Validation**")
            
            validate_email_format = st.checkbox(
                "Validate Email Format",
                value=bool(config.get("data_sources", {}).get("validate_emails", True)),
                help="Validate email address format"
            )
            
            min_content_length = st.number_input(
                "Minimum Content Length",
                min_value=1,
                max_value=1000,
                value=int(config.get("data_sources", {}).get("min_content_length", 10)),
                help="Minimum length for feedback content"
            )
            
            max_content_length = st.number_input(
                "Maximum Content Length",
                min_value=100,
                max_value=50000,
                value=int(config.get("data_sources", {}).get("max_content_length", 10000)),
                help="Maximum length for feedback content"
            )
            
            config["data_sources"].update({
                "validate_emails": validate_email_format,
                "min_content_length": min_content_length,
                "max_content_length": max_content_length
            })
    
    # Management Tab
    with tab6:
        st.subheader("Configuration Management")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Save & Load**")
            
            if st.button("💾 Save Configuration", type="primary"):
                if config_manager.save_config(config):
                    st.success("Configuration saved successfully!")
                    st.balloons()
                else:
                    st.error("Failed to save configuration")
            
            if st.button("🔄 Reset to Defaults"):
                if st.confirm("Are you sure you want to reset all settings to defaults?"):
                    if config_manager.reset_to_defaults():
                        st.success("Configuration reset to defaults!")
                        st.rerun()
                    else:
                        st.error("Failed to reset configuration")
            
            if st.button("📤 Export Configuration"):
                config_json = config_manager.export_config()
                st.download_button(
                    label="Download Config File",
                    data=config_json,
                    file_name=f"config_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            st.write("**Import & Restore**")
            
            uploaded_config = st.file_uploader(
                "Import Configuration",
                type="json",
                help="Upload a configuration JSON file"
            )
            
            if uploaded_config is not None:
                try:
                    config_content = uploaded_config.read().decode('utf-8')
                    if st.button("📥 Import Configuration"):
                        if config_manager.import_config(config_content):
                            st.success("Configuration imported successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to import configuration")
                except Exception as e:
                    st.error(f"Error reading config file: {str(e)}")
            
            # Backup management
            st.write("**Backup Management**")
            backup_files = list(Path(config_manager.backup_dir).glob("*.json"))
            
            if backup_files:
                backup_options = [f.name for f in sorted(backup_files, reverse=True)]
                selected_backup = st.selectbox(
                    "Available Backups",
                    options=backup_options,
                    help="Select a backup to restore"
                )
                
                if st.button("🔙 Restore from Backup"):
                    backup_path = Path(config_manager.backup_dir) / selected_backup
                    try:
                        with open(backup_path, 'r') as f:
                            backup_config = json.load(f)
                        if config_manager.save_config(backup_config):
                            st.success(f"Configuration restored from {selected_backup}!")
                            st.rerun()
                        else:
                            st.error("Failed to restore configuration")
                    except Exception as e:
                        st.error(f"Error restoring backup: {str(e)}")
            else:
                st.info("No backups available")
        
        # Configuration validation
        st.write("**Configuration Status**")
        
        # Validate current configuration
        validation_status = config_manager.validate_config(config)
        
        if validation_status:
            st.success("✅ Configuration is valid")
        else:
            st.error("❌ Configuration has issues")
        
        # Display configuration summary
        with st.expander("Configuration Summary"):
            st.json(config)
        
        # System information
        st.write("**System Information**")
        
        system_info = {
            "Config File": config_manager.config_file,
            "Last Modified": datetime.fromtimestamp(os.path.getmtime(config_manager.config_file)).isoformat() if os.path.exists(config_manager.config_file) else "Not found",
            "File Size": f"{os.path.getsize(config_manager.config_file) / 1024:.2f} KB" if os.path.exists(config_manager.config_file) else "N/A",
            "Backup Count": len(backup_files) if 'backup_files' in locals() else 0
        }
        
        for key, value in system_info.items():
            st.text(f"{key}: {value}")
    
    # Auto-save functionality
    if st.button("🔄 Apply Changes", type="secondary"):
        if config_manager.save_config(config):
            st.success("Changes applied successfully!")
        else:
            st.error("Failed to apply changes")
    
    # Configuration change detection
    if config != config_manager.current_config:
        st.warning("⚠️ You have unsaved changes. Click 'Save Configuration' to apply them.")


# Main function for standalone testing
def main():
    """Main function for testing the configuration component."""
    st.set_page_config(
        page_title="Configuration Manager",
        page_icon="⚙️",
        layout="wide"
    )
    
    render_configuration_ui()


if __name__ == "__main__":
    main()