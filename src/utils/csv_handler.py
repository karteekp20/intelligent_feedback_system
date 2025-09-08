"""
CSV handling utilities for the Intelligent Feedback Analysis System.
"""

import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional
import csv
from datetime import datetime

from .logger import get_logger


class CSVHandler:
    """Handles CSV file operations with error handling and validation."""
    
    def __init__(self, encoding: str = "utf-8"):
        """
        Initialize CSV handler.
        
        Args:
            encoding: Default encoding for CSV files
        """
        self.encoding = encoding
        self.logger = get_logger("csv_handler")
    
    def read_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """
        Read CSV file with error handling.
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional parameters for pd.read_csv
            
        Returns:
            DataFrame containing CSV data
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If CSV is malformed
        """
        try:
            if not Path(file_path).exists():
                raise FileNotFoundError(f"CSV file not found: {file_path}")
            
            # Default parameters
            default_params = {
                "encoding": self.encoding,
                "skipinitialspace": True,
                "quoting": csv.QUOTE_MINIMAL
            }
            
            # Merge with provided parameters
            params = {**default_params, **kwargs}
            
            # Read CSV
            df = pd.read_csv(file_path, **params)
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Clean string data
            string_columns = df.select_dtypes(include=['object']).columns
            for col in string_columns:
                df[col] = df[col].astype(str).str.strip()
                df[col] = df[col].replace('nan', '')
                df[col] = df[col].replace('None', '')
            
            self.logger.info(f"Successfully read CSV: {file_path} ({len(df)} rows)")
            return df
            
        except Exception as e:
            self.logger.error(f"Error reading CSV {file_path}: {str(e)}")
            raise
    
    def write_csv(self, df: pd.DataFrame, file_path: Path, **kwargs) -> bool:
        """
        Write DataFrame to CSV file.
        
        Args:
            df: DataFrame to write
            file_path: Output file path
            **kwargs: Additional parameters for pd.to_csv
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure directory exists
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Default parameters
            default_params = {
                "index": False,
                "encoding": self.encoding,
                "quoting": csv.QUOTE_MINIMAL
            }
            
            # Merge with provided parameters
            params = {**default_params, **kwargs}
            
            # Write CSV
            df.to_csv(file_path, **params)
            
            self.logger.info(f"Successfully wrote CSV: {file_path} ({len(df)} rows)")
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing CSV {file_path}: {str(e)}")
            return False
    
    def append_to_csv(self, data: List[Dict[str, Any]], file_path: Path, **kwargs) -> bool:
        """
        Append data to existing CSV file.
        
        Args:
            data: List of dictionaries to append
            file_path: CSV file path
            **kwargs: Additional parameters
            
        Returns:
            True if successful, False otherwise
        """
        try:
            df_new = pd.DataFrame(data)
            
            if Path(file_path).exists():
                # Read existing data and append
                df_existing = self.read_csv(file_path)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
            else:
                # Create new file
                df_combined = df_new
            
            return self.write_csv(df_combined, file_path, **kwargs)
            
        except Exception as e:
            self.logger.error(f"Error appending to CSV {file_path}: {str(e)}")
            return False
    
    def validate_csv_schema(self, file_path: Path, expected_columns: List[str]) -> Dict[str, Any]:
        """
        Validate CSV file schema.
        
        Args:
            file_path: Path to CSV file
            expected_columns: List of expected column names
            
        Returns:
            Validation results dictionary
        """
        try:
            if not Path(file_path).exists():
                return {"valid": False, "error": "File not found"}
            
            # Read only headers
            df_headers = pd.read_csv(file_path, nrows=0)
            actual_columns = list(df_headers.columns)
            
            missing_columns = set(expected_columns) - set(actual_columns)
            extra_columns = set(actual_columns) - set(expected_columns)
            
            validation_result = {
                "valid": len(missing_columns) == 0,
                "actual_columns": actual_columns,
                "expected_columns": expected_columns,
                "missing_columns": list(missing_columns),
                "extra_columns": list(extra_columns)
            }
            
            if missing_columns:
                validation_result["error"] = f"Missing required columns: {missing_columns}"
            
            return validation_result
            
        except Exception as e:
            return {"valid": False, "error": f"Validation failed: {str(e)}"}
    
    def get_csv_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Get information about CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Information dictionary
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return {"error": "File not found"}
            
            # File stats
            stat = path.stat()
            file_info = {
                "file_name": path.name,
                "file_size": stat.st_size,
                "created_time": datetime.fromtimestamp(stat.st_ctime),
                "modified_time": datetime.fromtimestamp(stat.st_mtime)
            }
            
            # CSV content info
            try:
                df = self.read_csv(file_path)
                file_info.update({
                    "row_count": len(df),
                    "column_count": len(df.columns),
                    "columns": list(df.columns),
                    "data_types": df.dtypes.to_dict(),
                    "null_counts": df.isnull().sum().to_dict(),
                    "memory_usage": df.memory_usage(deep=True).sum()
                })
            except Exception as e:
                file_info["csv_error"] = str(e)
            
            return file_info
            
        except Exception as e:
            return {"error": f"Failed to get file info: {str(e)}"}
    
    def clean_csv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean CSV data by removing/fixing common issues.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        df_clean = df.copy()
        
        # Remove completely empty rows
        df_clean = df_clean.dropna(how='all')
        
        # Clean string columns
        string_columns = df_clean.select_dtypes(include=['object']).columns
        for col in string_columns:
            # Strip whitespace
            df_clean[col] = df_clean[col].astype(str).str.strip()
            
            # Replace various null representations
            null_values = ['nan', 'NaN', 'None', 'null', 'NULL', '']
            df_clean[col] = df_clean[col].replace(null_values, pd.NA)
        
        # Remove duplicate rows
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        duplicates_removed = initial_count - len(df_clean)
        
        if duplicates_removed > 0:
            self.logger.info(f"Removed {duplicates_removed} duplicate rows")
        
        return df_clean
    
    def convert_datatypes(self, df: pd.DataFrame, type_mapping: Dict[str, str]) -> pd.DataFrame:
        """
        Convert column data types.
        
        Args:
            df: DataFrame to convert
            type_mapping: Dictionary mapping column names to target types
            
        Returns:
            DataFrame with converted types
        """
        df_converted = df.copy()
        
        for column, target_type in type_mapping.items():
            if column in df_converted.columns:
                try:
                    if target_type == 'datetime':
                        df_converted[column] = pd.to_datetime(df_converted[column], errors='coerce')
                    elif target_type == 'int':
                        df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce').astype('Int64')
                    elif target_type == 'float':
                        df_converted[column] = pd.to_numeric(df_converted[column], errors='coerce')
                    elif target_type == 'bool':
                        df_converted[column] = df_converted[column].astype('bool')
                    elif target_type == 'str':
                        df_converted[column] = df_converted[column].astype('string')
                    
                    self.logger.info(f"Converted column {column} to {target_type}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to convert column {column} to {target_type}: {e}")
        
        return df_converted
    
    def split_csv_file(self, file_path: Path, chunk_size: int, output_dir: Path) -> List[Path]:
        """
        Split large CSV file into smaller chunks.
        
        Args:
            file_path: Path to large CSV file
            chunk_size: Number of rows per chunk
            output_dir: Directory for output files
            
        Returns:
            List of paths to chunk files
        """
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            file_stem = Path(file_path).stem
            
            chunk_files = []
            
            # Read CSV in chunks
            for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
                chunk_file = output_dir / f"{file_stem}_chunk_{i+1}.csv"
                self.write_csv(chunk, chunk_file)
                chunk_files.append(chunk_file)
            
            self.logger.info(f"Split {file_path} into {len(chunk_files)} chunks")
            return chunk_files
            
        except Exception as e:
            self.logger.error(f"Error splitting CSV file: {e}")
            return []
    
    def merge_csv_files(self, file_paths: List[Path], output_path: Path) -> bool:
        """
        Merge multiple CSV files into one.
        
        Args:
            file_paths: List of CSV file paths to merge
            output_path: Output file path
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not file_paths:
                self.logger.warning("No files provided for merging")
                return False
            
            # Read and combine all CSV files
            dataframes = []
            for file_path in file_paths:
                if Path(file_path).exists():
                    df = self.read_csv(file_path)
                    dataframes.append(df)
                else:
                    self.logger.warning(f"File not found for merging: {file_path}")
            
            if not dataframes:
                self.logger.error("No valid files found for merging")
                return False
            
            # Combine all dataframes
            merged_df = pd.concat(dataframes, ignore_index=True)
            
            # Remove duplicates
            merged_df = merged_df.drop_duplicates()
            
            # Write merged file
            success = self.write_csv(merged_df, output_path)
            
            if success:
                self.logger.info(f"Successfully merged {len(file_paths)} files into {output_path}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error merging CSV files: {e}")
            return False