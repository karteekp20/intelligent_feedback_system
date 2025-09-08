#!/usr/bin/env python3
"""
Performance Testing Suite
Intelligent User Feedback Analysis and Action System
"""

import asyncio
import time
import statistics
import json
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import sys

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.core.pipeline import FeedbackProcessingPipeline
from src.core.data_models import FeedbackItem, PipelineResult
from src.utils.logger import get_logger, PerformanceLogger
from src.utils.csv_handler import CSVHandler
from config.settings import OUTPUT_DIR, INPUT_DIR, PROCESSING_SETTINGS

logger = get_logger(__name__)

@dataclass
class PerformanceMetrics:
    """Performance metrics data structure."""
    test_name: str
    total_time: float
    items_processed: int
    throughput: float  # items per second
    memory_usage_mb: float
    cpu_usage_percent: float
    success_rate: float
    error_count: int
    agent_timings: Dict[str, float]
    classification_accuracy: float
    confidence_scores: List[float]
    timestamp: str

@dataclass
class LoadTestResult:
    """Load test result data structure."""
    concurrent_users: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time: float
    min_response_time: float
    max_response_time: float
    percentile_95: float
    percentile_99: float
    throughput: float
    error_rate: float
    timestamp: str

class PerformanceTestSuite:
    """Comprehensive performance testing suite."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or OUTPUT_DIR / "performance_tests"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.performance_logger = PerformanceLogger()
        self.test_results: List[PerformanceMetrics] = []
        self.load_test_results: List[LoadTestResult] = []
        
        # Test configurations
        self.test_configs = {
            'small_load': {'file_size': 100, 'concurrent_agents': 2},
            'medium_load': {'file_size': 500, 'concurrent_agents': 5},
            'large_load': {'file_size': 1000, 'concurrent_agents': 8},
            'stress_test': {'file_size': 2000, 'concurrent_agents': 12},
            'endurance_test': {'file_size': 500, 'duration_minutes': 30}
        }
    
    async def run_comprehensive_tests(self) -> Dict[str, Any]:
        """Run comprehensive performance test suite."""
        logger.info("🚀 Starting comprehensive performance test suite...")
        
        results = {
            'test_suite_start': datetime.now().isoformat(),
            'performance_tests': {},
            'load_tests': {},
            'benchmark_results': {},
            'recommendations': []
        }
        
        try:
            # 1. Basic performance tests
            logger.info("📊 Running basic performance tests...")
            for test_name, config in self.test_configs.items():
                if test_name != 'endurance_test':  # Skip endurance for basic tests
                    result = await self.run_performance_test(test_name, config)
                    results['performance_tests'][test_name] = asdict(result)
            
            # 2. Load testing
            logger.info("⚡ Running load tests...")
            load_scenarios = [1, 5, 10, 20, 50]
            for concurrent_users in load_scenarios:
                load_result = await self.run_load_test(concurrent_users, requests_per_user=10)
                results['load_tests'][f'{concurrent_users}_users'] = asdict(load_result)
            
            # 3. Memory and CPU benchmarks
            logger.info("🧠 Running memory and CPU benchmarks...")
            benchmark_result = await self.run_resource_benchmark()
            results['benchmark_results'] = benchmark_result
            
            # 4. Accuracy testing
            logger.info("🎯 Running accuracy tests...")
            accuracy_result = await self.run_accuracy_test()
            results['accuracy_test'] = accuracy_result
            
            # 5. Generate recommendations
            results['recommendations'] = self.generate_performance_recommendations()
            
            # 6. Save comprehensive report
            await self.save_comprehensive_report(results)
            
            logger.info("✅ Comprehensive test suite completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"❌ Test suite failed: {e}")
            results['error'] = str(e)
            return results
    
    async def run_performance_test(self, test_name: str, config: Dict[str, Any]) -> PerformanceMetrics:
        """Run a single performance test."""
        logger.info(f"🔄 Running performance test: {test_name}")
        
        start_time = time.time()
        
        # Create test data
        test_file = await self.create_test_data(
            filename=f"test_{test_name}.csv",
            num_items=config['file_size']
        )
        
        # Initialize pipeline with test configuration
        pipeline_config = PROCESSING_SETTINGS.copy()
        pipeline_config['max_concurrent_agents'] = config['concurrent_agents']
        
        pipeline = FeedbackProcessingPipeline(pipeline_config)
        await pipeline.initialize()
        
        # Monitor resource usage
        initial_memory = self._get_memory_usage()
        
        try:
            # Process test file
            with self.performance_logger.measure(f"test_{test_name}"):
                result = await pipeline.process_file(test_file)
            
            # Calculate metrics
            end_time = time.time()
            total_time = end_time - start_time
            
            final_memory = self._get_memory_usage()
            memory_usage = final_memory - initial_memory
            
            # Extract performance data
            throughput = len(result.processed_items) / total_time if total_time > 0 else 0
            success_rate = 1.0 if result.success else 0.0
            
            # Get agent timings
            agent_timings = {}
            for agent_name, agent_result in result.agent_results.items():
                if hasattr(agent_result, 'processing_time'):
                    agent_timings[agent_name] = agent_result.processing_time
            
            # Calculate classification accuracy and confidence
            confidence_scores = []
            if hasattr(result, 'processed_items'):
                for item in result.processed_items:
                    if hasattr(item, 'confidence'):
                        confidence_scores.append(item.confidence)
            
            avg_confidence = statistics.mean(confidence_scores) if confidence_scores else 0.0
            
            metrics = PerformanceMetrics(
                test_name=test_name,
                total_time=total_time,
                items_processed=len(result.processed_items),
                throughput=throughput,
                memory_usage_mb=memory_usage,
                cpu_usage_percent=self._get_cpu_usage(),
                success_rate=success_rate,
                error_count=0 if result.success else 1,
                agent_timings=agent_timings,
                classification_accuracy=avg_confidence,
                confidence_scores=confidence_scores,
                timestamp=datetime.now().isoformat()
            )
            
            self.test_results.append(metrics)
            
            logger.info(f"✅ Test {test_name} completed: {throughput:.2f} items/sec")
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Test {test_name} failed: {e}")
            
            # Return error metrics
            return PerformanceMetrics(
                test_name=test_name,
                total_time=time.time() - start_time,
                items_processed=0,
                throughput=0.0,
                memory_usage_mb=0.0,
                cpu_usage_percent=0.0,
                success_rate=0.0,
                error_count=1,
                agent_timings={},
                classification_accuracy=0.0,
                confidence_scores=[],
                timestamp=datetime.now().isoformat()
            )
        
        finally:
            await pipeline.cleanup()
            # Clean up test file
            if test_file.exists():
                test_file.unlink()
    
    async def run_load_test(self, concurrent_users: int, requests_per_user: int = 10) -> LoadTestResult:
        """Run load test with specified concurrent users."""
        logger.info(f"⚡ Running load test: {concurrent_users} concurrent users")
        
        total_requests = concurrent_users * requests_per_user
        response_times = []
        successful_requests = 0
        failed_requests = 0
        
        start_time = time.time()
        
        async def simulate_user(user_id: int) -> List[float]:
            """Simulate a single user making requests."""
            user_response_times = []
            
            for request_id in range(requests_per_user):
                try:
                    request_start = time.time()
                    
                    # Create small test data for each request
                    test_file = await self.create_test_data(
                        filename=f"load_test_user_{user_id}_req_{request_id}.csv",
                        num_items=50  # Small load per request
                    )
                    
                    # Process request
                    pipeline = FeedbackProcessingPipeline(PROCESSING_SETTINGS)
                    await pipeline.initialize()
                    
                    result = await pipeline.process_file(test_file)
                    
                    request_time = time.time() - request_start
                    user_response_times.append(request_time)
                    
                    if result.success:
                        nonlocal successful_requests
                        successful_requests += 1
                    else:
                        nonlocal failed_requests
                        failed_requests += 1
                    
                    await pipeline.cleanup()
                    
                    # Clean up test file
                    if test_file.exists():
                        test_file.unlink()
                
                except Exception as e:
                    logger.error(f"❌ Load test request failed: {e}")
                    failed_requests += 1
                    user_response_times.append(60.0)  # Timeout value
            
            return user_response_times
        
        # Run concurrent users
        tasks = [simulate_user(user_id) for user_id in range(concurrent_users)]
        user_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect all response times
        for user_result in user_results:
            if isinstance(user_result, list):
                response_times.extend(user_result)
            else:
                logger.error(f"❌ User simulation failed: {user_result}")
                failed_requests += requests_per_user
        
        total_time = time.time() - start_time
        
        # Calculate statistics
        if response_times:
            avg_response_time = statistics.mean(response_times)
            min_response_time = min(response_times)
            max_response_time = max(response_times)
            percentile_95 = np.percentile(response_times, 95)
            percentile_99 = np.percentile(response_times, 99)
        else:
            avg_response_time = min_response_time = max_response_time = 0.0
            percentile_95 = percentile_99 = 0.0
        
        throughput = successful_requests / total_time if total_time > 0 else 0.0
        error_rate = failed_requests / total_requests * 100 if total_requests > 0 else 0.0
        
        load_result = LoadTestResult(
            concurrent_users=concurrent_users,
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time=avg_response_time,
            min_response_time=min_response_time,
            max_response_time=max_response_time,
            percentile_95=percentile_95,
            percentile_99=percentile_99,
            throughput=throughput,
            error_rate=error_rate,
            timestamp=datetime.now().isoformat()
        )
        
        self.load_test_results.append(load_result)
        
        logger.info(f"✅ Load test completed: {throughput:.2f} req/sec, {error_rate:.1f}% error rate")
        return load_result
    
    async def run_resource_benchmark(self) -> Dict[str, Any]:
        """Run comprehensive resource usage benchmark."""
        logger.info("🧠 Running resource usage benchmark...")
        
        benchmark_results = {
            'memory_profile': [],
            'cpu_profile': [],
            'agent_resource_usage': {},
            'scalability_analysis': {},
            'optimization_suggestions': []
        }
        
        # Test different data sizes
        data_sizes = [100, 500, 1000, 2000, 5000]
        
        for size in data_sizes:
            logger.info(f"📊 Testing with {size} items...")
            
            # Create test data
            test_file = await self.create_test_data(
                filename=f"benchmark_{size}.csv",
                num_items=size
            )
            
            # Monitor resources during processing
            memory_before = self._get_memory_usage()
            cpu_before = self._get_cpu_usage()
            
            start_time = time.time()
            
            try:
                pipeline = FeedbackProcessingPipeline(PROCESSING_SETTINGS)
                await pipeline.initialize()
                
                result = await pipeline.process_file(test_file)
                
                processing_time = time.time() - start_time
                memory_after = self._get_memory_usage()
                cpu_after = self._get_cpu_usage()
                
                benchmark_results['memory_profile'].append({
                    'data_size': size,
                    'memory_usage_mb': memory_after - memory_before,
                    'processing_time': processing_time,
                    'memory_efficiency': size / (memory_after - memory_before) if memory_after > memory_before else 0
                })
                
                benchmark_results['cpu_profile'].append({
                    'data_size': size,
                    'cpu_usage_percent': cpu_after - cpu_before,
                    'processing_time': processing_time,
                    'cpu_efficiency': size / processing_time if processing_time > 0 else 0
                })
                
                await pipeline.cleanup()
                
            except Exception as e:
                logger.error(f"❌ Benchmark failed for size {size}: {e}")
            
            finally:
                if test_file.exists():
                    test_file.unlink()
        
        # Analyze scalability
        benchmark_results['scalability_analysis'] = self._analyze_scalability(
            benchmark_results['memory_profile'],
            benchmark_results['cpu_profile']
        )
        
        # Generate optimization suggestions
        benchmark_results['optimization_suggestions'] = self._generate_optimization_suggestions(
            benchmark_results
        )
        
        return benchmark_results
    
    async def run_accuracy_test(self) -> Dict[str, Any]:
        """Run classification accuracy test with known data."""
        logger.info("🎯 Running accuracy test...")
        
        # Create test data with known classifications
        test_data = await self.create_labeled_test_data()
        
        # Process test data
        pipeline = FeedbackProcessingPipeline(PROCESSING_SETTINGS)
        await pipeline.initialize()
        
        try:
            result = await pipeline.process_file(test_data['file_path'])
            
            # Compare with expected results
            accuracy_metrics = self._calculate_accuracy_metrics(
                result.processed_items,
                test_data['expected_classifications']
            )
            
            return accuracy_metrics
            
        except Exception as e:
            logger.error(f"❌ Accuracy test failed: {e}")
            return {'error': str(e)}
        
        finally:
            await pipeline.cleanup()
            if test_data['file_path'].exists():
                test_data['file_path'].unlink()
    
    async def create_test_data(self, filename: str, num_items: int) -> Path:
        """Create synthetic test data for performance testing."""
        test_file = self.output_dir / filename
        
        # Generate diverse feedback data
        feedback_templates = [
            "The app crashes when I try to {action}. This is very frustrating!",
            "I love the new {feature} feature! It works perfectly.",
            "Could you please add {feature_request}? It would be very helpful.",
            "Bug: {error_description}. Steps to reproduce: {steps}",
            "The app is slow when {scenario}. Please fix this issue.",
            "Great app overall, but {improvement_suggestion}",
            "Feature request: {detailed_request}",
            "Praise: {positive_feedback}",
            "Complaint: {negative_feedback}",
            "The latest update broke {functionality}"
        ]
        
        actions = ["login", "search", "upload files", "send messages", "save data"]
        features = ["dark mode", "notifications", "search", "sharing", "sync"]
        feature_requests = ["offline mode", "better search", "bulk operations", "export feature"]
        errors = ["null pointer exception", "connection timeout", "invalid data format"]
        steps = ["1. Open app 2. Click button 3. Error appears", "1. Login 2. Navigate to settings 3. Crash"]
        scenarios = ["loading large files", "switching between tabs", "using multiple features"]
        improvements = ["the UI could be more intuitive", "loading times could be faster"]
        
        # Generate feedback items
        feedback_items = []
        for i in range(num_items):
            template = np.random.choice(feedback_templates)
            
            # Fill template with random content
            content = template.format(
                action=np.random.choice(actions),
                feature=np.random.choice(features),
                feature_request=np.random.choice(feature_requests),
                error_description=np.random.choice(errors),
                steps=np.random.choice(steps),
                scenario=np.random.choice(scenarios),
                improvement_suggestion=np.random.choice(improvements),
                detailed_request=f"Add {np.random.choice(feature_requests)} to improve user experience",
                positive_feedback=f"I really enjoy using {np.random.choice(features)}",
                negative_feedback=f"The {np.random.choice(features)} feature is confusing",
                functionality=np.random.choice(features)
            )
            
            feedback_items.append({
                'id': f'test_{i:06d}',
                'content': content,
                'source': np.random.choice(['app_store', 'support_email', 'user_feedback']),
                'timestamp': (datetime.now() - timedelta(days=np.random.randint(0, 30))).isoformat(),
                'user_id': f'user_{np.random.randint(1000, 9999)}',
                'rating': np.random.randint(1, 6) if np.random.random() > 0.3 else None
            })
        
        # Save to CSV
        csv_handler = CSVHandler()
        await csv_handler.save_data(test_file, feedback_items)
        
        return test_file
    
    async def create_labeled_test_data(self) -> Dict[str, Any]:
        """Create labeled test data for accuracy testing."""
        test_file = self.output_dir / "accuracy_test.csv"
        
        # Create test cases with known classifications
        test_cases = [
            # Bug reports
            {
                'content': "The app crashes every time I try to upload a photo. Error code: 500",
                'expected_category': 'Bug',
                'expected_priority': 'High',
                'expected_confidence': 0.9
            },
            {
                'content': "Login button doesn't work on iOS 15. Can't access my account.",
                'expected_category': 'Bug',
                'expected_priority': 'High',
                'expected_confidence': 0.85
            },
            # Feature requests
            {
                'content': "Please add dark mode support. It would be great for night usage.",
                'expected_category': 'Feature Request',
                'expected_priority': 'Medium',
                'expected_confidence': 0.9
            },
            {
                'content': "Would love to see offline sync capability in the next update.",
                'expected_category': 'Feature Request',
                'expected_priority': 'Medium',
                'expected_confidence': 0.8
            },
            # Praise
            {
                'content': "Absolutely love this app! The new design is fantastic.",
                'expected_category': 'Praise',
                'expected_priority': 'Low',
                'expected_confidence': 0.95
            },
            {
                'content': "Great job on the latest update. Everything works smoothly now.",
                'expected_category': 'Praise',
                'expected_priority': 'Low',
                'expected_confidence': 0.9
            },
            # Complaints
            {
                'content': "The app is too slow and the interface is confusing.",
                'expected_category': 'Complaint',
                'expected_priority': 'Medium',
                'expected_confidence': 0.8
            },
            {
                'content': "Disappointed with the recent changes. The app was better before.",
                'expected_category': 'Complaint',
                'expected_priority': 'Medium',
                'expected_confidence': 0.75
            }
        ]
        
        # Expand test cases
        expanded_cases = []
        for i, case in enumerate(test_cases * 25):  # Repeat to get ~200 test cases
            expanded_case = case.copy()
            expanded_case.update({
                'id': f'accuracy_test_{i:06d}',
                'source': 'accuracy_test',
                'timestamp': datetime.now().isoformat(),
                'user_id': f'test_user_{i % 50}'
            })
            expanded_cases.append(expanded_case)
        
        # Save test data
        csv_data = [{k: v for k, v in case.items() if k != 'expected_category' and 
                    k != 'expected_priority' and k != 'expected_confidence'} 
                   for case in expanded_cases]
        
        csv_handler = CSVHandler()
        await csv_handler.save_data(test_file, csv_data)
        
        # Return file path and expected results
        expected_classifications = {
            case['id']: {
                'category': case['expected_category'],
                'priority': case['expected_priority'],
                'confidence': case['expected_confidence']
            }
            for case in expanded_cases
        }
        
        return {
            'file_path': test_file,
            'expected_classifications': expected_classifications
        }
    
    def _calculate_accuracy_metrics(self, processed_items: List, expected_classifications: Dict) -> Dict[str, Any]:
        """Calculate classification accuracy metrics."""
        correct_categories = 0
        correct_priorities = 0
        total_items = len(processed_items)
        
        category_confusion_matrix = {}
        priority_confusion_matrix = {}
        confidence_scores = []
        
        for item in processed_items:
            item_id = getattr(item, 'id', None)
            if item_id and item_id in expected_classifications:
                expected = expected_classifications[item_id]
                
                # Check category accuracy
                predicted_category = getattr(item, 'category', None)
                if predicted_category == expected['category']:
                    correct_categories += 1
                
                # Check priority accuracy
                predicted_priority = getattr(item, 'priority', None)
                if predicted_priority == expected['priority']:
                    correct_priorities += 1
                
                # Collect confidence scores
                confidence = getattr(item, 'confidence', 0.0)
                confidence_scores.append(confidence)
                
                # Build confusion matrices
                category_key = f"{expected['category']}->{predicted_category}"
                category_confusion_matrix[category_key] = category_confusion_matrix.get(category_key, 0) + 1
                
                priority_key = f"{expected['priority']}->{predicted_priority}"
                priority_confusion_matrix[priority_key] = priority_confusion_matrix.get(priority_key, 0) + 1
        
        return {
            'total_items': total_items,
            'category_accuracy': correct_categories / total_items if total_items > 0 else 0.0,
            'priority_accuracy': correct_priorities / total_items if total_items > 0 else 0.0,
            'average_confidence': statistics.mean(confidence_scores) if confidence_scores else 0.0,
            'confidence_std': statistics.stdev(confidence_scores) if len(confidence_scores) > 1 else 0.0,
            'category_confusion_matrix': category_confusion_matrix,
            'priority_confusion_matrix': priority_confusion_matrix,
            'high_confidence_items': len([c for c in confidence_scores if c > 0.8]),
            'low_confidence_items': len([c for c in confidence_scores if c < 0.5])
        }
    
    def _analyze_scalability(self, memory_profile: List, cpu_profile: List) -> Dict[str, Any]:
        """Analyze system scalability based on resource profiles."""
        if not memory_profile or not cpu_profile:
            return {'error': 'Insufficient data for scalability analysis'}
        
        # Calculate growth rates
        data_sizes = [p['data_size'] for p in memory_profile]
        memory_usage = [p['memory_usage_mb'] for p in memory_profile]
        processing_times = [p['processing_time'] for p in memory_profile]
        
        # Linear regression for memory growth
        memory_growth_rate = np.polyfit(data_sizes, memory_usage, 1)[0] if len(data_sizes) > 1 else 0
        time_growth_rate = np.polyfit(data_sizes, processing_times, 1)[0] if len(data_sizes) > 1 else 0
        
        # Efficiency metrics
        memory_efficiency = [p['memory_efficiency'] for p in memory_profile]
        cpu_efficiency = [p['cpu_efficiency'] for p in cpu_profile]
        
        return {
            'memory_growth_rate_mb_per_item': memory_growth_rate,
            'time_growth_rate_sec_per_item': time_growth_rate,
            'avg_memory_efficiency': statistics.mean(memory_efficiency) if memory_efficiency else 0,
            'avg_cpu_efficiency': statistics.mean(cpu_efficiency) if cpu_efficiency else 0,
            'scalability_score': self._calculate_scalability_score(memory_growth_rate, time_growth_rate),
            'recommended_max_batch_size': self._recommend_batch_size(memory_profile, cpu_profile),
            'bottleneck_analysis': self._identify_bottlenecks(memory_profile, cpu_profile)
        }
    
    def _calculate_scalability_score(self, memory_growth: float, time_growth: float) -> float:
        """Calculate overall scalability score (0-100)."""
        # Lower growth rates = better scalability
        memory_score = max(0, 100 - (memory_growth * 100))
        time_score = max(0, 100 - (time_growth * 1000))
        
        return (memory_score + time_score) / 2
    
    def _recommend_batch_size(self, memory_profile: List, cpu_profile: List) -> int:
        """Recommend optimal batch size based on resource usage."""
        # Find the point where efficiency starts decreasing
        memory_efficiencies = [p['memory_efficiency'] for p in memory_profile]
        
        if len(memory_efficiencies) < 2:
            return 500  # Default recommendation
        
        # Find peak efficiency
        max_efficiency_idx = memory_efficiencies.index(max(memory_efficiencies))
        optimal_size = memory_profile[max_efficiency_idx]['data_size']
        
        return min(max(optimal_size, 100), 2000)  # Clamp between 100-2000
    
    def _identify_bottlenecks(self, memory_profile: List, cpu_profile: List) -> List[str]:
        """Identify system bottlenecks."""
        bottlenecks = []
        
        # Check memory bottlenecks
        memory_usage = [p['memory_usage_mb'] for p in memory_profile]
        if memory_usage and max(memory_usage) > 1000:  # > 1GB
            bottlenecks.append("High memory usage detected - consider reducing batch size")
        
        # Check processing time bottlenecks
        processing_times = [p['processing_time'] for p in memory_profile]
        if processing_times and max(processing_times) > 60:  # > 1 minute
            bottlenecks.append("Long processing times detected - consider optimizing algorithms")
        
        # Check efficiency trends
        memory_efficiencies = [p['memory_efficiency'] for p in memory_profile]
        if len(memory_efficiencies) > 2:
            efficiency_trend = np.polyfit(range(len(memory_efficiencies)), memory_efficiencies, 1)[0]
            if efficiency_trend < -0.1:  # Decreasing efficiency
                bottlenecks.append("Memory efficiency decreasing with scale - review memory management")
        
        return bottlenecks if bottlenecks else ["No significant bottlenecks detected"]
    
    def _generate_optimization_suggestions(self, benchmark_results: Dict) -> List[str]:
        """Generate optimization suggestions based on benchmark results."""
        suggestions = []
        
        # Memory optimization suggestions
        memory_profile = benchmark_results.get('memory_profile', [])
        if memory_profile:
            avg_memory = statistics.mean([p['memory_usage_mb'] for p in memory_profile])
            if avg_memory > 500:
                suggestions.append("Consider implementing memory pooling to reduce garbage collection overhead")
                suggestions.append("Implement streaming processing for large datasets")
        
        # CPU optimization suggestions
        cpu_profile = benchmark_results.get('cpu_profile', [])
        if cpu_profile:
            avg_cpu = statistics.mean([p['cpu_usage_percent'] for p in cpu_profile])
            if avg_cpu > 80:
                suggestions.append("High CPU usage detected - consider implementing async processing")
                suggestions.append("Review algorithm complexity and optimize hot paths")
        
        # Scalability suggestions
        scalability = benchmark_results.get('scalability_analysis', {})
        if scalability.get('scalability_score', 100) < 70:
            suggestions.append("Implement horizontal scaling with load balancing")
            suggestions.append("Consider database sharding for large datasets")
        
        return suggestions if suggestions else ["System performance is within acceptable parameters"]
    
    def generate_performance_recommendations(self) -> List[str]:
        """Generate overall performance recommendations."""
        recommendations = []
        
        if not self.test_results:
            return ["No test results available for recommendations"]
        
        # Analyze throughput trends
        throughputs = [result.throughput for result in self.test_results]
        avg_throughput = statistics.mean(throughputs)
        
        if avg_throughput < 10:  # < 10 items/sec
            recommendations.append("Low throughput detected - consider increasing concurrent agents")
            recommendations.append("Optimize database queries and API calls")
        
        # Analyze success rates
        success_rates = [result.success_rate for result in self.test_results]
        avg_success_rate = statistics.mean(success_rates)
        
        if avg_success_rate < 0.95:  # < 95% success rate
            recommendations.append("Error rate too high - implement better error handling and retries")
            recommendations.append("Add input validation to prevent processing failures")
        
        # Analyze memory usage
        memory_usages = [result.memory_usage_mb for result in self.test_results]
        max_memory = max(memory_usages) if memory_usages else 0
        
        if max_memory > 2000:  # > 2GB
            recommendations.append("High memory usage - implement memory optimization strategies")
            recommendations.append("Consider processing data in smaller chunks")
        
        # Analyze confidence scores
        all_confidences = []
        for result in self.test_results:
            all_confidences.extend(result.confidence_scores)
        
        if all_confidences:
            avg_confidence = statistics.mean(all_confidences)
            if avg_confidence < 0.8:
                recommendations.append("Low classification confidence - retrain models with more data")
                recommendations.append("Implement human-in-the-loop validation for low-confidence items")
        
        return recommendations if recommendations else ["System performance is optimal"]
    
    async def save_comprehensive_report(self, results: Dict[str, Any]):
        """Save comprehensive performance test report."""
        report_file = self.output_dir / f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Add summary statistics
        results['summary'] = self._generate_summary_statistics()
        
        # Save JSON report
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate visualizations
        await self._generate_performance_charts(results)
        
        # Generate CSV reports
        await self._generate_csv_reports()
        
        logger.info(f"📊 Comprehensive report saved to {report_file}")
    
    def _generate_summary_statistics(self) -> Dict[str, Any]:
        """Generate summary statistics from all tests."""
        if not self.test_results:
            return {}
        
        throughputs = [r.throughput for r in self.test_results]
        processing_times = [r.total_time for r in self.test_results]
        memory_usages = [r.memory_usage_mb for r in self.test_results]
        success_rates = [r.success_rate for r in self.test_results]
        
        return {
            'total_tests_run': len(self.test_results),
            'avg_throughput': statistics.mean(throughputs),
            'max_throughput': max(throughputs),
            'avg_processing_time': statistics.mean(processing_times),
            'avg_memory_usage': statistics.mean(memory_usages),
            'overall_success_rate': statistics.mean(success_rates),
            'total_items_processed': sum(r.items_processed for r in self.test_results)
        }
    
    async def _generate_performance_charts(self, results: Dict[str, Any]):
        """Generate performance visualization charts."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # Throughput chart
            if self.test_results:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                
                # Throughput over tests
                test_names = [r.test_name for r in self.test_results]
                throughputs = [r.throughput for r in self.test_results]
                
                ax1.bar(test_names, throughputs)
                ax1.set_title('Throughput by Test')
                ax1.set_ylabel('Items/Second')
                ax1.tick_params(axis='x', rotation=45)
                
                # Memory usage
                memory_usage = [r.memory_usage_mb for r in self.test_results]
                ax2.bar(test_names, memory_usage, color='orange')
                ax2.set_title('Memory Usage by Test')
                ax2.set_ylabel('Memory (MB)')
                ax2.tick_params(axis='x', rotation=45)
                
                # Success rate
                success_rates = [r.success_rate * 100 for r in self.test_results]
                ax3.bar(test_names, success_rates, color='green')
                ax3.set_title('Success Rate by Test')
                ax3.set_ylabel('Success Rate (%)')
                ax3.tick_params(axis='x', rotation=45)
                
                # Processing time
                processing_times = [r.total_time for r in self.test_results]
                ax4.bar(test_names, processing_times, color='red')
                ax4.set_title('Processing Time by Test')
                ax4.set_ylabel('Time (seconds)')
                ax4.tick_params(axis='x', rotation=45)
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'performance_charts.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # Load test results chart
            if self.load_test_results:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                users = [r.concurrent_users for r in self.load_test_results]
                response_times = [r.avg_response_time for r in self.load_test_results]
                error_rates = [r.error_rate for r in self.load_test_results]
                
                ax1.plot(users, response_times, marker='o')
                ax1.set_title('Response Time vs Concurrent Users')
                ax1.set_xlabel('Concurrent Users')
                ax1.set_ylabel('Avg Response Time (s)')
                
                ax2.plot(users, error_rates, marker='o', color='red')
                ax2.set_title('Error Rate vs Concurrent Users')
                ax2.set_xlabel('Concurrent Users')
                ax2.set_ylabel('Error Rate (%)')
                
                plt.tight_layout()
                plt.savefig(self.output_dir / 'load_test_charts.png', dpi=300, bbox_inches='tight')
                plt.close()
                
        except Exception as e:
            logger.error(f" Failed to generate charts: {e}")
    
    async def _generate_csv_reports(self):
        """Generate CSV reports for detailed analysis."""
        # Performance test results CSV
        if self.test_results:
            perf_data = []
            for result in self.test_results:
                row = asdict(result)
                # Flatten agent_timings and confidence_scores for CSV
                row['agent_timings'] = json.dumps(row['agent_timings'])
                row['confidence_scores'] = json.dumps(row['confidence_scores'])
                perf_data.append(row)
            
            perf_df = pd.DataFrame(perf_data)
            perf_df.to_csv(self.output_dir / 'performance_test_results.csv', index=False)
        
        # Load test results CSV
        if self.load_test_results:
            load_data = [asdict(result) for result in self.load_test_results]
            load_df = pd.DataFrame(load_data)
            load_df.to_csv(self.output_dir / 'load_test_results.csv', index=False)
    
    # Utility methods
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except:
            return 0.0


# Main execution
async def main():
    """Main performance testing execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Testing Suite")
    parser.add_argument("--test", choices=['basic', 'load', 'accuracy', 'comprehensive'], 
                       default='comprehensive', help="Type of test to run")
    parser.add_argument("--output", type=Path, help="Output directory for results")
    parser.add_argument("--concurrent-users", type=int, default=10, 
                       help="Number of concurrent users for load testing")
    
    args = parser.parse_args()
    
    # Initialize test suite
    test_suite = PerformanceTestSuite(args.output)
    
    try:
        if args.test == 'basic':
            logger.info(" Running basic performance tests...")
            for test_name, config in test_suite.test_configs.items():
                if test_name != 'endurance_test':
                    await test_suite.run_performance_test(test_name, config)
        
        elif args.test == 'load':
            logger.info("⚡ Running load tests...")
            await test_suite.run_load_test(args.concurrent_users)
        
        elif args.test == 'accuracy':
            logger.info(" Running accuracy tests...")
            await test_suite.run_accuracy_test()
        
        elif args.test == 'comprehensive':
            logger.info("🚀 Running comprehensive test suite...")
            results = await test_suite.run_comprehensive_tests()
            
            print("\n Test Suite Summary:")
            print(f"   Total tests: {results.get('summary', {}).get('total_tests_run', 0)}")
            print(f"   Avg throughput: {results.get('summary', {}).get('avg_throughput', 0):.2f} items/sec")
            print(f"   Success rate: {results.get('summary', {}).get('overall_success_rate', 0):.1%}")
            print(f"   Total items processed: {results.get('summary', {}).get('total_items_processed', 0)}")
            
            # Print recommendations
            recommendations = results.get('recommendations', [])
            if recommendations:
                print("\n💡 Recommendations:")
                for i, rec in enumerate(recommendations[:5], 1):
                    print(f"   {i}. {rec}")
        
        logger.info("Performance testing completed successfully!")
        
    except Exception as e:
        logger.error(f"Performance testing failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)