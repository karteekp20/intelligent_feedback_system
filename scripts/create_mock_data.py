#!/usr/bin/env python3
"""
Script to generate mock data for the Intelligent Feedback Analysis System.
"""

import pandas as pd
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys
import asyncio

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import INPUT_DIR
from src.utils.logger import get_logger


class MockDataGenerator:
    """Generates realistic mock data for testing the system."""
    
    def __init__(self):
        """Initialize mock data generator."""
        self.logger = get_logger("mock_data_generator")
        
        # Sample data templates
        self.bug_reviews = [
            "App crashes every time I try to {action}. This is very frustrating!",
            "Since the latest update {version}, I can't {action}. Please fix this bug.",
            "The app keeps freezing when I {action}. Running on {device}.",
            "Login doesn't work anymore. Error message says '{error}'. Please help!",
            "Data sync is broken. My {data_type} are not syncing across devices.",
            "App is constantly crashing on {device}. Started after update to {version}.",
            "Can't save my work. App crashes when I try to save. Very annoying!",
            "Performance is terrible since {version}. App takes forever to load.",
            "Getting error code {error_code} when trying to {action}. What does this mean?",
            "App won't start on my {device}. Just shows black screen then crashes."
        ]
        
        self.feature_requests = [
            "Please add {feature}. This would be very useful for my workflow.",
            "Would love to see {feature} in the next update. Thanks!",
            "Suggestion: Add {feature} functionality. Other apps have this.",
            "Can you please implement {feature}? It would make the app perfect!",
            "I wish the app had {feature}. This is a must-have feature.",
            "Please consider adding {feature}. Many users would benefit from this.",
            "Feature request: {feature}. This would save me so much time!",
            "The app needs {feature} to compete with other similar apps.",
            "Could you add {feature}? It would improve the user experience significantly.",
            "I suggest implementing {feature}. It's a common feature in productivity apps."
        ]
        
        self.praise_reviews = [
            "Amazing app! Love the {feature} functionality. Keep up the great work!",
            "This app is fantastic! The {feature} works perfectly. 5 stars!",
            "Best {app_type} app I've used. The interface is clean and intuitive.",
            "Love this app! {feature} is exactly what I needed. Thank you developers!",
            "Excellent app with great {feature}. Highly recommend to everyone!",
            "Perfect app for {use_case}. The {feature} is brilliant!",
            "Outstanding work on this app. {feature} makes my life so much easier.",
            "This app is a game-changer! {feature} is implemented beautifully.",
            "Wonderful app! {feature} works flawlessly. Thank you for creating this!",
            "Love everything about this app, especially {feature}. Amazing job!"
        ]
        
        self.complaint_reviews = [
            "This app is terrible. {complaint}. Waste of money!",
            "Very disappointed with this app. {complaint}. Considering uninstalling.",
            "Used to be good but now {complaint}. Very frustrating experience.",
            "The app is too {problem}. {complaint}. Please improve!",
            "Hate the new {feature}. {complaint}. Bring back the old version!",
            "This app is getting worse with each update. {complaint}.",
            "Poor user experience. {complaint}. Needs major improvements.",
            "Very buggy app. {complaint}. Not worth the price.",
            "Disappointed with the latest update. {complaint}.",
            "The app is becoming unusable. {complaint}. Fix these issues!"
        ]
        
        self.spam_reviews = [
            "Check out this amazing offer! Visit {url} for free money!",
            "Get rich quick! Click {url} to start earning today!",
            "Limited time offer! Free premium account at {url}!",
            "Make money from home! Visit {url} for details. Act now!",
            "Best deals online! Check {url} for amazing discounts!",
            "asdkjfhlaksjdhflkajshdflkjh random text here spam spam",
            "FREE MONEY Visit {url} now! Limited time!",
            "Click here {url} for the best deals online! Hurry up!",
            "Work from home opportunity! {url} - guaranteed income!",
            "Special promotion! Visit {url} and get instant rewards!"
        ]
        
        # Data pools for templates
        self.actions = [
            "login", "save my work", "sync data", "upload files", "share content",
            "export data", "search", "create new project", "delete items", "update profile"
        ]
        
        self.devices = [
            "iPhone 14", "iPhone 13", "iPhone 12", "iPhone SE", "iPad Pro",
            "iPad Air", "Samsung Galaxy S23", "Samsung Galaxy S22", "Pixel 7",
            "Pixel 6", "OnePlus 11", "Xiaomi 13"
        ]
        
        self.versions = [
            "v2.1.3", "v2.0.5", "v1.9.8", "v3.0.1", "v2.5.2",
            "version 2.1", "version 3.0", "latest update", "recent update"
        ]
        
        self.features = [
            "dark mode", "offline sync", "cloud backup", "collaboration tools",
            "advanced search", "custom themes", "voice notes", "file encryption",
            "bulk operations", "keyboard shortcuts", "widget support", "tablet mode"
        ]
        
        self.app_types = [
            "productivity", "note-taking", "task management", "file manager",
            "document editor", "project management"
        ]
        
        self.use_cases = [
            "work", "personal projects", "team collaboration", "study",
            "research", "creative work"
        ]
        
        self.complaints = [
            "the interface is confusing", "it's too slow", "ads are everywhere",
            "features are missing", "sync doesn't work", "it's overpriced",
            "customer support is poor", "updates break things"
        ]
        
        self.problems = [
            "slow", "expensive", "complicated", "buggy", "unreliable",
            "cluttered", "outdated", "limited"
        ]
        
        self.data_types = [
            "notes", "files", "projects", "tasks", "documents", "settings",
            "bookmarks", "photos", "contacts", "calendars"
        ]
        
        self.error_codes = [
            "E1001", "E2045", "E3021", "ERR_SYNC_FAILED", "AUTH_ERROR",
            "NETWORK_TIMEOUT", "FILE_NOT_FOUND", "PERMISSION_DENIED"
        ]
        
        self.urls = [
            "bit.ly/free-offer", "get-rich-now.com", "amazing-deals.net",
            "instant-money.org", "best-offers.co"
        ]
        
        self.platforms = ["Google Play", "App Store"]
        self.user_names = [
            "TechLover92", "ProductivityPro", "MobileUser", "AppReviewer",
            "DigitalNomad", "StudentLife", "WorkFromHome", "CreativeUser",
            "PowerUser", "CasualUser", "BusinessUser", "FreelancerLife"
        ]
        
        self.email_subjects = {
            "bug": [
                "App Crash Report - Urgent",
                "Login Issue - Need Help",
                "Data Loss Problem",
                "Sync Error - Critical",
                "App Not Working",
                "Bug Report: {feature}",
                "Critical Issue with {feature}",
                "App Freezing Problem",
                "Error Message: {error}",
                "Urgent: Cannot Access My Data"
            ],
            "feature": [
                "Feature Request: {feature}",
                "Suggestion for App Improvement",
                "Enhancement Request",
                "Feature Idea: {feature}",
                "App Enhancement Suggestion",
                "New Feature Proposal",
                "Improvement Suggestion: {feature}",
                "Feature Request - {feature}",
                "Suggestion: Add {feature}",
                "Enhancement: {feature} Support"
            ],
            "complaint": [
                "Very Disappointed with Recent Update",
                "Poor App Performance",
                "Frustrated with App Issues",
                "Complaint: App Quality",
                "Dissatisfied with Service",
                "App Issues - Need Resolution",
                "Complaint About {feature}",
                "Poor User Experience",
                "App Problems - Multiple Issues",
                "Feedback: App Needs Improvement"
            ]
        }
    
    def generate_app_store_reviews(self, count: int = 50) -> pd.DataFrame:
        """Generate mock app store reviews."""
        reviews = []
        
        for i in range(count):
            review_id = f"review_{i+1:04d}"
            platform = random.choice(self.platforms)
            
            # Determine review type and rating
            review_type = random.choices(
                ["bug", "feature", "praise", "complaint", "spam"],
                weights=[30, 25, 25, 15, 5]
            )[0]
            
            if review_type == "bug":
                rating = random.choices([1, 2, 3], weights=[50, 35, 15])[0]
                template = random.choice(self.bug_reviews)
                review_text = template.format(
                    action=random.choice(self.actions),
                    version=random.choice(self.versions),
                    device=random.choice(self.devices),
                    error=random.choice(self.error_codes),
                    data_type=random.choice(self.data_types),
                    error_code=random.choice(self.error_codes)
                )
            elif review_type == "feature":
                rating = random.choices([3, 4, 5], weights=[40, 35, 25])[0]
                template = random.choice(self.feature_requests)
                review_text = template.format(
                    feature=random.choice(self.features)
                )
            elif review_type == "praise":
                rating = random.choices([4, 5], weights=[30, 70])[0]
                template = random.choice(self.praise_reviews)
                review_text = template.format(
                    feature=random.choice(self.features),
                    app_type=random.choice(self.app_types),
                    use_case=random.choice(self.use_cases)
                )
            elif review_type == "complaint":
                rating = random.choices([1, 2], weights=[60, 40])[0]
                template = random.choice(self.complaint_reviews)
                review_text = template.format(
                    complaint=random.choice(self.complaints),
                    problem=random.choice(self.problems),
                    feature=random.choice(self.features)
                )
            else:  # spam
                rating = random.choice([1, 2, 3, 4, 5])
                template = random.choice(self.spam_reviews)
                review_text = template.format(
                    url=random.choice(self.urls)
                )
            
            user_name = random.choice(self.user_names)
            date = self._random_date(30)  # Within last 30 days
            app_version = random.choice(self.versions)
            
            reviews.append({
                "review_id": review_id,
                "platform": platform,
                "rating": rating,
                "review_text": review_text,
                "user_name": user_name,
                "date": date.strftime("%Y-%m-%d"),
                "app_version": app_version
            })
        
        return pd.DataFrame(reviews)
    
    def generate_support_emails(self, count: int = 30) -> pd.DataFrame:
        """Generate mock support emails."""
        emails = []
        
        for i in range(count):
            email_id = f"email_{i+1:04d}"
            
            # Determine email type
            email_type = random.choices(
                ["bug", "feature", "complaint"],
                weights=[50, 30, 20]
            )[0]
            
            # Generate subject
            subject_template = random.choice(self.email_subjects[email_type])
            subject = subject_template.format(
                feature=random.choice(self.features),
                error=random.choice(self.error_codes)
            )
            
            # Generate body based on type
            if email_type == "bug":
                body_parts = [
                    f"Hi Support Team,",
                    f"",
                    f"I'm experiencing a critical issue with the app. {random.choice(self.bug_reviews).format(action=random.choice(self.actions), version=random.choice(self.versions), device=random.choice(self.devices), error=random.choice(self.error_codes), data_type=random.choice(self.data_types), error_code=random.choice(self.error_codes))}",
                    f"",
                    f"Device: {random.choice(self.devices)}",
                    f"App Version: {random.choice(self.versions)}",
                    f"OS: {random.choice(['iOS 16.1', 'iOS 15.7', 'Android 13', 'Android 12'])}",
                    f"",
                    f"Steps to reproduce:",
                    f"1. Open the app",
                    f"2. Try to {random.choice(self.actions)}",
                    f"3. App crashes or shows error",
                    f"",
                    f"Please help resolve this issue as soon as possible.",
                    f"",
                    f"Thanks,",
                    f"{random.choice(self.user_names)}"
                ]
                priority = random.choices(["High", "Medium"], weights=[70, 30])[0]
                
            elif email_type == "feature":
                body_parts = [
                    f"Hello,",
                    f"",
                    f"I have a suggestion for improving the app. {random.choice(self.feature_requests).format(feature=random.choice(self.features))}",
                    f"",
                    f"This feature would be very helpful for users like me who use the app for {random.choice(self.use_cases)}.",
                    f"",
                    f"Use case: I often need to {random.choice(self.actions)} and having {random.choice(self.features)} would make this much easier.",
                    f"",
                    f"Please consider this for a future update.",
                    f"",
                    f"Best regards,",
                    f"{random.choice(self.user_names)}"
                ]
                priority = random.choices(["Medium", "Low"], weights=[60, 40])[0]
                
            else:  # complaint
                body_parts = [
                    f"Dear Team,",
                    f"",
                    f"I am very disappointed with the recent changes to the app. {random.choice(self.complaint_reviews).format(complaint=random.choice(self.complaints), problem=random.choice(self.problems), feature=random.choice(self.features))}",
                    f"",
                    f"The app used to work perfectly, but now it's {random.choice(self.problems)} and {random.choice(self.complaints)}.",
                    f"",
                    f"I hope you can address these issues soon, otherwise I might have to look for alternatives.",
                    f"",
                    f"Regards,",
                    f"{random.choice(self.user_names)}"
                ]
                priority = random.choices(["Medium", "High"], weights=[60, 40])[0]
            
            body = "\n".join(body_parts)
            sender_email = f"{random.choice(self.user_names).lower()}@{random.choice(['gmail.com', 'yahoo.com', 'outlook.com', 'hotmail.com'])}"
            timestamp = self._random_datetime(14)  # Within last 14 days
            
            emails.append({
                "email_id": email_id,
                "subject": subject,
                "body": body,
                "sender_email": sender_email,
                "timestamp": timestamp.isoformat(),
                "priority": priority
            })
        
        return pd.DataFrame(emails)
    
    def generate_expected_classifications(self, reviews_df: pd.DataFrame, emails_df: pd.DataFrame) -> pd.DataFrame:
        """Generate expected classifications for evaluation."""
        classifications = []
        
        # Process reviews
        for _, review in reviews_df.iterrows():
            category, priority = self._classify_review(review)
            
            classifications.append({
                "source_id": review["review_id"],
                "source_type": "app_store_review",
                "category": category,
                "priority": priority,
                "technical_details": self._generate_technical_details(review["review_text"], category),
                "suggested_title": self._generate_title(review["review_text"], category)
            })
        
        # Process emails
        for _, email in emails_df.iterrows():
            category, priority = self._classify_email(email)
            
            classifications.append({
                "source_id": email["email_id"],
                "source_type": "support_email",
                "category": category,
                "priority": priority,
                "technical_details": self._generate_technical_details(email["body"], category),
                "suggested_title": self._generate_title(email["subject"], category)
            })
        
        return pd.DataFrame(classifications)
    
    def _classify_review(self, review: pd.Series) -> tuple:
        """Classify a review and determine priority."""
        text = review["review_text"].lower()
        rating = review["rating"]
        
        if any(word in text for word in ["crash", "bug", "error", "broken", "not working"]):
            category = "Bug"
            priority = "Critical" if rating <= 2 else "High"
        elif any(word in text for word in ["request", "suggest", "would like", "please add"]):
            category = "Feature Request"
            priority = "Medium"
        elif any(word in text for word in ["love", "great", "awesome", "excellent", "amazing"]):
            category = "Praise"
            priority = "Low"
        elif any(word in text for word in ["free money", "click here", "visit", "promotion"]):
            category = "Spam"
            priority = "Low"
        else:
            category = "Complaint"
            priority = "Medium" if rating <= 2 else "Low"
        
        return category, priority
    
    def _classify_email(self, email: pd.Series) -> tuple:
        """Classify an email and determine priority."""
        text = (email["subject"] + " " + email["body"]).lower()
        
        if any(word in text for word in ["crash", "bug", "error", "broken", "not working", "critical", "urgent"]):
            category = "Bug"
            priority = "Critical" if "critical" in text or "urgent" in text else "High"
        elif any(word in text for word in ["request", "suggest", "would like", "please add", "feature", "enhancement"]):
            category = "Feature Request"
            priority = "Medium"
        else:
            category = "Complaint"
            priority = email.get("priority", "Medium")
        
        return category, priority
    
    def _generate_technical_details(self, text: str, category: str) -> str:
        """Generate technical details based on text and category."""
        if category == "Bug":
            details = []
            text_lower = text.lower()
            
            if any(device in text_lower for device in ["iphone", "ipad", "android", "samsung"]):
                device_match = next((device for device in self.devices if device.lower() in text_lower), None)
                if device_match:
                    details.append(f"Device: {device_match}")
            
            if any(version in text_lower for version in ["version", "v2.", "v3.", "update"]):
                version_match = next((version for version in self.versions if version.lower() in text_lower), None)
                if version_match:
                    details.append(f"Version: {version_match}")
            
            if "crash" in text_lower:
                details.append("Severity: Critical - App crashes")
            elif "error" in text_lower:
                details.append("Severity: High - Error encountered")
            
            return "; ".join(details) if details else "Technical issue reported"
        
        elif category == "Feature Request":
            feature_match = next((feature for feature in self.features if feature.lower() in text.lower()), "new functionality")
            return f"Requested feature: {feature_match}"
        
        return ""
    
    def _generate_title(self, text: str, category: str) -> str:
        """Generate a suggested ticket title."""
        if category == "Bug":
            if "crash" in text.lower():
                return "App crash issue needs investigation"
            elif "login" in text.lower():
                return "Login functionality not working"
            elif "sync" in text.lower():
                return "Data synchronization problem"
            else:
                return "Technical issue reported by user"
        
        elif category == "Feature Request":
            feature_match = next((feature for feature in self.features if feature.lower() in text.lower()), "new feature")
            return f"Feature request: Add {feature_match}"
        
        elif category == "Complaint":
            return "User complaint requires attention"
        
        elif category == "Praise":
            return "Positive user feedback"
        
        else:
            return "User feedback for review"
    
    def _random_date(self, days_back: int) -> datetime:
        """Generate a random date within the last N days."""
        start_date = datetime.now() - timedelta(days=days_back)
        random_days = random.randint(0, days_back)
        return start_date + timedelta(days=random_days)
    
    def _random_datetime(self, days_back: int) -> datetime:
        """Generate a random datetime within the last N days."""
        start_date = datetime.now() - timedelta(days=days_back)
        random_seconds = random.randint(0, days_back * 24 * 60 * 60)
        return start_date + timedelta(seconds=random_seconds)


async def create_all_mock_data():
    """Create all mock data files."""
    logger = get_logger("create_mock_data")
    
    # Ensure input directory exists
    INPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    generator = MockDataGenerator()
    
    try:
        # Generate app store reviews
        logger.info("Generating app store reviews...")
        reviews_df = generator.generate_app_store_reviews(50)
        reviews_file = INPUT_DIR / "app_store_reviews.csv"
        reviews_df.to_csv(reviews_file, index=False)
        logger.info(f"Created {reviews_file} with {len(reviews_df)} reviews")
        
        # Generate support emails
        logger.info("Generating support emails...")
        emails_df = generator.generate_support_emails(30)
        emails_file = INPUT_DIR / "support_emails.csv"
        emails_df.to_csv(emails_file, index=False)
        logger.info(f"Created {emails_file} with {len(emails_df)} emails")
        
        # Generate expected classifications
        logger.info("Generating expected classifications...")
        classifications_df = generator.generate_expected_classifications(reviews_df, emails_df)
        classifications_file = INPUT_DIR / "expected_classifications.csv"
        classifications_df.to_csv(classifications_file, index=False)
        logger.info(f"Created {classifications_file} with {len(classifications_df)} classifications")
        
        logger.info("Mock data generation completed successfully!")
        
        # Display summary
        total_items = len(reviews_df) + len(emails_df)
        logger.info(f"Generated {total_items} total feedback items:")
        logger.info(f"  - {len(reviews_df)} app store reviews")
        logger.info(f"  - {len(emails_df)} support emails")
        logger.info(f"  - {len(classifications_df)} expected classifications")
        
    except Exception as e:
        logger.error(f"Error generating mock data: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(create_all_mock_data())