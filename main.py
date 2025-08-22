"""
KYC Identity Verification PoC using Fireworks AI
=================================================
A proof-of-concept solution for extracting identity information from 
various document types (passports, driver's licenses) using Fireworks AI's
vision-language models and structured output capabilities.

Author: Michael Hopko
Date: August 21, 2025
Time to Build: ~3 hours
"""

import os
import base64
import json
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
import logging
import argparse
import time
import random

from dotenv import load_dotenv

from pydantic import BaseModel, Field, field_validator
from fireworks.client import Fireworks
import asyncio
from concurrent.futures import ThreadPoolExecutor
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =====================================================================
# CONFIGURATION
# =====================================================================

# Load environment variables from .env file
load_dotenv()

# Set your Fireworks API key as environment variable
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
if not FIREWORKS_API_KEY:
    raise ValueError("FIREWORKS_API_KEY environment variable is required. Please set it in your .env file or environment.")

# Model selection - using Qwen2.5 VL for better document processing
# Alternative: "llama-v3-2-11b-vision-instruct" for faster processing
VISION_MODEL = os.getenv("VISION_MODEL", "accounts/fireworks/models/qwen2p5-vl-32b-instruct")

# For structured extraction with JSON mode (text-only model)
TEXT_MODEL = os.getenv("TEXT_MODEL", "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507")

# Processing Configuration
MAX_TOKENS_DETECTION = 50
MAX_TOKENS_EXTRACTION = 500
MAX_TOKENS_STRUCTURING = 1000
TEMPERATURE_DETECTION = 0.1
TEMPERATURE_STRUCTURING = 0
CONFIDENCE_THRESHOLD = 0.8
VERIFIED_THRESHOLD = 0.7
MANUAL_REVIEW_THRESHOLD = 0.8
EXTRACTION_SUCCESS_THRESHOLD = 0.5
DEFAULT_CONFIDENCE = 0.3
CONFIDENCE_PENALTY_PER_WARNING = 0.1
MAX_WORKERS = 3
FILE_READ_CHUNK_SIZE = 4096
LOG_PREVIEW_LENGTH = 200
SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
CACHE_DIR = "cache"
ENABLE_CACHE = True

# Rate Limiting Configuration
REQUEST_DELAY_SECONDS = float(os.getenv("REQUEST_DELAY_SECONDS", "0.5"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "5"))
BASE_BACKOFF_SECONDS = float(os.getenv("BASE_BACKOFF_SECONDS", "1.0"))
MAX_BACKOFF_SECONDS = float(os.getenv("MAX_BACKOFF_SECONDS", "16.0"))
JITTER_ENABLED = os.getenv("JITTER_ENABLED", "true").lower() == "true"

# =====================================================================
# DATA MODELS (Pydantic Schemas)
# =====================================================================

class DocumentType(str, Enum):
    """Supported document types for KYC verification"""
    PASSPORT = "passport"
    DRIVERS_LICENSE = "drivers_license"
    NATIONAL_ID = "national_id"
    UNKNOWN = "unknown"

class Address(BaseModel):
    """Address information schema"""
    street: Optional[str] = Field(None, description="Street address")
    city: Optional[str] = Field(None, description="City")
    state: Optional[str] = Field(None, description="State or province")
    postal_code: Optional[str] = Field(None, description="Postal/ZIP code")
    country: Optional[str] = Field(None, description="Country")
    
    def __str__(self):
        parts = [self.street, self.city, self.state, self.postal_code, self.country]
        return ", ".join(filter(None, parts))

class PersonalInfo(BaseModel):
    """Personal information extracted from ID documents"""
    first_name: Optional[str] = Field(None, description="First/given name")
    last_name: Optional[str] = Field(None, description="Last/family name")
    middle_name: Optional[str] = Field(None, description="Middle name or initial")
    date_of_birth: Optional[str] = Field(None, description="Date of birth (YYYY-MM-DD)")
    gender: Optional[str] = Field(None, description="Gender (M/F/X)")
    nationality: Optional[str] = Field(None, description="Nationality/citizenship")
    
    @field_validator('date_of_birth')
    @classmethod
    def validate_dob(cls, v):
        """Validate and standardize date format"""
        if v:
            try:
                # Try to parse various date formats and standardize to YYYY-MM-DD
                for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%B %d, %Y', '%d %b %Y']:
                    try:
                        dt = datetime.strptime(v, fmt)
                        return dt.strftime('%Y-%m-%d')
                    except ValueError:
                        continue
            except Exception:
                pass
        return v

class DocumentInfo(BaseModel):
    """Document-specific information"""
    document_type: DocumentType = Field(..., description="Type of document")
    document_number: Optional[str] = Field(None, description="Document/ID number")
    issue_date: Optional[str] = Field(None, description="Issue date (YYYY-MM-DD)")
    expiry_date: Optional[str] = Field(None, description="Expiry date (YYYY-MM-DD)")
    issuing_authority: Optional[str] = Field(None, description="Issuing authority/country")
    machine_readable_zone: Optional[List[str]] = Field(None, description="MRZ lines if present")

class SecurityFeatures(BaseModel):
    """Security and validation features detected"""
    has_photo: bool = Field(False, description="Photo present")
    has_signature: bool = Field(False, description="Signature present")
    has_hologram: bool = Field(False, description="Hologram/security features detected")
    has_barcode: bool = Field(False, description="Barcode/QR code present")
    confidence_score: float = Field(0.0, description="Overall confidence score (0-1)")
    
    @field_validator('has_photo', 'has_signature', 'has_hologram', 'has_barcode', mode='before')
    @classmethod
    def convert_none_to_false(cls, v):
        """Convert None/null to False for boolean fields"""
        return False if v is None else v
    
    @field_validator('confidence_score', mode='before')
    @classmethod
    def convert_none_to_default(cls, v):
        """Convert None/null to default confidence score"""
        return 0.3 if v is None else v

class KYCExtractionResult(BaseModel):
    """Complete KYC extraction result"""
    personal_info: PersonalInfo = Field(default_factory=PersonalInfo)
    document_info: DocumentInfo = Field(default_factory=lambda: DocumentInfo(document_type=DocumentType.UNKNOWN))
    address: Optional[Address] = None
    security_features: SecurityFeatures = Field(default_factory=SecurityFeatures)
    extraction_timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    processing_time_ms: Optional[int] = None
    warnings: List[str] = Field(default_factory=list)
    
    @field_validator('personal_info', mode='before')
    @classmethod
    def ensure_personal_info(cls, v):
        """Ensure personal_info is valid"""
        if v is None or not isinstance(v, dict):
            return {}
        return v
    
    @field_validator('document_info', mode='before')
    @classmethod
    def ensure_document_info(cls, v):
        """Ensure document_info is valid"""
        if v is None or not isinstance(v, dict):
            return {"document_type": "unknown"}
        return v
    
    @field_validator('security_features', mode='before')
    @classmethod
    def ensure_security_features(cls, v):
        """Ensure security_features is valid"""
        if v is None or not isinstance(v, dict):
            return {}
        return v
    
    def to_kyc_record(self) -> Dict[str, Any]:
        """Convert to KYC-compliant record format"""
        return {
            "customer_identification": {
                "name": f"{self.personal_info.first_name or ''} {self.personal_info.middle_name or ''} {self.personal_info.last_name or ''}".strip(),
                "date_of_birth": self.personal_info.date_of_birth,
                "identification_number": self.document_info.document_number,
                "address": str(self.address) if self.address else None
            },
            "document_verification": {
                "type": self.document_info.document_type.value,
                "expiry": self.document_info.expiry_date,
                "issuer": self.document_info.issuing_authority,
                "verified": self.security_features.confidence_score > VERIFIED_THRESHOLD
            },
            "risk_assessment": {
                "confidence_score": self.security_features.confidence_score,
                "requires_manual_review": len(self.warnings) > 0 or self.security_features.confidence_score < MANUAL_REVIEW_THRESHOLD
            },
            "metadata": {
                "extraction_timestamp": self.extraction_timestamp,
                "processing_time_ms": self.processing_time_ms
            }
        }

# =====================================================================
# DOCUMENT PROCESSOR
# =====================================================================

class DocumentProcessor:
    """Main class for processing identity documents"""
    
    def __init__(self, api_key: str):
        """Initialize the processor with Fireworks client"""
        self.client = Fireworks(api_key=api_key)
        self.executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        
        # Setup cache directory
        if ENABLE_CACHE:
            self.cache_dir = Path(CACHE_DIR)
            self.cache_dir.mkdir(exist_ok=True)
        else:
            self.cache_dir = None
    
    def _get_cache_path(self, file_hash: str, cache_type: str) -> Path:
        """Get cache file path for a given file hash and type"""
        if not self.cache_dir:
            return None
        return self.cache_dir / f"{file_hash}_{cache_type}.json"
    
    def _load_from_cache(self, file_hash: str, cache_type: str) -> Optional[Dict[str, Any]]:
        """Load cached data if available"""
        if not ENABLE_CACHE or not self.cache_dir:
            return None
        
        cache_path = self._get_cache_path(file_hash, cache_type)
        if cache_path and cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_path}: {e}")
        return None
    
    def _save_to_cache(self, file_hash: str, cache_type: str, data: Dict[str, Any]) -> None:
        """Save data to cache"""
        if not ENABLE_CACHE or not self.cache_dir:
            return
        
        cache_path = self._get_cache_path(file_hash, cache_type)
        if cache_path:
            try:
                with open(cache_path, 'w') as f:
                    json.dump(data, f, indent=2)
                logger.debug(f"Saved cache: {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save cache {cache_path}: {e}")
        
    def encode_image(self, image_path: Union[str, Path]) -> str:
        """Encode image to base64 for API transmission"""
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            logger.error(f"Error encoding image: {e}")
            raise
    
    def detect_document_type(self, image_base64: str, file_hash: str = None) -> DocumentType:
        """Detect the type of document from the image"""
        
        # Check cache first
        if file_hash:
            cached_type = self._load_from_cache(file_hash, "doc_type")
            if cached_type:
                logger.info(f"Using cached document type: {cached_type['document_type']}")
                return DocumentType(cached_type['document_type'])
        
        try:
            response = api_request_with_retry(
                self.client.chat.completions.create,
                model=VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"""Identify the type of this identity document. 
                            Respond with ONLY one of: {', '.join([dt.value for dt in DocumentType])}."""
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        }
                    ]
                }],
                max_tokens=MAX_TOKENS_DETECTION,
                temperature=TEMPERATURE_DETECTION
            )
            
            doc_type_str = response.choices[0].message.content.strip().lower().replace(" ", "_").replace("-", "_")
            
            # Map response to enum
            if "passport" in doc_type_str:
                doc_type = DocumentType.PASSPORT
            elif "driver" in doc_type_str or "license" in doc_type_str:
                doc_type = DocumentType.DRIVERS_LICENSE
            elif "national" in doc_type_str or "id" in doc_type_str:
                doc_type = DocumentType.NATIONAL_ID
            else:
                doc_type = DocumentType.UNKNOWN
            
            # Save to cache
            if file_hash:
                self._save_to_cache(file_hash, "doc_type", {"document_type": doc_type.value})
            
            return doc_type
                
        except Exception as e:
            logger.error(f"Error detecting document type: {e}")
            return DocumentType.UNKNOWN
    
    def extract_with_vision_model(self, image_base64: str, doc_type: DocumentType, file_hash: str = None) -> Dict[str, Any]:
        """Extract information using vision model"""
        
        # Check cache first for raw vision output
        if file_hash:
            cached_raw_text = self._load_from_cache(file_hash, "vision_raw_text")
            if cached_raw_text:
                logger.info("Using cached vision output")
                # Still need to structure it with text model
                return self.structure_extraction(cached_raw_text['raw_text'], doc_type)
        
        # Customize prompt based on document type
        prompts = {
            DocumentType.PASSPORT: """Extract ALL the following information from this passport:
                - Full name (first, middle, last)
                - Date of birth
                - Passport number
                - Nationality/Country
                - Issue date
                - Expiry date
                - Gender
                - Place of birth
                - MRZ (Machine Readable Zone) if visible
                Be precise and extract exactly what you see.""",
            
            DocumentType.DRIVERS_LICENSE: """Extract ALL the following information from this driver's license:
                - Full name (first, middle, last)
                - Date of birth
                - License number
                - Address (street, city, state, postal code)
                - Issue date
                - Expiry date
                - Gender
                - License class/type
                - Restrictions
                - State/Country of issue
                Be precise and extract exactly what you see.""",
            
            DocumentType.NATIONAL_ID: """Extract ALL the following information from this ID card:
                - Full name (first, middle, last)
                - Date of birth
                - ID number
                - Address if present
                - Issue date
                - Expiry date
                - Gender
                - Nationality
                - Issuing authority
                Be precise and extract exactly what you see.""",
            
            DocumentType.UNKNOWN: """Extract all identifying information from this document including:
                - Any names
                - Any dates
                - Any ID numbers
                - Any addresses
                - Document type if identifiable"""
        }
        
        prompt = prompts.get(doc_type, prompts[DocumentType.UNKNOWN])
        
        try:
            response = api_request_with_retry(
                self.client.chat.completions.create,
                model=VISION_MODEL,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
                        }
                    ]
                }],
                max_tokens=MAX_TOKENS_EXTRACTION,
                temperature=TEMPERATURE_DETECTION
            )
            
            raw_text = response.choices[0].message.content
            logger.info(f"Raw extraction: {raw_text[:LOG_PREVIEW_LENGTH]}...")
            
            # Save raw vision output to cache (before text model processing)
            if file_hash:
                self._save_to_cache(file_hash, "vision_raw_text", {"raw_text": raw_text})
            
            # Parse the raw text into structured format using the text model with JSON mode
            structured_data = self.structure_extraction(raw_text, doc_type)
            
            return structured_data
            
        except Exception as e:
            logger.error(f"Error in vision extraction: {e}")
            raise
    
    def structure_extraction(self, raw_text: str, doc_type: DocumentType) -> Dict[str, Any]:
        """Use text model with JSON mode to structure the extracted data"""
        
        prompt = f"""Given this extracted text from a {doc_type.value}, structure it into the required JSON format.

        EXAMPLE 1 - Driver's License:
        Input: "First Name: John, Last Name: Smith, DOB: 01/15/1990, License: D123456789, Address: 123 Main St, Anytown, CA 90210"
        Output: {{
          "personal_info": {{
            "first_name": "John",
            "last_name": "Smith", 
            "middle_name": null,
            "date_of_birth": "1990-01-15",
            "gender": null,
            "nationality": null
          }},
          "document_info": {{
            "document_type": "drivers_license",
            "document_number": "D123456789",
            "issue_date": null,
            "expiry_date": null,
            "issuing_authority": null,
            "machine_readable_zone": []
          }},
          "address": {{
            "street": "123 Main St",
            "city": "Anytown", 
            "state": "CA",
            "postal_code": "90210",
            "country": null
          }},
          "security_features": {{
            "has_photo": true,
            "has_signature": false,
            "has_hologram": false,
            "has_barcode": false,
            "confidence_score": 0.85
          }}
        }}

        EXAMPLE 2 - Passport:
        Input: "Name: Jane Doe, DOB: March 5, 1985, Passport: P987654321, Nationality: USA, Issue: 2020, Expiry: 2030"
        Output: {{
          "personal_info": {{
            "first_name": "Jane",
            "last_name": "Doe",
            "middle_name": null, 
            "date_of_birth": "1985-03-05",
            "gender": null,
            "nationality": "USA"
          }},
          "document_info": {{
            "document_type": "passport",
            "document_number": "P987654321",
            "issue_date": "2020",
            "expiry_date": "2030", 
            "issuing_authority": null,
            "machine_readable_zone": []
          }},
          "address": null,
          "security_features": {{
            "has_photo": true,
            "has_signature": true,
            "has_hologram": true,
            "has_barcode": false,
            "confidence_score": 0.90
          }}
        }}

        Now process this extracted text:
        {raw_text}
        
        Return ONLY valid JSON in the same format as the examples above. Use null for missing fields."""
        
        try:
            response = api_request_with_retry(
                self.client.chat.completions.create,
                model=TEXT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a data structuring assistant. Always return valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                max_tokens=MAX_TOKENS_STRUCTURING,
                temperature=TEMPERATURE_STRUCTURING
            )
            
            structured_data = json.loads(response.choices[0].message.content)
            
            # Add document type if not present
            if "document_info" in structured_data:
                structured_data["document_info"]["document_type"] = doc_type.value
            
            return structured_data
            
        except Exception as e:
            logger.error(f"Error structuring data: {e}")
            # Return a basic structure with the raw text
            return {
                "personal_info": {},
                "document_info": {"document_type": doc_type.value},
                "security_features": {"confidence_score": DEFAULT_CONFIDENCE},
                "warnings": [f"Failed to structure data: {str(e)}"]
            }
    
    def validate_extraction(self, result: KYCExtractionResult) -> KYCExtractionResult:
        """Validate and add warnings for missing or suspicious data"""
        warnings = []
        
        # Check for required fields
        if not result.personal_info.first_name or not result.personal_info.last_name:
            warnings.append("Name information incomplete")
        
        if not result.personal_info.date_of_birth:
            warnings.append("Date of birth missing")
        
        if not result.document_info.document_number:
            warnings.append("Document number missing")
        
        if result.document_info.expiry_date:
            try:
                expiry = datetime.strptime(result.document_info.expiry_date, '%Y-%m-%d')
                if expiry < datetime.now():
                    warnings.append("Document appears to be expired")
            except Exception as e:
                warnings.append(f"Invalid expiry date format: {e}")
        
        # Check document type detection confidence
        if result.document_info.document_type == DocumentType.UNKNOWN:
            warnings.append("Document type could not be determined")
        
        # Add warnings to result
        result.warnings.extend(warnings)
        
        # Adjust confidence score based on warnings
        if warnings:
            result.security_features.confidence_score *= (1 - CONFIDENCE_PENALTY_PER_WARNING * len(warnings))
        
        return result
    
    async def process_document_async(self, image_path: Union[str, Path]) -> KYCExtractionResult:
        """Asynchronously process a document for KYC extraction"""
        start_time = datetime.now()
        
        try:
            # Compute file hash for caching
            file_hash = hash_file(image_path) if ENABLE_CACHE else None
            
            # Encode image
            logger.info(f"Processing document: {image_path}")
            image_base64 = self.encode_image(image_path)
            
            # Detect document type
            doc_type = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.detect_document_type, image_base64, file_hash
            )
            logger.info(f"Detected document type: {doc_type.value}")
            
            # Extract information
            extracted_data = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.extract_with_vision_model, image_base64, doc_type, file_hash
            )
            
            # Create result object
            result = KYCExtractionResult(**extracted_data)
            
            # Calculate processing time
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            result.processing_time_ms = processing_time
            
            # Validate and add warnings
            result = self.validate_extraction(result)
            
            logger.info(f"Processing completed in {processing_time}ms")
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise
    
    def process_document(self, image_path: Union[str, Path]) -> KYCExtractionResult:
        """Synchronous wrapper for document processing"""
        return asyncio.run(self.process_document_async(image_path))
    
    async def process_batch_async(self, image_paths: List[Union[str, Path]], sequential: bool = True) -> List[KYCExtractionResult]:
        """Process multiple documents either sequentially or in parallel"""
        if sequential:
            # Process sequentially to avoid rate limits
            results = []
            for i, path in enumerate(image_paths):
                try:
                    logger.info(f"Processing batch item {i + 1}/{len(image_paths)}: {path}")
                    result = await self.process_document_async(path)
                    results.append(result)
                    
                    # Add delay between documents except for the last one
                    if i < len(image_paths) - 1 and REQUEST_DELAY_SECONDS > 0:
                        logger.debug(f"Waiting {REQUEST_DELAY_SECONDS}s before next document")
                        await asyncio.sleep(REQUEST_DELAY_SECONDS)
                        
                except Exception as e:
                    logger.error(f"Error processing {path}: {e}")
                    results.append(e)
            return results
        else:
            # Original parallel processing (may hit rate limits)
            tasks = [self.process_document_async(path) for path in image_paths]
            return await asyncio.gather(*tasks, return_exceptions=True)
    
    def process_batch(self, image_paths: List[Union[str, Path]], sequential: bool = True) -> List[KYCExtractionResult]:
        """Synchronous wrapper for batch processing"""
        return asyncio.run(self.process_batch_async(image_paths, sequential=sequential))

# =====================================================================
# COMPLIANCE & AUDIT
# =====================================================================

class ComplianceAuditor:
    """Handle compliance and audit trail for KYC processes"""
    
    def __init__(self):
        self.audit_log = []
    
    def log_extraction(self, result: KYCExtractionResult, document_hash: str):
        """Log extraction for audit purposes"""
        audit_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "document_hash": document_hash,
            "document_type": result.document_info.document_type.value,
            "extraction_success": result.security_features.confidence_score > EXTRACTION_SUCCESS_THRESHOLD,
            "warnings_count": len(result.warnings),
            "processing_time_ms": result.processing_time_ms,
            "requires_manual_review": len(result.warnings) > 0 or result.security_features.confidence_score < MANUAL_REVIEW_THRESHOLD
        }
        self.audit_log.append(audit_entry)
        logger.info(f"Audit logged: {audit_entry}")
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report from audit log"""
        if not self.audit_log:
            return {"status": "No entries to report"}
        
        total_processed = len(self.audit_log)
        successful = sum(1 for e in self.audit_log if e["extraction_success"])
        requiring_review = sum(1 for e in self.audit_log if e["requires_manual_review"])
        avg_processing_time = sum(e["processing_time_ms"] for e in self.audit_log) / total_processed
        
        return {
            "report_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_documents_processed": total_processed,
            "successful_extractions": successful,
            "success_rate": f"{(successful/total_processed)*100:.2f}%",
            "requiring_manual_review": requiring_review,
            "average_processing_time_ms": avg_processing_time,
            "document_types_processed": list(set(e["document_type"] for e in self.audit_log))
        }

# =====================================================================
# MAIN EXECUTION & TESTING
# =====================================================================

def hash_file(filepath: Union[str, Path]) -> str:
    """Create SHA256 hash of file for audit purposes"""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(FILE_READ_CHUNK_SIZE), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def clear_cache_directory():
    """Clear all files in the cache directory"""
    cache_dir = Path(CACHE_DIR)
    if cache_dir.exists():
        for cache_file in cache_dir.iterdir():
            if cache_file.is_file():
                try:
                    cache_file.unlink()
                    logger.info(f"Removed cache file: {cache_file}")
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {cache_file}: {e}")
        print(f"🗑️  Cache directory cleared: {cache_dir}")
    else:
        print(f"📁 Cache directory does not exist: {cache_dir}")

def is_rate_limit_error(error_message: str) -> bool:
    """Check if the error is a rate limit error"""
    rate_limit_indicators = [
        "rate limit exceeded",
        "too many requests", 
        "429",
        "rate_limit",
        "quota exceeded"
    ]
    error_lower = str(error_message).lower()
    return any(indicator in error_lower for indicator in rate_limit_indicators)

def calculate_backoff_delay(attempt: int) -> float:
    """Calculate exponential backoff delay with optional jitter"""
    delay = min(BASE_BACKOFF_SECONDS * (2 ** attempt), MAX_BACKOFF_SECONDS)
    
    if JITTER_ENABLED:
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0, delay * 0.1)
        delay += jitter
    
    return delay

def api_request_with_retry(func, *args, **kwargs):
    """Execute API request with retry logic for rate limits"""
    last_exception = None
    
    for attempt in range(MAX_RETRIES + 1):
        try:
            # Add delay before request (except first attempt)
            if attempt > 0:
                delay = calculate_backoff_delay(attempt - 1)
                logger.info(f"Retrying in {delay:.2f} seconds (attempt {attempt + 1}/{MAX_RETRIES + 1})")
                time.sleep(delay)
            elif REQUEST_DELAY_SECONDS > 0:
                # Small delay even on first request to prevent overwhelming API
                time.sleep(REQUEST_DELAY_SECONDS)
            
            # Execute the function
            result = func(*args, **kwargs)
            
            # Success - log if this was a retry
            if attempt > 0:
                logger.info(f"Request succeeded on attempt {attempt + 1}")
            
            return result
            
        except Exception as e:
            last_exception = e
            error_message = str(e)
            
            # Check if this is a rate limit error
            if is_rate_limit_error(error_message):
                if attempt < MAX_RETRIES:
                    logger.warning(f"Rate limit hit (attempt {attempt + 1}): {error_message}")
                    continue
                else:
                    logger.error(f"Max retries exceeded for rate limit error: {error_message}")
            else:
                # Not a rate limit error, don't retry
                logger.error(f"Non-retryable error: {error_message}")
                break
    
    # If we get here, all retries failed
    raise last_exception

def main(no_cache=False):
    """Main execution function for testing the KYC system"""
    
    # Clear cache if requested
    if no_cache:
        clear_cache_directory()
        # Temporarily disable cache for this run
        global ENABLE_CACHE
        ENABLE_CACHE = False
    
    # Initialize components
    processor = DocumentProcessor(api_key=FIREWORKS_API_KEY)
    auditor = ComplianceAuditor()
    
    # Get all image files from the documents folder (relative to script location)
    script_dir = Path(__file__).parent
    documents_folder = script_dir / "documents"
    
    if not documents_folder.exists():
        documents_folder.mkdir()
        print(f"📁 Created documents folder: {documents_folder}")
        print("   Drop your document images in this folder and run again.")
        return
    
    # Supported image formats
    image_extensions = SUPPORTED_IMAGE_FORMATS
    test_documents = [
        f for f in documents_folder.iterdir() 
        if f.is_file() and f.suffix.lower() in image_extensions
    ]
    
    if not test_documents:
        print(f"📁 No image files found in {documents_folder}")
        print(f"   Supported formats: {', '.join(image_extensions)}")
        return
    
    print("=" * 80)
    print("KYC IDENTITY VERIFICATION SYSTEM - PROOF OF CONCEPT")
    print("=" * 80)
    print()
    
    # Process documents sequentially to avoid rate limits
    print(f"📊 Processing {len(test_documents)} documents sequentially...")
    print(f"⏱️  Rate limiting: {REQUEST_DELAY_SECONDS}s delay, {MAX_RETRIES} retries with exponential backoff")
    print()
    
    for i, doc_path in enumerate(test_documents):
        if not doc_path.exists():
            print(f"⚠️  Skipping {doc_path} - file not found")
            continue
        
        print(f"📄 Processing ({i + 1}/{len(test_documents)}): {doc_path}")
        print("-" * 40)
        
        try:
            # Process document
            result = processor.process_document(doc_path)
            
            # Log for audit
            doc_hash = hash_file(doc_path)
            auditor.log_extraction(result, doc_hash)
            
            # Display results
            print(f"✅ Document Type: {result.document_info.document_type.value}")
            print(f"👤 Name: {result.personal_info.first_name} {result.personal_info.last_name}")
            print(f"📅 DOB: {result.personal_info.date_of_birth}")
            print(f"🆔 Document #: {result.document_info.document_number}")
            print(f"📊 Confidence: {result.security_features.confidence_score:.2%}")
            
            if result.warnings:
                print(f"⚠️  Warnings: {', '.join(result.warnings)}")
            
            # Generate KYC record
            kyc_record = result.to_kyc_record()
            print(f"📋 KYC Compliant: {'✅' if kyc_record['document_verification']['verified'] else '❌'}")
            print(f"⏱️  Processing Time: {result.processing_time_ms}ms")
            
            # Note: Results displayed above - not saved to disk for security
            
        except Exception as e:
            error_msg = str(e)
            if is_rate_limit_error(error_msg):
                print(f"❌ Rate limit error (retries exhausted): {e}")
                print(f"💡 Try increasing REQUEST_DELAY_SECONDS or reducing batch size")
            else:
                print(f"❌ Error processing document: {e}")
        
        print()
    
    # Generate compliance report
    print("=" * 80)
    print("COMPLIANCE REPORT")
    print("=" * 80)
    report = auditor.generate_compliance_report()
    for key, value in report.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    # Note: Set your FIREWORKS_API_KEY environment variable before running
    # export FIREWORKS_API_KEY="your-api-key-here"
    
    # Setup command line argument parsing
    parser = argparse.ArgumentParser(
        description="KYC Identity Verification PoC using Fireworks AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run with cache enabled
  python main.py --no-cache         # Run without cache (fresh processing)
        """
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable cache usage and clear existing cache for fresh processing'
    )
    
    args = parser.parse_args()
    
    # Run main function with parsed arguments
    main(no_cache=args.no_cache)

# =====================================================================
# USAGE EXAMPLES
# =====================================================================

"""
EXAMPLE 1: Basic single document processing
-------------------------------------------
processor = DocumentProcessor(api_key="your-api-key")
result = processor.process_document("passport.jpg")
print(result.to_kyc_record())

EXAMPLE 2: Batch processing
---------------------------
processor = DocumentProcessor(api_key="your-api-key")
documents = ["passport1.jpg", "license1.jpg", "id_card1.jpg"]
results = processor.process_batch(documents)
for result in results:
    if isinstance(result, Exception):
        print(f"Error: {result}")
    else:
        print(result.to_kyc_record())

EXAMPLE 3: With compliance audit
--------------------------------
processor = DocumentProcessor(api_key="your-api-key")
auditor = ComplianceAuditor()

result = processor.process_document("document.jpg")
auditor.log_extraction(result, hash_file("document.jpg"))

report = auditor.generate_compliance_report()
print(report)
"""