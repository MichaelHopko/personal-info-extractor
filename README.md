# KYC Identity Verification PoC

A proof-of-concept solution for extracting identity information from various document types (passports, driver's licenses, national IDs) using Fireworks AI's vision-language models and structured output capabilities.

## Features

- =
 **Document Type Detection** - Automatically identifies passport, driver's license, or national ID
- =� **Information Extraction** - Extracts personal info, document details, and addresses
- =� **Security Features** - Detects photos, signatures, holograms, and barcodes
- =� **Confidence Scoring** - Provides reliability scores for extracted data
- = **Caching System** - Caches results to avoid redundant API calls
- � **Rate Limit Handling** - Built-in retry logic with exponential backoff
- =� **KYC Compliance** - Generates compliant records for regulatory requirements
- =� **Audit Trail** - Comprehensive logging and compliance reporting

## Quick Start

### 1. Install Dependencies

```bash
# Install using uv (recommended)
uv sync

# Or using pip
pip install fireworks-ai pydantic python-dotenv
```

### 2. Setup API Key

1. Get your Fireworks API key from [https://fireworks.ai/api-keys](https://fireworks.ai/api-keys)
2. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
3. Edit `.env` and add your API key:
   ```bash
   FIREWORKS_API_KEY=your-actual-api-key-here
   ```

### 3. Add Documents

Create a `documents/` folder and add your identity document images:

```bash
mkdir documents
# Copy your document images (PNG, JPG, JPEG supported) into the documents folder
```

### 4. Run the System

```bash
# Basic run with caching
uv run main.py

# Fresh run without cache
uv run main.py --no-cache
```

## Usage Examples

### Basic Document Processing

```python
from main import DocumentProcessor

processor = DocumentProcessor(api_key="your-api-key")
result = processor.process_document("passport.jpg")
print(result.to_kyc_record())
```

### Batch Processing

```python
documents = ["passport1.jpg", "license1.jpg", "id_card1.jpg"]
results = processor.process_batch(documents)
for result in results:
    if isinstance(result, Exception):
        print(f"Error: {result}")
    else:
        print(result.to_kyc_record())
```

### With Compliance Audit

```python
from main import DocumentProcessor, ComplianceAuditor, hash_file

processor = DocumentProcessor(api_key="your-api-key")
auditor = ComplianceAuditor()

result = processor.process_document("document.jpg")
auditor.log_extraction(result, hash_file("document.jpg"))

report = auditor.generate_compliance_report()
print(report)
```

## Configuration

You can customize behavior via environment variables in your `.env` file:

```bash
# API Configuration
FIREWORKS_API_KEY=your-api-key-here
VISION_MODEL=accounts/fireworks/models/qwen2p5-vl-32b-instruct
TEXT_MODEL=accounts/fireworks/models/qwen3-235b-a22b-instruct-2507

# Rate Limiting (helps avoid API limits)
REQUEST_DELAY_SECONDS=0.5
MAX_RETRIES=3
BASE_BACKOFF_SECONDS=1.0
MAX_BACKOFF_SECONDS=16.0
JITTER_ENABLED=true
```

## Command Line Options

```bash
python main.py --help
```

- `--no-cache` - Disable cache and force fresh processing of all documents

## Supported Document Types

-  **Passports** - Full extraction including MRZ data
-  **Driver's Licenses** - Personal info, address, license details
-  **National ID Cards** - Basic identity information
- � **Unknown Documents** - Basic text extraction for unrecognized types

## Supported Image Formats

- PNG, JPG, JPEG, BMP, TIFF, WebP

## Output Format

The system provides structured JSON output with:

```json
{
  "personal_info": {
    "first_name": "John",
    "last_name": "Doe",
    "date_of_birth": "1990-01-15",
    "gender": "M",
    "nationality": "USA"
  },
  "document_info": {
    "document_type": "passport",
    "document_number": "123456789",
    "issue_date": "2020-01-01",
    "expiry_date": "2030-01-01",
    "issuing_authority": "United States"
  },
  "security_features": {
    "has_photo": true,
    "has_signature": true,
    "confidence_score": 0.95
  }
}
```

## Security & Privacy

- � **No Data Persistence** - Results are displayed but not saved to disk
- = **Local Processing** - Documents stay on your machine (only sent to Fireworks API)
- =� **Cache Management** - Use `--no-cache` to clear cached results
- =� **Audit Logging** - All processing activities are logged for compliance

## Rate Limiting

The system automatically handles Fireworks API rate limits with:

- Exponential backoff retry logic
- Configurable delays between requests
- Sequential processing to avoid overwhelming the API
- Helpful error messages when limits are exceeded

## Development

Built in ~3 hours as a proof-of-concept by Michael Hopko on August 21, 2025.

### Architecture

- **Document Processor** - Main processing engine
- **Pydantic Models** - Type-safe data structures
- **Compliance Auditor** - Audit trail and reporting
- **Rate Limiting** - Robust API error handling
- **Caching System** - Performance optimization

## License

Proof-of-concept code - use at your own discretion.