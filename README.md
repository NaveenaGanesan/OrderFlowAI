# OrderFlowAI - Invoice Processing System

An AI-powered invoice processing system that extracts structured information from PDF invoices using Claude 3.5 Sonnet.

## Features

- PDF to image conversion for optimal processing
- Automated information extraction using Claude AI
- Structured JSON output
- Validation and data cleaning
- Comprehensive logging
- Support for both single invoice and batch processing

## Setup

1. Clone the repository:

```bash
git clone git@github.com:NaveenaGanesan/OrderFlowAI.git
cd OrderFlowAI
```

2. Create and activate a virtual environment:

```bash
python -m venv myenv
source myenv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Set up your environment variables:

```bash
echo "ANTHROPIC_API_KEY=<your-api-key>" >> .env
```

5. Run the script:

```bash
python invoice_processor.py
```

## Requirements

- Python 3.8+
- Poppler (for PDF processing)
- Anthropic API key

## Error Handling

The system includes comprehensive error handling and logging:
- PDF conversion errors
- API communication issues
- Data validation failures
- JSON parsing errors