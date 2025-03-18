# OrderFlowAI - Invoice Processing System
An AI-powered invoice processing system that extracts structured information from PDF invoices using Claude 3.5 Sonnet. This system leverages advanced natural language processing (NLP) capabilities to identify and extract key fields such as invoice numbers, dates, shipping details, line items, and totals. It ensures high accuracy and efficiency in handling single invoice processing tasks. The extracted data is output in a structured JSON format, making it easy to integrate with other systems or workflows.

The system also includes robust MongoDB integration for storing and managing extracted invoice data, as well as Elasticsearch support for advanced search capabilities. Users can perform full-text searches, execute complex queries, and filter invoice data efficiently. These features make it a scalable and versatile solution for various invoice processing and data retrieval needs.

## Features

- PDF to image conversion for optimal processing
- Automated information extraction using Claude AI
- Structured JSON output
- Validation and data cleaning
- Comprehensive logging
- MongoDB integration for storing and managing extracted invoice data
- Elasticsearch support for advanced search capabilities
- Full-text search and query support for invoice data
- Scalable and efficient data retrieval mechanisms
- Support for complex queries and filtering
- Integration with external systems for seamless data exchange

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

or 

```bash
chmod +x install_dependencies.sh
./install_dependencies.sh
```


4. Set up your environment variables:

```bash
echo "ANTHROPIC_API_KEY=<your-api-key>" >> .env
echo "MONGO_URI=<mongo-uri>" >> .env
echo "MONGO_DB_NAME=<mongodb-name>" >> .env
echo "MONGO_COLLECTION=<mongo-collection-name>" >> .env
echo "ES_URI=<elasticsearch-uri>" >> .env
echo "ES_INDEX=<elasticsearch-index>" >> .env
echo "ES_DOC_TYPE=<elasticsearch-doc-type>" >> .env
```

5. Run the script:

```bash
python3 invoice_pipeline.py process
```
```bash
python3 invoice_pipeline.py search "query_string"
```
```bash
python3 invoice_pipeline.py query "query_string"
```
```bash
python3 invoice_pipeline.py get invoice_number
```
```bash
python3 invoice_pipeline.py delete invoice_number
```
## Requirements

Ensure you have the following installed and configured before running the system:

- Python 3.8 or higher
- `pdf2image` library for PDF processing
- `poppler-utils` (required for `pdf2image` to work)
- An active Anthropic API key for Claude AI integration
- `pip` for managing Python packages
- A Unix-based system (Linux/MacOS) or WSL for Windows users 

## Error Handling

The system includes comprehensive error handling and logging:
- PDF conversion errors
- API communication issues
- Data validation failures
- JSON parsing errors
