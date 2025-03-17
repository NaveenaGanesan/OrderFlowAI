from typing import Dict, List, Optional
import anthropic
from datetime import datetime
import base64
import logging
import json

class InvoiceExtractor:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.logger = logging.getLogger(__name__)
        
    def _create_extraction_prompt(self) -> str:
        """Create a structured prompt for the VLM to extract invoice information."""
        return """Please analyze this invoice image and extract the following information in JSON format:
- Invoice number
- Order ID
- Date
- Bill to (customer name))
- Ship to (detailed shipping address including city, state, postal code, country)
- Shipping mode
- Items (with product details including product name, sub-category, category, product ID, quantity, unit price, and total amount)
- Subtotal
- Shipping cost
- Total amount
- Any additional fees or discounts
- Balance Due

Please return ONLY a valid JSON object with these exact keys, no other text:
{
    "invoice_number": "",
    "order_id": "",
    "date": "",
    "billing": {
        "name": ""
    },
    "shipping": {
        "city": "",
        "state": "",
        "postal_code": "",
        "country": "",
        "mode": "",
        "cost": 0.00
    },
    "items": [
        {
            "product_name": "",
            "sub_category": "",
            "category": "",
            "product_id": "",
            "quantity": 0,
            "unit_price": 0.00,
            "amount": 0.00
        }
    ],
    "financials": {
        "subtotal": 0.00,
        "additional_charges": [
            {
                "description": "",
                "amount": 0.00
            }
        ],
        "total": 0.00,
        "balance_due": 0.00
    }
}"""

    def extract_information(self, file_content: bytes, file_type: str) -> Dict:
        """Extract information from invoice using VLM."""
        try:
            self.logger.info("Starting information extraction from invoice")
            file_base64 = base64.b64encode(file_content).decode('utf-8')
            self.logger.debug(f"File converted to base64, MIME type: {file_type}")
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": file_type,
                                "data": file_base64
                            }
                        },
                        {
                            "type": "text",
                            "text": self._create_extraction_prompt()
                        }
                    ]
                }
            ]
            
            self.logger.info("Sending request to Claude API")
            response = self.client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=4000,
                temperature=0,
                system="You are an expert at extracting structured information from invoice images. Always return valid JSON.",
                messages=messages
            )
            
            json_str = response.content[0].text
            self.logger.info(f"Raw response from API: {json_str}")
            
            # Parse JSON string into dictionary
            extracted_data = json.loads(json_str)
            self.logger.info("Successfully extracted and parsed information from invoice")
            self.logger.debug(f"Extracted data: {extracted_data}")
            return extracted_data

        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON response: {str(e)}", exc_info=True)
            return None
        except Exception as e:
            self.logger.error(f"Error extracting information: {str(e)}", exc_info=True)
            return None

    def validate_extraction(self, extracted_data: Dict) -> List[str]:
        """Validate the extracted data for completeness and format."""
        self.logger.info("Starting validation of extracted data")
        missing_fields = []
        required_fields = {
            "invoice_number": str,
            "order_id": str,
            "date": str,
            "billing": dict,
            "shipping": dict,
            "items": list,
            "financials": dict
        }
        
        for field, expected_type in required_fields.items():
            if field not in extracted_data:
                missing_fields.append(f"Missing field: {field}")
            elif not isinstance(extracted_data[field], expected_type):
                missing_fields.append(f"Invalid type for {field}: expected {expected_type}")
        
        if missing_fields:
            self.logger.warning(f"Validation found issues: {', '.join(missing_fields)}")
        else:
            self.logger.info("Validation passed successfully")
        return missing_fields

    def clean_extracted_data(self, data: Dict) -> Dict:
        """Clean and standardize the extracted data."""
        self.logger.info("Starting data cleaning and standardization")
        if not data:
            self.logger.warning("No data provided for cleaning")
            return {}
            
        try:
            # Convert string amounts to float
            if "financials" in data:
                financials = data["financials"]
                for key in ["subtotal", "total", "balance_due"]:
                    if key in financials:
                        try:
                            financials[key] = float(str(financials[key]).replace("$", "").replace(",", ""))
                        except (ValueError, TypeError):
                            financials[key] = 0.0
                        
            # Clean items data
            if "items" in data:
                for item in data["items"]:
                    for key in ["quantity", "unit_price", "amount"]:
                        if key in item:
                            try:
                                item[key] = float(str(item[key]).replace("$", "").replace(",", ""))
                            except (ValueError, TypeError):
                                item[key] = 0.0
                        
            # Standardize date format
            if "date" in data:
                try:
                    date_obj = datetime.strptime(data["date"], "%b %d %Y")
                    data["date"] = date_obj.isoformat()
                except ValueError:
                    pass
                
            self.logger.info("Data cleaning completed successfully")
            return data
        except Exception as e:
            self.logger.error(f"Error during data cleaning: {str(e)}", exc_info=True)
            return data