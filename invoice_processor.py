import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import logging
from datetime import datetime
import mimetypes
from invoice_extractor import InvoiceExtractor
from dotenv import load_dotenv

import pdf2image  # For converting PDF to images
from PIL import Image
import io

load_dotenv()

class InvoiceProcessor:
    def __init__(self, api_key: str, output_dir: str = "results/processed_invoices"):
        self.extractor = InvoiceExtractor(api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.output_dir / 'processing.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def _convert_pdf_to_image(self, pdf_path: str) -> bytes:
        """Convert first page of PDF to JPEG image."""
        self.logger.info(f"Converting PDF to image: {pdf_path}")
        try:
            images = pdf2image.convert_from_path(pdf_path, first_page=1, last_page=1)
            if not images:
                self.logger.error("No images extracted from PDF")
                raise ValueError("No images extracted from PDF")
            
            self.logger.debug("PDF first page converted to image")
            img_byte_arr = io.BytesIO()
            images[0].save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr.seek(0)
            
            self.logger.info("PDF successfully converted to JPEG")
            return img_byte_arr.getvalue()
            
        except Exception as e:
            self.logger.error(f"Error converting PDF to image: {str(e)}", exc_info=True)
            raise

    def _get_file_info(self, file_path: str) -> Tuple[bytes, str]:
        """Get file content and appropriate MIME type."""
        self.logger.info(f"Processing file: {file_path}")
        file_path = Path(file_path)
        suffix = file_path.suffix.lower()
        
        try:
            if suffix == '.pdf':
                self.logger.debug("Processing PDF file")
                file_content = self._convert_pdf_to_image(str(file_path))
                mime_type = 'image/jpeg'
            else:
                self.logger.debug(f"Processing image file with extension: {suffix}")
                with open(file_path, 'rb') as f:
                    file_content = f.read()
                
                mime_types = {
                    '.jpg': 'image/jpeg',
                    '.jpeg': 'image/jpeg',
                    '.png': 'image/png',
                    '.gif': 'image/gif',
                    '.webp': 'image/webp'
                }
                mime_type = mime_types.get(suffix)
                
                if not mime_type:
                    self.logger.error(f"Unsupported file type: {suffix}")
                    raise ValueError(f"Unsupported file type: {suffix}")
            
            self.logger.info(f"File processed successfully. MIME type: {mime_type}")
            return file_content, mime_type
            
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}", exc_info=True)
            raise

    def process_single_invoice(self, invoice_path: str) -> Dict:
        """Process a single invoice and return extracted data."""
        self.logger.info(f"Processing invoice: {invoice_path}")
        
        try:
            # Get file content and MIME type
            file_content, mime_type = self._get_file_info(invoice_path)
            
            # Extract information using VLM
            extracted_data = self.extractor.extract_information(file_content, mime_type)
            
            if not extracted_data:
                self.logger.error(f"Failed to extract data from {invoice_path}")
                return {}
            
            # Validate extraction
            validation_errors = self.extractor.validate_extraction(extracted_data)
            if validation_errors:
                self.logger.warning(f"Validation errors for {invoice_path}: {validation_errors}")
            
            # Clean and standardize the data
            cleaned_data = self.extractor.clean_extracted_data(extracted_data)
            
            # Save the processed data
            self._save_processed_data(invoice_path, cleaned_data)
            
            return cleaned_data
            
        except Exception as e:
            self.logger.error(f"Error processing invoice {invoice_path}: {str(e)}")
            return {}

    def _save_processed_data(self, invoice_path: str, data: Dict):
        """Save processed data to JSON file."""
        try:
            invoice_name = Path(invoice_path).stem
            output_path = self.output_dir / f"{invoice_name}_processed.json"
            
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            self.logger.info(f"Saved processed data to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving processed data: {str(e)}")

def process_invoice(api_key: str, invoice_path: str):
    """Process a sample invoice and print the results."""
    processor = InvoiceProcessor(api_key)
    result = processor.process_single_invoice(invoice_path)
    print("\nExtracted Invoice Data:")
    print(json.dumps(result, indent=2))
    return result

if __name__ == "__main__":
    # Initialize the processor with your API key
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("Please set ANTHROPIC_API_KEY environment variable")
        exit(1)
        
    # Process a single invoice
    invoice_path = "Datasets/PDF_Invoice_Folder/invoice_Max Jones_5179.pdf"
    process_invoice(api_key, invoice_path)
