import os
from typing import List, Dict, Optional
from pathlib import Path
import json
import logging
from datetime import datetime
from invoice_processor import InvoiceProcessor
from dotenv import load_dotenv

load_dotenv()

class InvoiceProcessingPipeline:
    def __init__(self, api_key: str, output_dir: str = "results/processed_invoices", db_config: Optional[Dict] = None):
        # Set up logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Path(output_dir) / 'pipeline.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        self.logger.info("Initializing Invoice Processing Pipeline")
        try:
            self.processor = InvoiceProcessor(api_key, output_dir)
            self.db_config = db_config
            self.logger.info("Pipeline initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {str(e)}", exc_info=True)
            raise
    
    def process_invoice(self, invoice_path: str):
        """Process a invoice and print the results."""
        self.logger.info(f"Starting to process invoice: {invoice_path}")
        try:
            result = self.processor.process_single_invoice(invoice_path)
            self.logger.info("Successfully processed invoice")
            self.logger.info(f"Extracted data: {json.dumps(result, indent=2)}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process invoice {invoice_path}: {str(e)}", exc_info=True)
            return None

if __name__ == "__main__":
    try:
        # Initialize the processor with your API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logging.error("ANTHROPIC_API_KEY environment variable not set")
            exit(1)
        
        # Process a single invoice
        invoice_path = "Datasets/PDF_Invoice_Folder/invoice_Aaron Bergman_36258.pdf"
        pipeline = InvoiceProcessingPipeline(api_key)
        pipeline.process_invoice(invoice_path)
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        exit(1)
