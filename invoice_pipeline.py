import os
from typing import List, Dict, Optional
from pathlib import Path
import json
import logging
import argparse
from datetime import datetime
from invoice_processor import InvoiceProcessor
from invoice_database import InvoiceDatabase
from dotenv import load_dotenv

load_dotenv()

class InvoiceProcessingPipeline:
    def __init__(self, api_key: str, output_dir: str = "results/processed_invoices"):
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

            self.mongodb_uri = os.getenv("MONGO_URI")
            self.es_uri = os.getenv("ES_URI")
            self.database = InvoiceDatabase(
                mongodb_uri = self.mongodb_uri,
                es_uri = self.es_uri,
            )
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
            if result:
                db_result = self.database.store_invoice(result)
                self.logger.info(f"Invoice stored in database with IDs: {db_result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to process invoice {invoice_path}: {str(e)}", exc_info=True)
            return None
    
    def search_invoices(self, query: str, search_type: str = "hybrid"):
        """
        Search for invoices in the database using the specified search type.
        
        Args:
            query: Search query string
            search_type: Type of search to perform ("standard", "semantic", "hybrid")
            
        Returns:
            List of matching invoice documents
        """
        self.logger.info(f"Searching for invoices with query: '{query}', search_type: {search_type}")
        try:
            if search_type == "standard":
                results = self.database.standard_search(query)
            elif search_type == "semantic":
                results = self.database.semantic_search(query)
            else:  # Default to hybrid
                results = self.database.hybrid_search(query)
                
            self.logger.info(f"Found {len(results)} matching invoices")
            return results
        except Exception as e:
            self.logger.error(f"Failed to search invoices: {str(e)}", exc_info=True)
            return []
    
    def query_with_llm(self, query: str, search_type: str = None):
        """
        Query invoices with natural language and get an LLM-generated response.
        
        Args:
            query: Natural language query
            search_type: Override the search type (optional)
            
        Returns:
            Dict with LLM response and source documents
        """
        self.logger.info(f"Processing natural language query: '{query}'")
        try:
            result = self.database.query_with_llm(query, search_type)
            self.logger.info(f"Query processed with search type: {result['search_type']}")
            return result
        except Exception as e:
            self.logger.error(f"Failed to process query: {str(e)}", exc_info=True)
            return {
                "answer": f"An error occurred while processing your query: {str(e)}",
                "documents": [],
                "search_type": "error"
            }
    
    def get_invoice(self, identifier: str, id_type: str = "number"):
        """
        Retrieve an invoice by its identifier.
        
        Args:
            identifier: The invoice identifier (number or MongoDB ID)
            id_type: Type of identifier ("number" or "id")
            
        Returns:
            The invoice document or None if not found
        """
        self.logger.info(f"Retrieving invoice with {id_type}: {identifier}")
        try:
            if id_type == "id":
                return self.database.get_invoice_by_id(identifier)
            else:
                return self.database.get_invoice_by_number(identifier)
        except Exception as e:
            self.logger.error(f"Failed to retrieve invoice: {str(e)}", exc_info=True)
            return None
    
    def delete_invoice(self, invoice_number: str):
        """
        Delete an invoice by its number.
        
        Args:
            invoice_number: The invoice number to delete
            
        Returns:
            Boolean indicating success
        """
        self.logger.info(f"Deleting invoice with number: {invoice_number}")
        try:
            result = self.database.delete_invoice(invoice_number)
            if result:
                self.logger.info(f"Successfully deleted invoice {invoice_number}")
            else:
                self.logger.warning(f"Failed to delete invoice {invoice_number}")
            return result
        except Exception as e:
            self.logger.error(f"Error deleting invoice: {str(e)}", exc_info=True)
            return False
        
if __name__ == "__main__":
    try:
        # Initialize the processor with your API key
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            logging.error("ANTHROPIC_API_KEY environment variable not set")
            exit(1)
        
        # Process a single invoice
        invoice_path = "Datasets/PDF_Invoice_Folder/invoice_Alan Shonely_31902.pdf"
        pipeline = InvoiceProcessingPipeline(api_key)

        # Set up argument parser
        parser = argparse.ArgumentParser(description='Invoice Processing Pipeline')
    
        # Create subparsers for different commands
        subparsers = parser.add_subparsers(dest='command', help='Command to execute')
        
        # Process a single invoice
        process_parser = subparsers.add_parser('process', help='Process a single invoice')
        # process_parser.add_argument('invoice_path', help='Path to the invoice PDF file')
        
        # Search for invoices
        search_parser = subparsers.add_parser('search', help='Search for invoices')
        search_parser.add_argument('query', help='Search query string')
        search_parser.add_argument('--type', choices=['standard', 'semantic', 'hybrid'], 
                                default='hybrid', help='Search type (default: hybrid)')
        # search_parser.add_argument('--limit', type=int, default=10, help='Maximum number of results')
        
        # Query with natural language
        query_parser = subparsers.add_parser('query', help='Natural language query with LLM response')
        query_parser.add_argument('question', help='Natural language question')
        query_parser.add_argument('--type', choices=['standard', 'semantic', 'hybrid'], 
                                help='Override search type')
        
        # Get an invoice by number or ID
        get_parser = subparsers.add_parser('get', help='Get an invoice by number or ID')
        get_parser.add_argument('identifier', help='Invoice number or MongoDB ID')
        get_parser.add_argument('--type', choices=['number', 'id'], default='number', 
                            help='Identifier type (default: number)')
        
        # Delete an invoice
        delete_parser = subparsers.add_parser('delete', help='Delete an invoice')
        delete_parser.add_argument('invoice_number', help='Invoice number to delete')

        args = parser.parse_args()

        # If no command provided, show help
        if args.command is None:
            parser.print_help() 
        elif args.command == 'process': # Perform actions based on arguments
            result = pipeline.process_invoice(invoice_path)
            if result:
                print(json.dumps(str(result), indent=2))
                print(f"Invoice processed successfully and stored in database")
            else:
                print("Failed to process invoice")
        elif args.command == 'search':
            results = pipeline.search_invoices(args.query, search_type=args.type)
            print(f"Found {len(results)} matching invoices:")
            print(json.dumps(results, indent=2))
            # for i, doc in enumerate(results):
            #     print(f"\nResult {i+1}:")
            #     print(f"Invoice Number: {doc.get('invoice_number')}")
            #     print(f"Order ID: {doc.get('order_id')}")
            #     print(f"Billing Name: {doc.get('billing', {}).get('name')}")
            #     if '_score' in doc:
            #         print(f"Relevance Score: {doc.get('_score')}")
            #     print("-" * 40)
        elif args.command == 'query':
            response = pipeline.query_with_llm(args.question, search_type=args.type)
            print("\n" + "=" * 80)
            print("QUESTION:")
            print(args.question)
            print("\nANSWER:")
            print(response['answer'])
            print("\nBased on search type: " + response['search_type'])
            print("=" * 80)
            
            # Print source documents if any
            if response['documents']:
                print(f"\nBased on {len(response['documents'])} source documents:")
                for i, doc in enumerate(response['documents'][:3]):
                    print(f"\nSource {i+1}: Invoice {doc.get('invoice_number')} for {doc.get('billing', {}).get('name')}")
        elif args.command == 'get':
            invoice = pipeline.get_invoice(args.identifier, id_type=args.type)
            if invoice:
                print(json.dumps(invoice, indent=2))
            else:
                print(f"No invoice found with {args.type} {args.identifier}")
        elif args.command == 'delete':
            success = pipeline.delete_invoice(args.invoice_number)
            if success:
                print(f"Invoice {args.invoice_number} deleted successfully")
            else:
                print(f"Failed to delete invoice {args.invoice_number}")
        else:
            print("No valid action provided. Use --help for usage information.")
    except Exception as e:
        logging.error(f"Pipeline execution failed: {str(e)}", exc_info=True)
        exit(1)