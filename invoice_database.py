import logging
from typing import Dict, List, Optional, Any, Union
import anthropic
import os
import json
# import numpy as np
from datetime import datetime
from pymongo import MongoClient
from elasticsearch import Elasticsearch
from bson import ObjectId
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import time

load_dotenv()
logger = logging.getLogger(__name__)

class InvoiceDatabase:
    """
    Manages connections and operations for both MongoDB and Elasticsearch
    to store and search invoice data.
    """
    def __init__(self, mongodb_uri: str = None, es_uri: str = None):
        """
        Initialize database connections for both MongoDB and Elasticsearch.
        
        Args:
            mongodb_uri: Connection string for MongoDB
            es_uri: Connection string for Elasticsearch
        """       
        # Connect to MongoDB
        try:
            self.mongo_uri = mongodb_uri or os.getenv("MONGO_URI")
            self.mongo_db_name = os.getenv("MONGO_DB_NAME")
            self.mongo_collection = os.getenv("MONGO_COLLECTION")

            self.mongo_client = MongoClient(self.mongo_uri)
            self.mongo_db = self.mongo_client[self.mongo_db_name]
            self.invoices_collection = self.mongo_db[self.mongo_collection]
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
        
        # Connect to Elasticsearch
        try:
            self.es_uri = es_uri or os.getenv("ES_URI")
            self.index_name = os.getenv("ES_INDEX")
            print(f"Connecting to Elasticsearch at {self.es_uri} with index {self.index_name}")
            if not self.es_uri:
                raise ValueError("Elasticsearch URI is not provided")
            if not self.index_name: 
                raise ValueError("Elasticsearch index name is not provided")
            self.es_client = Elasticsearch(self.es_uri)
            if not self.es_client.ping():
                raise ConnectionError("Could not connect to Elasticsearch")
            logger.info("Successfully connected to Elasticsearch")
        except Exception as e:
            logger.error(f"Failed to connect to Elasticsearch: {str(e)}")
            raise
        
        # Initialize embedding model for semantic search
        try:
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("Successfully loaded embedding model")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            self.embedding_model = None
            
        # Initialize LLM client if API key provided
        self.llm_client = None
        llm_api_key = os.getenv("ANTHROPIC_API_KEY")
        if llm_api_key:
            try:
                self.llm_client = anthropic.Anthropic(api_key=llm_api_key)
                logger.info("Successfully initialized LLM client")
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {str(e)}")
        
        # Create Elasticsearch index if it doesn't exist
        self._create_es_index()
    
    def _create_es_index(self):
        """Create the Elasticsearch index with appropriate mappings."""
        try:
            if not self.es_client.indices.exists(index=self.index_name):
                # Define the mapping for invoice data
                mappings = {
                    "mappings": {
                        "properties": {
                            "invoice_number": {"type": "keyword"},
                            "order_id": {"type": "keyword"},
                            "date": {"type": "date"},
                            "billing": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "text"}
                                }
                            },
                            "shipping": {
                                "type": "object",
                                "properties": {
                                    "city": {"type": "keyword"},
                                    "state": {"type": "keyword"},
                                    "postal_code": {"type": "keyword"},
                                    "country": {"type": "keyword"},
                                    "mode": {"type": "keyword"},
                                    "cost": {"type": "float"}
                                }
                            },
                            "items": {
                                "type": "nested",
                                "properties": {
                                    "product_name": {"type": "text"},
                                    "sub_category": {"type": "keyword"},
                                    "category": {"type": "keyword"},
                                    "product_id": {"type": "keyword"},
                                    "quantity": {"type": "float"},
                                    "unit_price": {"type": "float"},
                                    "amount": {"type": "float"}
                                }
                            },
                            "financials": {
                                "type": "object",
                                "properties": {
                                    "subtotal": {"type": "float"},
                                    "total": {"type": "float"},
                                    "balance_due": {"type": "float"}
                                }
                            },
                            "mongodb_id": {"type": "keyword"},
                            "stored_at": {"type": "date"},
                            "embedding": {
                                "type": "dense_vector",
                                "dims": 384,  # Dimension for all-MiniLM-L6-v2
                                "index": True,
                                "similarity": "cosine"
                            },
                            "content_for_embedding": {"type": "text"}
                        }
                    },
                    "settings": {
                        "index": {
                            "number_of_shards": 1,
                            "number_of_replicas": 0
                        }
                    }
                }
                
                # Create the index
                self.es_client.indices.create(index=self.index_name, body=mappings)
                logger.info(f"Created Elasticsearch index: {self.index_name}")
            else:
                logger.info(f"Elasticsearch index '{self.index_name}' already exists")
                
        except Exception as e:
            logger.error(f"Failed to create Elasticsearch index: {str(e)}")
            raise
    
    def _generate_text_for_embedding(self, invoice_data: Dict) -> str:
        """
        Generate a text representation of the invoice for embedding.
        
        Args:
            invoice_data: The invoice data dictionary
            
        Returns:
            Text string representing the invoice content
        """
        text_parts = []
        
        # Add invoice identifiers
        text_parts.append(f"Invoice number: {invoice_data.get('invoice_number', '')}")
        text_parts.append(f"Order ID: {invoice_data.get('order_id', '')}")
        
        # Add date
        text_parts.append(f"Date: {invoice_data.get('date', '')}")
        
        # Add billing information
        billing = invoice_data.get('billing', {})
        if billing:
            text_parts.append(f"Billed to: {billing.get('name', '')}")
        
        # Add shipping information
        shipping = invoice_data.get('shipping', {})
        if shipping:
            ship_parts = []
            if shipping.get('city'):
                ship_parts.append(shipping['city'])
            if shipping.get('state'):
                ship_parts.append(shipping['state'])
            if shipping.get('country'):
                ship_parts.append(shipping['country'])
            if ship_parts:
                text_parts.append(f"Shipped to: {', '.join(ship_parts)}")
            if shipping.get('mode'):
                text_parts.append(f"Shipping mode: {shipping['mode']}")
        
        # Add items
        items = invoice_data.get('items', [])
        if items:
            text_parts.append("Items:")
            for item in items:
                item_text = f"{item.get('quantity', 1)} {item.get('product_name', 'Unknown product')}"
                if item.get('category'):
                    item_text += f", Category: {item['category']}"
                if item.get('sub_category'):
                    item_text += f", Sub-category: {item['sub_category']}"
                text_parts.append(item_text)
        
        # Add financial information
        financials = invoice_data.get('financials', {})
        if financials:
            if 'subtotal' in financials:
                text_parts.append(f"Subtotal: {financials['subtotal']}")
            if 'total' in financials:
                text_parts.append(f"Total: {financials['total']}")
            if 'balance_due' in financials:
                text_parts.append(f"Balance due: {financials['balance_due']}")
        
        return " ".join(text_parts)
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate an embedding vector for the given text.
        
        Args:
            text: Text to embed
            
        Returns:
            List of floats representing the embedding vector
        """
        if not self.embedding_model:
            logger.warning("Embedding model not available")
            return []
        
        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            return []
    
    def store_invoice(self, invoice_data: Dict) -> Dict:
        """
        Store invoice data in both MongoDB and Elasticsearch.
        
        Args:
            invoice_data: Processed invoice data
            
        Returns:
            Dict with MongoDB ID and Elasticsearch ID
        """
        try:
            # Add timestamp
            invoice_data["stored_at"] = datetime.utcnow().isoformat()
            
            # Store in MongoDB
            mongo_result = self.invoices_collection.insert_one(invoice_data)
            mongodb_id = str(mongo_result.inserted_id)
            logger.info(f"Invoice stored in MongoDB with ID: {mongodb_id}")
            
            # Prepare for Elasticsearch
            es_document = invoice_data.copy()
            if "_id" in es_document:
                logger.info(f"Deleting the _id field from the invoice data: {es_document['_id']}")
                del es_document["_id"]

            es_document["mongodb_id"] = mongodb_id
            
            # Generate text for embedding
            content_for_embedding = self._generate_text_for_embedding(invoice_data)
            es_document["content_for_embedding"] = content_for_embedding
            
            # Generate and add embedding
            if self.embedding_model:
                embedding = self._generate_embedding(content_for_embedding)
                if embedding:
                    es_document["embedding"] = embedding
            
            # Ensure all ObjectId fields are converted to strings
            for key, value in es_document.items():
                if isinstance(value, ObjectId):
                    es_document[key] = str(value)

            # Store in Elasticsearch
            es_result = self.es_client.index(
                index=self.index_name,
                document=es_document
            )
            es_id = es_result["_id"]
            logger.info(f"Invoice indexed in Elasticsearch with ID: {es_id}")
            
            return {
                "mongodb_id": mongodb_id,
                "elasticsearch_id": es_id
            }
            
        except Exception as e:
            logger.error(f"Failed to store invoice: {str(e)}")
            raise
    
    def get_invoice_by_id(self, mongodb_id: str) -> Optional[Dict]:
        """
        Retrieve an invoice by its MongoDB ID.
        
        Args:
            mongodb_id: The MongoDB ID string
            
        Returns:
            The invoice document or None if not found
        """
        try:
            # Convert string ID to ObjectId
            object_id = ObjectId(mongodb_id)
            
            # Query MongoDB
            result = self.invoices_collection.find_one({"_id": object_id})
            
            if result:
                # Convert ObjectId to string for JSON serialization
                result["_id"] = str(result["_id"])
                return result
            
            logger.warning(f"No invoice found with MongoDB ID: {mongodb_id}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get invoice by ID: {str(e)}")
            raise
    
    def get_invoice_by_number(self, invoice_number: str) -> Optional[Dict]:
        """
        Retrieve an invoice by its invoice number.
        
        Args:
            invoice_number: The invoice number to search for
            
        Returns:
            The invoice document or None if not found
        """
        try:
            search_body = {
                "query": {
                    "term": {
                        "invoice_number": invoice_number
                    }
                }
            }
            
            result = self.es_client.search(index=self.index_name, body=search_body)
            hits = result["hits"]["hits"]
            
            if hits:
                return hits[0]["_source"]
            
            logger.warning(f"No invoice found with invoice number: {invoice_number}")
            return None
            
        except Exception as e:
            logger.error(f"Failed to get invoice by number: {str(e)}")
            raise
    
    def standard_search(self, query: str, fields: List[str] = None, 
                      size: int = 10) -> List[Dict]:
        """
        Perform standard text search for invoices in Elasticsearch.
        
        Args:
            query: Search query string
            fields: List of fields to search in (defaults to all text fields)
            size: Maximum number of results to return
            
        Returns:
            List of matching invoice documents
        """
        try:
            if fields is None:
                fields = [
                    "invoice_number^3", 
                    "order_id^2",
                    "billing.name^3", 
                    "items.product_name^2", 
                    "items.category",
                    "shipping.city",
                    "shipping.state",
                    "shipping.country",
                    "content_for_embedding"
                ]
                
            search_body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": fields,
                        "type": "best_fields"
                    }
                },
                "size": size
            }
            
            results = self.es_client.search(index=self.index_name, body=search_body)
            hits = results["hits"]["hits"]
            
            return [hit["_source"] for hit in hits]
            
        except Exception as e:
            logger.error(f"Failed to perform standard search: {str(e)}")
            raise
    
    def semantic_search(self, query: str, size: int = 10) -> List[Dict]:
        """
        Perform semantic search using vector embeddings.
        
        Args:
            query: Natural language query
            size: Maximum number of results to return
            
        Returns:
            List of matching invoice documents
        """
        try:
            if not self.embedding_model:
                logger.warning("Embedding model not available, falling back to standard search")
                return self.standard_search(query, size=size)
            
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            if not query_embedding:
                logger.warning("Failed to generate embedding for query, falling back to standard search")
                return self.standard_search(query, size=size)
            
            # Perform vector search
            search_body = {
                "query": {
                    "script_score": {
                        "query": {"match_all": {}},
                        "script": {
                            "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                            "params": {"query_vector": query_embedding}
                        }
                    }
                },
                "size": size
            }
            
            results = self.es_client.search(index=self.index_name, body=search_body)
            hits = results["hits"]["hits"]
            
            return [hit["_source"] for hit in hits]
            
        except Exception as e:
            logger.error(f"Failed to perform semantic search: {str(e)}")
            # Fall back to standard search in case of error
            logger.info("Falling back to standard search")
            return self.standard_search(query, size=size)
    
    def hybrid_search(self, query: str, size: int = 10) -> List[Dict]:
        """
        Perform hybrid search combining text and semantic search.
        
        Args:
            query: Natural language query
            size: Maximum number of results to return
            
        Returns:
            List of matching invoice documents
        """
        try:
            if not self.embedding_model:
                logger.warning("Embedding model not available, falling back to standard search")
                return self.standard_search(query, size=size)
            
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            
            if not query_embedding:
                logger.warning("Failed to generate embedding for query, falling back to standard search")
                return self.standard_search(query, size=size)
            
            # Define fields for text search
            fields = [
                "invoice_number^3", 
                "order_id^2",
                "billing.name^3", 
                "items.product_name^2", 
                "items.category",
                "shipping.city",
                "shipping.state",
                "shipping.country",
                "content_for_embedding"
            ]
            
            # Perform hybrid search
            search_body = {
                "query": {
                    "bool": {
                        "should": [
                            # Text search component
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": fields,
                                    "type": "best_fields",
                                    "boost": 0.4
                                }
                            },
                            # Exact match boosting
                            {
                                "multi_match": {
                                    "query": query,
                                    "fields": ["invoice_number^4", "order_id^3", "billing.name^3"],
                                    "type": "phrase",
                                    "boost": 0.3
                                }
                            },
                            # Vector search component
                            {
                                "script_score": {
                                    "query": {"match_all": {}},
                                    "script": {
                                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                        "params": {"query_vector": query_embedding}
                                    },
                                    "boost": 0.7
                                }
                            }
                        ]
                    }
                },
                "size": size
            }
            
            results = self.es_client.search(index="invoices", body=search_body)
            hits = results["hits"]["hits"]
            
            # Return documents with their scores
            return [
                {
                    **hit["_source"],
                    "_score": hit["_score"]
                } for hit in hits
            ]
            
        except Exception as e:
            logger.error(f"Failed to perform hybrid search: {str(e)}")
            # Fall back to standard search in case of error
            logger.info("Falling back to standard search")
            return self.standard_search(query, size=size)
    
    def _classify_query(self, query: str) -> str:
        """
        Classify the query type to determine the best search method.
        
        Args:
            query: The user query
            
        Returns:
            Search type: "standard", "semantic", or "hybrid"
        """
        query_lower = query.lower()
        
        # Check for exact ID searches
        if "invoice number" in query_lower or "order id" in query_lower:
            return "standard"
        
        # Check for semantic searches
        semantic_indicators = [
            "similar to", "like", "related to", "about", 
            "concerning", "regarding", "pertaining to",
            "what", "who", "when", "where", "why", "how"
        ]
        
        if any(indicator in query_lower for indicator in semantic_indicators):
            return "semantic"
        
        # Default to hybrid for best results
        return "hybrid"
    
    def _format_invoices_for_context(self, invoices: List[Dict]) -> str:
        """
        Format invoice data as context for LLM.
        
        Args:
            invoices: List of invoice documents
            
        Returns:
            Formatted context string
        """
        if not invoices:
            return "No relevant invoices found."
        
        context = "Here are the details of the relevant invoices:\n\n"
        
        for i, invoice in enumerate(invoices):
            context += f"INVOICE {i+1}:\n"
            context += f"Invoice Number: {invoice.get('invoice_number', 'N/A')}\n"
            context += f"Order ID: {invoice.get('order_id', 'N/A')}\n"
            context += f"Date: {invoice.get('date', 'N/A')}\n"
            
            # Billing info
            billing = invoice.get('billing', {})
            if billing:
                context += f"Billed to: {billing.get('name', 'N/A')}\n"
            
            # Shipping info
            shipping = invoice.get('shipping', {})
            if shipping:
                ship_to = []
                if shipping.get('city'):
                    ship_to.append(shipping['city'])
                if shipping.get('state'):
                    ship_to.append(shipping['state'])
                if shipping.get('country'):
                    ship_to.append(shipping['country'])
                
                if ship_to:
                    context += f"Shipped to: {', '.join(ship_to)}\n"
                
                if shipping.get('mode'):
                    context += f"Shipping mode: {shipping['mode']}\n"
                
                if shipping.get('cost') is not None:
                    context += f"Shipping cost: ${shipping['cost']}\n"
            
            # Items
            items = invoice.get('items', [])
            if items:
                context += "Items:\n"
                for j, item in enumerate(items):
                    context += f"  - {item.get('quantity', 1)} x {item.get('product_name', 'Unknown product')}\n"
                    context += f"    Category: {item.get('category', 'N/A')}, Sub-category: {item.get('sub_category', 'N/A')}\n"
                    context += f"    Unit price: ${item.get('unit_price', 0)}, Amount: ${item.get('amount', 0)}\n"
            
            # Financial info
            financials = invoice.get('financials', {})
            if financials:
                context += "Financial Summary:\n"
                if 'subtotal' in financials:
                    context += f"  Subtotal: ${financials['subtotal']}\n"
                
                # Additional charges
                add_charges = financials.get('additional_charges', [])
                for charge in add_charges:
                    context += f"  {charge.get('description', 'Charge')}: ${charge.get('amount', 0)}\n"
                
                if 'total' in financials:
                    context += f"  Total: ${financials['total']}\n"
                if 'balance_due' in financials:
                    context += f"  Balance due: ${financials['balance_due']}\n"
            
            context += "\n"
        
        return context
    
    def query_with_llm(self, query: str, search_type: str = None) -> Dict:
        """
        Query invoices with natural language and generate a response using LLM.
        
        Args:
            query: Natural language query
            search_type: Override the search type classification
            
        Returns:
            Dict with LLM response and source documents
        """
        try:
            # Check if LLM client is available
            if not self.llm_client:
                logger.warning("LLM client not available, returning raw search results")
                return {
                    "answer": "LLM response generation is not available. Here are the raw search results.",
                    "documents": self.hybrid_search(query)
                }
            
            # Step 1: Determine search type if not provided
            if not search_type:
                search_type = self._classify_query(query)
            
            # Step 2: Execute appropriate search
            if search_type == "semantic":
                results = self.semantic_search(query)
            elif search_type == "standard":
                results = self.standard_search(query)
            else:  # Default to hybrid
                results = self.hybrid_search(query)
            
            # Step 3: Generate response with LLM
            context = self._format_invoices_for_context(results[:3])
            
            # Create the prompt for Claude
            prompt = f"""
                Human: I need information from these invoices:

                {context}

                My question is: {query}

                Please answer based only on the invoice data provided above.
                Format currency values with $ sign and proper decimals.
                If you're analyzing multiple invoices, summarize patterns or calculate totals as needed.
                If the information isn't in the provided invoices, clearly state that.
                Keep your response concise and focused on answering my question.
            """

            response = self.llm_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0,
                system="You are a helpful assistant that answers questions about invoice data based solely on the information provided. Don't make up information that isn't in the data provided.",
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
        
            # Extract response text
            answer = response.content[0].text if response.content else "Sorry, I couldn't generate a response."
            
            # Step 4: Return comprehensive result
            return {
                "answer": answer,
                "documents": results,
                "search_type": search_type
            }
        
        except Exception as e:
            logger.error(f"Failed to generate LLM response: {str(e)}")
            # Return a fallback response with raw search results
            return {
                "answer": f"An error occurred while generating a response: {str(e)}",
                "documents": self.hybrid_search(query),
                "search_type": "hybrid (fallback)"
            }

    def delete_invoice(self, invoice_number: str) -> bool:
        """
        Delete an invoice from both MongoDB and Elasticsearch using the invoice number.
        
        Args:
            invoice_number: The invoice number to delete
            
        Returns:
            Boolean indicating success
        """
        try:
            # First, get the MongoDB ID from Elasticsearch
            es_doc = self.get_invoice_by_number(invoice_number)
            
            if not es_doc or "mongodb_id" not in es_doc:
                logger.warning(f"Invoice with number {invoice_number} not found")
                return False
            
            mongodb_id = es_doc["mongodb_id"]
            
            # Delete from MongoDB using ID
            mongo_result = self.invoices_collection.delete_one({"_id": ObjectId(mongodb_id)})
            
            # Delete from Elasticsearch using invoice number
            es_result = self.es_client.delete_by_query(
                index=self.index_name,
                body={"query": {"term": {"invoice_number": invoice_number}}}
            )
            
            deleted_from_mongo = mongo_result.deleted_count > 0
            deleted_from_es = es_result.get("deleted", 0) > 0
            
            success = deleted_from_mongo and deleted_from_es
            logger.info(f"Invoice {invoice_number} deleted: {success}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to delete invoice: {str(e)}")
            raise
