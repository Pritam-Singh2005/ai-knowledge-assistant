def build_context(documents):
    """
    Build a combined context string from reranked documents.
    
    Args:
        documents: List of document objects or dictionaries
        
    Returns:
        str: Combined context from all documents
    """
    
    if not documents:
        return ""
    
    context_parts = []
    
    for doc in documents:
        
        # Extract text from different document formats
        text = ""
        
        if doc is None:
            continue
            
        # String format
        if isinstance(doc, str):
            text = doc
            
        # Dictionary format
        elif isinstance(doc, dict):
            for key in ["text", "content", "document", "page_content"]:
                if key in doc and doc[key]:
                    text = str(doc[key])
                    break
                    
        # LangChain-style document
        elif hasattr(doc, "page_content"):
            text = str(doc.page_content)
            
        # Generic object
        else:
            text = str(doc)
        
        # Add non-empty text to context
        if text and text.strip():
            context_parts.append(text.strip())
    
    # Join all parts with separator
    return "\n\n".join(context_parts)