"""
Tests for Brevit.py
"""
import pytest
import asyncio
from src.brevit import (
    BrevitClient,
    BrevitConfig,
    JsonOptimizationMode,
    TextOptimizationMode,
    ImageOptimizationMode
)


@pytest.mark.asyncio
async def test_flatten_json_object():
    """Test flattening a JSON object."""
    config = BrevitConfig(json_mode=JsonOptimizationMode.Flatten)
    brevit = BrevitClient(config)
    
    test_object = {
        "user": {
            "name": "Javian",
            "email": "support@javianpicardo.com"
        }
    }
    
    result = await brevit.optimize(test_object)
    
    assert "user.name:Javian" in result
    assert "user.email:support@javianpicardo.com" in result


@pytest.mark.asyncio
async def test_flatten_json_string():
    """Test flattening a JSON string."""
    config = BrevitConfig(json_mode=JsonOptimizationMode.Flatten)
    brevit = BrevitClient(config)
    
    json_string = '{"order": {"orderId": "o-456", "status": "SHIPPED"}}'
    result = await brevit.optimize(json_string)
    
    assert "order.orderId:o-456" in result
    assert "order.status:SHIPPED" in result


@pytest.mark.asyncio
async def test_short_text_returns_as_is():
    """Test that short text is returned as-is."""
    config = BrevitConfig(long_text_threshold=500)
    brevit = BrevitClient(config)
    
    short_text = "Hello World"
    result = await brevit.optimize(short_text)
    
    assert result == "Hello World"


@pytest.mark.asyncio
async def test_array_handling():
    """Test array flattening."""
    config = BrevitConfig(json_mode=JsonOptimizationMode.Flatten)
    brevit = BrevitClient(config)
    
    test_object = {
        "items": [
            {"sku": "A-88", "name": "Brevit Pro"},
            {"sku": "T-22", "name": "Toon Handbook"}
        ]
    }
    
    result = await brevit.optimize(test_object)
    
    # Tabular format is now used for uniform arrays
    assert "items[2]" in result or "items[0].sku:A-88" in result
    assert ("A-88" in result and "T-22" in result) or ("Brevit Pro" in result and "Toon Handbook" in result)


@pytest.mark.asyncio
async def test_nested_objects():
    """Test deeply nested objects."""
    config = BrevitConfig(json_mode=JsonOptimizationMode.Flatten)
    brevit = BrevitClient(config)
    
    test_object = {
        "order": {
            "customer": {
                "contact": {
                    "email": "test@example.com"
                }
            }
        }
    }
    
    result = await brevit.optimize(test_object)
    
    assert "order.customer.contact.email:test@example.com" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

