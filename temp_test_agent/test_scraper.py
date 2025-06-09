import asyncio
import csv
import json
import logging
from io import StringIO
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from your_module import (
    fetch_page,
    parse_products,
    save_to_csv,
    save_to_json,
    scrape_site,
)

# Replace 'your_module' with the actual module name where the code is located


@pytest.mark.asyncio
async def test_fetch_page_success():
    session = AsyncMock()
    response = AsyncMock()
    response.status = 200
    response.text.return_value = "HTML content"
    session.get.return_value.__aenter__.return_value = response

    result = await fetch_page(session, "http://example.com")
    assert result == "HTML content"


@pytest.mark.asyncio
async def test_fetch_page_http_error():
    session = AsyncMock()
    response = AsyncMock()
    response.status = 404
    session.get.return_value.__aenter__.return_value = response

    result = await fetch_page(session, "http://example.com")
    assert result == ""


@pytest.mark.asyncio
async def test_fetch_page_network_error():
    session = AsyncMock()
    session.get.side_effect = aiohttp.ClientError("Network error")
    result = await fetch_page(session, "http://example.com")
    assert result == ""


@pytest.mark.asyncio
async def test_parse_products_returns_sample():
    html = "<html></html>"
    result = await parse_products(html)
    assert result == [
        {
            "name": "Sample Product",
            "price": 19.99,
            "rating": 4.5,
            "image_url": "https://example.com/image.jpg",
        }
    ]


@pytest.mark.asyncio
async def test_scrape_site_scrapes_two_pages():
    base_url = "http://example.com"
    max_depth = 2
    rate_limit = 1

    with patch("your_module.fetch_page") as mock_fetch:
        mock_fetch.side_effect = lambda session, url: (
            "html1"
            if url == base_url
            else "html2" if url == f"{base_url}/page-1" else ""
        )

        with patch("your_module.parse_products") as mock_parse:
            mock_parse.return_value = [{"name": "Product A"}]

            products = await scrape_site(base_url, max_depth, rate_limit)
            assert len(products) == 2


@pytest.mark.asyncio
async def test_scrape_site_respects_max_depth():
    base_url = "http://example.com"
    max_depth = 1
    rate_limit = 1

    with patch("your_module.fetch_page") as mock_fetch:
        mock_fetch.side_effect = lambda session, url: "html" if url == base_url else ""

        with patch("your_module.parse_products") as mock_parse:
            mock_parse.return_value = [{"name": "Product A"}]

            products = await scrape_site(base_url, max_depth, rate_limit)
            assert len(products) == 1


def test_save_to_csv(tmpdir):
    products = [{"name": "Product A", "price": 19.99}]
    filename = tmpdir.join("products.csv")
    save_to_csv(products, filename)
    with open(filename, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["name"] == "Product A"
        assert rows[0]["price"] == "19.99"


def test_save_to_json(tmpdir):
    products = [{"name": "Product A", "price": 19.99}]
    filename = tmpdir.join("products.json")
    save_to_json(products, filename)
    with open(filename, "r") as f:
        data = json.load(f)
        assert data == products


def test_save_to_csv_empty_products():
    with pytest.raises(IndexError):
        save_to_csv([], "file.csv")


def test_save_to_json_empty_products():
    with pytest.raises(IndexError):
        save_to_json([], "file.json")
