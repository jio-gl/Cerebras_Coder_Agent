import argparse
import asyncio
import csv
import json
import logging
from typing import Any, Dict, List

import aiohttp

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def fetch_page(session: aiohttp.ClientSession, url: str) -> str:
    """Fetch HTML content from the given URL using an aiohttp session.

    Args:
        session (aiohttp.ClientSession): The aiohttp session to use.
        url (str): The URL to fetch.

    Returns:
        str: The HTML content, or empty string if the request fails.
    """
    try:
        async with session.get(url, timeout=10) as response:
            if response.status == 200:
                return await response.text()
            else:
                logger.warning(f"Failed to fetch {url}: Status {response.status}")
    except aiohttp.ClientError as e:
        logger.error(f"Network error for {url}: {e}")
    return ""


async def parse_products(html: str) -> List[Dict[str, Any]]:
    """Parse product data from HTML content (placeholder implementation).

    Args:
        html (str): The HTML content to parse.

    Returns:
        List[Dict[str, Any]]: A list of product dictionaries with sample data.
    """
    return [
        {
            "name": "Sample Product",
            "price": 19.99,
            "rating": 4.5,
            "image_url": "https://example.com/image.jpg",
        }
    ]


async def scrape_site(base_url: str, max_depth: int, rate_limit: int):
    """Scrape e-commerce site using breadth-first search up to specified depth.

    Args:
        base_url (str): The root URL of the site to scrape.
        max_depth (int): Maximum number of pages to scrape.
        rate_limit (int): Maximum concurrent requests per host.

    Returns:
        List[Dict[str, Any]]: List of scraped product data.
    """
    connector = aiohttp.TCPConnector(limit_per_host=rate_limit)
    async with aiohttp.ClientSession(connector=connector) as session:
        visited = set()
        queue = [base_url]
        products = []

        while queue and len(visited) < max_depth:
            url = queue.pop(0)
            if url in visited:
                continue

            logger.info(f"Scraping {url}")
            html = await fetch_page(session, url)
            if not html:
                continue

            try:
                new_products = await parse_products(html)
                products.extend(new_products)

                # Extract links for further scraping (placeholder logic)
                new_url = f"{base_url}/page-{len(visited) + 1}"
                if new_url not in visited and new_url.startswith(base_url):
                    queue.append(new_url)
            except Exception as e:
                logger.error(f"Error parsing {url}: {e}")

            visited.add(url)

        return products


def save_to_csv(products: List[Dict[str, Any]], filename: str):
    """Save product data to a CSV file.

    Args:
        products (List[Dict[str, Any]]): The product data to save.
        filename (str): The output CSV filename.
    """
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=products[0].keys())
        writer.writeheader()
        writer.writerows(products)
    logger.info(f"Saved {len(products)} products to {filename}")


def save_to_json(products: List[Dict[str, Any]], filename: str):
    """Save product data to a JSON file.

    Args:
        products (List[Dict[str, Any]]): The product data to save.
        filename (str): The output JSON filename.
    """
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(products, f, indent=2)
    logger.info(f"Saved {len(products)} products to {filename}")


def main():
    """Parse command-line arguments and run the web scraping process.

    Configures logging, scrapes the specified site, and saves results in CSV/JSON format.
    """
    parser = argparse.ArgumentParser(
        description="Web scraping utility for e-commerce sites."
    )
    parser.add_argument(
        "--url", type=str, required=True, help="Base URL of the e-commerce site"
    )
    parser.add_argument(
        "--depth", type=int, default=5, help="Maximum number of pages to scrape"
    )
    parser.add_argument(
        "--rate-limit",
        type=int,
        default=3,
        help="Maximum number of concurrent requests",
    )
    parser.add_argument("--output-csv", type=str, help="Output file for CSV format")
    parser.add_argument("--output-json", type=str, help="Output file for JSON format")
    parser.add_argument(
        "--verbose", action="store_true", help="Increase output verbosity"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info(
        f"Starting scrape of {args.url} with max depth {args.depth} and rate limit {args.rate_limit}"
    )

    products = asyncio.run(scrape_site(args.url, args.depth, args.rate_limit))

    if products:
        if args.output_csv:
            save_to_csv(products, args.output_csv)
        if args.output_json:
            save_to_json(products, args.output_json)
    else:
        logger.warning("No products were scraped.")


if __name__ == "__main__":
    main()
