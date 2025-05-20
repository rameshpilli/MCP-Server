import logging
import os
from app.tools.clientview_financials import (
    RisersDecliners,
    ClientValueByProduct,
    ClientValueByTimePeriod,
    CCYEnum,
    RDSortingCriteriaEnum,
)


def main():
    """Run ClientView financial tool tests against the dummy server."""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("clientview_demo")

    base_url = os.getenv("CLIENTVIEW_BASE_URL", "http://localhost:8001")
    logger.info("Using ClientView base URL: %s", base_url)

    # RisersDecliners demo
    logger.info("\n-- RisersDecliners --")
    rd = RisersDecliners(
        ccy_code=CCYEnum.usd,
        sorting_criteria=RDSortingCriteriaEnum.top,
    )
    result = rd.execute()
    logger.info("status: %s", result.get("status"))
    for i, row in enumerate(result.get("data", [])[:3]):
        logger.info("row %d: %s", i + 1, row)
    df = rd.display()
    logger.info("\nDataFrame:\n%s", df.head().to_string(index=False))

    # ClientValueByProduct demo
    logger.info("\n-- ClientValueByProduct --")
    cvbp = ClientValueByProduct(ccy_code=CCYEnum.usd, client_cdrid=34960)
    result = cvbp.execute()
    logger.info("status: %s", result.get("status"))
    for i, row in enumerate(result.get("data", [])[:3]):
        logger.info("row %d: %s", i + 1, row)
    df = cvbp.display()
    logger.info("\nDataFrame:\n%s", df.head().to_string(index=False))

    # ClientValueByTimePeriod demo
    logger.info("\n-- ClientValueByTimePeriod --")
    cvbt = ClientValueByTimePeriod(ccy_code=CCYEnum.usd, time_period_year=2025)
    result = cvbt.execute()
    logger.info("status: %s", result.get("status"))
    for i, row in enumerate(result.get("data", [])[:3]):
        logger.info("row %d: %s", i + 1, row)
    df = cvbt.display()
    logger.info("\nDataFrame:\n%s", df.head().to_string(index=False))


if __name__ == "__main__":
    main()
