from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Any
import random

app = FastAPI(title="Dummy Financial Service")

class RequestPayload(BaseModel):
    appCode: str
    values: List[Any]

# In-memory tables
TOP_CLIENTS = []
REVENUE_BY_TIME = []
CLIENT_VALUE_BY_PRODUCT = []

REGIONS = ["CAN", "USA", "EUR", "APAC", "LATAM", "OTHER"]
FOCUS_LISTS = ["Focus40", "FS30", "Corp100"]
PRODUCTS = ["Bonds", "Equities", "FX", "Derivatives", "Commodities"]


def _generate_tables():
    random.seed(0)
    for i in range(150):
        TOP_CLIENTS.append({
            "ClientName": f"Client {i}",
            "ClientCDRID": 1000 + i,
            "RevenueYTD": round(random.uniform(1_000_000, 10_000_000), 2),
            "RegionName": random.choice(REGIONS),
            "FocusList": random.choice(FOCUS_LISTS),
            "InteractionCMOCYTD": random.randint(0, 20),
            "InteractionGMOCYTD": random.randint(0, 20),
            "InteractionYTD": random.randint(0, 40),
            "InteractionCMOCPrevYTD": random.randint(0, 20),
            "InteractionGMOCPrevYTD": random.randint(0, 20),
            "InteractionPrevYTD": random.randint(0, 40),
        })

        REVENUE_BY_TIME.append({
            "ClientName": f"Client {i}",
            "ClientCDRID": 1000 + i,
            "RevenueYTD": round(random.uniform(1_000_000, 10_000_000), 2),
            "RevenuePrevYTD": round(random.uniform(500_000, 9_000_000), 2),
            "InteractionCMOCYTD": random.randint(0, 20),
            "InteractionGMOCYTD": random.randint(0, 20),
            "InteractionYTD": random.randint(0, 40),
            "TimePeriodList": [2023, 2024, 2025],
            "TimePeriodCategory": random.choice(["FY", "CY"]),
        })

        CLIENT_VALUE_BY_PRODUCT.append({
            "ProductName": random.choice(PRODUCTS),
            "RevenueYTD": round(random.uniform(500_000, 5_000_000), 2),
            "RevenuePrevYTD": round(random.uniform(500_000, 5_000_000), 2),
            "ProductID": 2000 + i,
            "ProductHierarchyDepth": random.randint(1, 3),
            "ParentProductID": random.randint(1000, 1999),
            "TimePeriodList": [2023, 2024, 2025],
        })

_generate_tables()

@app.post("/procedure/memsql__client1__getTopClients")
def get_top_clients(_: RequestPayload):
    return {"status": "success", "data": TOP_CLIENTS}

@app.post("/procedure/memsql__client1__getRevenueTotalByTimePeriod")
def get_revenue_by_time(_: RequestPayload):
    return {"status": "success", "data": REVENUE_BY_TIME}

@app.post("/procedure/memsql__client1__getClientValueRevenueByProduct")
def get_client_value_by_product(_: RequestPayload):
    return {"status": "success", "data": CLIENT_VALUE_BY_PRODUCT}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
