# --- FastAPI + Helius Backend for Solana Risk Checker ---
from fastapi import FastAPI
import os
import time
import httpx
from solana.rpc.async_api import AsyncClient
import base64

app = FastAPI()

# -------------------------
# Environment & Config
# -------------------------
HELIUS_API_KEY = os.getenv("HELIUS_API_KEY")  # Set this in Railway or .env
HELIUS_BASE_URL = "https://api-devnet.helius.xyz/v0"  # Helius REST API (devnet)
HELIUS_RPC = f"https://devnet.helius-rpc.com/?api-key={HELIUS_API_KEY}"  # Helius RPC for simulation

# In-memory cache to avoid hitting Helius too often
CACHE = {}
CACHE_TTL = 300  # seconds (5 min)

# -------------------------
# Health Check Endpoint
# -------------------------
@app.get("/health")
def health():
    """
    Simple endpoint to verify server is running.
    """
    return {"ok": True}

# -------------------------
# Fetch transactions from Helius
# -------------------------
async def get_transactions(address: str):
    """
    Fetches Solana transactions for a given address using the Helius Enhanced Transactions API.
    Uses caching to reduce API calls.
    """
    # Return cached data if valid
    if address in CACHE and time.time() - CACHE[address]["timestamp"] < CACHE_TTL:
        return CACHE[address]["data"]

    # Build Helius REST URL
    url = f"{HELIUS_BASE_URL}/addresses/{address}/transactions"
    params = {"api-key": HELIUS_API_KEY}

    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"[ERROR] Helius request failed: {e}")
            return []

    # Cache successful response
    CACHE[address] = {"timestamp": time.time(), "data": data}
    return data

# -------------------------
# Extract heuristic features
# -------------------------
def extract_features_from_helius(data, address: str):
    """
    Convert Helius transaction data into simple features for risk scoring.
    """
    tx_count = len(data)
    incoming, outgoing = 0, 0
    last_tx_timestamp = None

    for tx in data:
        for t in tx.get("nativeTransfers", []):
            if t.get("toUserAccount") == address:
                incoming += 1
            elif t.get("fromUserAccount") == address:
                outgoing += 1
        ts = tx.get("timestamp")
        if ts and (last_tx_timestamp is None or ts > last_tx_timestamp):
            last_tx_timestamp = ts

    incoming_ratio = incoming / tx_count if tx_count > 0 else 0
    recent_days = (time.time() - last_tx_timestamp) / 86400 if last_tx_timestamp else None

    return {
        "tx_count": tx_count,
        "incoming_tx_count": incoming,
        "outgoing_tx_count": outgoing,
        "incoming_ratio": incoming_ratio,
        "recent_activity_days": recent_days,
    }

# -------------------------
# Risk Scoring Endpoint
# -------------------------
@app.post("/score")
async def score(body: dict):
    """
    POST /score
    Input: {"address": "SolanaAddress"}
    Output: {"score": float, "label": str, "features": {...}}

    Fetches address history via Helius REST,
    extracts simple heuristic features,
    and returns a basic risk score.
    """
    address = body.get("address")
    if not address:
        return {"error": "Missing 'address' field in request."}

    # Step 1: Get transactions
    tx_data = await get_transactions(address)
    if not tx_data:
        return {"error": "Failed to fetch transaction data from Helius."}

    # Step 2: Compute heuristic features
    features = extract_features_from_helius(tx_data, address)

    # Step 3: Basic risk score (replace later with LLM + heuristics)
    score = min(1.0, 0.1 + 0.8 * features["incoming_ratio"])
    label = "high" if score > 0.65 else "medium" if score > 0.35 else "low"

    # Step 4: Return structured JSON
    return {
        "score": round(score, 2),
        "label": label,
        "features": features,
    }

# -------------------------
# Transaction Simulation Endpoint
# -------------------------
@app.post("/simulate")
async def simulate(body: dict):
    """
    POST /simulate
    Input: {"tx": "<base64-encoded transaction>"}
    Output: simulation logs, units consumed, and errors (if any)

    Uses Helius RPC to simulate a transaction before sending.
    """
    encoded_tx = body.get("tx")
    if not encoded_tx:
        return {"error": "Missing 'tx' field — expected base64-encoded transaction"}

    try:
        async with AsyncClient(HELIUS_RPC) as client:
            tx_bytes = base64.b64decode(encoded_tx)
            resp = await client.simulate_transaction(tx_bytes)

        return {
            "logs": resp.value.logs if resp.value else [],
            "units_consumed": resp.value.units_consumed if resp.value else None,
            "err": resp.value.err if resp.value else None,
        }

    except Exception as e:
        print(f"[ERROR] Simulation failed: {e}")
        return {"error": str(e)}
