# AI-Assisted Location Decision Support System (ALSDS) — Instructor Baseline

This repository provides the **baseline application and infrastructure** for the ALSDS capstone project.

It includes:

- Core **data files** for Worcester, MA (CBGs, POIs, visits, parameters)
- A **baseline Huff-style model implementation** (`huff_engine.py`)
- A **Flask-based web application** with:
  - Guided chatbot (Azure OpenAI / GPT-4o)
  - Interactive map (Leaflet + GeoJSON)
  - Model execution API
- Azure deployment configuration (App Service compatible)


## Important Setup Instructions

### This is NOT a fork-based workflow

Each team must:

- Create a NEW repository in your GitHub Organization
- Copy (clone or download/upload) this repository into your repo


**Do NOT fork this repository.**



## Required Repository Naming

Each team repository must be named:


alsds-teamX-app


Example:


alsds-team3-app



## Baseline Version


Stable infrastructure release: v1-baseline


All teams must start from this version before making any changes.



## Do NOT Modify (Critical Infrastructure)

The following files must remain unchanged:

startup.sh
requirements.txt (do not remove any current requirements, you can add more items if required.)
app.py (initially, unless explicitly instructed)
deployment configuration (GitHub Actions / Azure settings)
environment variable naming
API route structure (/api/run_huff, /api/ask)


These are required for:
- Azure deployment
- database connectivity
- OpenAI integration
- instructor testing


## What Teams ARE Expected to Modify

Teams should focus on:


- huff_engine.py
- database design (Azure SQL)
- data preprocessing / optimization
- optional UI enhancements


## Huff Model Interface Requirement

Your implementation must preserve this function:

`def run_huff_model(candidate_lat, candidate_lon, business_category, floor_area, db_connection):`

Return format must include:

predicted_visits
market_share
competitors
runtime_ms
notes

You are free to completely redesign the internal logic.

## Map Integration

The app uses:

static/data/worcester_cbgs_map.geojson

Ensure this file exists and is not removed.

### The map supports:

- click-to-select candidate location
- competitor visualization

## Chatbot Behavior

The chatbot will:

1. Guide the user to input:
- business category
- location (map click or coordinates)
- floor area
2. Run the model
3. Provide explanation
4. Answer follow-up questions


## Deployment Workflow

After your repository is ready:

1. Instructor connects your repo to Azure Web App
2. GitHub Actions handles deployment
3. App is available at your Azure URL

## Required Endpoints

Your deployed app must support:

`/health` → returns `{"status":"ok"}`

`/dbcheck` → verifies database connection

`/`  → loads UI (chat + map)

## Evaluation Focus

Your project will be evaluated on:

- Model quality (Huff implementation)
- Database design and efficiency
- Query performance / runtime
- Code organization
- Ability to explain results (chatbot)

## Notes
- Keep your repository clean and well-structured
- Commit changes regularly
- Create a `dev` branch and test locally before deployment

