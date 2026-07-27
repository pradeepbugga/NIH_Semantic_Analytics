from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

import logging
from core.logging_config import configure_logging

configure_logging()

import app.startup as startup

from core.services.search_service import search as search_service
from core.services.activity_service import get_activity_portfolio
from core.services.agency_service import get_agency_portfolio
from core.services.portfolio_service import (
    get_category_distribution,
    get_grants_by_category,
    search_portfolio,
    SearchRequest,
)
from core.services.grant_service import fetch_grant_abstract

from core.search.modal_reranker import distributed_rerank_fn, rerank_fn

load_dotenv()


# ===============================================================
# FRAMEWORK INITIALIZATION AND ROUTER SETUP
# ===============================================================
app = FastAPI(title="NIH Grant Search API", lifespan=startup.application_lifespan)

logger = logging.getLogger(__name__)

app.mount("/static", StaticFiles(directory="./app/static"), name="static")
templates = Jinja2Templates(directory="./app/templates")


# ===============================================================
# API APPLICATION CORE ENDPOINTS
# ===============================================================
@app.get("/", response_class=HTMLResponse)
def home_page(request: Request):
    """
    Renders the home page with a search bar and agency/activity code filters.

    Parameters
    ----------
    request : Request
        The FastAPI request object, used to pass context to the template.

    Returns
    -------
    TemplateResponse
        A Jinja2 template response rendering the home page with the search bar and filters.
    """

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "agencies": startup.GLOBAL_AGENCIES_LIST,
            "valid_activity_codes": startup.GLOBAL_VALID_ACTIVITY_CODES,
        },
    )


@app.get("/portfolio")
def portfolio_page(request: Request):
    """
    Renders the portfolio page (global analysis of NIH grants).

    Parameters
    ----------
    request : Request
        The FastAPI request object, used to pass context to the template.

    Returns
    -------
    TemplateResponse
        A Jinja2 template response rendering the portfolio page.
    """

    return templates.TemplateResponse("portfolio.html", {"request": request})


@app.get("/categories")
def categories_page(request: Request):
    """
    Renders the categories page (look-up table for category definitions).

    Parameters
    ----------
    request : Request
        The FastAPI request object, used to pass context to the template.

    Returns
    -------
    TemplateResponse
        A Jinja2 template response rendering the categories page.

    """
    return templates.TemplateResponse("categories.html", {"request": request})


@app.get("/contact_us")
def contact_page(request: Request):
    """
    Renders the contact us page for user inquiries.

    Parameters
    ----------
    request : Request
        The FastAPI request object, used to pass context to the template.

    Returns
    -------
    TemplateResponse
        A Jinja2 template response rendering the contact us page.
    """

    return templates.TemplateResponse("contact_us.html", {"request": request})


@app.get("/search")
async def search(
    request: Request, query: str = Query(..., description="Search query string")
):
    """
    Handles the search endpoint, performing a hybrid search over NIH grant documents.

    Parameters
    ----------

    request : Request
        The FastAPI request object, used to pass context to the template.
    query : str
        The search query string provided by the user.

    Returns
    -------
    TemplateResponse
        A Jinja2 template response rendering the search results page with ranked grants and visualizations.
    """

    try:
        context = await search_service(
            query=query,
            rerank_fn=distributed_rerank_fn,
            synonym_registry=startup.GLOBAL_SYNONYM_REGISTRY,
        )
    except Exception:
        logger.exception("Search request failed")
        raise HTTPException(
            status_code=500,
            detail="Search pipeline execution failed.",
        )

    context["request"] = request

    return templates.TemplateResponse("results.html", context)


@app.get("/activity_codes", response_class=HTMLResponse)
async def activity_codes_multi_search(
    request: Request,
    codes: str = Query(
        ..., description="Comma-separated 3-character NIH Activity codes"
    ),
):
    """
    Handles the activity codes search endpoint, retrieving and aggregating grants based on a list of NIH activity codes.

    Parameters
    ----------
    request : Request
        The FastAPI request object, used to pass context to the template.
    codes : str
        A comma-separated string of NIH activity codes (e.g., "R01,R21,R03") to filter grants by.

    Returns
    -------
    TemplateResponse
        A Jinja2 template response rendering the search results page with ranked grants and visualizations for the specified activity codes.
    """

    try:

        code_list = [c.strip().upper() for c in codes.split(",") if c.strip()]
        context = await get_activity_portfolio(
            code_list, code_registry=startup.GLOBAL_VALID_ACTIVITY_CODES
        )

    except ValueError as e:
        logger.exception("Invalid activity code(s) provided")
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    except Exception:
        logger.exception("Activity code search failed")
        raise HTTPException(
            status_code=500,
            detail="Database lookup error.",
        )

    context["request"] = request

    return templates.TemplateResponse("results.html", context)


@app.get("/agency/{agency_code}", response_class=HTMLResponse)
async def agency_portal(request: Request, agency_code: str):
    """
    Handles the agency portal endpoint, retrieving and aggregating grants based on a specific NIH agency code.

    Parameters
    ----------
    request : Request
        The FastAPI request object, used to pass context to the template.

    agency_code : str
        The NIH agency code (e.g., "NCI", "NIAID") to filter grants by.

    Returns
    -------
    TemplateResponse
        A Jinja2 template response rendering the search results page with ranked grants and visualizations for the specified agency.
    """

    try:
        context = await get_agency_portfolio(
            agency_code, code_registry=startup.GLOBAL_AGENCIES_LIST
        )
    except Exception:
        logger.exception("Agency search failed")
        raise HTTPException(
            status_code=500,
            detail="Database lookup error.",
        )

    context["request"] = request

    return templates.TemplateResponse("results.html", context)


@app.get("/api/portfolio/categories")
async def portfolio_categories(
    year: int = Query(..., description="Target fiscal year for categorization analysis")
):
    """
    API endpoint to retrieve the funding distribution across abstract categories for a specific fiscal year.

    Parameters
    ----------
    year : int
        The fiscal year for which to retrieve the category distribution.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - ``year``: The fiscal year used for filtering.
        - ``ontology_labels``: A list of ontology category labels.
        - ``ontology_values``: A list of total funding amounts corresponding to each ontology category.
    """

    try:
        context = await get_category_distribution(year)
    except Exception:
        logger.exception("Category distribution request failed")
        raise HTTPException(
            status_code=500,
            detail="Database lookup error.",
        )

    return context


@app.get("/api/portfolio/grants")
async def portfolio_grants(
    year: int = Query(..., description="Target fiscal year for grant retrieval"),
    category: str = Query(
        ..., description="Target abstract category for grant retrieval"
    ),
):
    """
    API endpoint to retrieve grants for a specific fiscal year and abstract category.

    Parameters
    ----------
    year : int
        The fiscal year for which to retrieve grants.
    category : str
        The abstract category for which to retrieve grants. Must be one of the valid categories defined in VALID_CATEGORY_COLUMNS.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - ``category``: The abstract category used for filtering.
        - ``year``: The fiscal year used for filtering.
        - ``grants``: A list of dictionaries, each containing detailed information about a grant, including:
            - ``grant_id``: The unique identifier of the grant.
            - ``title``: The title of the grant project.
            - ``organization``: The name of the organization receiving the grant.
            - ``pi``: The contact principal investigator's ID.
            - ``funding``: The total award amount for the grant.
            - ``agency_ic``: The agency or institute code associated with the grant.
            - ``activity_code``: The activity code associated with the grant.
            - ``summary``: A two-sentence summary of the grant project.
    Raises
    ------
    ValueError
        If the provided category is not in the list of valid categories defined in VALID_CATEGORY_COLUMNS.
    """

    try:

        context = await get_grants_by_category(year, category)

    except ValueError as e:
        logger.exception("Invalid category provided")
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )
    except Exception:
        logger.exception("Grant retrieval by category failed")
        raise HTTPException(
            status_code=500,
            detail="Database lookup error.",
        )

    return context


@app.post("/api/portfolio/grants/search")
async def portfolio_grants_search(payload: SearchRequest) -> dict:
    """
    API endpoint to perform a semantic/keyword search filter over NIH grants based on a query string, fiscal year, and optional abstract category.
    We note that this is not the initial semantic search of the entire NIH corpus,
    but rather a search filter over a pre-filtered set of grants (e.g., by year and category).

    Parameters
    ----------
    payload : SearchRequest
        A Pydantic model containing the following fields:
        - ``year``: The fiscal year for which to filter grants.
        - ``category``: An optional abstract category for filtering grants.
        - ``query``: The search query string provided by the user.
        - ``existing_ids``: A list of grant IDs to filter the search results against.
        - ``query_history_count``: An integer tracking the number of queries made, used to determine whether to perform a semantic or hybrid search.

    Returns
    -------
    dict
        A dictionary containing the following keys:
        - ``query``: The search query string.
        - ``year``: The fiscal year used for filtering.
        - ``category``: The abstract category used for filtering (if provided).
        - ``results``: A list of dictionaries, each containing detailed information about a grant, including:
            - ``grant_id``: The unique identifier of the grant.
            - ``title``: The title of the grant project.
            - ``organization``: The name of the organization receiving the grant.
            - ``pi``: The contact principal investigator's ID.
            - ``funding``: The total award amount for the grant.
            - ``agency_ic``: The agency or institute code associated with the grant.
            - ``activity_code``: The activity code associated with the grant.
            - ``summary``: A two-sentence summary of the grant project.

    Raises
    ------
    ValueError
        If the provided category is not in the list of valid categories defined in VALID_CATEGORY_COLUMNS.

    """

    try:
        return await search_portfolio(
            payload=payload,
            rerank_fn=rerank_fn,
        )

    except ValueError as e:
        logger.exception("Invalid category provided")
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception:
        logger.exception("Portfolio search failed")
        raise HTTPException(
            status_code=500,
            detail="Search pipeline execution failed.",
        )


@app.get("/api/grant/{grant_id}/abstract")
async def get_grant_abstract(grant_id: str) -> dict:
    """
    API endpoint to retrieve the abstract for a specific grant ID.

    Parameters
    ----------
    grant_id : str
        The unique identifier of the grant for which to retrieve the abstract.

    Returns
    -------
    dict
        A dictionary containing the following key:
        - ``abstract``: The abstract text of the grant. If the grant is not found, raises a 404 HTTPException.

    Raises
    ------
    HTTPException
        If the grant is not found (404) or if there is a database lookup error (500).
    """

    try:
        return await fetch_grant_abstract(grant_id)

    except ValueError:
        logger.exception("Grant not found for ID: %s", grant_id)
        raise HTTPException(
            status_code=404,
            detail="Grant not found.",
        )

    except Exception:
        logger.exception("Failed to retrieve abstract")
        raise HTTPException(
            status_code=500,
            detail="Database lookup error.",
        )
